"""
Variational Preference Learning (VPL) for Binary Choice Trainer
Implements VPL with variational method for binary selector.
Based on: https://github.com/WEIRDLabUW/vpl
"""
import torch
import torch.nn.functional as F
import logging
import copy
import numpy as np
import gc

from federatedscope.register import register_trainer
from federatedscope.llm.trainer.reward_choice_trainer import (
    RewardChoiceTrainer, cal_loss)
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.llm.model.variational_encoder import VariationalEncoder
import torch.nn as nn
from federatedscope.llm.dataset.llm_dataset import DefaultToken

import sys

sys.setrecursionlimit(100000)

logger = logging.getLogger(__name__)


class VPLRewardChoiceTrainer(RewardChoiceTrainer):
    """
    Variational Preference Learning trainer for binary choice tasks.
    
    Extends RewardChoiceTrainer with variational inference over user-specific
    latents. The model learns to:
    1. Infer user-specific latent z from preference data
    2. Make choices conditioned on the latent z
    3. Optimize ELBO: E_q(z|x)[log p(y|z,x)] - KL(q(z|x) || p(z))
    """
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super().__init__(model, data, device, config, only_for_eval, monitor)
        
        # VPL hyperparameters
        self.vpl_latent_dim = getattr(config.llm, 'vpl_latent_dim', 32)
        self.vpl_kl_weight = getattr(config.llm, 'vpl_kl_weight', 0.1)
        self.vpl_feature_method = getattr(config.llm, 'vpl_feature_method', 'choice_logits')
        
        # Check if using feature difference (embedding difference)
        self.vpl_use_feature_difference = getattr(config.llm, 'vpl_use_feature_difference', False)
        self.vpl_use_llm_feature_extractor = getattr(config.llm, 'vpl_use_llm_feature_extractor', True)
        self.vpl_use_difference_only = getattr(config.llm, 'vpl_use_difference_only', False)  # Use only difference embedding (no chosen/rejected)
        
        # VPL-GP hyperparameters (integrated into main trainer)
        self.vpl_use_gp_prior = getattr(config.llm, 'vpl_use_gp_prior', False)
        self.vpl_gp_temperature = getattr(config.llm, 'vpl_gp_temperature', 1.0)
        self.num_clients = getattr(config.federate, 'client_num', 10)
        
        # Orthogonal loss hyperparameters (CLOP-based)
        self.vpl_orthogonal_weight = getattr(config.llm, 'vpl_orthogonal_weight', 0.0)
        self.vpl_orthogonal_orthonorm_weight = getattr(config.llm, 'vpl_orthogonal_orthonorm_weight', 0.1)
        self.vpl_use_manual_orthogonal_labels = getattr(config.llm, 'vpl_use_manual_orthogonal_labels', False)
        
        # Initialize orthonormal prototypes if orthogonal loss is enabled
        # NOTE: Prototypes are FIXED (CLOP standard), not learnable
        if self.vpl_orthogonal_weight > 0.0:
            num_prototypes = getattr(config.llm, 'vpl_num_prototypes', self.num_clients)
            # Get prototype scale (distance from origin)
            prototype_scale = getattr(config.llm, 'vpl_prototype_scale', 5.0)  # Default: 5.0 (further from origin)
            
            # Fixed prototypes: Initialize as buffer (not updated by gradients)
            # Create orthonormal basis using identity matrix scaled by prototype_scale
            prototypes = torch.eye(num_prototypes, self.vpl_latent_dim, device=device) * prototype_scale
            # If latent_dim > num_prototypes, pad with zeros
            if self.vpl_latent_dim > num_prototypes:
                padding = torch.zeros(num_prototypes, self.vpl_latent_dim - num_prototypes, device=device)
                prototypes = torch.cat([prototypes, padding], dim=1)
            self.register_buffer('orthogonal_prototypes', prototypes)
            self.orthogonal_label = None  # Will be set by server
            logger.info(f"Initialized {num_prototypes} FIXED orthonormal prototypes for CLOP loss (scale={prototype_scale})")
        else:
            self.orthogonal_prototypes = None
            self.orthogonal_label = None
        
        # Get model dimensions
        try:
            embedding_dim = self.model.get_input_embeddings().embedding_dim
        except:
            try:
                embedding_dim = self.model.config.hidden_size
            except:
                embedding_dim = 2048  # Default for gemma-2b
        
        # Initialize feature extractor
        # Strategy: Reuse hidden_states from main forward pass, no additional forward passes
        # According to original VPL paper: encoder takes [chosen_emb, rejected_emb] or similar
        if self.vpl_use_llm_feature_extractor and self.vpl_use_feature_difference:
            if self.vpl_use_difference_only:
                # Use only difference embedding (removes general information, keeps only preference)
                # Input: embedding_dim (difference only)
                self.feature_extractor = nn.Sequential(
                    nn.Linear(embedding_dim, 512),  # difference only
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128)
                ).to(device)
                feature_extractor_output_dim = 128
                logger.info("Using projection-based feature extractor with [difference only] (removes general information, keeps only preference)")
            else:
                # Original VPL uses: concat([chosen_emb, rejected_emb]) or [chosen_emb, rejected_emb, diff]
                # We'll use: [chosen_emb, rejected_emb, chosen_emb - rejected_emb] for richer representation
                # Input: 3 * embedding_dim (chosen + rejected + difference)
                self.feature_extractor = nn.Sequential(
                    nn.Linear(embedding_dim * 3, 512),  # chosen + rejected + difference
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128)
                ).to(device)
                feature_extractor_output_dim = 128
                logger.info("Using projection-based feature extractor with [chosen, rejected, difference] (reuses hidden_states from main forward pass)")
        else:
            # Use MLP feature extractor
            if self.vpl_use_feature_difference:
                raw_feature_dim = embedding_dim
            elif self.vpl_feature_method == 'choice_logits':
                raw_feature_dim = len(self.choices) * 2
            else:
                raw_feature_dim = 2
            
            # Deep feature extraction network
            self.feature_extractor = nn.Sequential(
                nn.Linear(raw_feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128)
            ).to(device)
            feature_extractor_output_dim = 128
        
        # Store output dimension
        self.feature_extractor_output_dim = feature_extractor_output_dim
        
        # Initialize variational encoder with deeper layers
        # Use VariationalEncoderGP if GP prior is enabled, otherwise use standard VariationalEncoder
        if self.vpl_use_gp_prior:
            from federatedscope.llm.model.variational_encoder_gp import VariationalEncoderGP
            self.variational_encoder = VariationalEncoderGP(
                input_dim=self.feature_extractor_output_dim,
                latent_dim=self.vpl_latent_dim,
                hidden_dims=[512, 256, 128],
                temperature=self.vpl_gp_temperature,
                num_clients=self.num_clients
            ).to(device)
        else:
            self.variational_encoder = VariationalEncoder(
                input_dim=self.feature_extractor_output_dim,
                latent_dim=self.vpl_latent_dim,
                hidden_dims=[512, 256, 128]
            ).to(device)
        
        # Latent conditioning: project latent z to modify model behavior
        # Option 1: Add latent to embeddings
        # Option 2: Use latent to scale/adjust logits
        # We'll use Option 2: scale logits based on latent
        self.latent_projection = nn.Linear(
            self.vpl_latent_dim, 
            len(self.choices)
        ).to(device)
        
        # Initialize GP prior related attributes (only if GP prior is enabled)
        if self.vpl_use_gp_prior:
            # Z history for visualization
            self.z_history = []
            self.z_mu_history = []
            self.z_logvar_history = []
            
            # Client z distribution (average over batches)
            self.client_z_mu = None
            self.client_z_logvar = None
        
        logger.info(f'VPLRewardChoiceTrainer initialized with latent_dim={self.vpl_latent_dim}, '
                   f'kl_weight={self.vpl_kl_weight}, '
                   f'use_feature_difference={self.vpl_use_feature_difference}, '
                   f'use_llm_feature_extractor={self.vpl_use_llm_feature_extractor}, '
                   f'use_gp_prior={self.vpl_use_gp_prior}')

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        ctx.vpl_kl_loss_total = CtxVar(0.0, LIFECYCLE.ROUTINE)
        ctx.vpl_reconstruction_loss_total = CtxVar(0.0, LIFECYCLE.ROUTINE)
        if self.vpl_orthogonal_weight > 0.0:
            ctx.vpl_orthogonal_loss_total = CtxVar(0.0, LIFECYCLE.ROUTINE)

    def _extract_preference_features(self, logits, labels, choices, hidden_states=None):
        """
        Extract features from preference data for variational encoder.
        
        If vpl_use_feature_difference is True, extracts embedding difference:
        feature = positive_embedding - negative_embedding
        This removes general information and captures only preference information.
        
        Args:
            logits: Model logits (batch, seq_len, vocab_size)
            labels: Labels (batch, seq_len)
            choices: Choice token indices [A_token, B_token]
            hidden_states: Model hidden states (batch, seq_len, hidden_dim) if available
            
        Returns:
            features: Extracted features (batch, feature_dim)
            - If feature_difference: (batch, embedding_dim)
            - If choice_logits: (batch, len(choices) * 2) = (batch, 4)
        """
        if self.vpl_use_feature_difference and hidden_states is not None:
            # Extract embedding difference: positive - negative
            # This removes general information and keeps only preference information
            return self._extract_embedding_difference(hidden_states, labels, choices)
        else:
            # Fallback to original logits-based extraction
            return self._extract_logits_features(logits, labels, choices)
    
    def _extract_embedding_difference(self, hidden_states, labels, choices):
        """
        Extract preference features: [chosen_emb, rejected_emb, chosen_emb - rejected_emb]
        
        According to original VPL paper, the encoder should receive both chosen and rejected
        embeddings to capture preference information. We use:
        - chosen_emb: embedding of the chosen response
        - rejected_emb: embedding of the rejected response  
        - difference: chosen_emb - rejected_emb (removes general info, keeps preference)
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            labels: (batch, seq_len)
            choices: [A_token, B_token]
            
        Returns:
            features: (batch, hidden_dim * 3) - [chosen, rejected, difference]
            OR (batch, hidden_dim) if only difference is used
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        shift_labels = labels[..., 1:].contiguous()  # (batch, seq_len-1)
        shift_hidden = hidden_states[..., :-1, :].contiguous()  # (batch, seq_len-1, hidden_dim)
        
        # Find choice token positions
        A_token, B_token = choices[0], choices[1]
        A_positions = (shift_labels == A_token)  # (batch, seq_len-1)
        B_positions = (shift_labels == B_token)  # (batch, seq_len-1)
        
        features_list = []
        for b in range(batch_size):
            # Get embeddings at choice positions
            A_pos = A_positions[b]  # (seq_len-1,)
            B_pos = B_positions[b]  # (seq_len-1,)
            
            # Determine which choice was selected (chosen) and which was rejected
            A_found = A_pos.any()
            B_found = B_pos.any()
            
            if A_found and B_found:
                # Both choices found: determine which is chosen based on label
                # For now, use A as chosen if both exist (can be improved by checking actual choice)
                chosen_emb = shift_hidden[b, A_pos].mean(dim=0)  # (hidden_dim,)
                rejected_emb = shift_hidden[b, B_pos].mean(dim=0)  # (hidden_dim,)
            elif A_found:
                # Only A found: A is chosen, B is rejected (use mean as rejected)
                chosen_emb = shift_hidden[b, A_pos].mean(dim=0)
                rejected_emb = shift_hidden[b].mean(dim=0)  # Use mean as rejected representation
            elif B_found:
                # Only B found: B is chosen, A is rejected
                chosen_emb = shift_hidden[b, B_pos].mean(dim=0)
                rejected_emb = shift_hidden[b].mean(dim=0)  # Use mean as rejected representation
            else:
                # No choice found: use mean for both (fallback)
                chosen_emb = shift_hidden[b].mean(dim=0)
                rejected_emb = shift_hidden[b].mean(dim=0)
            
            # Compute difference: chosen - rejected (removes general info, keeps preference)
            feature_diff = chosen_emb - rejected_emb
            
            # According to original VPL: use [chosen, rejected, difference] for richer representation
            # This allows the encoder to see both responses and their difference
            # However, if vpl_use_difference_only=True, use only difference to remove general information
            if self.vpl_use_llm_feature_extractor:
                if self.vpl_use_difference_only:
                    # Use only difference (removes general information, keeps only preference)
                    feature_combined = feature_diff  # (hidden_dim,)
                else:
                    # Concatenate: [chosen_emb, rejected_emb, difference]
                    feature_combined = torch.cat([chosen_emb, rejected_emb, feature_diff], dim=0)  # (hidden_dim * 3,)
            else:
                # Use only difference for efficiency
                feature_combined = feature_diff  # (hidden_dim,)
            
            features_list.append(feature_combined)
        
        features = torch.stack(features_list, dim=0)  # (batch, hidden_dim * 3) or (batch, hidden_dim)
        
        # Ensure features are float32
        if features.dtype != torch.float32:
            features = features.float()
        
        return features
    
    def _extract_logits_features(self, logits, labels, choices):
        """
        Extract features from logits (original method).
        
        Args:
            logits: Model logits (batch, seq_len, vocab_size)
            labels: Labels (batch, seq_len)
            choices: Choice token indices
            
        Returns:
            features: Extracted features (batch, len(choices) * 2)
        """
        # Get logits at choice positions
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        batch_size = logits.shape[0]
        
        # Extract logits for each choice token
        choice_logits_per_token = shift_logits[..., choices]  # (batch, seq_len, len(choices))
        
        # Find positions where each choice appears
        features_list = []
        for choice_idx, choice_token in enumerate(choices):
            choice_positions = (shift_labels == choice_token)  # (batch, seq_len)
            
            if choice_positions.any():
                batch_features = []
                for b in range(batch_size):
                    sample_positions = choice_positions[b]
                    if sample_positions.any():
                        sample_choice_logits = choice_logits_per_token[b, sample_positions, choice_idx]
                        avg_logit = sample_choice_logits.mean()
                    else:
                        avg_logit = choice_logits_per_token[b, :, choice_idx].mean()
                    batch_features.append(avg_logit)
                features_list.append(torch.stack(batch_features))
            else:
                features_list.append(choice_logits_per_token[:, :, choice_idx].mean(dim=1))
        
        features = torch.stack(features_list, dim=1)  # (batch, len(choices))
        features = torch.cat([features, features], dim=1)  # (batch, len(choices) * 2)
        
        if features.dtype != torch.float32:
            features = features.float()
        
        return features

    def _hook_on_batch_forward(self, ctx):
        """
        Forward pass with variational inference.
        Supports both standard VPL and VPL-GP (mixture prior).
        """
        # Get hidden states for embedding difference extraction
        output_hidden_states = self.vpl_use_feature_difference
        
        if ctx.cfg.llm.accelerator.use:
            input_ids = ctx.data_batch['input_ids']
            labels = ctx.data_batch['labels']
            attention_mask = ctx.data_batch['attention_mask']
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask,
                                output_hidden_states=output_hidden_states)

        elif ctx.cfg.llm.deepspeed.use:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask,
                                       output_hidden_states=output_hidden_states)

        else:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            # Get hidden states for embedding difference extraction
            output_hidden_states = self.vpl_use_feature_difference
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask,
                                output_hidden_states=output_hidden_states)

        logits = outputs.logits
        
        # Get hidden states for embedding difference extraction
        hidden_states = None
        if self.vpl_use_feature_difference:
            # Try to get hidden states from outputs
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # Use last hidden state (from the last transformer layer)
                hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            elif hasattr(outputs, 'last_hidden_state'):
                hidden_states = outputs.last_hidden_state
            else:
                # Fallback: get embeddings from model input embeddings
                try:
                    input_embeddings = ctx.model.get_input_embeddings()
                    hidden_states = input_embeddings(input_ids)  # (batch, seq_len, embedding_dim)
                except:
                    logger.warning("Could not extract hidden states, falling back to logits")
                    hidden_states = None
        
        # Extract preference features for variational encoder
        # If feature_difference=True: uses embedding difference (positive - negative)
        # Otherwise: uses logits-based features
        preference_features = self._extract_preference_features(
            logits, labels, self.choices, hidden_states=hidden_states
        )
        
        # Ensure preference_features are on the correct device and dtype
        # Note: _extract_preference_features already converts to float32
        preference_features = preference_features.to(ctx.device)
        
        # Deep feature extraction: raw features -> richer representation
        # Note: We reuse hidden_states from the main forward pass (no additional forward pass)
        # The embedding difference already contains information processed by the main model
        extracted_features = self.feature_extractor(preference_features)
        
        # Variational inference: encode to latent z
        z, mu, logvar = self.variational_encoder(extracted_features)
        
        # Store z in ctx (for GP prior collection if enabled)
        ctx.vpl_z = CtxVar(z.detach(), LIFECYCLE.BATCH)
        ctx.vpl_mu = CtxVar(mu.detach(), LIFECYCLE.BATCH)
        ctx.vpl_logvar = CtxVar(logvar.detach(), LIFECYCLE.BATCH)
        
        # Collect z values for visualization (always collect, not just for GP prior)
        # This allows t-SNE visualization even when GP prior is disabled
        if not hasattr(self, 'z_history'):
            self.z_history = []
        z_cpu = z.detach().cpu()
        self.z_history.append(z_cpu)
        
        # Compute KL divergence: KL(q(z|x) || p(z)) or KL(q(z|x) || p_mixture(z))
        # If GP prior is enabled, uses mixture prior; otherwise uses standard normal prior
        kl_loss = self.variational_encoder.kl_divergence(mu, logvar)
        
        # Condition model on latent z
        # Project latent to choice logit adjustments
        latent_adjustment = self.latent_projection(z)  # (batch, num_choices)
        
        # Apply latent conditioning to logits
        # Option: add latent adjustment to choice logits
        new_logits, new_labels, base_loss = cal_loss(logits, labels, self.choices)
        
        # Adjust logits with latent
        # latent_adjustment is (batch, num_choices), need to expand to match new_logits
        batch_size, seq_len, num_choices = new_logits.shape
        latent_adjustment_expanded = latent_adjustment.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # (batch, seq_len, num_choices)
        
        # Add latent adjustment to logits
        conditioned_logits = new_logits + latent_adjustment_expanded
        
        # Compute reconstruction loss (negative log likelihood)
        loss_fn = torch.nn.CrossEntropyLoss()
        reconstruction_loss = loss_fn(
            conditioned_logits.view(-1, num_choices),
            new_labels.view(-1)
        )
        
        # VPL loss = reconstruction loss + KL divergence
        vpl_loss = reconstruction_loss + self.vpl_kl_weight * kl_loss
        
        # Add orthogonal loss (CLOP-based) if enabled
        orthogonal_loss_val = torch.tensor(0.0, device=z.device)
        if self.vpl_orthogonal_weight > 0.0 and self.orthogonal_prototypes is not None:
            orthogonal_loss_val, pull_loss, orthonorm_loss = self._compute_clop_orthogonal_loss(z)
            vpl_loss = vpl_loss + orthogonal_loss_val
        
        # Store for monitoring
        ctx.vpl_kl_loss = CtxVar(kl_loss.item(), LIFECYCLE.BATCH)
        ctx.vpl_reconstruction_loss = CtxVar(reconstruction_loss.item(), LIFECYCLE.BATCH)
        if self.vpl_orthogonal_weight > 0.0:
            ctx.vpl_orthogonal_loss = CtxVar(orthogonal_loss_val.item(), LIFECYCLE.BATCH)
        
        # Clean up intermediate tensors to save memory (keep only what's needed for backward)
        # Note: Don't delete tensors that are part of the computation graph
        if ctx.cur_mode != MODE.TRAIN:
            # In eval mode, we can safely delete intermediate tensors
            del preference_features, z, mu, logvar, latent_adjustment, latent_adjustment_expanded

        if torch.isnan(vpl_loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        # Compute predictions for evaluation
        new_labels_flat = new_labels.view(-1)
        conditioned_logits_flat = conditioned_logits.view(-1, len(self.choices))
        conditioned_logits_flat = conditioned_logits_flat[(
            new_labels_flat != DefaultToken.IGNORE_INDEX.value), :]
        new_labels_flat = new_labels_flat[(new_labels_flat !=
                                          DefaultToken.IGNORE_INDEX.value)]
        _, predicted = conditioned_logits_flat.max(1)

        ctx.y_true = CtxVar(new_labels_flat, LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(predicted, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(conditioned_logits_flat, LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(vpl_loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if ctx.cfg.llm.retry_on_nan_loss:
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        # Update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        
        # Update VPL-specific statistics
        ctx.vpl_kl_loss_total += ctx.vpl_kl_loss * ctx.batch_size
        ctx.vpl_reconstruction_loss_total += ctx.vpl_reconstruction_loss * ctx.batch_size
        if self.vpl_orthogonal_weight > 0.0 and hasattr(ctx, 'vpl_orthogonal_loss'):
            ctx.vpl_orthogonal_loss_total += ctx.vpl_orthogonal_loss * ctx.batch_size
        
        # NOTE: Fixed prototypes don't need orthonormal constraint updates
        # They are already orthonormal and fixed
        
        # Cache label for evaluate
        ctx.ys_true.append(ctx.y_true)
        ctx.ys_pred.append(ctx.y_pred)
        
        # Periodic memory cleanup to prevent OOM
        # Clean up every 5 batches to reduce memory usage
        if ctx.cur_batch_i % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _hook_on_fit_end(self, ctx):
        ctx.ys_true = CtxVar(torch.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar(torch.concatenate(ctx.ys_pred), LIFECYCLE.ROUTINE)
        results = ctx.monitor.eval(ctx)
        
        # Add VPL-specific metrics
        if hasattr(ctx, 'vpl_kl_loss_total') and ctx.num_samples > 0:
            results['vpl_kl_loss'] = ctx.vpl_kl_loss_total / ctx.num_samples
            results['vpl_reconstruction_loss'] = ctx.vpl_reconstruction_loss_total / ctx.num_samples
            # Add orthogonal loss if enabled
            if self.vpl_orthogonal_weight > 0.0 and hasattr(ctx, 'vpl_orthogonal_loss_total'):
                results['vpl_orthogonal_loss'] = ctx.vpl_orthogonal_loss_total / ctx.num_samples
            # Add vpl_total for monitor formatting (required by format_eval_res)
            if 'vpl_total' not in results:
                results['vpl_total'] = ctx.num_samples
            
            # Log VPL metrics to wandb if enabled (client-side logging)
            if ctx.cfg.wandb.use and ctx.cfg.wandb.online_track and ctx.cfg.wandb.client_train_info:
                try:
                    import wandb
                    client_id = getattr(ctx, 'client_id', None)
                    client_prefix = f'client_{client_id}/' if client_id is not None else 'client/'
                    
                    wandb_metrics = {
                        f'{client_prefix}{ctx.cur_mode}/vpl_total_loss': results.get('loss', 0.0),
                        f'{client_prefix}{ctx.cur_mode}/vpl_reconstruction_loss': results['vpl_reconstruction_loss'],
                        f'{client_prefix}{ctx.cur_mode}/vpl_kl_loss': results['vpl_kl_loss'],
                    }
                    
                    # Add orthogonal loss if enabled
                    if self.vpl_orthogonal_weight > 0.0 and 'vpl_orthogonal_loss' in results:
                        wandb_metrics[f'{client_prefix}{ctx.cur_mode}/vpl_orthogonal_loss'] = results['vpl_orthogonal_loss']
                    
                    # Use round as step for x-axis in wandb (if available)
                    step = None
                    if hasattr(ctx, 'cur_round'):
                        step = ctx.cur_round
                    
                    if step is not None:
                        wandb.log(wandb_metrics, step=step)
                    else:
                        wandb.log(wandb_metrics)
                except ImportError:
                    logger.warning("wandb not installed, skipping VPL metrics logging")
                except Exception as e:
                    logger.warning(f"Failed to log VPL metrics to wandb: {e}")
        
        # Collect z values for visualization (always, if available)
        # This is separate from GP prior collection - we always want z values for t-SNE
        if hasattr(self, 'z_history') and len(self.z_history) > 0:
            z_values = torch.cat(self.z_history, dim=0)  # (num_batches * batch_size, latent_dim)
            
            # Store z values for visualization (sample a reasonable number)
            num_samples = min(100, len(z_values))  # 최대 100개 샘플 저장
            sampled_indices = torch.randperm(len(z_values))[:num_samples]
            self.client_z_values = z_values[sampled_indices]  # (num_samples, latent_dim)
            
            logger.info(f"Collected z values for visualization: "
                       f"{self.client_z_values.shape} from {len(self.z_history)} batches "
                       f"(round {ctx.cur_round if hasattr(ctx, 'cur_round') else 'unknown'})")
            
            # Clear history for next round
            self.z_history = []
        
        # Collect z distribution for GP prior (if enabled)
        if self.vpl_use_gp_prior and hasattr(self, 'client_z_values') and self.client_z_values is not None:
            # Compute mean and logvar over z values for GP prior
            z_values = self.client_z_values  # Use the sampled values
            self.client_z_mu = z_values.mean(dim=0)  # (latent_dim,)
            z_var = z_values.var(dim=0)  # (latent_dim,)
            self.client_z_logvar = torch.log(z_var + 1e-8)  # (latent_dim,)
        
        setattr(ctx, 'eval_metrics', results)
    
    # GP Prior methods (only used when vpl_use_gp_prior=True)
    def get_client_z_distribution(self):
        """
        Get client's z distribution (mu, logvar) for server aggregation.
        Only available when GP prior is enabled.
        
        Returns:
            (mu, logvar): Tuple of mean and log variance tensors, or None
        """
        if not self.vpl_use_gp_prior:
            return None
        if not hasattr(self, 'client_z_mu') or self.client_z_mu is None:
            return None
        if not hasattr(self, 'client_z_logvar') or self.client_z_logvar is None:
            return None
        return (self.client_z_mu.clone(), self.client_z_logvar.clone())
    
    def get_client_z_values(self):
        """
        Get client's z values for visualization.
        Available for all VPL experiments (not just GP prior).
        
        Returns:
            z_values: Tensor of z values (num_samples, latent_dim), or None
        """
        if not hasattr(self, 'client_z_values') or self.client_z_values is None:
            return None
        return self.client_z_values.clone()
    
    def update_prior_from_server(self, client_mus, client_logvars, client_weights):
        """
        Update the mixture prior from server.
        Only used when GP prior is enabled.
        
        Args:
            client_mus: Mean vectors from other clients (num_clients, latent_dim)
            client_logvars: Log variance vectors from other clients (num_clients, latent_dim)
            client_weights: Weights for each client distribution (num_clients,)
        """
        if not self.vpl_use_gp_prior:
            return
        if hasattr(self.variational_encoder, 'update_prior'):
            self.variational_encoder.update_prior(client_mus, client_logvars, client_weights)
    
    def update_orthogonal_label_from_server(self, label):
        """
        Update orthogonal label from server.
        
        Args:
            label: Orthogonal label (int)
        """
        self.orthogonal_label = label
    
    def get_client_orthogonal_prototypes(self):
        """
        Get client's orthogonal prototypes for visualization.
        
        Returns:
            prototypes: Tensor of prototypes (num_prototypes, latent_dim) or None
        """
        if hasattr(self, 'orthogonal_prototypes') and self.orthogonal_prototypes is not None:
            return self.orthogonal_prototypes
        return None
    
    def _compute_clop_orthogonal_loss(self, z, labels=None):
        """
        Compute CLOP orthogonal loss.
        
        Args:
            z: Latent embeddings (batch_size, latent_dim)
            labels: Optional labels for each sample (batch_size,)
            
        Returns:
            orthogonal_loss: Total orthogonal loss
            pull_loss: Pull loss component
            orthonorm_loss: Orthonormal constraint loss component
        """
        if self.orthogonal_prototypes is None:
            return torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device), torch.tensor(0.0, device=z.device)
        
        batch_size, latent_dim = z.shape
        num_prototypes, _ = self.orthogonal_prototypes.shape
        
        # NOTE: QR decomposition을 forward에서 수행하면 gradient가 차단됨
        # 대신 orthonormal constraint를 loss로만 적용하여 prototype이 학습되도록 함
        # QR decomposition은 backward 후에만 수행 (gradient 유지)
        
        # Determine orthogonal labels
        if self.vpl_use_manual_orthogonal_labels and self.orthogonal_label is not None:
            # Use manual label from server (same for all samples in batch)
            orthogonal_labels = torch.full((batch_size,), self.orthogonal_label, device=z.device, dtype=torch.long)
        else:
            # Automatically assign to closest prototype
            similarities = torch.matmul(z, self.orthogonal_prototypes.T)  # (batch_size, num_prototypes)
            orthogonal_labels = torch.argmax(similarities, dim=1)  # (batch_size,)
        
        # Pull loss: z를 해당 prototype에 가깝게
        # NOTE: Prototypes are FIXED, so only z moves toward prototypes
        selected_prototypes = self.orthogonal_prototypes[orthogonal_labels]  # (batch_size, latent_dim)
        pull_loss = torch.mean((z - selected_prototypes) ** 2)
        
        # Orthonormal constraint: P^T P = I
        # NOTE: Fixed prototypes are already orthonormal, but we compute this for monitoring
        PTP = torch.matmul(self.orthogonal_prototypes, self.orthogonal_prototypes.T)  # (num_prototypes, num_prototypes)
        identity = torch.eye(num_prototypes, device=z.device)
        orthonorm_loss = torch.norm(PTP - identity, p='fro') ** 2
        
        # Total orthogonal loss
        orthogonal_loss = self.vpl_orthogonal_weight * pull_loss + \
                         self.vpl_orthogonal_orthonorm_weight * orthonorm_loss
        
        return orthogonal_loss, pull_loss, orthonorm_loss
    


def call_vpl_reward_choice_trainer(trainer_type):
    if trainer_type == 'vplrewardchoicetrainer':
        trainer_builder = VPLRewardChoiceTrainer
        return trainer_builder
    # Also support vplgprewardchoicetrainer for backward compatibility
    # But it will use the same unified trainer with GP prior enabled via config
    elif trainer_type == 'vplgprewardchoicetrainer':
        trainer_builder = VPLRewardChoiceTrainer
        return trainer_builder


register_trainer('vplrewardchoicetrainer', call_vpl_reward_choice_trainer)
register_trainer('vplgprewardchoicetrainer', call_vpl_reward_choice_trainer)