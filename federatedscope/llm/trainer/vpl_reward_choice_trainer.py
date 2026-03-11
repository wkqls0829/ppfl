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

        # Temperature annealing for Gumbel-Softmax (VMTL-style)
        self.vpl_gp_tau_anneal = getattr(config.llm, 'vpl_gp_tau_anneal', True)
        self.vpl_gp_tau_start = getattr(config.llm, 'vpl_gp_tau_start', self.vpl_gp_temperature)
        self.vpl_gp_tau_end = getattr(config.llm, 'vpl_gp_tau_end', 0.1)
        self.total_round_num = getattr(config.federate, 'total_round_num', 30)
        
        # Orthogonal loss hyperparameters (CLOP-based)
        self.vpl_orthogonal_weight = getattr(config.llm, 'vpl_orthogonal_weight', 0.0)
        self.vpl_orthogonal_orthonorm_weight = getattr(config.llm, 'vpl_orthogonal_orthonorm_weight', 0.1)
        self.vpl_use_manual_orthogonal_labels = getattr(config.llm, 'vpl_use_manual_orthogonal_labels', False)
        
        # Initialize orthonormal prototypes if orthogonal loss is enabled
        # Prototypes are LEARNABLE nn.Parameters (CLOP standard)
        # Orthonormality is maintained via the orthonorm constraint loss
        if self.vpl_orthogonal_weight > 0.0:
            num_prototypes = getattr(config.llm, 'vpl_num_prototypes', 2)
            # Get prototype scale (distance from origin)
            prototype_scale = getattr(config.llm, 'vpl_prototype_scale', 5.0)
            self.prototype_scale = prototype_scale

            # Create orthonormal basis via QR decomposition, then scale
            if num_prototypes <= self.vpl_latent_dim:
                # Random matrix -> QR -> orthonormal rows, padded to latent_dim
                rand_mat = torch.randn(
                    num_prototypes, self.vpl_latent_dim,
                    device=device
                )
                q, _ = torch.linalg.qr(rand_mat.T)
                # q is (latent_dim, num_prototypes), transpose
                prototypes = q.T[:num_prototypes] * prototype_scale
            else:
                rand_mat = torch.randn(
                    self.vpl_latent_dim, self.vpl_latent_dim,
                    device=device
                )
                q, _ = torch.linalg.qr(rand_mat)
                prototypes = q.T[:num_prototypes, :self.vpl_latent_dim] \
                    * prototype_scale

            # Learnable prototypes (updated via gradient descent)
            self.orthogonal_prototypes = nn.Parameter(prototypes)
            self.orthogonal_label = None  # Will be set by server
            logger.info(
                f"Initialized {num_prototypes} LEARNABLE "
                f"orthonormal prototypes for CLOP loss "
                f"(scale={prototype_scale}, "
                f"shape={self.orthogonal_prototypes.shape})"
            )
        else:
            self.orthogonal_prototypes = None
            self.orthogonal_label = None
        
        # Get model dimensions
        # Try multiple methods to get embedding dimension
        embedding_dim = None
        try:
            embedding_dim = self.model.get_input_embeddings().embedding_dim
            logger.info(f"Got embedding_dim={embedding_dim} from get_input_embeddings()")
        except Exception as e:
            logger.debug(f"Could not get embedding_dim from get_input_embeddings(): {e}")
            try:
                # Most HuggingFace models (including LLaMA) expose hidden_size on config
                embedding_dim = getattr(self.model, "config", None)
                if embedding_dim is not None:
                    embedding_dim = embedding_dim.hidden_size
                    logger.info(f"Got embedding_dim={embedding_dim} from model.config.hidden_size")
            except Exception as e2:
                logger.debug(f"Could not get embedding_dim from config.hidden_size: {e2}")
                embedding_dim = None

        # If direct queries failed, fall back based on model type
        if embedding_dim is None:
            model_type = getattr(config.model, 'type', '')
            mt_lower = model_type.lower()
            if 'llama' in mt_lower:
                # LLaMA hidden size (e.g., LLaMA-2 7B) is typically 4096
                embedding_dim = 4096
            elif 'gemma' in mt_lower:
                embedding_dim = 2048  # Gemma-2B
            elif 'qwen' in mt_lower:
                # Qwen2 models: default to 0.5B (896) for main table experiments
                embedding_dim = 896  # Qwen2-0.5B (default for main table)
            else:
                embedding_dim = 2048  # Generic fallback
            logger.info(f"Using inferred embedding_dim={embedding_dim} for model type: {model_type}")
        
        if embedding_dim is None:
            embedding_dim = 2048  # Final fallback
            logger.warning(f"Could not determine embedding_dim, using default: {embedding_dim}")
        
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
        # Get max_logvar to limit variance (sigma) - smaller values = tighter distribution
        # max_logvar=0.0 means sigma <= 1.0, max_logvar=-2.0 means sigma <= 0.368
        vpl_max_logvar = getattr(config.llm, 'vpl_max_logvar', -2.0)  # Default: -2.0 for tighter distribution
        if self.vpl_use_gp_prior:
            from federatedscope.llm.model.variational_encoder_gp import VariationalEncoderGP
            self.variational_encoder = VariationalEncoderGP(
                input_dim=self.feature_extractor_output_dim,
                latent_dim=self.vpl_latent_dim,
                hidden_dims=[512, 256, 128],
                temperature=self.vpl_gp_temperature,
                num_clients=self.num_clients,
                max_logvar=vpl_max_logvar,
                tau_anneal=self.vpl_gp_tau_anneal,
                tau_start=self.vpl_gp_tau_start,
                tau_end=self.vpl_gp_tau_end,
            ).to(device)
        else:
            self.variational_encoder = VariationalEncoder(
                input_dim=self.feature_extractor_output_dim,
                latent_dim=self.vpl_latent_dim,
                hidden_dims=[512, 256, 128],
                max_logvar=vpl_max_logvar
            ).to(device)
        
        # Latent conditioning: project latent z to modify model behavior
        # Option 1: Add latent to embeddings
        # Option 2: Use latent to scale/adjust logits
        # We'll use Option 2: scale logits based on latent
        self.latent_projection = nn.Linear(
            self.vpl_latent_dim,
            len(self.choices)
        ).to(device)

        # Cached loss function (avoid re-instantiation per batch)
        self._ce_loss_fn = torch.nn.CrossEntropyLoss()

        # Cache ortho target (avoid recreating torch.eye per batch)
        if self.vpl_orthogonal_weight > 0.0:
            scale_sq = self.prototype_scale ** 2
            num_p = self.orthogonal_prototypes.shape[0]
            self._ortho_target = (
                torch.eye(num_p, device=device) * scale_sq
            )

        # Cache VPL param list for grad clipping
        self._vpl_params_for_clip = []
        for comp_name in ('feature_extractor',
                          'variational_encoder',
                          'latent_projection'):
            comp = getattr(self, comp_name, None)
            if comp is not None:
                self._vpl_params_for_clip.extend(comp.parameters())

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
        
        # Freeze base model (LLM) for binary selector training
        # This prevents base model from being updated during VPL training
        # Only VPL components (feature_extractor, variational_encoder, latent_projection) will be trained
        freeze_base_model = getattr(ctx.cfg.llm, 'vpl_freeze_base_model', True)  # Default: True
        if freeze_base_model and ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            # Use ctx.model instead of self.model since model may not be set yet
            model = getattr(ctx, 'model', None) or getattr(self, 'model', None)
            if model is not None:
                # Freeze all parameters of the base model
                for param in model.parameters():
                    param.requires_grad = False
                logger.info("Frozen base model (LLM) parameters for binary selector training. "
                           "Only VPL components will be trained.")
            else:
                logger.warning("Model not available yet for freezing. Will freeze later if needed.")
        
        # Create separate optimizer for VPL components (feature_extractor + variational_encoder + latent_projection)
        # This allows VPL components to learn faster while LLM is frozen or learns slowly
        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            vpl_lr_multiplier = getattr(ctx.cfg.llm, 'vpl_lr_multiplier', 10.0)  # Default: 10x faster learning
            
            # Collect VPL component parameters
            vpl_params = []
            if hasattr(self, 'feature_extractor'):
                vpl_params.extend(self.feature_extractor.parameters())
            if hasattr(self, 'variational_encoder'):
                vpl_params.extend(self.variational_encoder.parameters())
            if hasattr(self, 'latent_projection'):
                vpl_params.extend(self.latent_projection.parameters())
            if hasattr(self, 'orthogonal_prototypes') \
                    and self.orthogonal_prototypes is not None \
                    and isinstance(self.orthogonal_prototypes,
                                   nn.Parameter):
                vpl_params.append(self.orthogonal_prototypes)
            
            if len(vpl_params) > 0:
                # Get base learning rate from config
                base_lr = ctx.cfg[ctx.cur_mode].optimizer.get('lr', 1e-5)
                vpl_lr = base_lr * vpl_lr_multiplier
                
                # Create separate optimizer for VPL components
                from torch.optim import AdamW
                ctx.vpl_optimizer = AdamW(
                    vpl_params,
                    lr=vpl_lr,
                    betas=ctx.cfg[ctx.cur_mode].optimizer.get('betas', (0.9, 0.95)),
                    weight_decay=ctx.cfg[ctx.cur_mode].optimizer.get('weight_decay', 0.0)
                )
                logger.info(f"Created separate VPL optimizer with lr={vpl_lr:.2e} (base_lr={base_lr:.2e} * {vpl_lr_multiplier}x)")
            else:
                ctx.vpl_optimizer = None
                logger.warning("No VPL components found for separate optimizer")

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
        # Detach hidden_states to prevent LLM gradient flow
        # Only feature_extractor and variational_encoder will be trained
        hidden_states = hidden_states.detach()
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        shift_labels = labels[..., 1:].contiguous()  # (batch, seq_len-1)
        shift_hidden = hidden_states[..., :-1, :].contiguous()  # (batch, seq_len-1, hidden_dim)
        
        # Find choice token positions
        A_token, B_token = choices[0], choices[1]
        A_positions = (shift_labels == A_token)  # (batch, seq_len-1)
        B_positions = (shift_labels == B_token)  # (batch, seq_len-1)
        
        # Vectorized extraction: compute masked means without Python loop
        # This avoids per-sample GPU syncs from .any() calls
        # Compute masked mean for A and B positions (batch, hidden_dim)
        A_mask = A_positions.unsqueeze(-1).float()  # (B, seq, 1)
        B_mask = B_positions.unsqueeze(-1).float()
        A_count = A_mask.sum(dim=1).clamp(min=1)  # (B, 1)
        B_count = B_mask.sum(dim=1).clamp(min=1)
        A_emb = (shift_hidden * A_mask).sum(dim=1) / A_count  # (B, H)
        B_emb = (shift_hidden * B_mask).sum(dim=1) / B_count  # (B, H)

        # For samples where A or B is missing, fall back to sequence mean
        seq_mean = shift_hidden.mean(dim=1)  # (B, H)
        A_found = A_positions.any(dim=1, keepdim=True)  # (B, 1)
        B_found = B_positions.any(dim=1, keepdim=True)  # (B, 1)

        # chosen = A where A found, else B where B found, else seq_mean
        # rejected = B where B found, else seq_mean
        chosen_emb = torch.where(A_found, A_emb, torch.where(B_found, B_emb, seq_mean))
        rejected_emb = torch.where(A_found & B_found, B_emb, seq_mean)

        feature_diff = chosen_emb - rejected_emb

        if self.vpl_use_llm_feature_extractor:
            if self.vpl_use_difference_only:
                features = feature_diff  # (B, H)
            else:
                features = torch.cat(
                    [chosen_emb, rejected_emb, feature_diff],
                    dim=-1)  # (B, H*3)
        else:
            features = feature_diff  # (B, H)
        
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
        is_eval = ctx.cur_mode not in (MODE.TRAIN, MODE.FINETUNE)
        if is_eval:
            with torch.no_grad():
                self._hook_on_batch_forward_impl(ctx)
        else:
            self._hook_on_batch_forward_impl(ctx)

    def _hook_on_batch_forward_impl(self, ctx):
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
        
        # preference_features are already on ctx.device from model output
        
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
        
        # Collect z values for visualization (subsample to avoid
        # per-batch GPU->CPU transfer; 100 samples suffices for t-SNE)
        if not hasattr(self, 'z_history'):
            self.z_history = []
        if len(self.z_history) < 100:
            self.z_history.append(z.detach().cpu())
        
        # Compute KL divergence: KL(q(z|x) || p(z)) or KL(q(z|x) || p_mixture(z))
        # If GP prior is enabled, uses mixture prior; otherwise uses standard normal prior
        kl_loss = self.variational_encoder.kl_divergence(mu, logvar)
        
        # Condition model on latent z
        # Project latent to choice logit adjustments
        latent_adjustment = self.latent_projection(z)  # (batch, num_choices)
        
        # Apply latent conditioning to logits
        # Option: add latent adjustment to choice logits
        # Extract choice logits/labels without computing unused base loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        new_labels = torch.full_like(shift_labels,
                                     DefaultToken.IGNORE_INDEX.value)
        for idx, choice in enumerate(self.choices):
            new_labels[shift_labels == choice] = idx
        new_logits = shift_logits[..., self.choices]
        
        # Adjust logits with latent
        # latent_adjustment is (batch, num_choices), need to expand to match new_logits
        batch_size, seq_len, num_choices = new_logits.shape
        latent_adjustment_expanded = latent_adjustment.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # (batch, seq_len, num_choices)
        
        # Add latent adjustment to logits
        conditioned_logits = new_logits + latent_adjustment_expanded
        
        # Compute reconstruction loss (negative log likelihood)
        reconstruction_loss = self._ce_loss_fn(
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
        
        # Store for monitoring (use detach() to avoid GPU sync from .item())
        ctx.vpl_kl_loss = CtxVar(kl_loss.detach(), LIFECYCLE.BATCH)
        ctx.vpl_reconstruction_loss = CtxVar(
            reconstruction_loss.detach(), LIFECYCLE.BATCH)
        if self.vpl_orthogonal_weight > 0.0:
            ctx.vpl_orthogonal_loss = CtxVar(
                orthogonal_loss_val.detach(), LIFECYCLE.BATCH)
        
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

    def _hook_on_batch_backward(self, ctx):
        """
        Override backward pass to use separate VPL optimizer.
        This allows VPL components (feature_extractor, variational_encoder, latent_projection)
        to learn faster while LLM is frozen (hidden_states.detach()).
        """
        if ctx.skip_this_batch:
            return

        # Use separate VPL optimizer if available
        # This allows VPL components (feature_extractor, variational_encoder, latent_projection)
        # to learn faster while LLM learns slowly or is frozen
        use_vpl_optimizer = hasattr(ctx, 'vpl_optimizer') and ctx.vpl_optimizer is not None
        
        if ctx.cfg.llm.accelerator.use:
            self.accelerator.backward(ctx.loss_task)
            if use_vpl_optimizer:
                # Update VPL components with separate optimizer
                ctx.vpl_optimizer.step()
                ctx.vpl_optimizer.zero_grad()
            else:
                # Update all parameters with main optimizer
                ctx.optimizer.step()
                ctx.optimizer.zero_grad()
            if ctx.scheduler is not None:
                ctx.scheduler.step()

        elif ctx.cfg.llm.deepspeed.use:
            ctx.model_engine.backward(ctx.loss_task)
            ctx.model_engine.step()
            if ctx.scheduler is not None:
                ctx.scheduler.step()

        else:
            (ctx.loss_task / self.grad_accum_step).backward()

            if (ctx.cur_batch_i + 1) % self.grad_accum_step == 0:
                if use_vpl_optimizer:
                    # Update VPL components with separate optimizer (faster learning)
                    if ctx.grad_clip > 0 and self._vpl_params_for_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self._vpl_params_for_clip, ctx.grad_clip)
                    ctx.vpl_optimizer.step()
                    ctx.vpl_optimizer.zero_grad()
                    
                    # LLM parameters are not updated (hidden_states was detached)
                    # If you want to update LLM as well, you can add:
                    # ctx.optimizer.step()
                    # ctx.optimizer.zero_grad()
                else:
                    # Update all parameters with main optimizer
                    if ctx.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                                       ctx.grad_clip)
                    ctx.optimizer.step()
                    ctx.optimizer.zero_grad()
                
                if ctx.scheduler is not None:
                    ctx.scheduler.step()

        # Free training data from GPU (del instead of .cpu() which
        # was a no-op bug — return values were discarded)
        del ctx.data_batch
    
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
        
        # NOTE: gc.collect() + empty_cache() removed here — calling
        # every 5 batches was adding ~1-2s overhead per round.
        # If OOM occurs, reduce batch_size or enable gradient
        # checkpointing instead.

    def _hook_on_fit_end(self, ctx):
        # Handle case where all batches were skipped (e.g. NaN loss): avoid empty concatenate
        if not ctx.ys_true:
            device = getattr(ctx, 'device', None) or getattr(self, 'device', None) or 'cpu'
            ctx.ys_true = CtxVar(
                torch.tensor([], dtype=torch.long, device=device),
                LIFECYCLE.ROUTINE,
            )
            ctx.ys_pred = CtxVar(
                torch.tensor([], dtype=torch.long, device=device),
                LIFECYCLE.ROUTINE,
            )
            logger.warning(
                "Evaluation had no valid batches (all skipped e.g. due to NaN loss). "
                "Reporting empty metrics for this client."
            )
        else:
            ctx.ys_true = CtxVar(torch.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
            ctx.ys_pred = CtxVar(torch.concatenate(ctx.ys_pred), LIFECYCLE.ROUTINE)
        # Set tokenizer in ctx for evaluation metrics that need it (e.g., reward model evaluation)
        if not hasattr(ctx, 'tokenizer') and hasattr(self, 'tokenizer'):
            ctx.tokenizer = self.tokenizer
        results = ctx.monitor.eval(ctx)
        
        # Add VPL-specific metrics
        if hasattr(ctx, 'vpl_kl_loss_total') and ctx.num_samples > 0:
            kl_avg = ctx.vpl_kl_loss_total / ctx.num_samples
            recon_avg = ctx.vpl_reconstruction_loss_total / ctx.num_samples
            results['vpl_kl_loss'] = kl_avg.item() if torch.is_tensor(kl_avg) else kl_avg
            results['vpl_reconstruction_loss'] = recon_avg.item() if torch.is_tensor(recon_avg) else recon_avg
            # Add orthogonal loss if enabled
            if self.vpl_orthogonal_weight > 0.0 and hasattr(ctx, 'vpl_orthogonal_loss_total'):
                ortho_avg = ctx.vpl_orthogonal_loss_total / ctx.num_samples
                results['vpl_orthogonal_loss'] = ortho_avg.item() if torch.is_tensor(ortho_avg) else ortho_avg
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
    
    def update_prior_from_server(self, client_mus, client_logvars, client_weights, current_round=None):
        """
        Update the mixture prior from server.
        Only used when GP prior is enabled.

        After updating the prior, ensures prior_logits (learnable Gumbel-Softmax
        weights) are included in the VPL optimizer, and anneals temperature.

        Args:
            client_mus: Mean vectors from other clients (num_clients, latent_dim)
            client_logvars: Log variance vectors from other clients (num_clients, latent_dim)
            client_weights: Sample-size weights (used for initialization only) (num_clients,)
            current_round: Current FL round (for temperature annealing)
        """
        if not self.vpl_use_gp_prior:
            return
        if hasattr(self.variational_encoder, 'update_prior'):
            self.variational_encoder.update_prior(client_mus, client_logvars, client_weights)

            # Ensure prior_logits is in the VPL optimizer
            if hasattr(self.variational_encoder, 'prior_logits') and \
               self.variational_encoder.prior_logits is not None:
                self._ensure_prior_logits_in_optimizer()

            # Anneal temperature
            if current_round is not None and hasattr(self.variational_encoder, 'anneal_temperature'):
                self.variational_encoder.anneal_temperature(current_round, self.total_round_num)

    def _ensure_prior_logits_in_optimizer(self):
        """Add prior_logits to the VPL optimizer if not already present."""
        logits_param = self.variational_encoder.prior_logits
        if logits_param is None:
            return

        # Try VPL optimizer first, then fall back to ctx optimizer
        optimizer = None
        if hasattr(self, 'ctx') and hasattr(self.ctx, 'vpl_optimizer') and self.ctx.vpl_optimizer is not None:
            optimizer = self.ctx.vpl_optimizer
        elif hasattr(self, 'ctx') and hasattr(self.ctx, 'optimizer'):
            optimizer = self.ctx.optimizer

        if optimizer is None:
            logger.warning("No optimizer found to add prior_logits to")
            return

        # Check if already tracked
        param_id = id(logits_param)
        existing_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
        if param_id not in existing_ids:
            optimizer.add_param_group({
                'params': [logits_param],
                'lr': optimizer.param_groups[0]['lr'],
            })
            logger.info(f"Added prior_logits to optimizer (lr={optimizer.param_groups[0]['lr']:.2e})")
    
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
        Compute CLOP orthogonal loss (Eq. 9).

        L_ortho = λ·||z - p_{y*}||² + γ·||PP^T - I||²_F

        Prototypes are learnable nn.Parameters. The pull loss moves
        both z and the assigned prototype closer together, while the
        orthonormality constraint keeps prototypes separated.

        Label assignment priority:
          1. Server-assigned label (k-means on client z-means)
          2. Local nearest-prototype fallback (round 1, before
             server has computed labels)

        Args:
            z: Latent embeddings (batch_size, latent_dim)
            labels: Optional labels for each sample (batch_size,)

        Returns:
            orthogonal_loss: Total orthogonal loss
            pull_loss: Pull loss component
            orthonorm_loss: Orthonormal constraint loss component
        """
        if self.orthogonal_prototypes is None:
            return (torch.tensor(0.0, device=z.device),
                    torch.tensor(0.0, device=z.device),
                    torch.tensor(0.0, device=z.device))

        batch_size, latent_dim = z.shape
        num_prototypes = self.orthogonal_prototypes.shape[0]

        # Determine orthogonal labels
        # Use server-assigned label when available (k-means or manual)
        if self.orthogonal_label is not None:
            orthogonal_labels = torch.full(
                (batch_size,), self.orthogonal_label,
                device=z.device, dtype=torch.long
            )
        else:
            # Fallback: assign to closest prototype (e.g. round 1)
            with torch.no_grad():
                similarities = torch.matmul(
                    z.detach(),
                    self.orthogonal_prototypes.detach().T
                )
                orthogonal_labels = torch.argmax(
                    similarities, dim=1
                )

        # Pull loss: move z toward assigned prototype (and vice versa)
        selected_prototypes = self.orthogonal_prototypes[
            orthogonal_labels
        ]  # (batch_size, latent_dim)
        pull_loss = torch.mean((z - selected_prototypes) ** 2)

        # Orthonormal constraint: ||PP^T - s²·I||²_F
        # Uses scaled identity since prototypes have norm ~prototype_scale
        PTP = torch.matmul(
            self.orthogonal_prototypes,
            self.orthogonal_prototypes.T
        )  # (num_prototypes, num_prototypes)
        orthonorm_loss = torch.norm(
            PTP - self._ortho_target, p='fro') ** 2

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