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
        
        # Initialize variational encoder
        # Input dim depends on feature extraction method
        if self.vpl_feature_method == 'choice_logits':
            input_dim = len(self.choices) * 2  # chosen + rejected
        else:
            # For pooling methods, use a small feature dim
            input_dim = 2  # mean/max of chosen and rejected
        
        self.variational_encoder = VariationalEncoder(
            input_dim=input_dim,
            latent_dim=self.vpl_latent_dim,
            hidden_dims=[128, 64]  # Smaller for efficiency
        ).to(device)
        
        # Latent conditioning: project latent z to modify model behavior
        # Option 1: Add latent to embeddings
        # Option 2: Use latent to scale/adjust logits
        # We'll use Option 2: scale logits based on latent
        self.latent_projection = nn.Linear(
            self.vpl_latent_dim, 
            len(self.choices)
        ).to(device)
        
        logger.info(f'VPLRewardChoiceTrainer initialized with latent_dim={self.vpl_latent_dim}, '
                   f'kl_weight={self.vpl_kl_weight}')

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        ctx.vpl_kl_loss_total = CtxVar(0.0, LIFECYCLE.ROUTINE)
        ctx.vpl_reconstruction_loss_total = CtxVar(0.0, LIFECYCLE.ROUTINE)

    def _extract_preference_features(self, logits, labels, choices):
        """
        Extract features from preference data for variational encoder.
        
        Args:
            logits: Model logits (batch, seq_len, vocab_size)
            labels: Labels (batch, seq_len)
            choices: Choice token indices
            
        Returns:
            features: Extracted features (batch, feature_dim)
            For choice_logits method: (batch, len(choices) * 2) = (batch, 4)
        """
        # Get logits at choice positions
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        batch_size = logits.shape[0]
        
        # Extract logits for each choice token
        # For each choice token, get the logit value at that position
        # Shape: (batch, seq_len, vocab_size) -> (batch, len(choices))
        choice_logits_per_token = shift_logits[..., choices]  # (batch, seq_len, len(choices))
        
        # Find positions where each choice appears
        features_list = []
        for choice_idx, choice_token in enumerate(choices):
            # Find positions where this choice token appears
            choice_positions = (shift_labels == choice_token)  # (batch, seq_len)
            
            if choice_positions.any():
                # Get logits at positions where this choice appears
                # For each sample, get logits at choice positions
                batch_features = []
                for b in range(batch_size):
                    sample_positions = choice_positions[b]  # (seq_len,)
                    if sample_positions.any():
                        # Get logits at choice positions for this choice token
                        sample_choice_logits = choice_logits_per_token[b, sample_positions, choice_idx]  # (num_positions,)
                        # Average over positions
                        avg_logit = sample_choice_logits.mean()
                    else:
                        # No choice token found, use mean of all logits for this choice
                        avg_logit = choice_logits_per_token[b, :, choice_idx].mean()
                    batch_features.append(avg_logit)
                
                features_list.append(torch.stack(batch_features))  # (batch,)
            else:
                # No choice token found in any sample, use mean
                features_list.append(choice_logits_per_token[:, :, choice_idx].mean(dim=1))  # (batch,)
        
        # Stack to get (batch, len(choices))
        features = torch.stack(features_list, dim=1)  # (batch, len(choices))
        
        # For choice_logits method, we want (batch, len(choices) * 2)
        # Use [A_logit, B_logit, A_logit, B_logit] or [A_logit, B_logit, max(A,B), min(A,B)]
        # Simple approach: duplicate the features
        features = torch.cat([features, features], dim=1)  # (batch, len(choices) * 2)
        
        # Ensure features are float32 for variational encoder
        if features.dtype != torch.float32:
            features = features.float()
        
        return features

    def _hook_on_batch_forward(self, ctx):
        """
        Forward pass with variational inference.
        """
        if ctx.cfg.llm.accelerator.use:
            input_ids = ctx.data_batch['input_ids']
            labels = ctx.data_batch['labels']
            attention_mask = ctx.data_batch['attention_mask']
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        elif ctx.cfg.llm.deepspeed.use:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model_engine(input_ids=input_ids,
                                       labels=labels,
                                       attention_mask=attention_mask)

        else:
            input_ids = ctx.data_batch['input_ids'].to(ctx.device)
            labels = ctx.data_batch['labels'].to(ctx.device)
            attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
            outputs = ctx.model(input_ids=input_ids,
                                labels=labels,
                                attention_mask=attention_mask)

        logits = outputs.logits
        
        # Extract preference features for variational encoder
        preference_features = self._extract_preference_features(
            logits, labels, self.choices
        )
        
        # Ensure preference_features are on the correct device and dtype
        # Note: _extract_preference_features already converts to float32
        preference_features = preference_features.to(ctx.device)
        
        # Variational inference: encode to latent z
        z, mu, logvar = self.variational_encoder(preference_features)
        
        # Compute KL divergence: KL(q(z|x) || p(z))
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
        
        # Store for monitoring
        ctx.vpl_kl_loss = CtxVar(kl_loss.item(), LIFECYCLE.BATCH)
        ctx.vpl_reconstruction_loss = CtxVar(reconstruction_loss.item(), LIFECYCLE.BATCH)
        
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
            
            # Log VPL metrics to wandb if enabled
            if ctx.cfg.wandb.use and ctx.cfg.wandb.online_track:
                try:
                    import wandb
                    wandb_metrics = {
                        f'{ctx.cur_mode}/vpl_kl_loss': results['vpl_kl_loss'],
                        f'{ctx.cur_mode}/vpl_reconstruction_loss': results['vpl_reconstruction_loss'],
                        f'{ctx.cur_mode}/vpl_elbo_loss': results.get('loss', 0.0),  # Total loss includes ELBO
                    }
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
        
        setattr(ctx, 'eval_metrics', results)


def call_vpl_reward_choice_trainer(trainer_type):
    if trainer_type == 'vplrewardchoicetrainer':
        trainer_builder = VPLRewardChoiceTrainer
        return trainer_builder


register_trainer('vplrewardchoicetrainer', call_vpl_reward_choice_trainer)
