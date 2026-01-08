"""
Variational Preference Learning (VPL) for Preference Learning Model
Implements VPL with variational method for preference learning (DPO-style).
Based on: https://github.com/WEIRDLabUW/vpl
"""
import torch
import torch.nn.functional as F
import logging
import numpy as np
import gc

from federatedscope.register import register_trainer
from federatedscope.llm.trainer.reward_trainer import (
    DPORewardTrainer, _get_batch_logps, dpo_loss)
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE, MODE
from federatedscope.llm.model.variational_encoder import (
    VariationalEncoder, PreferenceFeatureExtractor)
import torch.nn as nn

logger = logging.getLogger(__name__)


class VPLRewardTrainer(DPORewardTrainer):
    """
    Variational Preference Learning trainer for preference learning tasks.
    
    Extends DPORewardTrainer with variational inference over user-specific
    latents. The model learns to:
    1. Infer user-specific latent z from preference pairs
    2. Compute rewards conditioned on the latent z
    3. Optimize ELBO: E_q(z|x)[log p(preference|z,x)] - KL(q(z|x) || p(z))
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
        
        # Get model hidden dimension (approximate from model config)
        # For LLM models, we'll use a reasonable default
        model_hidden_dim = getattr(config.model, 'hidden', 2048)
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            model_hidden_dim = model.config.hidden_size
        
        # Initialize preference feature extractor
        self.feature_extractor = PreferenceFeatureExtractor(
            model_hidden_dim=model_hidden_dim,
            feature_dim=128
        ).to(device)
        
        # Initialize variational encoder
        self.variational_encoder = VariationalEncoder(
            input_dim=128,  # Output dim of feature extractor
            latent_dim=self.vpl_latent_dim,
            hidden_dims=[256, 128]
        ).to(device)
        
        # Latent conditioning: project latent to reward adjustment
        self.latent_to_reward = nn.Sequential(
            nn.Linear(self.vpl_latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single reward adjustment
        ).to(device)
        
        logger.info(f'VPLRewardTrainer initialized with latent_dim={self.vpl_latent_dim}, '
                   f'kl_weight={self.vpl_kl_weight}')

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        ctx.vpl_kl_loss_total = CtxVar(0.0, LIFECYCLE.ROUTINE)
        ctx.vpl_reconstruction_loss_total = CtxVar(0.0, LIFECYCLE.ROUTINE)

    def _extract_preference_features(self, ctx, win_outputs, lose_outputs):
        """
        Extract features from win/lose pairs for variational encoder.
        
        Args:
            ctx: Training context
            win_outputs: Model outputs for winning responses
            lose_outputs: Model outputs for losing responses
            
        Returns:
            features: Extracted preference features
        """
        # Use hidden states if available, otherwise use logits
        if hasattr(win_outputs, 'hidden_states') and win_outputs.hidden_states is not None:
            # Use last hidden state
            win_hidden = win_outputs.hidden_states[-1].mean(dim=1)  # (batch, hidden_dim)
            lose_hidden = lose_outputs.hidden_states[-1].mean(dim=1)
        else:
            # Fallback: use logits mean as proxy for hidden representation
            win_hidden = win_outputs.logits.mean(dim=1)  # (batch, vocab_size)
            lose_hidden = lose_outputs.logits.mean(dim=1)
            # Project to smaller dimension if needed
            if win_hidden.shape[-1] > 2048:
                # Simple projection
                win_hidden = win_hidden[..., :2048]
                lose_hidden = lose_hidden[..., :2048]
        
        # Ensure hidden states are float32 for feature extractor
        if win_hidden.dtype != torch.float32:
            win_hidden = win_hidden.float()
        if lose_hidden.dtype != torch.float32:
            lose_hidden = lose_hidden.float()
        
        # Extract features using feature extractor
        features = self.feature_extractor(win_hidden, lose_hidden)
        
        # Ensure features are float32 for variational encoder
        if features.dtype != torch.float32:
            features = features.float()
        
        return features

    def _hook_on_batch_forward(self, ctx):
        """
        Forward pass with variational inference for preference learning.
        """
        if ctx.cfg.llm.accelerator.use:
            win_input_ids = ctx.data_batch['win_input_ids'].to(ctx.device)
            win_labels = ctx.data_batch['win_labels'].to(ctx.device)
            win_attention_mask = ctx.data_batch['win_attention_mask'].to(ctx.device)
            lose_input_ids = ctx.data_batch['lose_input_ids'].to(ctx.device)
            lose_labels = ctx.data_batch['lose_labels'].to(ctx.device)
            lose_attention_mask = ctx.data_batch['lose_attention_mask'].to(ctx.device)

            with torch.no_grad():
                ref_win_outputs = ctx.model(
                    disable_adapter=True,
                    input_ids=win_input_ids,
                    labels=win_labels,
                    attention_mask=win_attention_mask
                )
                ref_lose_outputs = ctx.model(
                    disable_adapter=True,
                    input_ids=lose_input_ids,
                    labels=lose_labels,
                    attention_mask=lose_attention_mask
                )
                ref_win_logps = _get_batch_logps(
                    ref_win_outputs.logits, win_labels, average_log_prob=False)
                ref_lose_logps = _get_batch_logps(
                    ref_lose_outputs.logits, lose_labels, average_log_prob=False)

            adap_win_outputs = ctx.model(
                disable_adapter=False,
                input_ids=win_input_ids,
                labels=win_labels,
                attention_mask=win_attention_mask
            )
            adap_lose_outputs = ctx.model(
                disable_adapter=False,
                input_ids=lose_input_ids,
                labels=lose_labels,
                attention_mask=lose_attention_mask
            )
            adap_win_logps = _get_batch_logps(
                adap_win_outputs.logits, win_labels, average_log_prob=False)
            adap_lose_logps = _get_batch_logps(
                adap_lose_outputs.logits, lose_labels, average_log_prob=False)

        elif ctx.cfg.llm.deepspeed.use:
            win_input_ids = ctx.data_batch['win_input_ids'].to(ctx.device)
            win_labels = ctx.data_batch['win_labels'].to(ctx.device)
            win_attention_mask = ctx.data_batch['win_attention_mask'].to(ctx.device)
            lose_input_ids = ctx.data_batch['lose_input_ids'].to(ctx.device)
            lose_labels = ctx.data_batch['lose_labels'].to(ctx.device)
            lose_attention_mask = ctx.data_batch['lose_attention_mask'].to(ctx.device)

            with torch.no_grad():
                ref_win_outputs = ctx.model_engine(
                    disable_adapter=True,
                    input_ids=win_input_ids,
                    labels=win_labels,
                    attention_mask=win_attention_mask
                )
                ref_lose_outputs = ctx.model_engine(
                    disable_adapter=True,
                    input_ids=lose_input_ids,
                    labels=lose_labels,
                    attention_mask=lose_attention_mask
                )
                ref_win_logps = _get_batch_logps(
                    ref_win_outputs.logits, win_labels, average_log_prob=False)
                ref_lose_logps = _get_batch_logps(
                    ref_lose_outputs.logits, lose_labels, average_log_prob=False)

            adap_win_outputs = ctx.model_engine(
                disable_adapter=False,
                input_ids=win_input_ids,
                labels=win_labels,
                attention_mask=win_attention_mask
            )
            adap_lose_outputs = ctx.model_engine(
                disable_adapter=False,
                input_ids=lose_input_ids,
                labels=lose_labels,
                attention_mask=lose_attention_mask
            )
            adap_win_logps = _get_batch_logps(
                adap_win_outputs.logits, win_labels, average_log_prob=False)
            adap_lose_logps = _get_batch_logps(
                adap_lose_outputs.logits, lose_labels, average_log_prob=False)

        else:
            win_input_ids = ctx.data_batch['win_input_ids'].to(ctx.device)
            win_labels = ctx.data_batch['win_labels'].to(ctx.device)
            win_attention_mask = ctx.data_batch['win_attention_mask'].to(ctx.device)
            lose_input_ids = ctx.data_batch['lose_input_ids'].to(ctx.device)
            lose_labels = ctx.data_batch['lose_labels'].to(ctx.device)
            lose_attention_mask = ctx.data_batch['lose_attention_mask'].to(ctx.device)

            with torch.no_grad():
                ref_win_outputs = ctx.model(
                    disable_adapter=True,
                    input_ids=win_input_ids,
                    labels=win_labels,
                    attention_mask=win_attention_mask
                )
                ref_lose_outputs = ctx.model(
                    disable_adapter=True,
                    input_ids=lose_input_ids,
                    labels=lose_labels,
                    attention_mask=lose_attention_mask
                )
                ref_win_logps = _get_batch_logps(
                    ref_win_outputs.logits, win_labels, average_log_prob=False)
                ref_lose_logps = _get_batch_logps(
                    ref_lose_outputs.logits, lose_labels, average_log_prob=False)

            adap_win_outputs = ctx.model(
                disable_adapter=False,
                input_ids=win_input_ids,
                labels=win_labels,
                attention_mask=win_attention_mask
            )
            adap_lose_outputs = ctx.model(
                disable_adapter=False,
                input_ids=lose_input_ids,
                labels=lose_labels,
                attention_mask=lose_attention_mask
            )
            adap_win_logps = _get_batch_logps(
                adap_win_outputs.logits, win_labels, average_log_prob=False)
            adap_lose_logps = _get_batch_logps(
                adap_lose_outputs.logits, lose_labels, average_log_prob=False)

        # Extract preference features for variational encoder
        preference_features = self._extract_preference_features(
            ctx, adap_win_outputs, adap_lose_outputs
        )
        
        # Ensure preference_features are on the correct device and dtype
        preference_features = preference_features.to(ctx.device)
        # Convert to float32 to match variational encoder dtype
        if preference_features.dtype != torch.float32:
            preference_features = preference_features.float()
        
        # Variational inference: encode to latent z
        z, mu, logvar = self.variational_encoder(preference_features)
        
        # Compute KL divergence: KL(q(z|x) || p(z))
        kl_loss = self.variational_encoder.kl_divergence(mu, logvar)
        
        # Condition rewards on latent z
        # Get reward adjustment from latent
        reward_adjustment = self.latent_to_reward(z).squeeze(-1)  # (batch,)
        
        # Adjust log probabilities with latent-conditioned reward
        # The reward adjustment acts as a bias term
        adjusted_win_logps = adap_win_logps + reward_adjustment
        adjusted_lose_logps = adap_lose_logps - reward_adjustment  # Opposite for lose
        
        # Compute DPO loss with adjusted logps
        dpo_loss_value, win_rewards, lose_rewards = dpo_loss(
            adjusted_win_logps,
            adjusted_lose_logps,
            ref_win_logps,
            ref_lose_logps,
            beta=self.reward_coeff
        )
        
        # VPL loss = DPO loss (reconstruction) + KL divergence
        vpl_loss = dpo_loss_value + self.vpl_kl_weight * kl_loss
        
        # Store for monitoring
        ctx.vpl_kl_loss = CtxVar(kl_loss.item(), LIFECYCLE.BATCH)
        ctx.vpl_reconstruction_loss = CtxVar(dpo_loss_value.item(), LIFECYCLE.BATCH)
        
        # Clean up intermediate tensors to save memory
        # Note: Don't delete tensors that are part of the computation graph
        if ctx.cur_mode != MODE.TRAIN:
            # In eval mode, we can safely delete intermediate tensors
            del preference_features, z, mu, logvar

        if torch.isnan(vpl_loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to the loss is NaN, '
                           'it may be caused by exceeding the precision or '
                           'invalid labels.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(torch.zeros(len(win_input_ids)), LIFECYCLE.BATCH)
        ctx.y_pred = CtxVar(
            torch.where(win_rewards.cpu() > lose_rewards.cpu(),
                        torch.zeros(len(win_input_ids)),
                        torch.ones(len(win_input_ids))), LIFECYCLE.BATCH)

        ctx.loss_batch = CtxVar(vpl_loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(win_input_ids), LIFECYCLE.BATCH)

    def _hook_on_batch_end(self, ctx):
        # Update statistics
        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))
        
        # Update VPL-specific statistics
        ctx.vpl_kl_loss_total += ctx.vpl_kl_loss * ctx.batch_size
        ctx.vpl_reconstruction_loss_total += ctx.vpl_reconstruction_loss * ctx.batch_size
        
        # Cache label for evaluate
        ctx.ys_true.append(ctx.y_true.detach().cpu().numpy())
        ctx.ys_pred.append(ctx.y_pred.detach().cpu().numpy())
        
        # Periodic memory cleanup to prevent OOM
        # Clean up every 5 batches to reduce memory usage
        if ctx.cur_batch_i % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _hook_on_fit_end(self, ctx):
        ctx.ys_true = CtxVar(np.concatenate(ctx.ys_true), LIFECYCLE.ROUTINE)
        ctx.ys_pred = CtxVar(np.concatenate(ctx.ys_pred), LIFECYCLE.ROUTINE)
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


def call_vpl_reward_trainer(trainer_type):
    if trainer_type == 'vplrewardtrainer':
        trainer_builder = VPLRewardTrainer
        return trainer_builder


register_trainer('vplrewardtrainer', call_vpl_reward_trainer)
