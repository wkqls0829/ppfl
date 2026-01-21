"""
Utility functions to load VPL components (variational encoder, feature extractor, latent projection)
from a trained selector model checkpoint.
"""
import torch
import torch.nn as nn
import logging
import os

from federatedscope.llm.model.variational_encoder import VariationalEncoder
from federatedscope.llm.model.variational_encoder_gp import VariationalEncoderGP

logger = logging.getLogger(__name__)


def load_vpl_components_from_checkpoint(checkpoint_path, config, device='cuda:0'):
    """
    Load VPL components (variational encoder, feature extractor, latent projection)
    from a trained selector model checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: Configuration object with VPL hyperparameters
        device: Device to load models on
        
    Returns:
        tuple: (variational_encoder, feature_extractor, latent_projection, z_to_embedding)
        or (None, None, None, None) if not found
    """
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None, None, None
    
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model_state_dict = ckpt.get('model', ckpt)
        
        # Extract VPL hyperparameters from config
        vpl_latent_dim = getattr(config.llm, 'vpl_latent_dim', 32)
        vpl_use_gp_prior = getattr(config.llm, 'vpl_use_gp_prior', False)
        vpl_use_feature_difference = getattr(config.llm, 'vpl_use_feature_difference', True)
        vpl_use_llm_feature_extractor = getattr(config.llm, 'vpl_use_llm_feature_extractor', True)
        
        # Determine input dimensions
        if vpl_use_feature_difference:
            try:
                embedding_dim = config.model.get('hidden_size', 2048)
            except Exception:
                embedding_dim = 2048  # Default for gemma-2b
            if vpl_use_llm_feature_extractor:
                raw_feature_dim = embedding_dim * 3  # [chosen, rejected, difference]
            else:
                raw_feature_dim = embedding_dim  # Only difference
        else:
            raw_feature_dim = len(config.trainer.choices) * 2  # choice_logits
        
        feature_extractor_output_dim = 128  # From VPLRewardChoiceTrainer
        
        # Initialize variational encoder
        if vpl_use_gp_prior:
            num_clients = getattr(config.federate, 'client_num', 10)
            vpl_gp_temperature = getattr(config.llm, 'vpl_gp_temperature', 1.0)
            variational_encoder = VariationalEncoderGP(
                input_dim=feature_extractor_output_dim,
                latent_dim=vpl_latent_dim,
                hidden_dims=[512, 256, 128],
                temperature=vpl_gp_temperature,
                num_clients=num_clients
            ).to(device)
        else:
            variational_encoder = VariationalEncoder(
                input_dim=feature_extractor_output_dim,
                latent_dim=vpl_latent_dim,
                hidden_dims=[512, 256, 128]
            ).to(device)
        
        # Initialize feature extractor
        feature_extractor = nn.Sequential(
            nn.Linear(raw_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, feature_extractor_output_dim)
        ).to(device)
        
        # Initialize latent projection
        choices = getattr(config.trainer, 'choices', ['A', 'B'])
        latent_projection = nn.Linear(vpl_latent_dim, len(choices)).to(device)

        # Initialize z_to_embedding for conditional generation
        try:
            embedding_dim = getattr(config.model, 'hidden_size', None) or getattr(config.model, 'embed_size', None) \
                or config.model.get('hidden_size', None) or config.model.get('embed_size', None)
        except Exception:
            embedding_dim = None
        if embedding_dim is None:
            embedding_dim = 2048  # Fallback for gemma-2b
        z_to_embedding = nn.Linear(vpl_latent_dim, embedding_dim).to(device)
        
        # Try to load from checkpoint
        variational_encoder_loaded = False
        feature_extractor_loaded = False
        latent_projection_loaded = False
        z_to_embedding_loaded = False
        
        for key in model_state_dict.keys():
            if 'variational_encoder' in key:
                # Load variational encoder weights
                try:
                    variational_encoder.load_state_dict(
                        {k.replace('variational_encoder.', ''): v 
                         for k, v in model_state_dict.items() 
                         if 'variational_encoder' in k}, strict=False)
                    variational_encoder_loaded = True
                    logger.info(f"Loaded variational encoder from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load variational encoder: {e}")
            
            if 'feature_extractor' in key:
                # Load feature extractor weights
                try:
                    feature_extractor.load_state_dict(
                        {k.replace('feature_extractor.', ''): v 
                         for k, v in model_state_dict.items() 
                         if 'feature_extractor' in k}, strict=False)
                    feature_extractor_loaded = True
                    logger.info(f"Loaded feature extractor from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load feature extractor: {e}")
            
            if 'latent_projection' in key:
                # Load latent projection weights
                try:
                    latent_projection.load_state_dict(
                        {k.replace('latent_projection.', ''): v 
                         for k, v in model_state_dict.items() 
                         if 'latent_projection' in k}, strict=False)
                    latent_projection_loaded = True
                    logger.info(f"Loaded latent projection from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load latent projection: {e}")

            if 'z_to_embedding' in key:
                try:
                    z_to_embedding.load_state_dict(
                        {k.replace('z_to_embedding.', ''): v
                         for k, v in model_state_dict.items()
                         if 'z_to_embedding' in k}, strict=False)
                    z_to_embedding_loaded = True
                    logger.info(f"Loaded z_to_embedding from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load z_to_embedding: {e}")
        
        if not (variational_encoder_loaded or feature_extractor_loaded or latent_projection_loaded or z_to_embedding_loaded):
            logger.warning("No VPL components found in checkpoint. Using randomly initialized components.")
        
        return variational_encoder, feature_extractor, latent_projection, z_to_embedding
        
    except Exception as e:
        logger.error(f"Failed to load VPL components from checkpoint: {e}")
        return None, None, None, None


def load_vpl_components_from_trainer(trainer):
    """
    Extract VPL components directly from a VPLRewardChoiceTrainer instance.
    
    Args:
        trainer: VPLRewardChoiceTrainer instance
        
    Returns:
        tuple: (variational_encoder, feature_extractor, latent_projection)
    """
    if not hasattr(trainer, 'variational_encoder'):
        logger.error("Trainer does not have variational_encoder. Not a VPL trainer?")
        return None, None, None
    
    variational_encoder = trainer.variational_encoder
    feature_extractor = trainer.feature_extractor if hasattr(trainer, 'feature_extractor') else None
    latent_projection = trainer.latent_projection if hasattr(trainer, 'latent_projection') else None
    
    return variational_encoder, feature_extractor, latent_projection
