"""
Variational Encoder with Gumbel-Softmax Prior for VPL-GP
Implements mixture prior from other clients' z distributions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

from federatedscope.llm.model.variational_encoder import VariationalEncoder

logger = logging.getLogger(__name__)


class VariationalEncoderGP(VariationalEncoder):
    """
    Variational Encoder with Gumbel-Softmax Prior for Federated VPL.
    
    Extends VariationalEncoder to use a mixture prior from other clients'
    z distributions instead of a fixed standard normal prior.
    
    The mixture prior is: p_mixture(z) = Σ_i w_i * N(z; μ_i, σ_i²)
    where μ_i, σ_i² are learned from other clients' z distributions.
    """
    def __init__(self, input_dim, latent_dim=32, hidden_dims=[256, 128], 
                 temperature=1.0, num_clients=10):
        super(VariationalEncoderGP, self).__init__(input_dim, latent_dim, hidden_dims)
        self.temperature = temperature
        self.num_clients = num_clients
        
        # Prior distributions (will be updated from server)
        self.prior_mus = None  # (num_clients, latent_dim)
        self.prior_logvars = None  # (num_clients, latent_dim)
        self.prior_weights = None  # (num_clients,)
        
    def update_prior(self, client_mus, client_logvars, client_weights):
        """
        Update the mixture prior from other clients' z distributions.
        
        Args:
            client_mus: Mean vectors from other clients (num_clients, latent_dim)
            client_logvars: Log variance vectors from other clients (num_clients, latent_dim)
            client_weights: Weights for each client distribution (num_clients,)
        """
        if isinstance(client_mus, list):
            client_mus = torch.stack(client_mus)
        if isinstance(client_logvars, list):
            client_logvars = torch.stack(client_logvars)
        if isinstance(client_weights, list):
            client_weights = torch.tensor(client_weights, dtype=torch.float32)
        
        # Ensure tensors are on the same device
        device = next(self.parameters()).device
        self.prior_mus = client_mus.to(device)
        self.prior_logvars = client_logvars.to(device)
        self.prior_weights = client_weights.to(device)
        
        # Normalize weights
        if self.prior_weights.sum() > 0:
            self.prior_weights = self.prior_weights / self.prior_weights.sum()
        
        # Log statistics
        mu_norm = torch.norm(self.prior_mus, dim=-1).mean().item()
        logvar_mean = self.prior_logvars.mean().item()
        num_clients = len(self.prior_mus)
        
        # Compute average distance between mus
        if num_clients > 1:
            mu_distances = []
            for i in range(num_clients):
                for j in range(i + 1, num_clients):
                    dist = torch.norm(self.prior_mus[i] - self.prior_mus[j]).item()
                    mu_distances.append(dist)
            avg_mu_distance = np.mean(mu_distances) if mu_distances else 0.0
        else:
            avg_mu_distance = 0.0
        
        logger.info(f"Updated VPL-GP prior: mu_norm={mu_norm:.4f}, "
                   f"logvar_mean={logvar_mean:.4f}, num_clients={num_clients}, "
                   f"avg_mu_distance={avg_mu_distance:.4f}")
    
    def sample_prior(self, batch_size, use_gumbel=True):
        """
        Sample from the mixture prior using Gumbel-Softmax.
        
        Args:
            batch_size: Number of samples to generate
            use_gumbel: Whether to use Gumbel-Softmax (default: True)
            
        Returns:
            z_samples: Sampled latent vectors (batch_size, latent_dim)
        """
        if self.prior_mus is None:
            # Fallback to standard normal
            return torch.randn(batch_size, self.latent_dim, 
                             device=next(self.parameters()).device)
        
        num_components = len(self.prior_mus)
        device = self.prior_mus.device
        
        if use_gumbel and num_components > 1:
            # Gumbel-Softmax to select which component to sample from
            log_weights = torch.log(self.prior_weights + 1e-8)
            gumbel_noise = -torch.log(-torch.log(torch.rand(batch_size, num_components, device=device) + 1e-8) + 1e-8)
            gumbel_logits = (log_weights.unsqueeze(0) + gumbel_noise) / self.temperature
            component_probs = F.softmax(gumbel_logits, dim=-1)  # (batch_size, num_components)
            
            # Sample from each component
            z_samples_list = []
            for i in range(num_components):
                mu_i = self.prior_mus[i]  # (latent_dim,)
                logvar_i = self.prior_logvars[i]  # (latent_dim,)
                std_i = torch.exp(0.5 * logvar_i)
                eps = torch.randn(batch_size, self.latent_dim, device=device)
                z_i = mu_i.unsqueeze(0) + eps * std_i.unsqueeze(0)  # (batch_size, latent_dim)
                
                # Weight by component probability
                weight = component_probs[:, i:i+1]  # (batch_size, 1)
                z_samples_list.append(z_i * weight)
            
            # Combine weighted samples
            z_samples = sum(z_samples_list)  # (batch_size, latent_dim)
        else:
            # Simple weighted average (no Gumbel-Softmax)
            # Sample from each component and take weighted average
            z_samples_list = []
            for i in range(num_components):
                mu_i = self.prior_mus[i]
                logvar_i = self.prior_logvars[i]
                std_i = torch.exp(0.5 * logvar_i)
                eps = torch.randn(batch_size, self.latent_dim, device=device)
                z_i = mu_i.unsqueeze(0) + eps * std_i.unsqueeze(0)
                weight = self.prior_weights[i]
                z_samples_list.append(z_i * weight)
            
            z_samples = sum(z_samples_list)
        
        return z_samples
    
    def kl_divergence(self, mu, logvar, use_gumbel_prior=True):
        """
        Compute KL divergence KL(q(z|x) || p_mixture(z)) where p_mixture is the
        mixture prior from other clients.
        
        Uses log-sum-exp trick for numerical stability.
        
        Args:
            mu: Mean of posterior q(z|x) (batch_size, latent_dim)
            logvar: Log variance of posterior q(z|x) (batch_size, latent_dim)
            use_gumbel_prior: Whether to use Gumbel-Softmax prior (default: True)
            
        Returns:
            kl: KL divergence (scalar)
        """
        if self.prior_mus is None:
            # Fallback to standard KL divergence
            return super().kl_divergence(mu, logvar)
        
        batch_size = mu.shape[0]
        num_components = len(self.prior_mus)
        device = mu.device
        
        # Compute log q(z|x) for sampled z
        # We'll use the reparameterization: z = mu + eps * std
        # For KL, we need E_q[log q(z|x) - log p_mixture(z)]
        
        # Sample z from q(z|x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # (batch_size, latent_dim)
        
        # Compute log q(z|x)
        # log q(z|x) = -0.5 * sum(log(2π) + logvar + (z - mu)^2 / var)
        log_q = -0.5 * torch.sum(
            np.log(2 * np.pi) + logvar + (z - mu).pow(2) / torch.exp(logvar),
            dim=-1
        )  # (batch_size,)
        
        # Compute log p_mixture(z) = log(Σ_i w_i * N(z; μ_i, σ_i²))
        # Use log-sum-exp trick: log(Σ exp(a_i)) = max(a_i) + log(Σ exp(a_i - max(a_i)))
        log_p_components = []
        for i in range(num_components):
            mu_i = self.prior_mus[i]  # (latent_dim,)
            logvar_i = self.prior_logvars[i]  # (latent_dim,)
            weight_i = self.prior_weights[i]
            
            # log N(z; μ_i, σ_i²) = -0.5 * sum(log(2π) + logvar_i + (z - μ_i)^2 / var_i)
            log_p_i = -0.5 * torch.sum(
                np.log(2 * np.pi) + logvar_i + (z - mu_i.unsqueeze(0)).pow(2) / torch.exp(logvar_i.unsqueeze(0)),
                dim=-1
            )  # (batch_size,)
            
            # Add log weight
            log_p_i = log_p_i + torch.log(weight_i + 1e-8)
            log_p_components.append(log_p_i)
        
        # Stack and use log-sum-exp
        log_p_stack = torch.stack(log_p_components, dim=0)  # (num_components, batch_size)
        log_p_max = torch.max(log_p_stack, dim=0, keepdim=True)[0]  # (1, batch_size)
        log_p_mixture = log_p_max.squeeze(0) + torch.log(
            torch.sum(torch.exp(log_p_stack - log_p_max), dim=0) + 1e-8
        )  # (batch_size,)
        
        # KL = E_q[log q(z|x) - log p_mixture(z)]
        kl = (log_q - log_p_mixture).mean()
        
        # Log comparison with standard KL for debugging
        if torch.rand(1).item() < 0.01:  # Log 1% of the time
            standard_kl = super().kl_divergence(mu, logvar)
            diff = kl.item() - standard_kl.item()
            
            # Compute statistics
            active_clients = num_components
            active_weights_sum = self.prior_weights.sum().item()
            
            # Average mu distance
            if num_components > 1:
                mu_distances = []
                for i in range(num_components):
                    for j in range(i + 1, num_components):
                        dist = torch.norm(self.prior_mus[i] - self.prior_mus[j]).item()
                        mu_distances.append(dist)
                avg_mu_distance = np.mean(mu_distances) if mu_distances else 0.0
            else:
                avg_mu_distance = 0.0
            
            # Top mixing weights
            top_weights, top_indices = torch.topk(self.prior_weights, k=min(3, num_components))
            top_mixing_str = ', '.join([f"client_{idx.item()}:{w.item():.4f}" 
                                       for w, idx in zip(top_weights, top_indices)])
            
            logger.info(f"KL divergence comparison: mixture={kl.item():.4f}, "
                       f"standard={standard_kl.item():.4f}, diff={diff:.4f}, "
                       f"active_clients={active_clients}, active_weights_sum={active_weights_sum:.4f}, "
                       f"avg_mu_distance={avg_mu_distance:.4f}, top_mixing_weights=[{top_mixing_str}]")
        
        return kl
