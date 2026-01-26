"""
t-SNE visualization for cross-client z values in VPL-GP.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging
import os
import json

logger = logging.getLogger(__name__)


def visualize_cross_client_z(z_values, client_labels, orthogonal_labels=None,
                             orthogonal_prototypes=None, round_num=0,
                             output_dir=None, wandb_project=None):
    """
    Visualize cross-client z values using t-SNE.
    
    Args:
        z_values: Array of z values (num_points, latent_dim)
        client_labels: List of client IDs for each z value (num_points,)
        orthogonal_labels: Optional list of orthogonal labels (num_points,)
        orthogonal_prototypes: Optional array of orthogonal prototypes (num_prototypes, latent_dim)
        round_num: Current round number
        output_dir: Output directory for saving plots
        wandb_project: WandB project name (optional)
    """
    if len(z_values) == 0:
        logger.warning("No z values to visualize")
        return
    
    # Convert to numpy if needed
    if not isinstance(z_values, np.ndarray):
        z_values = np.array(z_values)
    
    num_points = len(z_values)
    num_clients = len(set(client_labels))
    
    logger.info(f"Round {round_num}: Visualizing z from {num_clients} clients "
               f"at round {round_num} ({num_points} total points, shape: {z_values.shape})")
    
    # Apply t-SNE
    if num_points < 2:
        logger.warning("Not enough points for t-SNE (need at least 2)")
        return
    
    # Reduce perplexity if we have few points
    perplexity = min(30, max(5, num_points - 1))
    
    tsne_successful = True
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        z_2d = tsne.fit_transform(z_values)
    except Exception as e:
        logger.warning(f"t-SNE failed: {e}, using PCA instead")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_values)
        tsne_successful = False
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot z values colored by client
    unique_clients = sorted(set(client_labels))
    
    # Color mapping based on orthogonal labels (harmlessness=red, helpfulness=blue)
    # If orthogonal_labels are available, use them; otherwise infer from client_id
    client_color_map = {}
    if orthogonal_labels is not None and len(orthogonal_labels) == len(client_labels):
        # Map orthogonal labels to colors
        for client_id in unique_clients:
            client_mask = np.array(client_labels) == client_id
            if client_mask.sum() > 0:
                # Get the first orthogonal label for this client (all should be the same)
                orth_label = orthogonal_labels[np.where(client_mask)[0][0]]
                if orth_label == 0:  # Harmlessness -> red/orange-red
                    client_color_map[client_id] = '#DC143C'  # Crimson red (more vivid)
                elif orth_label == 1:  # Helpfulness -> sky blue
                    client_color_map[client_id] = '#00BFFF'  # Deep sky blue (more vivid)
                else:
                    # Default color for unlabeled
                    client_color_map[client_id] = '#808080'  # Gray
        # Fallback: use default colors if mapping failed
        if len(client_color_map) < len(unique_clients):
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clients)))
            for i, cid in enumerate(unique_clients):
                if cid not in client_color_map:
                    client_color_map[cid] = colors[i % len(colors)]
    else:
        # If orthogonal_labels not available, infer from client_id
        # Assume first half are harmlessness, second half are helpfulness
        # This matches the data distribution in hh-rlhf dataset
        max_client_id = max(unique_clients) if unique_clients else 0
        split_point = max_client_id // 2 if max_client_id > 0 else 0
        
        for client_id in unique_clients:
            if client_id <= split_point:
                client_color_map[client_id] = '#DC143C'  # Crimson red for harmlessness (more vivid)
            else:
                client_color_map[client_id] = '#00BFFF'  # Deep sky blue for helpfulness (more vivid)
    
    for client_id in unique_clients:
        mask = np.array(client_labels) == client_id
        if mask.sum() > 0:
            # Get orthogonal label for legend
            if orthogonal_labels is not None and len(orthogonal_labels) == len(client_labels):
                orth_label = orthogonal_labels[np.where(mask)[0][0]]
                if orth_label == 0:
                    label = f'Client {client_id} (Harmlessness)'
                elif orth_label == 1:
                    label = f'Client {client_id} (Helpfulness)'
                else:
                    label = f'Client {client_id}'
            else:
                # Infer from client_id if orthogonal_labels not available
                max_client_id = max(unique_clients) if unique_clients else 0
                split_point = max_client_id // 2 if max_client_id > 0 else 0
                if client_id <= split_point:
                    label = f'Client {client_id} (Harmlessness)'
                else:
                    label = f'Client {client_id} (Helpfulness)'
            
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1], 
                      c=[client_color_map[client_id]], 
                      label=label,
                      alpha=0.8, s=60, edgecolors='white', linewidths=0.5)  # Increased alpha and size, added edge
    
    # Plot orthogonal prototypes if available
    if orthogonal_prototypes is not None:
        if not isinstance(orthogonal_prototypes, np.ndarray):
            orthogonal_prototypes = np.array(orthogonal_prototypes)
        
        if len(orthogonal_prototypes) > 0:
            # Project prototypes to 2D using the same t-SNE transform
            # Note: We need to refit with prototypes included, or use a different approach
            # For simplicity, we'll project prototypes separately
            try:
                # Combine z_values and prototypes for t-SNE
                combined = np.vstack([z_values, orthogonal_prototypes])
                tsne_combined = TSNE(n_components=2, random_state=42, 
                                    perplexity=perplexity, n_iter=1000)
                combined_2d = tsne_combined.fit_transform(combined)
                prototypes_2d = combined_2d[-len(orthogonal_prototypes):]
                
                # Plot prototypes with distinct markers
                for i, prototype_2d in enumerate(prototypes_2d):
                    ax.scatter(prototype_2d[0], prototype_2d[1],
                             marker='*', s=500, c='red', 
                             edgecolors='black', linewidths=2,
                             label=f'Prototype {i}' if i < 2 else None,
                             zorder=10)
            except Exception as e:
                logger.warning(f"Failed to project prototypes: {e}")
    
    # Plot orthogonal labels if available
    if orthogonal_labels is not None:
        unique_orth_labels = sorted(set(orthogonal_labels))
        if len(unique_orth_labels) > 1:
            # Add a second plot or overlay
            for orth_label in unique_orth_labels:
                if orth_label >= 0:  # Skip -1 (unlabeled)
                    mask = np.array(orthogonal_labels) == orth_label
                    if mask.sum() > 0:
                        # Draw contour or highlight
                        ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                                 edgecolors='black', linewidths=1,
                                 alpha=0.3, s=60, zorder=5)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    if round_num < 0:
        ax.set_title('Cross-Client Z Visualization (Generation Phase)', fontsize=14)
    else:
        ax.set_title(f'Cross-Client Z Visualization (Round {round_num})', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if round_num < 0:
            # Use "generation" suffix for pre-training visualization
            save_path = os.path.join(output_dir, 'cross_client_z_tsne_generation.png')
            z_data_path = os.path.join(output_dir, 'cross_client_z_tsne_generation.json')
        else:
            save_path = os.path.join(output_dir, f'cross_client_z_tsne_round_{round_num}.png')
            z_data_path = os.path.join(output_dir, f'cross_client_z_tsne_round_{round_num}.json')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cross-client z visualization to {save_path}")
        
        # Save z values to JSON file for later analysis
        try:
            z_data = {
                'z_values': z_values.tolist(),  # Convert numpy array to list
                'z_values_2d': z_2d.tolist(),  # Save 2D t-SNE coordinates
                'client_labels': client_labels,
                'orthogonal_labels': orthogonal_labels.tolist() if orthogonal_labels is not None else None,
                'round_num': round_num,
                'num_points': num_points,
                'num_clients': num_clients,
                'latent_dim': z_values.shape[1] if len(z_values.shape) > 1 else z_values.shape[0],
                'metadata': {
                    'perplexity': perplexity,
                    'tsne_successful': tsne_successful
                }
            }
            
            # Add orthogonal prototypes if available
            if orthogonal_prototypes is not None:
                z_data['orthogonal_prototypes'] = orthogonal_prototypes.tolist()
            
            with open(z_data_path, 'w') as f:
                json.dump(z_data, f, indent=2)
            logger.info(f"Saved z values data to {z_data_path} ({num_points} points, {num_clients} clients)")
        except Exception as e:
            logger.warning(f"Failed to save z values to JSON: {e}")
    
    # Log to WandB if available
    if wandb_project:
        try:
            import wandb
            wandb.log({
                f'visualization/cross_client_z_tsne_round_{round_num}': wandb.Image(fig)
            }, step=round_num)
            logger.info(f"Logged cross-client z t-SNE visualization to wandb at round {round_num}")
        except ImportError:
            logger.warning("wandb not installed, skipping visualization logging")
        except Exception as e:
            logger.warning(f"Failed to log visualization to wandb: {e}")
    
    plt.close(fig)
