"""
t-SNE visualization for cross-client z values in VPL-GP.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no X server needed)
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
    
    # Create figure with better styling
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor('white')
    
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
                      alpha=0.85, s=80, edgecolors='white', linewidths=1.0, 
                      marker='o', zorder=3)  # Enhanced styling for better visibility
    
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
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    if round_num < 0:
        title = 'Cross-Client Z Visualization (Generation Phase)'
    else:
        title = f'Cross-Client Z Visualization (Round {round_num})'
        # Add statistics to title
        if orthogonal_labels is not None:
            harmless_count = sum(1 for l in orthogonal_labels if l == 0)
            helpful_count = sum(1 for l in orthogonal_labels if l == 1)
            title += f'\n({harmless_count} Harmlessness, {helpful_count} Helpfulness clients)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
              framealpha=0.9, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#FAFAFA')  # Light gray background for better contrast
    
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
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved cross-client z visualization to {save_path}")
        
        # Save z values to JSON file for later analysis
        try:
            # Helper function to safely convert to list
            def to_list_safe(arr):
                if arr is None:
                    return None
                if isinstance(arr, np.ndarray):
                    return arr.tolist()
                elif isinstance(arr, list):
                    return arr
                else:
                    return list(arr)
            
            z_data = {
                'z_values': to_list_safe(z_values),  # Convert numpy array to list
                'z_values_2d': to_list_safe(z_2d),  # Save 2D t-SNE coordinates
                'client_labels': client_labels if isinstance(client_labels, list) else list(client_labels),
                'orthogonal_labels': to_list_safe(orthogonal_labels),
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
                z_data['orthogonal_prototypes'] = to_list_safe(orthogonal_prototypes)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            with open(z_data_path, 'w') as f:
                json.dump(z_data, f, indent=2)
            logger.info(f"Saved z values data to {z_data_path} ({num_points} points, {num_clients} clients)")
        except Exception as e:
            logger.error(f"Failed to save z values to JSON: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Log to WandB if available with enhanced visualization
    if wandb_project:
        try:
            import wandb
            
            # Calculate statistics for each client
            client_stats = []
            for client_id in unique_clients:
                mask = np.array(client_labels) == client_id
                if mask.sum() > 0:
                    client_z_2d = z_2d[mask]
                    client_z_original = z_values[mask]
                    
                    # Calculate statistics
                    mean_x = float(np.mean(client_z_2d[:, 0]))
                    mean_y = float(np.mean(client_z_2d[:, 1]))
                    std_x = float(np.std(client_z_2d[:, 0]))
                    std_y = float(np.std(client_z_2d[:, 1]))
                    
                    # Get orthogonal label
                    if orthogonal_labels is not None and len(orthogonal_labels) == len(client_labels):
                        orth_label = orthogonal_labels[np.where(mask)[0][0]]
                        orth_type = "Harmlessness" if orth_label == 0 else "Helpfulness"
                    else:
                        max_client_id = max(unique_clients) if unique_clients else 0
                        split_point = max_client_id // 2 if max_client_id > 0 else 0
                        orth_type = "Harmlessness" if client_id <= split_point else "Helpfulness"
                    
                    client_stats.append({
                        "Client ID": int(client_id),
                        "Type": orth_type,
                        "Num Points": int(mask.sum()),
                        "Mean X": round(mean_x, 4),
                        "Mean Y": round(mean_y, 4),
                        "Std X": round(std_x, 4),
                        "Std Y": round(std_y, 4),
                        "Color": client_color_map[client_id]
                    })
            
            # Create WandB Table
            table_columns = ["Client ID", "Type", "Num Points", "Mean X", "Mean Y", "Std X", "Std Y", "Color"]
            table_data = [[row[col] for col in table_columns] for row in client_stats]
            table = wandb.Table(columns=table_columns, data=table_data)
            
            # Calculate separation metrics
            harmless_mask = np.array([l == 0 for l in orthogonal_labels]) if orthogonal_labels is not None else np.array([cid <= max(unique_clients) // 2 for cid in client_labels])
            helpful_mask = np.array([l == 1 for l in orthogonal_labels]) if orthogonal_labels is not None else np.array([cid > max(unique_clients) // 2 for cid in client_labels])
            
            separation_metrics = {}
            if harmless_mask.sum() > 0 and helpful_mask.sum() > 0:
                harmless_center = np.mean(z_2d[harmless_mask], axis=0)
                helpful_center = np.mean(z_2d[helpful_mask], axis=0)
                separation_distance = float(np.linalg.norm(harmless_center - helpful_center))
                
                harmless_std = float(np.mean(np.std(z_2d[harmless_mask], axis=0)))
                helpful_std = float(np.mean(np.std(z_2d[helpful_mask], axis=0)))
                avg_std = (harmless_std + helpful_std) / 2
                
                # Separation ratio (higher is better)
                separation_ratio = separation_distance / (avg_std + 1e-8) if avg_std > 0 else 0
                
                separation_metrics = {
                    "z_separation/distance": separation_distance,
                    "z_separation/ratio": separation_ratio,
                    "z_separation/harmless_std": harmless_std,
                    "z_separation/helpful_std": helpful_std
                }
            
            # Enhanced logging with multiple visualizations
            log_dict = {
                f'z_visualization/t-SNE_round_{round_num}': wandb.Image(fig),
                f'z_visualization/client_stats': table,
                f'z_visualization/num_clients': num_clients,
                f'z_visualization/num_points': num_points,
                f'z_visualization/latent_dim': z_values.shape[1] if len(z_values.shape) > 1 else z_values.shape[0]
            }
            
            # Add separation metrics
            log_dict.update(separation_metrics)
            
            # Log to WandB
            wandb.log(log_dict, step=round_num)
            logger.info(f"Logged enhanced cross-client z visualization to wandb at round {round_num} "
                       f"({num_clients} clients, {num_points} points, separation_ratio={separation_metrics.get('z_separation/ratio', 0):.3f})")
        except ImportError:
            logger.warning("wandb not installed, skipping visualization logging")
        except Exception as e:
            logger.warning(f"Failed to log visualization to wandb: {e}")
    
    plt.close(fig)


def compare_algorithms_tsne(json_path1, json_path2, algorithm_name1, algorithm_name2, 
                            output_path=None):
    """
    Compare two algorithms by loading their t-SNE results and plotting side by side.
    
    Args:
        json_path1: Path to first algorithm's JSON file (round 30)
        json_path2: Path to second algorithm's JSON file (round 30)
        algorithm_name1: Name of first algorithm (e.g., "FedVPL")
        algorithm_name2: Name of second algorithm (e.g., "FedVPL-GP")
        output_path: Path to save the comparison plot
    """
    # Load JSON files
    with open(json_path1, 'r') as f:
        data1 = json.load(f)
    with open(json_path2, 'r') as f:
        data2 = json.load(f)
    
    # Extract data
    z_values_2d_1 = np.array(data1['z_values_2d'])
    client_labels_1 = np.array(data1['client_labels'])
    orthogonal_labels_1 = np.array(data1['orthogonal_labels']) if data1.get('orthogonal_labels') is not None else None
    
    z_values_2d_2 = np.array(data2['z_values_2d'])
    client_labels_2 = np.array(data2['client_labels'])
    orthogonal_labels_2 = np.array(data2['orthogonal_labels']) if data2.get('orthogonal_labels') is not None else None
    
    # Create figure with two subplots side by side
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor('white')
    
    # Helper function to plot on an axis
    def plot_tsne_on_axis(ax, z_2d, client_labels, orthogonal_labels, algorithm_name):
        unique_clients = sorted(set(client_labels))
        
        # Color mapping based on orthogonal labels
        client_color_map = {}
        if orthogonal_labels is not None and len(orthogonal_labels) == len(client_labels):
            for client_id in unique_clients:
                client_mask = np.array(client_labels) == client_id
                if client_mask.sum() > 0:
                    orth_label = orthogonal_labels[np.where(client_mask)[0][0]]
                    if orth_label == 0:  # Harmlessness -> red
                        client_color_map[client_id] = '#DC143C'  # Crimson red
                    elif orth_label == 1:  # Helpfulness -> blue
                        client_color_map[client_id] = '#00BFFF'  # Deep sky blue
                    else:
                        client_color_map[client_id] = '#808080'  # Gray
        else:
            # Infer from client_id
            max_client_id = max(unique_clients) if unique_clients else 0
            split_point = max_client_id // 2 if max_client_id > 0 else 0
            for client_id in unique_clients:
                if client_id <= split_point:
                    client_color_map[client_id] = '#DC143C'  # Crimson red
                else:
                    client_color_map[client_id] = '#00BFFF'  # Deep sky blue
        
        # Plot each client
        for client_id in unique_clients:
            mask = np.array(client_labels) == client_id
            if mask.sum() > 0:
                # Get label for legend (only show first few to avoid clutter)
                if orthogonal_labels is not None and len(orthogonal_labels) == len(client_labels):
                    orth_label = orthogonal_labels[np.where(mask)[0][0]]
                    if orth_label == 0:
                        label = f'Client {client_id} (Harmlessness)' if client_id <= 2 else None
                    elif orth_label == 1:
                        label = f'Client {client_id} (Helpfulness)' if client_id <= 2 else None
                    else:
                        label = f'Client {client_id}' if client_id <= 2 else None
                else:
                    max_client_id = max(unique_clients) if unique_clients else 0
                    split_point = max_client_id // 2 if max_client_id > 0 else 0
                    if client_id <= split_point:
                        label = f'Client {client_id} (Harmlessness)' if client_id <= 2 else None
                    else:
                        label = f'Client {client_id} (Helpfulness)' if client_id <= 2 else None
                
                ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                          c=[client_color_map[client_id]],
                          label=label,
                          alpha=0.85, s=80, edgecolors='white', linewidths=1.0,
                          marker='o', zorder=3)
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
        ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
        ax.set_title(algorithm_name, fontsize=18, fontweight='bold', pad=15)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10,
                  framealpha=0.9, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
        ax.set_facecolor('#FAFAFA')
    
    # Plot both algorithms
    plot_tsne_on_axis(ax1, z_values_2d_1, client_labels_1, orthogonal_labels_1, algorithm_name1)
    plot_tsne_on_axis(ax2, z_values_2d_2, client_labels_2, orthogonal_labels_2, algorithm_name2)
    
    plt.tight_layout()
    
    # Save plot
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved comparison plot to {output_path}")
    
    plt.close(fig)
