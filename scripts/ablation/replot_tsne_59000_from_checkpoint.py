#!/usr/bin/env python3
"""
Replot t-SNE visualizations for experiment 59000 using checkpoint data.
Enhanced visualization with better styling matching the new WandB logging format.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def load_z_from_checkpoint(ckpt_path):
    """Load z values from checkpoint."""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Try to find client_average_z_dict
        client_average_z_dict = None
        if 'client_average_z_dict' in ckpt:
            client_average_z_dict = ckpt['client_average_z_dict']
        elif 'model' in ckpt and 'client_average_z_dict' in ckpt['model']:
            client_average_z_dict = ckpt['model']['client_average_z_dict']
        
        if client_average_z_dict is None:
            print(f"No client_average_z_dict found in checkpoint")
            return None
        
        # Convert to numpy arrays
        z_values_list = []
        client_labels_list = []
        orthogonal_labels_list = []
        
        for client_id, z_mu in client_average_z_dict.items():
            if isinstance(z_mu, torch.Tensor):
                z_np = z_mu.cpu().numpy()
            else:
                z_np = np.array(z_mu)
            
            z_values_list.append(z_np)
            client_labels_list.append(int(client_id))
            
            # Infer orthogonal label from client_id (first half = harmlessness, second half = helpfulness)
            max_client_id = max(client_average_z_dict.keys())
            split_point = max_client_id // 2
            if client_id <= split_point:
                orthogonal_labels_list.append(0)  # Harmlessness
            else:
                orthogonal_labels_list.append(1)  # Helpfulness
        
        z_values = np.array(z_values_list)
        return {
            'z_values': z_values,
            'client_labels': client_labels_list,
            'orthogonal_labels': orthogonal_labels_list
        }
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

def replot_tsne_from_data(z_values, client_labels, orthogonal_labels, output_dir, round_num, suffix=""):
    """
    Replot t-SNE visualization with enhanced styling.
    """
    num_points = len(z_values)
    num_clients = len(set(client_labels))
    
    print(f"Round {round_num}: {num_clients} clients, {num_points} points")
    
    # Apply t-SNE
    if num_points < 2:
        print(f"Not enough points for t-SNE (need at least 2)")
        return
    
    perplexity = min(30, max(5, num_points - 1))
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        z_2d = tsne.fit_transform(z_values)
    except Exception as e:
        print(f"t-SNE failed: {e}, using PCA instead")
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_values)
    
    # Create figure with better styling
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor('white')
    
    # Color mapping based on orthogonal labels
    unique_clients = sorted(set(client_labels))
    client_color_map = {}
    
    if orthogonal_labels is not None and len(orthogonal_labels) == len(client_labels):
        for i, client_id in enumerate(unique_clients):
            client_idx = client_labels.index(client_id) if client_id in client_labels else None
            if client_idx is not None:
                orth_label = orthogonal_labels[client_idx]
                if orth_label == 0:  # Harmlessness
                    client_color_map[client_id] = '#DC143C'  # Crimson red
                elif orth_label == 1:  # Helpfulness
                    client_color_map[client_id] = '#00BFFF'  # Deep sky blue
                else:
                    client_color_map[client_id] = '#808080'  # Gray
    else:
        # Infer from client_id
        max_client_id = max(unique_clients) if unique_clients else 0
        split_point = max_client_id // 2 if max_client_id > 0 else 0
        for client_id in unique_clients:
            if client_id <= split_point:
                client_color_map[client_id] = '#DC143C'  # Harmlessness
            else:
                client_color_map[client_id] = '#00BFFF'  # Helpfulness
    
    # Plot z values
    for i, client_id in enumerate(unique_clients):
        # Find index of this client
        client_indices = [j for j, cid in enumerate(client_labels) if cid == client_id]
        if len(client_indices) == 0:
            continue
        
        idx = client_indices[0]  # Use first occurrence
        
        # Get label
        if orthogonal_labels is not None and len(orthogonal_labels) > idx:
            orth_label = orthogonal_labels[idx]
            if orth_label == 0:
                label = f'Client {client_id} (Harmlessness)'
            elif orth_label == 1:
                label = f'Client {client_id} (Helpfulness)'
            else:
                label = f'Client {client_id}'
        else:
            max_client_id = max(unique_clients) if unique_clients else 0
            split_point = max_client_id // 2 if max_client_id > 0 else 0
            if client_id <= split_point:
                label = f'Client {client_id} (Harmlessness)'
            else:
                label = f'Client {client_id} (Helpfulness)'
        
        ax.scatter(z_2d[idx, 0], z_2d[idx, 1], 
                  c=[client_color_map[client_id]], 
                  label=label,
                  alpha=0.85, s=80, edgecolors='white', linewidths=1.0, 
                  marker='o', zorder=3)
    
    # Calculate statistics for title
    if orthogonal_labels is not None:
        harmless_count = sum(1 for l in orthogonal_labels if l == 0)
        helpful_count = sum(1 for l in orthogonal_labels if l == 1)
        title = f'Cross-Client Z Visualization (Round {round_num})\n({harmless_count} Harmlessness, {helpful_count} Helpfulness clients)'
    else:
        title = f'Cross-Client Z Visualization (Round {round_num})'
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
              framealpha=0.9, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    
    # Save with enhanced suffix
    save_path = os.path.join(output_dir, f'cross_client_z_tsne_round_{round_num}_enhanced{suffix}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved enhanced visualization to {save_path}")
    
    plt.close(fig)

def main():
    exp_dir = "/home/kjb/ppfl/exp/vplgp_ortho_hhst_n10_t59000/sub_exp_20260127011436"
    checkpoint_dir = "/home/kjb/ppfl/checkpoints"
    
    # Find final checkpoint
    ckpt_patterns = [
        f"final_hhrl_choice_gemma_vplgp_ortho_n10_t59000.ckpt",
        f"hhrl_choice_gemma_vplgp_ortho_n10_t59000.ckpt",
        f"40_hhrl_choice_gemma_vplgp_ortho_n10_t59000.ckpt"
    ]
    
    ckpt_path = None
    for pattern in ckpt_patterns:
        path = os.path.join(checkpoint_dir, pattern)
        if os.path.exists(path):
            ckpt_path = path
            print(f"Found checkpoint: {ckpt_path}")
            break
    
    if ckpt_path is None:
        print(f"No checkpoint found. Tried: {ckpt_patterns}")
        return
    
    # Load z values from checkpoint
    data = load_z_from_checkpoint(ckpt_path)
    if data is None:
        print("Failed to load z values from checkpoint")
        return
    
    z_values = data['z_values']
    client_labels = data['client_labels']
    orthogonal_labels = data['orthogonal_labels']
    
    # Replot for round 40 (final round)
    print(f"\nReplotting t-SNE visualization from checkpoint...")
    replot_tsne_from_data(z_values, client_labels, orthogonal_labels, exp_dir, 40, "_from_checkpoint")
    
    print("\n✅ Enhanced visualization created!")

if __name__ == "__main__":
    main()
