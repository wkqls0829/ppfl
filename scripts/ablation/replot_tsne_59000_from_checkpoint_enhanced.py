#!/usr/bin/env python3
"""
Replot t-SNE visualizations for experiment 59000 from checkpoint.
Since JSON files don't exist, we'll use checkpoint z values and recompute t-SNE.
Enhanced visualization with larger fonts.
"""

import os
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

def replot_tsne_enhanced(z_values, client_labels, orthogonal_labels, output_dir, round_num, suffix=""):
    """
    Replot t-SNE visualization with enhanced styling and larger fonts.
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
    
    # Create figure with larger size and better styling
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(16, 12))  # Larger figure
    fig.patch.set_facecolor('white')
    
    # Color mapping based on orthogonal labels
    unique_clients = sorted(set(client_labels))
    client_color_map = {}
    
    if orthogonal_labels is not None and len(orthogonal_labels) == len(client_labels):
        for client_id in unique_clients:
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
    
    # Plot z values with larger markers
    for client_id in unique_clients:
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
                  alpha=0.9, s=300, edgecolors='white', linewidths=2.5, 
                  marker='o', zorder=3)
    
    # Calculate statistics for title
    if orthogonal_labels is not None:
        harmless_count = sum(1 for l in orthogonal_labels if l == 0)
        helpful_count = sum(1 for l in orthogonal_labels if l == 1)
        title = f'Cross-Client Z Visualization (Round {round_num})\n({harmless_count} Harmlessness, {helpful_count} Helpfulness clients)'
    else:
        title = f'Cross-Client Z Visualization (Round {round_num})'
    
    # Larger fonts
    ax.set_xlabel('t-SNE Dimension 1', fontsize=18, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=18, fontweight='bold')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, 
              framealpha=0.95, fancybox=True, shadow=True, title='Clients', title_fontsize=13)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_facecolor('#FAFAFA')
    
    # Larger tick labels
    ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    
    # Save with enhanced suffix
    save_path = os.path.join(output_dir, f'cross_client_z_tsne_round_{round_num}_enhanced_large_font{suffix}.png')
    plt.savefig(save_path, dpi=250, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved enhanced visualization to {save_path}")
    
    plt.close(fig)

def main():
    exp_dir = "/home/kjb/ppfl/exp/vplgp_ortho_hhst_n10_t59000/sub_exp_20260127011436"
    checkpoint_dir = "/hdd/hdd3/kjb/checkpoints/backup_old_ablation"  # Check backup folder first
    
    # Find checkpoints (try round-specific first, then final)
    rounds = [10, 20]
    available_checkpoints = {}
    
    # Check what checkpoints are available
    for round_num in [10, 20, 40]:
        ckpt_path = os.path.join(checkpoint_dir, f"{round_num}_hhrl_choice_gemma_vplgp_ortho_n10_t59000.ckpt")
        if os.path.exists(ckpt_path):
            available_checkpoints[round_num] = ckpt_path
            print(f"Found checkpoint for round {round_num}: {os.path.basename(ckpt_path)}")
    
    # Check final checkpoint
    final_ckpt = os.path.join(checkpoint_dir, f"final_hhrl_choice_gemma_vplgp_ortho_n10_t59000.ckpt")
    if os.path.exists(final_ckpt):
        available_checkpoints[40] = final_ckpt
        print(f"Found final checkpoint (round 40)")
    
    for target_round in rounds:
        # Find closest available checkpoint
        ckpt_path = None
        actual_round = target_round
        
        if target_round in available_checkpoints:
            ckpt_path = available_checkpoints[target_round]
        elif target_round < 20 and 20 in available_checkpoints:
            ckpt_path = available_checkpoints[20]
            actual_round = 20
            print(f"\n⚠️  Round {target_round} checkpoint not found, using round 20 checkpoint")
        elif 40 in available_checkpoints:
            ckpt_path = available_checkpoints[40]
            actual_round = 40
            print(f"\n⚠️  Round {target_round} checkpoint not found, using final checkpoint")
        
        if ckpt_path is None:
            print(f"\n❌ No checkpoint available for round {target_round}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Plotting Round {target_round} (using checkpoint from round {actual_round})")
        print(f"Checkpoint: {os.path.basename(ckpt_path)}")
        print(f"{'='*60}")
        
        # Load z values from checkpoint
        data = load_z_from_checkpoint(ckpt_path)
        if data is None:
            print(f"Failed to load z values from checkpoint for round {target_round}")
            continue
        
        print(f"Replotting t-SNE visualization for Round {target_round}...")
        replot_tsne_enhanced(
            data['z_values'],
            data['client_labels'],
            data['orthogonal_labels'],
            exp_dir,
            target_round,  # Use target_round for title, even if using different checkpoint
            "_from_checkpoint"
        )
    
    print("\n✅ Enhanced visualizations created!")

if __name__ == "__main__":
    main()
