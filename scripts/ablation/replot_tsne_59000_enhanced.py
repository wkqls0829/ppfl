#!/usr/bin/env python3
"""
Replot t-SNE visualizations for experiment 59000 from JSON files.
Enhanced visualization with larger fonts and better styling.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

def load_z_from_json(json_path):
    """Load z values from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        z_values = np.array(data['z_values'])
        z_values_2d = np.array(data['z_values_2d'])  # Already computed t-SNE
        client_labels = data['client_labels']
        orthogonal_labels = np.array(data['orthogonal_labels']) if data.get('orthogonal_labels') else None
        round_num = data['round_num']
        
        return {
            'z_values': z_values,
            'z_values_2d': z_values_2d,
            'client_labels': client_labels,
            'orthogonal_labels': orthogonal_labels,
            'round_num': round_num
        }
    except Exception as e:
        print(f"Error loading JSON: {e}")
        import traceback
        traceback.print_exc()
        return None

def replot_tsne_enhanced(z_values_2d, client_labels, orthogonal_labels, output_dir, round_num, suffix=""):
    """
    Replot t-SNE visualization with enhanced styling and larger fonts.
    """
    num_points = len(z_values_2d)
    num_clients = len(set(client_labels))
    
    print(f"Round {round_num}: {num_clients} clients, {num_points} points")
    
    # Create figure with larger size and better styling
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(16, 12))  # Larger figure
    fig.patch.set_facecolor('white')
    
    # Color mapping based on orthogonal labels
    unique_clients = sorted(set(client_labels))
    client_color_map = {}
    
    if orthogonal_labels is not None and len(orthogonal_labels) == len(client_labels):
        for client_id in unique_clients:
            client_mask = np.array(client_labels) == client_id
            if client_mask.sum() > 0:
                orth_label = orthogonal_labels[np.where(client_mask)[0][0]]
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
        mask = np.array(client_labels) == client_id
        if mask.sum() == 0:
            continue
        
        client_z_2d = z_values_2d[mask]
        
        # Get label
        if orthogonal_labels is not None and len(orthogonal_labels) > 0:
            orth_label = orthogonal_labels[np.where(mask)[0][0]]
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
        
        # Use mean position for single point per client
        if len(client_z_2d) == 1:
            ax.scatter(client_z_2d[0, 0], client_z_2d[0, 1], 
                      c=[client_color_map[client_id]], 
                      label=label,
                      alpha=0.9, s=200, edgecolors='white', linewidths=2.0, 
                      marker='o', zorder=3)
        else:
            # Multiple points per client - plot all with mean highlighted
            ax.scatter(client_z_2d[:, 0], client_z_2d[:, 1], 
                      c=[client_color_map[client_id]], 
                      label=label,
                      alpha=0.7, s=120, edgecolors='white', linewidths=1.5, 
                      marker='o', zorder=2)
            # Highlight mean
            mean_pos = np.mean(client_z_2d, axis=0)
            ax.scatter(mean_pos[0], mean_pos[1], 
                      c=[client_color_map[client_id]], 
                      alpha=1.0, s=300, edgecolors='black', linewidths=2.5, 
                      marker='*', zorder=4)
    
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
    
    # Try to find JSON files for rounds 10 and 20
    rounds = [10, 20]
    
    for round_num in rounds:
        json_path = os.path.join(exp_dir, f'cross_client_z_tsne_round_{round_num}.json')
        
        if os.path.exists(json_path):
            print(f"\n{'='*60}")
            print(f"Loading JSON for Round {round_num}")
            print(f"{'='*60}")
            
            data = load_z_from_json(json_path)
            if data is None:
                print(f"Failed to load JSON for round {round_num}")
                continue
            
            print(f"Replotting t-SNE visualization for Round {round_num}...")
            replot_tsne_enhanced(
                data['z_values_2d'],
                data['client_labels'],
                data['orthogonal_labels'],
                exp_dir,
                round_num,
                "_from_json"
            )
        else:
            print(f"\n⚠️  JSON file not found: {json_path}")
            print(f"   Skipping round {round_num}")
    
    print("\n✅ Enhanced visualizations created!")

if __name__ == "__main__":
    main()
