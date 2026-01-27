#!/usr/bin/env python3
"""
Replot t-SNE visualizations for experiment 59000 using saved JSON data.
Enhanced visualization with better styling matching the new WandB logging format.
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

def load_z_data(json_path):
    """Load z values from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def replot_tsne_from_json(json_path, output_dir, round_num):
    """
    Replot t-SNE visualization from saved JSON data with enhanced styling.
    """
    # Load data
    data = load_z_data(json_path)
    
    z_values = np.array(data['z_values'])
    z_values_2d = np.array(data.get('z_values_2d', None))  # Use saved 2D if available
    client_labels = data['client_labels']
    orthogonal_labels = np.array(data['orthogonal_labels']) if data.get('orthogonal_labels') else None
    round_num_from_file = data.get('round_num', round_num)
    
    num_points = len(z_values)
    num_clients = len(set(client_labels))
    
    print(f"Round {round_num_from_file}: {num_clients} clients, {num_points} points")
    
    # Apply t-SNE if 2D coordinates not available
    if z_values_2d is None:
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
    else:
        z_2d = z_values_2d
    
    # Create figure with better styling
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(14, 11))
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
    
    # Plot z values
    for client_id in unique_clients:
        mask = np.array(client_labels) == client_id
        if mask.sum() > 0:
            # Get label
            if orthogonal_labels is not None and len(orthogonal_labels) == len(client_labels):
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
            
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1], 
                      c=[client_color_map[client_id]], 
                      label=label,
                      alpha=0.85, s=80, edgecolors='white', linewidths=1.0, 
                      marker='o', zorder=3)
    
    # Calculate statistics for title
    if orthogonal_labels is not None:
        harmless_count = sum(1 for l in orthogonal_labels if l == 0)
        helpful_count = sum(1 for l in orthogonal_labels if l == 1)
        title = f'Cross-Client Z Visualization (Round {round_num_from_file})\n({harmless_count} Harmlessness, {helpful_count} Helpfulness clients)'
    else:
        title = f'Cross-Client Z Visualization (Round {round_num_from_file})'
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
              framealpha=0.9, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    
    # Save with enhanced suffix
    save_path = os.path.join(output_dir, f'cross_client_z_tsne_round_{round_num_from_file}_enhanced.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved enhanced visualization to {save_path}")
    
    plt.close(fig)

def main():
    exp_dir = "/home/kjb/ppfl/exp/vplgp_ortho_hhst_n10_t59000/sub_exp_20260127011436"
    
    if not os.path.exists(exp_dir):
        print(f"Experiment directory not found: {exp_dir}")
        return
    
    # Find all JSON files
    json_files = []
    for f in os.listdir(exp_dir):
        if f.startswith('cross_client_z_tsne_round_') and f.endswith('.json'):
            json_files.append(f)
    
    if not json_files:
        print(f"No JSON files found in {exp_dir}")
        return
    
    json_files.sort()
    print(f"Found {len(json_files)} JSON files")
    
    # Replot each JSON file
    for json_file in json_files:
        json_path = os.path.join(exp_dir, json_file)
        # Extract round number from filename
        round_num = int(json_file.split('_round_')[1].split('.')[0])
        
        print(f"\nProcessing {json_file} (Round {round_num})...")
        try:
            replot_tsne_from_json(json_path, exp_dir, round_num)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ All visualizations replotted!")

if __name__ == "__main__":
    main()
