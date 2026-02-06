#!/usr/bin/env python3
"""
Analyze UltraFeedback client distribution strategies.
Compare proportional vs equal distribution for 10 clients.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import datasets
except ImportError:
    print("ERROR: datasets library not found. Please install it:")
    print("  pip install datasets")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy library not found. Please install it:")
    print("  pip install numpy")
    sys.exit(1)

from collections import defaultdict

def extract_score_from_annotation(annotation):
    """Extract numeric score from annotation dict."""
    if isinstance(annotation, dict):
        if 'Rating' in annotation:
            try:
                return float(annotation['Rating'])
            except (ValueError, TypeError):
                pass
    elif isinstance(annotation, (int, float)):
        return float(annotation)
    return None

def analyze_distribution():
    """Analyze different client distribution strategies."""
    
    print("=" * 80)
    print("UltraFeedback Client Distribution Analysis (Threshold 3.0)")
    print("=" * 80)
    
    try:
        print("\nLoading dataset...")
        dataset = datasets.load_dataset("openbmb/UltraFeedback")
        data = dataset['train']
        
        annotation_dims = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
        threshold = 3.0
        
        # Collect conflicting pairs
        large_diff_pairs = []
        
        print(f"\nAnalyzing samples with threshold >= {threshold}...")
        
        for idx, sample in enumerate(data):
            if idx % 10000 == 0 and idx > 0:
                print(f"  Processing {idx}/{len(data)}...")
            
            if 'completions' not in sample or len(sample['completions']) < 2:
                continue
            
            completions = sample['completions']
            
            completion_scores = []
            for comp in completions:
                if 'annotations' not in comp:
                    continue
                
                annotations = comp['annotations']
                if not isinstance(annotations, dict):
                    continue
                
                dim_scores = {}
                for dim in annotation_dims:
                    if dim in annotations:
                        score = extract_score_from_annotation(annotations[dim])
                        if score is not None:
                            dim_scores[dim] = score
                
                overall_score = comp.get('overall_score', None)
                if overall_score is None:
                    overall_score = comp.get('fine-grained_score', None)
                
                if dim_scores:
                    completion_scores.append({
                        'dim_scores': dim_scores,
                        'overall_score': overall_score
                    })
            
            if len(completion_scores) < 2:
                continue
            
            def get_sort_key(cs):
                if cs['overall_score'] is not None:
                    return cs['overall_score']
                if cs['dim_scores']:
                    return np.mean(list(cs['dim_scores'].values()))
                return -float('inf')
            
            completion_scores.sort(key=get_sort_key, reverse=True)
            
            best = completion_scores[0]
            best_scores = best['dim_scores']
            
            for other in completion_scores[1:]:
                other_scores = other['dim_scores']
                
                dim_diffs = {}
                best_better_dims = []
                other_better_dims = []
                
                for dim in annotation_dims:
                    if dim in best_scores and dim in other_scores:
                        diff = best_scores[dim] - other_scores[dim]
                        dim_diffs[dim] = diff
                        if diff > 0:
                            best_better_dims.append(dim)
                        elif diff < 0:
                            other_better_dims.append(dim)
                
                if best_better_dims and other_better_dims:
                    max_diff = -float('inf')
                    winning_dim = None
                    
                    for dim in annotation_dims:
                        if dim in dim_diffs:
                            diff = abs(dim_diffs[dim])
                            if diff > max_diff:
                                max_diff = diff
                                winning_dim = dim
                    
                    if winning_dim and max_diff >= threshold:
                        large_diff_pairs.append({
                            'winning_dim': winning_dim,
                            'max_diff': max_diff
                        })
        
        print(f"\nTotal pairs with difference >= {threshold}: {len(large_diff_pairs)}")
        
        # Count by dimension
        dim_counts = defaultdict(int)
        for pair in large_diff_pairs:
            dim_counts[pair['winning_dim']] += 1
        
        print(f"\nPairs by dimension:")
        for dim in annotation_dims:
            print(f"  {dim}: {dim_counts[dim]} pairs")
        
        # Strategy 1: Equal distribution (3, 3, 2, 2)
        print("\n" + "=" * 80)
        print("Strategy 1: Equal Distribution (3, 3, 2, 2)")
        print("=" * 80)
        
        dim_client_counts_equal = {
            'helpfulness': 3,
            'honesty': 3,
            'instruction_following': 2,
            'truthfulness': 2
        }
        
        client_assignments_equal = defaultdict(list)
        
        for dim in annotation_dims:
            dim_pairs_list = [p for p in large_diff_pairs if p['winning_dim'] == dim]
            dim_pairs_list.sort(key=lambda x: x['max_diff'], reverse=True)
            
            num_clients = dim_client_counts_equal[dim]
            pairs_per_client = len(dim_pairs_list) // num_clients
            remainder = len(dim_pairs_list) % num_clients
            
            idx = 0
            for client_idx in range(num_clients):
                num_pairs = pairs_per_client + (1 if client_idx < remainder else 0)
                client_id = f"{dim}_client_{client_idx}"
                
                for _ in range(num_pairs):
                    if idx < len(dim_pairs_list):
                        client_assignments_equal[client_id].append(dim_pairs_list[idx])
                        idx += 1
        
        sizes_equal = [len(pairs) for pairs in client_assignments_equal.values()]
        
        print(f"\nClient distribution:")
        for dim in annotation_dims:
            dim_clients = [k for k in client_assignments_equal.keys() if k.startswith(dim)]
            dim_total = sum(len(client_assignments_equal[k]) for k in dim_clients)
            print(f"  {dim}: {len(dim_clients)} clients, {dim_total} pairs total")
            for client_id in sorted(dim_clients):
                print(f"    {client_id}: {len(client_assignments_equal[client_id])} pairs")
        
        print(f"\nStatistics:")
        print(f"  Total clients: {len(client_assignments_equal)}")
        print(f"  Mean pairs per client: {np.mean(sizes_equal):.1f}")
        print(f"  Min: {np.min(sizes_equal)}, Max: {np.max(sizes_equal)}")
        
        # Strategy 2: Proportional distribution
        print("\n" + "=" * 80)
        print("Strategy 2: Proportional Distribution (by data amount)")
        print("=" * 80)
        
        total_pairs = len(large_diff_pairs)
        total_dim_pairs = sum(dim_counts.values())
        
        dim_client_counts_prop = {}
        remaining_clients = 10
        
        # Allocate based on proportion
        for dim in annotation_dims:
            proportion = dim_counts[dim] / total_dim_pairs
            allocated = max(1, int(proportion * 10))
            dim_client_counts_prop[dim] = allocated
            remaining_clients -= allocated
        
        # Distribute remaining
        if remaining_clients > 0:
            sorted_dims = sorted(dim_counts.items(), key=lambda x: x[1], reverse=True)
            for i in range(remaining_clients):
                dim_client_counts_prop[sorted_dims[i % len(sorted_dims)][0]] += 1
        
        client_assignments_prop = defaultdict(list)
        
        for dim in annotation_dims:
            dim_pairs_list = [p for p in large_diff_pairs if p['winning_dim'] == dim]
            dim_pairs_list.sort(key=lambda x: x['max_diff'], reverse=True)
            
            num_clients = dim_client_counts_prop[dim]
            pairs_per_client = len(dim_pairs_list) // num_clients
            remainder = len(dim_pairs_list) % num_clients
            
            idx = 0
            for client_idx in range(num_clients):
                num_pairs = pairs_per_client + (1 if client_idx < remainder else 0)
                client_id = f"{dim}_client_{client_idx}"
                
                for _ in range(num_pairs):
                    if idx < len(dim_pairs_list):
                        client_assignments_prop[client_id].append(dim_pairs_list[idx])
                        idx += 1
        
        sizes_prop = [len(pairs) for pairs in client_assignments_prop.values()]
        
        print(f"\nClient distribution:")
        for dim in annotation_dims:
            dim_clients = [k for k in client_assignments_prop.keys() if k.startswith(dim)]
            dim_total = sum(len(client_assignments_prop[k]) for k in dim_clients)
            print(f"  {dim}: {len(dim_clients)} clients, {dim_total} pairs total")
            for client_id in sorted(dim_clients):
                print(f"    {client_id}: {len(client_assignments_prop[client_id])} pairs")
        
        print(f"\nStatistics:")
        print(f"  Total clients: {len(client_assignments_prop)}")
        print(f"  Mean pairs per client: {np.mean(sizes_prop):.1f}")
        print(f"  Min: {np.min(sizes_prop)}, Max: {np.max(sizes_prop)}")
        
        # Comparison
        print("\n" + "=" * 80)
        print("Comparison")
        print("=" * 80)
        
        print(f"\nEqual Distribution (3, 3, 2, 2):")
        print(f"  Min: {np.min(sizes_equal)}, Mean: {np.mean(sizes_equal):.1f}, Max: {np.max(sizes_equal)}")
        print(f"  Std: {np.std(sizes_equal):.1f}")
        
        print(f"\nProportional Distribution:")
        print(f"  Min: {np.min(sizes_prop)}, Mean: {np.mean(sizes_prop):.1f}, Max: {np.max(sizes_prop)}")
        print(f"  Std: {np.std(sizes_prop):.1f}")
        
        print(f"\nRecommendation:")
        if np.std(sizes_equal) < np.std(sizes_prop):
            print(f"  ✓ Equal distribution (3, 3, 2, 2) provides more balanced client sizes")
        else:
            print(f"  ✓ Proportional distribution provides more balanced client sizes")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_distribution()
