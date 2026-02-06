#!/usr/bin/env python3
"""
Final analysis: How many clients can we create with threshold 3.0?
Shows actual distribution and recommendations.
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

def analyze_final():
    """Final analysis with threshold 3.0."""
    
    print("=" * 80)
    print("UltraFeedback Multi-Client Analysis (Threshold 3.0)")
    print("=" * 80)
    
    try:
        print("\nLoading dataset...")
        dataset = datasets.load_dataset("openbmb/UltraFeedback")
        data = dataset['train']
        
        annotation_dims = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
        threshold = 3.0
        
        # Collect conflicting pairs with large differences
        large_diff_pairs = []
        
        print(f"\nAnalyzing {len(data)} samples...")
        
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
                            'max_diff': max_diff,
                            'dim_diffs': dim_diffs
                        })
        
        print(f"\nTotal pairs with difference >= {threshold}: {len(large_diff_pairs)}")
        
        # Analyze score difference distribution
        max_diffs = [p['max_diff'] for p in large_diff_pairs]
        print(f"\nScore difference statistics:")
        print(f"  Range: {min(max_diffs):.2f} - {max(max_diffs):.2f}")
        print(f"  Mean: {np.mean(max_diffs):.2f}, Median: {np.median(max_diffs):.2f}")
        print(f"  Unique values: {sorted(set(max_diffs))}")
        
        # Count by exact score difference
        diff_counts = defaultdict(int)
        for pair in large_diff_pairs:
            diff_counts[pair['max_diff']] += 1
        
        print(f"\nDistribution by exact score difference:")
        for diff in sorted(diff_counts.keys()):
            print(f"  {diff:.2f}: {diff_counts[diff]} pairs")
        
        # Analyze by dimension
        print(f"\n{'='*80}")
        print("Distribution by Dimension")
        print(f"{'='*80}")
        
        dim_counts = defaultdict(int)
        for pair in large_diff_pairs:
            dim_counts[pair['winning_dim']] += 1
        
        for dim in annotation_dims:
            count = dim_counts[dim]
            pct = count / len(large_diff_pairs) * 100
            print(f"  {dim}: {count} pairs ({pct:.1f}%)")
        
        # Now analyze client splits
        print(f"\n{'='*80}")
        print("Client Split Analysis")
        print(f"{'='*80}")
        
        def analyze_split(num_clients):
            print(f"\n--- {num_clients} Clients ---")
            
            # Strategy: Split each dimension proportionally
            # helpfulness: 2018 pairs
            # honesty: 4854 pairs  
            # instruction_following: 3754 pairs
            # truthfulness: 1762 pairs
            # Total: 12388 pairs
            
            total_pairs = len(large_diff_pairs)
            
            # Calculate how many clients per dimension based on data amount
            dim_pairs = {dim: dim_counts[dim] for dim in annotation_dims}
            total_dim_pairs = sum(dim_pairs.values())
            
            # Allocate clients proportionally
            dim_client_counts = {}
            remaining_clients = num_clients
            
            # First pass: allocate based on proportion
            for dim in annotation_dims:
                proportion = dim_pairs[dim] / total_dim_pairs
                allocated = max(1, int(proportion * num_clients))
                dim_client_counts[dim] = allocated
                remaining_clients -= allocated
            
            # Distribute remaining clients to dimensions with most data
            if remaining_clients > 0:
                sorted_dims = sorted(dim_pairs.items(), key=lambda x: x[1], reverse=True)
                for i in range(remaining_clients):
                    dim_client_counts[sorted_dims[i % len(sorted_dims)][0]] += 1
            
            print(f"\nClient allocation by dimension:")
            for dim in annotation_dims:
                print(f"  {dim}: {dim_client_counts[dim]} clients, {dim_pairs[dim]} pairs")
            
            # Assign pairs to clients
            client_assignments = defaultdict(list)
            
            for dim in annotation_dims:
                dim_pairs_list = [p for p in large_diff_pairs if p['winning_dim'] == dim]
                
                if len(dim_pairs_list) == 0:
                    continue
                
                # Sort by score difference
                dim_pairs_list.sort(key=lambda x: x['max_diff'], reverse=True)
                
                num_clients_for_dim = dim_client_counts[dim]
                pairs_per_client = len(dim_pairs_list) // num_clients_for_dim
                remainder = len(dim_pairs_list) % num_clients_for_dim
                
                idx = 0
                for client_idx in range(num_clients_for_dim):
                    # Distribute remainder to first few clients
                    num_pairs = pairs_per_client + (1 if client_idx < remainder else 0)
                    client_id = f"{dim}_client_{client_idx}"
                    
                    for _ in range(num_pairs):
                        if idx < len(dim_pairs_list):
                            client_assignments[client_id].append(dim_pairs_list[idx])
                            idx += 1
            
            # Statistics
            client_sizes = [len(pairs) for pairs in client_assignments.values()]
            
            print(f"\nClient size statistics:")
            print(f"  Total clients: {len(client_assignments)}")
            print(f"  Mean: {np.mean(client_sizes):.1f}")
            print(f"  Median: {np.median(client_sizes):.1f}")
            print(f"  Min: {np.min(client_sizes)}")
            print(f"  Max: {np.max(client_sizes)}")
            print(f"  Std: {np.std(client_sizes):.1f}")
            
            # Show distribution
            print(f"\nTop 10 clients:")
            sorted_clients = sorted(client_assignments.items(), key=lambda x: len(x[1]), reverse=True)
            for client_id, pairs in sorted_clients[:10]:
                print(f"  {client_id}: {len(pairs)} pairs")
            
            if len(sorted_clients) > 10:
                print(f"\nBottom 10 clients:")
                for client_id, pairs in sorted_clients[-10:]:
                    print(f"  {client_id}: {len(pairs)} pairs")
            
            # Evaluation
            min_size = np.min(client_sizes)
            mean_size = np.mean(client_sizes)
            
            print(f"\n{'='*80}")
            if min_size >= 1000:
                print(f"✓ Sufficient: Min {min_size} pairs per client (>= 1000)")
            elif min_size >= 500:
                print(f"⚠️  May be sufficient: Min {min_size} pairs per client (>= 500)")
                print(f"   Mean: {mean_size:.1f} pairs per client")
            else:
                print(f"✗ Insufficient: Min {min_size} pairs per client (< 500)")
                print(f"   Mean: {mean_size:.1f} pairs per client")
            
            return client_assignments
        
        client_10 = analyze_split(10)
        client_100 = analyze_split(100)
        
        # Final recommendation
        print(f"\n{'='*80}")
        print("Final Recommendation")
        print(f"{'='*80}")
        
        sizes_10 = [len(pairs) for pairs in client_10.values()]
        sizes_100 = [len(pairs) for pairs in client_100.values()]
        
        print(f"\nWith threshold 3.0 ({len(large_diff_pairs)} pairs):")
        print(f"  10 clients: Min {np.min(sizes_10)}, Mean {np.mean(sizes_10):.1f}")
        print(f"  100 clients: Min {np.min(sizes_100)}, Mean {np.mean(sizes_100):.1f}")
        
        if np.min(sizes_10) < 500:
            print(f"\n⚠️  Warning: Threshold 3.0 may be too high for {len(large_diff_pairs)} pairs.")
            print(f"   Consider:")
            print(f"   - Using threshold 2.5 ({len(large_diff_pairs) * 2} pairs estimated)")
            print(f"   - Using threshold 2.0 ({len(large_diff_pairs) * 2} pairs estimated)")
            print(f"   - Using fewer clients (e.g., 4-8 clients)")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_final()
