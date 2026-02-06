#!/usr/bin/env python3
"""
Analyze UltraFeedback dataset for multiple client scenarios (10, 100 clients).
This analyzes how data would be distributed when splitting into more clients.
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

def analyze_multi_client():
    """Analyze data distribution for 10 and 100 clients."""
    
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
        
        print(f"\nAnalyzing {len(data)} samples with threshold >= {threshold}...")
        
        for idx, sample in enumerate(data):
            if idx % 5000 == 0 and idx > 0:
                print(f"  Processing {idx}/{len(data)}...")
            
            if 'completions' not in sample or len(sample['completions']) < 2:
                continue
            
            completions = sample['completions']
            
            # Extract scores for each completion
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
                        'comp': comp,
                        'dim_scores': dim_scores,
                        'overall_score': overall_score
                    })
            
            if len(completion_scores) < 2:
                continue
            
            # Sort by overall_score
            def get_sort_key(cs):
                if cs['overall_score'] is not None:
                    return cs['overall_score']
                if cs['dim_scores']:
                    return np.mean(list(cs['dim_scores'].values()))
                return -float('inf')
            
            completion_scores.sort(key=get_sort_key, reverse=True)
            
            # Check pairs
            best = completion_scores[0]
            best_scores = best['dim_scores']
            
            for other in completion_scores[1:]:
                other_scores = other['dim_scores']
                
                # Calculate differences
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
                
                # Conflicting pair
                if best_better_dims and other_better_dims:
                    # Find which dimension has the largest difference
                    max_diff = -float('inf')
                    winning_dim = None
                    
                    for dim in annotation_dims:
                        if dim in dim_diffs:
                            diff = abs(dim_diffs[dim])
                            if diff > max_diff:
                                max_diff = diff
                                winning_dim = dim
                    
                    # Only include if difference >= threshold
                    if winning_dim and max_diff >= threshold:
                        large_diff_pairs.append({
                            'best_scores': best_scores,
                            'other_scores': other_scores,
                            'dim_diffs': dim_diffs,
                            'winning_dim': winning_dim,
                            'max_diff': max_diff,
                            'all_diffs': {dim: abs(dim_diffs.get(dim, 0)) for dim in annotation_dims}
                        })
        
        print(f"\nTotal pairs with difference >= {threshold}: {len(large_diff_pairs)}")
        
        # Strategy 1: Split each dimension into multiple clients (by score difference magnitude)
        print("\n" + "=" * 80)
        print("Strategy 1: Split by Dimension + Score Difference Magnitude")
        print("=" * 80)
        
        # For 10 clients: 4 dimensions * 2-3 groups = 8-12 clients (use 10)
        # For 100 clients: 4 dimensions * 25 groups = 100 clients
        
        def analyze_client_split(num_clients):
            print(f"\n--- {num_clients} Clients ---")
            
            if num_clients == 10:
                # 4 dimensions, split each into ~2-3 groups
                # helpfulness: 2 groups, honesty: 2 groups, instruction_following: 2 groups, truthfulness: 2 groups = 8
                # Or: 3, 3, 2, 2 = 10
                dim_groups = {
                    'helpfulness': 3,
                    'honesty': 3,
                    'instruction_following': 2,
                    'truthfulness': 2
                }
            elif num_clients == 100:
                # 4 dimensions, split each into 25 groups
                dim_groups = {
                    'helpfulness': 25,
                    'honesty': 25,
                    'instruction_following': 25,
                    'truthfulness': 25
                }
            else:
                # General case: distribute evenly
                groups_per_dim = num_clients // 4
                remainder = num_clients % 4
                dim_groups = {
                    'helpfulness': groups_per_dim + (1 if remainder > 0 else 0),
                    'honesty': groups_per_dim + (1 if remainder > 1 else 0),
                    'instruction_following': groups_per_dim + (1 if remainder > 2 else 0),
                    'truthfulness': groups_per_dim
                }
            
            # Assign pairs to clients based on dimension and score difference
            client_assignments = defaultdict(list)
            
            for pair in large_diff_pairs:
                winning_dim = pair['winning_dim']
                max_diff = pair['max_diff']
                
                # Get score difference range for this dimension
                dim_pairs = [p for p in large_diff_pairs if p['winning_dim'] == winning_dim]
                dim_diffs = [p['max_diff'] for p in dim_pairs]
                
                if len(dim_diffs) == 0:
                    continue
                
                min_diff = min(dim_diffs)
                max_diff_range = max(dim_diffs)
                diff_range = max_diff_range - min_diff
                
                # Determine which group this pair belongs to
                num_groups = dim_groups[winning_dim]
                if diff_range > 0:
                    normalized_diff = (max_diff - min_diff) / diff_range
                    group_idx = min(int(normalized_diff * num_groups), num_groups - 1)
                else:
                    group_idx = 0
                
                client_id = f"{winning_dim}_{group_idx}"
                client_assignments[client_id].append(pair)
            
            # Print statistics
            print(f"\nTotal pairs: {len(large_diff_pairs)}")
            print(f"Number of clients: {len(client_assignments)}")
            
            client_sizes = [len(pairs) for pairs in client_assignments.values()]
            print(f"\nClient size statistics:")
            print(f"  Mean: {np.mean(client_sizes):.1f}")
            print(f"  Median: {np.median(client_sizes):.1f}")
            print(f"  Min: {np.min(client_sizes)}")
            print(f"  Max: {np.max(client_sizes)}")
            print(f"  Std: {np.std(client_sizes):.1f}")
            
            # Show distribution by dimension
            print(f"\nDistribution by dimension:")
            for dim in annotation_dims:
                dim_clients = [k for k in client_assignments.keys() if k.startswith(dim)]
                dim_total = sum(len(client_assignments[k]) for k in dim_clients)
                print(f"  {dim}: {len(dim_clients)} clients, {dim_total} pairs")
            
            # Show top and bottom clients
            sorted_clients = sorted(client_assignments.items(), key=lambda x: len(x[1]), reverse=True)
            print(f"\nTop 5 clients (by size):")
            for client_id, pairs in sorted_clients[:5]:
                print(f"  {client_id}: {len(pairs)} pairs")
            
            print(f"\nBottom 5 clients (by size):")
            for client_id, pairs in sorted_clients[-5:]:
                print(f"  {client_id}: {len(pairs)} pairs")
            
            # Check if sufficient
            min_size = np.min(client_sizes)
            if min_size >= 1000:
                print(f"\n✓ Sufficient: Min {min_size} pairs per client (>= 1000)")
            elif min_size >= 500:
                print(f"\n⚠️  May be sufficient: Min {min_size} pairs per client (>= 500)")
            else:
                print(f"\n✗ Insufficient: Min {min_size} pairs per client (< 500)")
            
            return client_assignments
        
        # Analyze for 10 and 100 clients
        client_10 = analyze_client_split(10)
        client_100 = analyze_client_split(100)
        
        # Strategy 2: Alternative - Split by score difference ranges only
        print("\n" + "=" * 80)
        print("Strategy 2: Split by Score Difference Ranges Only")
        print("=" * 80)
        
        # For 10 clients: split score differences into 10 ranges
        # For 100 clients: split into 100 ranges
        
        def analyze_by_diff_ranges(num_clients):
            print(f"\n--- {num_clients} Clients (by difference ranges) ---")
            
            # Get all max differences
            max_diffs = [p['max_diff'] for p in large_diff_pairs]
            min_diff = min(max_diffs)
            max_diff = max(max_diffs)
            diff_range = max_diff - min_diff
            
            # Create ranges
            client_assignments = defaultdict(list)
            
            for pair in large_diff_pairs:
                max_diff_val = pair['max_diff']
                if diff_range > 0:
                    normalized = (max_diff_val - min_diff) / diff_range
                    client_idx = min(int(normalized * num_clients), num_clients - 1)
                else:
                    client_idx = 0
                
                client_id = f"client_{client_idx}"
                client_assignments[client_id].append(pair)
            
            client_sizes = [len(pairs) for pairs in client_assignments.values()]
            print(f"\nTotal pairs: {len(large_diff_pairs)}")
            print(f"Number of clients: {len(client_assignments)}")
            print(f"\nClient size statistics:")
            print(f"  Mean: {np.mean(client_sizes):.1f}")
            print(f"  Median: {np.median(client_sizes):.1f}")
            print(f"  Min: {np.min(client_sizes)}")
            print(f"  Max: {np.max(client_sizes)}")
            print(f"  Std: {np.std(client_sizes):.1f}")
            
            min_size = np.min(client_sizes)
            if min_size >= 1000:
                print(f"\n✓ Sufficient: Min {min_size} pairs per client")
            elif min_size >= 500:
                print(f"\n⚠️  May be sufficient: Min {min_size} pairs per client")
            else:
                print(f"\n✗ Insufficient: Min {min_size} pairs per client")
        
        analyze_by_diff_ranges(10)
        analyze_by_diff_ranges(100)
        
        # Save results
        import json
        output_file = "ultrafeedback_multi_client_analysis.json"
        
        # Prepare summary
        summary = {
            'threshold': threshold,
            'total_pairs': len(large_diff_pairs),
            'strategy1_10clients': {
                'num_clients': len(client_10),
                'min_pairs': min(len(pairs) for pairs in client_10.values()),
                'max_pairs': max(len(pairs) for pairs in client_10.values()),
                'mean_pairs': np.mean([len(pairs) for pairs in client_10.values()])
            },
            'strategy1_100clients': {
                'num_clients': len(client_100),
                'min_pairs': min(len(pairs) for pairs in client_100.values()),
                'max_pairs': max(len(pairs) for pairs in client_100.values()),
                'mean_pairs': np.mean([len(pairs) for pairs in client_100.values()])
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_multi_client()
