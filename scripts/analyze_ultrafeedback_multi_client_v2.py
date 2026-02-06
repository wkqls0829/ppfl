#!/usr/bin/env python3
"""
Analyze UltraFeedback dataset for multiple client scenarios (10, 100 clients).
Improved version that properly handles client splitting.
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
    """Analyze data distribution for 10 and 100 clients with threshold 3.0."""
    
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
                            'max_diff': max_diff
                        })
        
        print(f"\nTotal pairs with difference >= {threshold}: {len(large_diff_pairs)}")
        
        # Check score difference distribution
        max_diffs = [p['max_diff'] for p in large_diff_pairs]
        print(f"\nScore difference range: {min(max_diffs):.2f} - {max(max_diffs):.2f}")
        print(f"Mean: {np.mean(max_diffs):.2f}, Median: {np.median(max_diffs):.2f}")
        
        # Strategy: Split by dimension, then by score difference within each dimension
        print("\n" + "=" * 80)
        print("Client Split Strategy: Dimension + Score Difference")
        print("=" * 80)
        
        def analyze_client_split(num_clients):
            print(f"\n{'='*80}")
            print(f"{num_clients} Clients")
            print(f"{'='*80}")
            
            # Distribute clients across dimensions
            # For 10: 3, 3, 2, 2 = 10
            # For 100: 25, 25, 25, 25 = 100
            if num_clients == 10:
                dim_client_counts = {
                    'helpfulness': 3,
                    'honesty': 3,
                    'instruction_following': 2,
                    'truthfulness': 2
                }
            elif num_clients == 100:
                dim_client_counts = {
                    'helpfulness': 25,
                    'honesty': 25,
                    'instruction_following': 25,
                    'truthfulness': 25
                }
            else:
                # General case
                base = num_clients // 4
                remainder = num_clients % 4
                dim_client_counts = {
                    'helpfulness': base + (1 if remainder > 0 else 0),
                    'honesty': base + (1 if remainder > 1 else 0),
                    'instruction_following': base + (1 if remainder > 2 else 0),
                    'truthfulness': base
                }
            
            # Assign pairs to clients
            client_assignments = defaultdict(list)
            
            for dim in annotation_dims:
                # Get all pairs for this dimension
                dim_pairs = [p for p in large_diff_pairs if p['winning_dim'] == dim]
                
                if len(dim_pairs) == 0:
                    continue
                
                # Get score differences for this dimension
                dim_diffs = [p['max_diff'] for p in dim_pairs]
                min_diff = min(dim_diffs)
                max_diff = max(dim_diffs)
                diff_range = max_diff - min_diff
                
                num_clients_for_dim = dim_client_counts[dim]
                
                # Assign each pair to a client based on score difference
                for pair in dim_pairs:
                    max_diff_val = pair['max_diff']
                    
                    if diff_range > 0:
                        # Normalize to [0, 1]
                        normalized = (max_diff_val - min_diff) / diff_range
                        # Map to client index
                        client_idx = min(int(normalized * num_clients_for_dim), num_clients_for_dim - 1)
                    else:
                        client_idx = 0
                    
                    client_id = f"{dim}_client_{client_idx}"
                    client_assignments[client_id].append(pair)
            
            # Print statistics
            print(f"\nTotal pairs: {len(large_diff_pairs)}")
            print(f"Number of clients created: {len(client_assignments)}")
            print(f"Expected: {num_clients}")
            
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
                if len(dim_clients) > 0:
                    dim_mean = dim_total / len(dim_clients)
                    print(f"  {dim}: {len(dim_clients)} clients, {dim_total} pairs total, {dim_mean:.1f} pairs/client avg")
            
            # Show some examples
            sorted_clients = sorted(client_assignments.items(), key=lambda x: len(x[1]), reverse=True)
            print(f"\nTop 10 clients (by size):")
            for client_id, pairs in sorted_clients[:10]:
                print(f"  {client_id}: {len(pairs)} pairs")
            
            if len(sorted_clients) > 10:
                print(f"\nBottom 10 clients (by size):")
                for client_id, pairs in sorted_clients[-10:]:
                    print(f"  {client_id}: {len(pairs)} pairs")
            
            # Check if sufficient
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
                print(f"   Consider using lower threshold or fewer clients")
            
            return client_assignments
        
        # Analyze for 10 and 100 clients
        client_10 = analyze_client_split(10)
        client_100 = analyze_client_split(100)
        
        # Summary comparison
        print("\n" + "=" * 80)
        print("Summary Comparison")
        print("=" * 80)
        
        print(f"\nThreshold 3.0 Results:")
        print(f"  Total pairs: {len(large_diff_pairs)}")
        print(f"\n  10 Clients:")
        sizes_10 = [len(pairs) for pairs in client_10.values()]
        print(f"    Min: {np.min(sizes_10)}, Mean: {np.mean(sizes_10):.1f}, Max: {np.max(sizes_10)}")
        print(f"\n  100 Clients:")
        sizes_100 = [len(pairs) for pairs in client_100.values()]
        print(f"    Min: {np.min(sizes_100)}, Mean: {np.mean(sizes_100):.1f}, Max: {np.max(sizes_100)}")
        
        # Save results
        import json
        output_file = "ultrafeedback_multi_client_analysis_v2.json"
        
        summary = {
            'threshold': threshold,
            'total_pairs': len(large_diff_pairs),
            'score_diff_range': {
                'min': float(min(max_diffs)),
                'max': float(max(max_diffs)),
                'mean': float(np.mean(max_diffs)),
                'median': float(np.median(max_diffs))
            },
            '10_clients': {
                'num_clients': len(client_10),
                'min_pairs': int(np.min(sizes_10)),
                'mean_pairs': float(np.mean(sizes_10)),
                'max_pairs': int(np.max(sizes_10))
            },
            '100_clients': {
                'num_clients': len(client_100),
                'min_pairs': int(np.min(sizes_100)),
                'mean_pairs': float(np.mean(sizes_100)),
                'max_pairs': int(np.max(sizes_100))
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
