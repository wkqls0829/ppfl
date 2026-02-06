#!/usr/bin/env python3
"""
Analyze how many conflicting pairs remain when filtering by large preference differences.
This helps determine how many high-quality conflicting pairs are available for client splitting.
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

def analyze_large_diff_pairs():
    """Analyze conflicting pairs with large preference differences."""
    
    print("=" * 80)
    print("UltraFeedback Large Preference Difference Analysis")
    print("=" * 80)
    
    try:
        print("\nLoading dataset...")
        dataset = datasets.load_dataset("openbmb/UltraFeedback")
        data = dataset['train']
        
        annotation_dims = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
        
        # Collect all conflicting pairs with their score differences
        conflicting_pairs = []
        
        print(f"\nAnalyzing {len(data)} samples...")
        
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
                
                # Check if this is a conflicting pair
                best_better_dims = []
                other_better_dims = []
                dim_diffs = {}
                
                for dim in annotation_dims:
                    if dim in best_scores and dim in other_scores:
                        diff = best_scores[dim] - other_scores[dim]
                        dim_diffs[dim] = diff
                        if diff > 0:
                            best_better_dims.append(dim)
                        elif diff < 0:
                            other_better_dims.append(dim)
                
                # Conflicting: best is better in some dims, other is better in others
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
                    
                    if winning_dim:
                        conflicting_pairs.append({
                            'best_scores': best_scores,
                            'other_scores': other_scores,
                            'dim_diffs': dim_diffs,
                            'winning_dim': winning_dim,
                            'max_diff': max_diff
                        })
        
        print(f"\nTotal conflicting pairs: {len(conflicting_pairs)}")
        
        # Analyze with different thresholds
        print("\n" + "=" * 80)
        print("Filtering by Minimum Score Difference Threshold")
        print("=" * 80)
        
        thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        results = {}
        
        for threshold in thresholds:
            filtered_pairs = []
            dim_counts = defaultdict(int)
            
            for pair in conflicting_pairs:
                winning_dim = pair['winning_dim']
                max_diff = pair['max_diff']
                
                # Check if the winning dimension has difference >= threshold
                if max_diff >= threshold:
                    filtered_pairs.append(pair)
                    dim_counts[winning_dim] += 1
            
            results[threshold] = {
                'total': len(filtered_pairs),
                'dim_counts': dict(dim_counts),
                'percentage': len(filtered_pairs) / len(conflicting_pairs) * 100 if conflicting_pairs else 0
            }
            
            print(f"\nThreshold >= {threshold}:")
            print(f"  Total pairs: {len(filtered_pairs)} ({len(filtered_pairs)/len(conflicting_pairs)*100:.2f}% of conflicting pairs)")
            print(f"  Per dimension:")
            for dim in annotation_dims:
                count = dim_counts[dim]
                if len(filtered_pairs) > 0:
                    pct = count / len(filtered_pairs) * 100
                    print(f"    {dim}: {count} ({pct:.1f}%)")
        
        # Show distribution of score differences
        print("\n" + "=" * 80)
        print("Score Difference Distribution")
        print("=" * 80)
        
        max_diffs = [p['max_diff'] for p in conflicting_pairs]
        print(f"\nMaximum score difference statistics:")
        print(f"  Mean: {np.mean(max_diffs):.3f}")
        print(f"  Median: {np.median(max_diffs):.3f}")
        print(f"  Std: {np.std(max_diffs):.3f}")
        print(f"  Min: {np.min(max_diffs):.3f}")
        print(f"  Max: {np.max(max_diffs):.3f}")
        
        # Percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            val = np.percentile(max_diffs, p)
            print(f"  {p}th: {val:.3f}")
        
        # Recommendation
        print("\n" + "=" * 80)
        print("Recommendation")
        print("=" * 80)
        
        # Find threshold that gives reasonable amount of data per client
        for threshold in [1.0, 1.5, 2.0]:
            res = results[threshold]
            min_per_client = min(res['dim_counts'].values()) if res['dim_counts'] else 0
            print(f"\nThreshold >= {threshold}:")
            print(f"  Total pairs: {res['total']}")
            print(f"  Min pairs per client: {min_per_client}")
            if min_per_client >= 5000:
                print(f"  ✓ Sufficient for training (>= 5000 per client)")
            elif min_per_client >= 2000:
                print(f"  ⚠️  May be sufficient (>= 2000 per client)")
            else:
                print(f"  ✗ Too few pairs per client (< 2000)")
        
        # Save results
        import json
        output_file = "ultrafeedback_large_diff_analysis.json"
        with open(output_file, 'w') as f:
            json.dump({
                'total_conflicting_pairs': len(conflicting_pairs),
                'threshold_results': results,
                'max_diff_stats': {
                    'mean': float(np.mean(max_diffs)),
                    'median': float(np.median(max_diffs)),
                    'std': float(np.std(max_diffs)),
                    'min': float(np.min(max_diffs)),
                    'max': float(np.max(max_diffs)),
                    'percentiles': {str(p): float(np.percentile(max_diffs, p)) for p in percentiles}
                }
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_large_diff_pairs()
