#!/usr/bin/env python3
"""
Analyze UltraFeedback dataset for client splitting by preference dimensions.
This script checks if we can split clients by 4 preference dimensions using only conflicting pairs.
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
        for key in ['score', 'rating', 'value']:
            if key in annotation:
                try:
                    return float(annotation[key])
                except (ValueError, TypeError):
                    pass
    elif isinstance(annotation, (int, float)):
        return float(annotation)
    return None

def analyze_client_split():
    """Analyze if we can split clients by preference dimensions using conflicting pairs."""
    
    print("=" * 80)
    print("UltraFeedback Client Split Analysis")
    print("=" * 80)
    
    try:
        print("\nLoading dataset...")
        dataset = datasets.load_dataset("openbmb/UltraFeedback")
        data = dataset['train']
        
        print(f"Total samples: {len(data)}")
        
        # Find annotation dimensions
        annotation_dims = None
        if len(data) > 0 and 'completions' in data[0]:
            completions = data[0]['completions']
            if len(completions) > 0 and 'annotations' in completions[0]:
                annotations = completions[0]['annotations']
                if isinstance(annotations, dict):
                    annotation_dims = list(annotations.keys())
                    print(f"\nAnnotation dimensions: {annotation_dims}")
        
        if annotation_dims is None:
            print("ERROR: Could not find annotation dimensions.")
            return
        
        # Statistics for client splitting
        stats = {
            'total_samples': len(data),
            'conflicting_pairs': [],
            'dimension_preferences': defaultdict(list),  # dim -> list of pairs where chosen is better in this dim
            'client_split_stats': {}
        }
        
        print(f"\nAnalyzing samples for conflicting pairs...")
        
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
            
            # Check all pairs (best vs others)
            best = completion_scores[0]
            best_scores = best['dim_scores']
            
            for other in completion_scores[1:]:
                other_scores = other['dim_scores']
                
                # Check if this is a conflicting pair
                best_better_dims = []
                other_better_dims = []
                
                for dim in annotation_dims:
                    if dim in best_scores and dim in other_scores:
                        if best_scores[dim] > other_scores[dim]:
                            best_better_dims.append(dim)
                        elif other_scores[dim] > best_scores[dim]:
                            other_better_dims.append(dim)
                
                # Conflicting: best is better in some dims, other is better in others
                if best_better_dims and other_better_dims:
                    stats['conflicting_pairs'].append({
                        'best_scores': best_scores,
                        'other_scores': other_scores,
                        'best_better_dims': best_better_dims,
                        'other_better_dims': other_better_dims
                    })
                    
                    # For each dimension where best is better, add to that dimension's preference list
                    for dim in best_better_dims:
                        stats['dimension_preferences'][dim].append({
                            'best_score': best_scores[dim],
                            'other_score': other_scores[dim],
                            'dim': dim
                        })
        
        # Print statistics
        print("\n" + "=" * 80)
        print("Client Split Analysis Results")
        print("=" * 80)
        
        print(f"\nTotal conflicting pairs found: {len(stats['conflicting_pairs'])}")
        print(f"Percentage of total samples: {len(stats['conflicting_pairs'])/len(data)*100:.2f}%")
        
        print("\n" + "-" * 80)
        print("Dimension-wise Preference Distribution")
        print("-" * 80)
        
        for dim in annotation_dims:
            pairs = stats['dimension_preferences'][dim]
            print(f"\n{dim}:")
            print(f"  Total pairs where chosen is better: {len(pairs)}")
            if pairs:
                avg_diff = np.mean([p['best_score'] - p['other_score'] for p in pairs])
                print(f"  Average score difference: {avg_diff:.3f}")
        
        # Analyze client split feasibility
        print("\n" + "-" * 80)
        print("Client Split Feasibility")
        print("-" * 80)
        
        # For each dimension, count how many conflicting pairs have this dimension as the "winning" dimension
        dim_win_counts = defaultdict(int)
        for pair in stats['conflicting_pairs']:
            best_dims = pair['best_better_dims']
            other_dims = pair['other_better_dims']
            
            # Count which dimension has the largest difference (strongest preference)
            max_diff = -float('inf')
            winning_dim = None
            
            for dim in annotation_dims:
                if dim in pair['best_scores'] and dim in pair['other_scores']:
                    diff = pair['best_scores'][dim] - pair['other_scores'][dim]
                    if diff > max_diff:
                        max_diff = diff
                        winning_dim = dim
            
            if winning_dim:
                dim_win_counts[winning_dim] += 1
        
        print("\nPairs where each dimension has the largest score difference:")
        total_winning = sum(dim_win_counts.values())
        for dim in annotation_dims:
            count = dim_win_counts[dim]
            if total_winning > 0:
                pct = count / total_winning * 100
                print(f"  {dim}: {count} ({pct:.2f}%)")
        
        # Estimate client distribution
        print("\n" + "-" * 80)
        print("Estimated Client Distribution (4 clients, one per dimension)")
        print("-" * 80)
        
        if total_winning > 0:
            for dim in annotation_dims:
                count = dim_win_counts[dim]
                pct = count / total_winning * 100
                print(f"  Client {dim}: ~{pct:.1f}% of conflicting pairs")
        
        # Check if we have enough data per client
        min_pairs_per_client = min(dim_win_counts.values()) if dim_win_counts else 0
        print(f"\nMinimum pairs per client: {min_pairs_per_client}")
        
        if min_pairs_per_client < 1000:
            print("⚠️  WARNING: Some clients may have too few pairs (< 1000)")
        else:
            print("✓ Sufficient pairs per client for training")
        
        # Save results
        output_file = "ultrafeedback_client_split_analysis.json"
        import json
        with open(output_file, 'w') as f:
            json.dump({
                'total_samples': stats['total_samples'],
                'conflicting_pairs_count': len(stats['conflicting_pairs']),
                'dimension_preferences_count': {k: len(v) for k, v in stats['dimension_preferences'].items()},
                'dim_win_counts': dict(dim_win_counts),
                'annotation_dims': annotation_dims
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Recommendation
        print("\n" + "=" * 80)
        print("Recommendation")
        print("=" * 80)
        
        if len(stats['conflicting_pairs']) > 10000 and min_pairs_per_client > 1000:
            print("\n✓ YES, you can split clients by preference dimensions!")
            print(f"  - Use only conflicting pairs ({len(stats['conflicting_pairs'])} pairs)")
            print(f"  - Assign each pair to the client whose preferred dimension has the largest score difference")
            print(f"  - This will create 4 clients, each preferring one dimension:")
            for dim in annotation_dims:
                print(f"    * Client {dim}: prefers {dim}")
        else:
            print("\n⚠️  May need to adjust the splitting strategy:")
            print(f"  - Conflicting pairs: {len(stats['conflicting_pairs'])}")
            print(f"  - Min pairs per client: {min_pairs_per_client}")
            if len(stats['conflicting_pairs']) < 10000:
                print("  - Consider using all pairs, not just conflicting ones")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_client_split()
