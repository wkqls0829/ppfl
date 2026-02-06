#!/usr/bin/env python3
"""
Show examples of conflicting pairs in UltraFeedback dataset.
This helps understand how conflicting pairs work for client splitting.
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

import numpy as np
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

def show_conflicting_examples():
    """Show examples of conflicting pairs for each dimension."""
    
    print("=" * 80)
    print("UltraFeedback Conflicting Pairs Examples")
    print("=" * 80)
    
    try:
        print("\nLoading dataset...")
        dataset = datasets.load_dataset("openbmb/UltraFeedback")
        data = dataset['train']
        
        # Find annotation dimensions
        annotation_dims = ['helpfulness', 'honesty', 'instruction_following', 'truthfulness']
        
        # Collect examples for each dimension
        examples_by_dim = defaultdict(list)
        
        print(f"\nFinding conflicting pairs examples...")
        
        for idx, sample in enumerate(data):
            if len(examples_by_dim) >= 4:  # One example per dimension
                break
            
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
                        'overall_score': overall_score,
                        'response': comp.get('response', '')
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
                
                for dim in annotation_dims:
                    if dim in best_scores and dim in other_scores:
                        if best_scores[dim] > other_scores[dim]:
                            best_better_dims.append(dim)
                        elif other_scores[dim] > best_scores[dim]:
                            other_better_dims.append(dim)
                
                # Conflicting: best is better in some dims, other is better in others
                if best_better_dims and other_better_dims:
                    # Find which dimension has the largest difference
                    max_diff = -float('inf')
                    winning_dim = None
                    
                    for dim in annotation_dims:
                        if dim in best_scores and dim in other_scores:
                            diff = best_scores[dim] - other_scores[dim]
                            if diff > max_diff:
                                max_diff = diff
                                winning_dim = dim
                    
                    # Only add if we don't have an example for this dimension yet
                    if winning_dim and winning_dim not in examples_by_dim:
                        examples_by_dim[winning_dim] = {
                            'instruction': sample.get('instruction', '')[:200],
                            'chosen': {
                                'response': best['response'][:300],
                                'scores': best_scores,
                                'better_in': best_better_dims
                            },
                            'rejected': {
                                'response': other['response'][:300],
                                'scores': other_scores,
                                'better_in': other_better_dims
                            },
                            'winning_dim': winning_dim
                        }
                        break
        
        # Print examples
        print("\n" + "=" * 80)
        print("Conflicting Pairs Examples by Dimension")
        print("=" * 80)
        
        for dim in annotation_dims:
            if dim not in examples_by_dim:
                continue
            
            example = examples_by_dim[dim]
            print(f"\n{'='*80}")
            print(f"Example for {dim.upper()} Client")
            print(f"{'='*80}")
            
            print(f"\nInstruction: {example['instruction']}...")
            
            print(f"\n--- CHOSEN (better in {dim}) ---")
            print(f"Response: {example['chosen']['response']}...")
            print(f"\nScores:")
            for d in annotation_dims:
                if d in example['chosen']['scores']:
                    score = example['chosen']['scores'][d]
                    marker = " ✓" if d in example['chosen']['better_in'] else ""
                    print(f"  {d}: {score:.2f}{marker}")
            
            print(f"\n--- REJECTED (better in other dimensions) ---")
            print(f"Response: {example['rejected']['response']}...")
            print(f"\nScores:")
            for d in annotation_dims:
                if d in example['rejected']['scores']:
                    score = example['rejected']['scores'][d]
                    marker = " ✓" if d in example['rejected']['better_in'] else ""
                    print(f"  {d}: {score:.2f}{marker}")
            
            print(f"\n📊 Analysis:")
            print(f"  - Chosen is better in: {', '.join(example['chosen']['better_in'])}")
            print(f"  - Rejected is better in: {', '.join(example['rejected']['better_in'])}")
            print(f"  - This pair is assigned to {dim} client because {dim} has the largest score difference")
            
            # Show score differences
            print(f"\n  Score differences:")
            for d in annotation_dims:
                if d in example['chosen']['scores'] and d in example['rejected']['scores']:
                    diff = example['chosen']['scores'][d] - example['rejected']['scores'][d]
                    marker = " ⭐" if d == dim else ""
                    print(f"    {d}: {diff:+.2f}{marker}")
        
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print("\nConflicting pair의 의미:")
        print("  - Chosen answer는 특정 차원(예: truthfulness)에서 높은 점수를 받음")
        print("  - Rejected answer는 그 차원에서는 낮지만, 다른 차원(예: helpfulness)에서는 더 높음")
        print("  - 각 쌍은 '가장 큰 점수 차이'를 보이는 차원의 클라이언트에 할당됨")
        print("\n예를 들어 Truthfulness 클라이언트에 할당된 쌍:")
        print("  - Chosen: truthfulness 점수가 높음 (예: 4.5)")
        print("  - Rejected: truthfulness 점수는 낮지만 (예: 2.0)")
        print("             다른 차원(예: helpfulness)에서는 더 높을 수 있음 (예: 4.0 vs 3.5)")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    show_conflicting_examples()
