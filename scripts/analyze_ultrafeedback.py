#!/usr/bin/env python3
"""
Analyze UltraFeedback dataset to check for preference conflicts.
This script checks if high-scoring responses have high scores across all dimensions,
or if there are conflicting preferences (like HH-RLHF where harmless and helpful can conflict).

Usage:
    python scripts/analyze_ultrafeedback.py

Requirements:
    pip install datasets numpy
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
import json

def extract_score_from_annotation(annotation):
    """Extract numeric score from annotation dict."""
    if isinstance(annotation, dict):
        # Try to get Rating field
        if 'Rating' in annotation:
            try:
                return float(annotation['Rating'])
            except (ValueError, TypeError):
                pass
        # Try to get direct numeric value
        for key in ['score', 'rating', 'value']:
            if key in annotation:
                try:
                    return float(annotation[key])
                except (ValueError, TypeError):
                    pass
    elif isinstance(annotation, (int, float)):
        return float(annotation)
    return None

def analyze_ultrafeedback():
    """Load and analyze UltraFeedback dataset statistics."""
    
    print("=" * 80)
    print("UltraFeedback Dataset Analysis")
    print("=" * 80)
    
    try:
        # Try different possible dataset names
        dataset_names = [
            "openbmb/UltraFeedback",
            "allenai/ultrafeedback_binarized",
            "openbmb/ultrafeedback",
        ]
        
        dataset = None
        dataset_name_used = None
        
        for name in dataset_names:
            try:
                print(f"\nTrying to load: {name}")
                dataset = datasets.load_dataset(name)
                dataset_name_used = name
                print(f"✓ Successfully loaded: {name}")
                break
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
                continue
        
        if dataset is None:
            print("\nERROR: Could not load UltraFeedback dataset from any source.")
            print("Please check if the dataset is available on Hugging Face.")
            return
        
        print(f"\nDataset structure: {list(dataset.keys())}")
        
        # Analyze train split (or main split)
        split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
        data = dataset[split_name]
        
        print(f"\nAnalyzing split: {split_name}")
        print(f"Total samples: {len(data)}")
        
        # Check data structure
        if len(data) > 0:
            sample = data[0]
            print(f"\nSample keys: {list(sample.keys())}")
            if 'completions' in sample and len(sample['completions']) > 0:
                comp = sample['completions'][0]
                print(f"\nFirst completion keys: {list(comp.keys())}")
                if 'annotations' in comp:
                    print(f"Annotation structure: {list(comp['annotations'].keys())}")
        
        # Analyze preference conflicts
        print("\n" + "=" * 80)
        print("Analyzing Preference Conflicts")
        print("=" * 80)
        
        # Find annotation dimensions
        annotation_dims = None
        if len(data) > 0 and 'completions' in data[0]:
            completions = data[0]['completions']
            if len(completions) > 0 and 'annotations' in completions[0]:
                annotations = completions[0]['annotations']
                if isinstance(annotations, dict):
                    annotation_dims = list(annotations.keys())
                    print(f"\nFound annotation dimensions: {annotation_dims}")
        
        if annotation_dims is None:
            print("\nERROR: Could not find annotation dimensions in the dataset.")
            return
        
        conflict_stats = {
            'total_pairs': 0,
            'conflicting_pairs': 0,
            'non_conflicting_pairs': 0,
            'dimension_scores': defaultdict(list),
            'chosen_vs_rejected': defaultdict(lambda: {'chosen_higher': 0, 'rejected_higher': 0, 'equal': 0}),
            'score_extraction_method': 'overall_score'
        }
        
        print(f"\nAnalyzing {len(data)} samples for conflicts...")
        print("Using overall_score to determine chosen/rejected pairs...")
        
        for idx, sample in enumerate(data):
            if idx % 5000 == 0 and idx > 0:
                print(f"  Processing sample {idx}/{len(data)}...")
            
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
                
                # Extract scores for each dimension
                dim_scores = {}
                for dim in annotation_dims:
                    if dim in annotations:
                        score = extract_score_from_annotation(annotations[dim])
                        if score is not None:
                            dim_scores[dim] = score
                
                # Get overall score if available
                overall_score = comp.get('overall_score', None)
                if overall_score is None:
                    overall_score = comp.get('fine-grained_score', None)
                
                if dim_scores:  # Only add if we have at least one dimension score
                    completion_scores.append({
                        'comp': comp,
                        'dim_scores': dim_scores,
                        'overall_score': overall_score
                    })
            
            # Need at least 2 completions with scores
            if len(completion_scores) < 2:
                continue
            
            # Sort by overall_score (or average of dimension scores if overall_score not available)
            def get_sort_key(cs):
                if cs['overall_score'] is not None:
                    return cs['overall_score']
                # Use average of dimension scores
                if cs['dim_scores']:
                    return np.mean(list(cs['dim_scores'].values()))
                return -float('inf')
            
            completion_scores.sort(key=get_sort_key, reverse=True)
            
            # Best completion (chosen) and worst completion (rejected)
            chosen = completion_scores[0]
            rejected = completion_scores[-1]
            
            # Skip if same completion
            if chosen == rejected:
                continue
            
            chosen_scores = chosen['dim_scores']
            rejected_scores = rejected['dim_scores']
            
            # Check for conflicts
            conflict_stats['total_pairs'] += 1
            has_conflict = False
            
            chosen_better_dims = 0
            rejected_better_dims = 0
            
            for dim in annotation_dims:
                if dim in chosen_scores and dim in rejected_scores:
                    chosen_score = chosen_scores[dim]
                    rejected_score = rejected_scores[dim]
                    
                    conflict_stats['dimension_scores'][dim].append({
                        'chosen': chosen_score,
                        'rejected': rejected_score
                    })
                    
                    if chosen_score > rejected_score:
                        conflict_stats['chosen_vs_rejected'][dim]['chosen_higher'] += 1
                        chosen_better_dims += 1
                    elif rejected_score > chosen_score:
                        conflict_stats['chosen_vs_rejected'][dim]['rejected_higher'] += 1
                        rejected_better_dims += 1
                    else:
                        conflict_stats['chosen_vs_rejected'][dim]['equal'] += 1
            
            # Conflict: chosen is better in some dims, rejected is better in others
            if chosen_better_dims > 0 and rejected_better_dims > 0:
                has_conflict = True
                conflict_stats['conflicting_pairs'] += 1
            else:
                conflict_stats['non_conflicting_pairs'] += 1
        
        # Print statistics
        print("\n" + "=" * 80)
        print("Statistics Summary")
        print("=" * 80)
        
        print(f"\nTotal pairs analyzed: {conflict_stats['total_pairs']}")
        print(f"Conflicting pairs (chosen better in some dims, rejected in others): {conflict_stats['conflicting_pairs']}")
        print(f"Non-conflicting pairs (chosen better in all dims): {conflict_stats['non_conflicting_pairs']}")
        
        if conflict_stats['total_pairs'] > 0:
            conflict_ratio = conflict_stats['conflicting_pairs'] / conflict_stats['total_pairs']
            print(f"\n⚠️  Conflict ratio: {conflict_ratio:.2%}")
            if conflict_ratio < 0.05:
                print(f"\n⚠️  IMPORTANT: Conflict ratio is very low (< 5%)!")
                print(f"   This means that high-scoring responses score high in ALL dimensions.")
                print(f"   It will be difficult to split clients by preference type (like HH-RLHF).")
                print(f"   Consider using a different approach for client assignment.")
            elif conflict_ratio > 0.20:
                print(f"\n✓ Conflict ratio is reasonable (> 20%).")
                print(f"   There are enough conflicting preferences to split clients by dimension.")
        
        print("\n" + "-" * 80)
        print("Dimension-wise Analysis (chosen vs rejected)")
        print("-" * 80)
        
        for dim in annotation_dims:
            if dim in conflict_stats['chosen_vs_rejected']:
                stats = conflict_stats['chosen_vs_rejected'][dim]
                total = stats['chosen_higher'] + stats['rejected_higher'] + stats['equal']
                if total > 0:
                    print(f"\n{dim}:")
                    print(f"  Chosen higher: {stats['chosen_higher']} ({stats['chosen_higher']/total:.2%})")
                    print(f"  Rejected higher: {stats['rejected_higher']} ({stats['rejected_higher']/total:.2%})")
                    print(f"  Equal: {stats['equal']} ({stats['equal']/total:.2%})")
        
        # Analyze score distributions
        if conflict_stats['dimension_scores']:
            print("\n" + "-" * 80)
            print("Score Distribution Analysis")
            print("-" * 80)
            
            for dim in annotation_dims:
                if dim in conflict_stats['dimension_scores']:
                    scores = conflict_stats['dimension_scores'][dim]
                    if scores:
                        chosen_scores = [s['chosen'] for s in scores]
                        rejected_scores = [s['rejected'] for s in scores]
                        
                        print(f"\n{dim}:")
                        print(f"  Chosen - Mean: {np.mean(chosen_scores):.3f}, Std: {np.std(chosen_scores):.3f}")
                        print(f"  Rejected - Mean: {np.mean(rejected_scores):.3f}, Std: {np.std(rejected_scores):.3f}")
                        print(f"  Difference - Mean: {np.mean([c-r for c, r in zip(chosen_scores, rejected_scores)]):.3f}")
        
        # Save detailed results
        output_file = "ultrafeedback_analysis.json"
        with open(output_file, 'w') as f:
            json.dump({
                'dataset_name': dataset_name_used,
                'total_samples': len(data),
                'annotation_dims': annotation_dims,
                'conflict_stats': {k: dict(v) if isinstance(v, defaultdict) else v 
                                for k, v in conflict_stats.items()}
            }, f, indent=2, default=str)
        
        print(f"\n✓ Detailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_ultrafeedback()
