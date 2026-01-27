#!/usr/bin/env python3
"""
Check how many clients have z values in ablation experiment checkpoints.
"""
import torch
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

checkpoints = [
    ('VPL Baseline', '/hdd/hdd3/kjb/checkpoints/final_hhrl_choice_gemma_vpl_n10_t39000.ckpt'),
    ('VPL-GP-Ortho', '/hdd/hdd3/kjb/checkpoints/final_hhrl_choice_gemma_vplgp_ortho_n10_t59000.ckpt'),
    ('VPL-GP (no ortho)', '/hdd/hdd3/kjb/checkpoints/final_hhrl_choice_gemma_ablation_vplgp_n10_t62300.ckpt'),
    ('VPL-Ortho (no GP)', '/hdd/hdd3/kjb/checkpoints/final_hhrl_choice_gemma_ablation_vplortho_n10_t62310.ckpt'),
]

print("="*70)
print("Ablation 실험 체크포인트 z 값 확인")
print("="*70)

for name, ckpt_path in checkpoints:
    print(f"\n📁 {name}")
    print("-" * 70)
    
    if not os.path.exists(ckpt_path):
        print(f"  ❌ 체크포인트 파일 없음: {ckpt_path}")
        continue
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        if 'client_average_z_dict' in ckpt:
            z_dict = ckpt['client_average_z_dict']
            if isinstance(z_dict, dict):
                client_ids = sorted([int(k) for k in z_dict.keys()])
                missing = set(range(1, 11)) - set(client_ids)
                print(f"  클라이언트 수: {len(z_dict)}/10")
                print(f"  클라이언트 ID: {client_ids}")
                if missing:
                    print(f"  ⚠️  누락된 클라이언트: {sorted(missing)}")
                    print(f"  ⚠️  이 체크포인트는 수정 전 코드로 저장되어 일부 클라이언트만 z 값을 가집니다.")
                else:
                    print(f"  ✅ 모든 클라이언트 z 값 존재")
            else:
                print(f"  ❌ client_average_z_dict가 dict가 아님: {type(z_dict)}")
        else:
            print(f"  ❌ client_average_z_dict 없음")
    except Exception as e:
        print(f"  ❌ 오류: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("결론:")
print("  - 수정 전 코드로 저장된 체크포인트들은 마지막 라운드에")
print("    참여한 클라이언트만 z 값을 가지고 있을 수 있습니다.")
print("  - RL training에서는 z 값이 없는 클라이언트에 대해")
print("    전체 평균 z를 사용하는 fallback 로직이 적용됩니다.")
print("="*70)
