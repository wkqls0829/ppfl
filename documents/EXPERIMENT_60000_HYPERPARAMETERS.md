# 실험 60000 Hyperparameter 설정 설명

## 개요

실험 60000은 VPL-GP 모델에서 orthogonal loss의 효과를 비교하기 위한 실험입니다.
- **hhst**: Orthogonal loss 비활성화 (baseline)
- **hhst-ortho**: Orthogonal loss 활성화 (CLOP 기반)

## 공통 Hyperparameters

### 기본 설정
- **모델**: `google/gemma-2b@huggingface_llm`
- **Task**: HH-RLHF binary choice (A vs B)
- **클라이언트 수**: 10
- **샘플링 클라이언트 수**: 5
- **총 라운드**: 50
- **체크포인트 저장 주기**: 10 라운드

### 데이터 설정
- **데이터셋**: HH-RLHF
- **Split**: [0.9, 0.09, 0.01] (train/val/test)
- **Splitter**: meta

### 모델 설정
- **Tokenizer 길이**: 1024
- **최대 새 토큰**: 512
- **Adapter**: LoRA (r=8, alpha=16, dropout=0.05)
- **Adapter 개수**: 3
- **Warmup 라운드**: 2

### 학습 설정
- **로컬 업데이트 스텝**: 30
- **배치 크기**: 4
- **Gradient Accumulation**: 2 (effective batch size = 8)
- **Optimizer**: AdamW
- **Learning Rate**: 0.00001
- **Betas**: (0.9, 0.95)
- **Half Precision**: True (FP16)

### VPL-GP 설정
- **GP Prior 사용**: True
- **Latent Dimension**: 32
- **KL Weight**: 0.1
- **Gumbel-Softmax Temperature**: 1.0
- **Feature Method**: 'choice_logits'
- **Feature Difference 사용**: True (embedding difference 사용)

## 차이점: Orthogonal Loss 설정

### hhst.yaml (Orthogonal Loss 비활성화)

```yaml
vpl_orthogonal_weight: 0.0  # Orthogonal loss disabled
vpl_orthogonal_orthonorm_weight: 0.0  # Orthonormal constraint disabled
vpl_use_manual_orthogonal_labels: False  # Not needed when orthogonal loss is disabled
```

**설명**:
- `vpl_orthogonal_weight: 0.0`: Pull loss 비활성화 (embedding을 prototype에 pull하지 않음)
- `vpl_orthogonal_orthonorm_weight: 0.0`: Orthonormal constraint 비활성화
- `vpl_use_manual_orthogonal_labels: False`: 서버에서 계산한 orthogonal label 사용 안 함

**효과**: 
- Neural collapse 방지 메커니즘 없음
- Embedding들이 자유롭게 학습됨
- Baseline 성능 측정

### hhst-ortho.yaml (Orthogonal Loss 활성화)

```yaml
vpl_orthogonal_weight: 10.0  # Pull loss weight (CLOP paper)
vpl_orthogonal_orthonorm_weight: 0.1  # Orthonormal constraint weight
vpl_use_manual_orthogonal_labels: True  # Use server-computed orthogonal labels
```

**설명**:
- `vpl_orthogonal_weight: 10.0`: Pull loss 가중치 (CLOP 논문 기반)
  - Embedding `z`를 해당 prototype `p_y`에 가깝게 pull
  - Loss: `L_pull = (1/|B|) Σ ||z(x) - p_y||²`
- `vpl_orthogonal_orthonorm_weight: 0.1`: Orthonormal constraint 가중치
  - Prototypes가 orthonormal이 되도록 제약
  - Loss: `L_orthonorm = ||P^T P - I||²_F`
- `vpl_use_manual_orthogonal_labels: True`: 서버에서 계산한 orthogonal label 사용
  - 서버에서 balanced k-means로 클라이언트들을 그룹화
  - 각 클라이언트에 orthogonal label 할당

**효과**:
- Neural collapse 방지
- Full-rank space 활용
- 더 구분 가능한 embedding
- Orthogonal subspaces 형성

## Loss 구성

### hhst (Baseline)
```
Total Loss = Reconstruction Loss + λ_KL * KL Loss
```

### hhst-ortho (With Orthogonal Loss)
```
Total Loss = Reconstruction Loss + λ_KL * KL Loss 
           + λ_pull * L_pull + λ_orthonorm * L_orthonorm
```

여기서:
- `Reconstruction Loss`: Cross-entropy loss for choice prediction
- `KL Loss`: KL divergence between posterior and prior
- `L_pull`: Pull loss (embedding을 prototype에 가깝게)
- `L_orthonorm`: Orthonormal constraint loss

## 평가 지표

- **loss**: 전체 loss
- **acc**: 정확도
- **avg_harmlessness**: 평균 무해성 점수
- **avg_helpfulness**: 평균 도움성 점수
- **vpl_kl_loss**: VPL KL divergence loss
- **vpl_reconstruction_loss**: VPL reconstruction loss

## 실행 방법

### Baseline (Orthogonal Loss 비활성화)
```bash
bash scripts/server/vpl-gp/hhst.sh
```

### With Orthogonal Loss
```bash
bash scripts/server/vpl-gp/hhst-ortho.sh
```

## 출력 파일

### Baseline
- **로그**: `outputs/60000.log`
- **체크포인트**: `/hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_vplgp_60000.ckpt`
- **WandB 프로젝트**: `fvpl-selector`
- **실험 이름**: `vplgp_hhst_t60000`

### With Orthogonal Loss
- **로그**: `outputs/60000-ortho.log`
- **체크포인트**: `/hdd/hdd3/kjb/checkpoints/hhrl_choice_gemma_fedbiscuit_u3_vplgp_ortho_60000.ckpt`
- **WandB 프로젝트**: `fvpl-selector`
- **실험 이름**: `vplgp_hhst_ortho_t60000`

## 예상 결과

### Baseline (hhst)
- Neural collapse 가능성 있음
- Embedding들이 낮은 차원으로 수렴할 수 있음
- 클래스 구분이 어려울 수 있음

### With Orthogonal Loss (hhst-ortho)
- Neural collapse 방지
- Full-rank space 활용
- 더 구분 가능한 embedding
- 더 안정적인 학습

## 참고

- CLOP 논문: [Preventing Collapse in Contrastive Learning with Orthonormal Prototypes](https://arxiv.org/pdf/2403.18699)
- VPL-GP 구현: `federatedscope/llm/trainer/vpl_reward_choice_trainer.py`
- Orthogonal loss 구현: `federatedscope/llm/llm_local/z_visualization.py`
