# Ablation Study vs Main Table: Config & Hyperparameter 차이점

## 1. t-SNE Visualization 문제

### 현재 상황
- **Ablation study config 파일들** (`cfg/vpl-gp-no-ortho/hhst.yaml`, `cfg/vpl-ortho/hhst.yaml`)에는 `vpl_tsne_visualize_freq: 10`이 **설정되어 있습니다**.
- **Main table config 파일들** (`cfg/vpl/hhst.yaml`, `cfg/vpl-gp/hhst.yaml`)에도 `vpl_tsne_visualize_freq: 10`이 설정되어 있습니다.

### 문제 원인 추정
1. **예전 버전 코드**: Ablation study가 실행될 때 코드가 예전 버전이었을 가능성
2. **Config override 누락**: Ablation study 스크립트에서 config를 직접 사용하지만, main table 스크립트는 Python으로 config를 수정하여 `vpl_tsne_visualize_freq`를 명시적으로 설정

### 해결 방법
- Ablation study config 파일들에는 이미 `vpl_tsne_visualize_freq: 10`이 설정되어 있으므로, 코드가 최신 버전이라면 정상 작동해야 합니다.
- 만약 여전히 작동하지 않는다면, ablation study 스크립트에서도 명시적으로 설정하도록 수정 필요.

---

## 2. Config & Hyperparameter 차이점

### A. VPL + GP (no orthogonal loss) vs FedVPA-GP

| 항목 | Main Table (FedVPA-GP) | Ablation (VPL + GP) |
|------|------------------------|---------------------|
| **Config 파일** | `cfg/vpl-gp/hhst.yaml` | `cfg/vpl-gp-no-ortho/hhst.yaml` |
| `vpl_use_gp_prior` | `True` | `True` ✅ |
| `vpl_kl_weight` | `0.02` | `0.02` ✅ |
| `vpl_gp_temperature` | `1.0` | `1.0` ✅ |
| `vpl_orthogonal_weight` | `1.0` | `0.0` ❌ **차이점** |
| `vpl_orthogonal_orthonorm_weight` | `0.0` | `0.0` ✅ |
| `vpl_use_manual_orthogonal_labels` | `True` | `False` ❌ **차이점** |
| `vpl_num_prototypes` | `2` | `0` ❌ **차이점** |
| `vpl_prototype_scale` | `5.0` | `0.0` ❌ **차이점** |
| `vpl_tsne_visualize_freq` | `10` | `10` ✅ |
| `grad_accum_step` | `2` | `4` ❌ **차이점** |
| `dataloader.batch_size` | `16` | `8` ❌ **차이점** |
| `train.optimizer.lr` | `0.0001` | `0.0001` ✅ |
| `train.local_update_steps` | `30` | `30` ✅ |
| `eval.freq` | `5` | `5` ✅ |
| `eval.max_samples_for_reward` | `100` | `100` ✅ |

### B. VPL + Ortho (no GP prior) vs FedVPL

| 항목 | Main Table (FedVPL) | Ablation (VPL + Ortho) |
|------|---------------------|------------------------|
| **Config 파일** | `cfg/vpl/hhst.yaml` | `cfg/vpl-ortho/hhst.yaml` |
| `vpl_use_gp_prior` | `False` | `False` ✅ |
| `vpl_kl_weight` | `0.1` | `0.1` ✅ |
| `vpl_orthogonal_weight` | `0.0` | `1.0` ❌ **차이점** |
| `vpl_orthogonal_orthonorm_weight` | `0.0` | `0.0` ✅ |
| `vpl_use_manual_orthogonal_labels` | `False` | `True` ❌ **차이점** |
| `vpl_num_prototypes` | `0` | `2` ❌ **차이점** |
| `vpl_prototype_scale` | `0.0` | `5.0` ❌ **차이점** |
| `vpl_tsne_visualize_freq` | `10` | `10` ✅ |
| `grad_accum_step` | `4` | `4` ✅ |
| `dataloader.batch_size` | `8` | `8` ✅ |
| `train.optimizer.lr` | `0.0001` | `0.0001` ✅ |
| `train.local_update_steps` | `30` | `30` ✅ |
| `eval.freq` | `5` | `5` ✅ |
| `eval.max_samples_for_reward` | `100` | `100` ✅ |

### C. Qwen 2 모델 사용 시 차이점

| 항목 | Main Table (Qwen 2) | Ablation (Qwen 2) |
|------|---------------------|-------------------|
| `model.type` | `Qwen/Qwen2-0.5B@huggingface_llm` | `Qwen/Qwen2-0.5B@huggingface_llm` ✅ |
| `train.optimizer.lr` | `0.00001` (스크립트에서 설정) | Config 기본값 사용 (확인 필요) |
| `dataloader.batch_size` | `16` (스크립트에서 설정) | Config 기본값 사용 (확인 필요) |
| `llm.grad_accum_step` | `1` (스크립트에서 설정) | Config 기본값 사용 (확인 필요) |

---

## 3. 주요 차이점 요약

### A. VPL + GP (no orthogonal loss)
1. **Orthogonal loss 비활성화**: `vpl_orthogonal_weight: 0.0`, `vpl_num_prototypes: 0`
2. **Batch size 차이**: `8` vs `16` (main table)
3. **Gradient accumulation 차이**: `4` vs `2` (main table)

### B. VPL + Ortho (no GP prior)
1. **GP prior 비활성화**: `vpl_use_gp_prior: False` (standard normal prior)
2. **Orthogonal loss 활성화**: `vpl_orthogonal_weight: 1.0`, `vpl_num_prototypes: 2`
3. **KL weight 차이**: `0.1` (VPL baseline과 동일)

### C. Qwen 2 모델
- Ablation study 스크립트에서 Qwen 2 관련 hyperparameter override가 없을 수 있음
- Main table 스크립트는 Python으로 명시적으로 설정

---

## 4. 권장 사항

1. **t-SNE Visualization**: Ablation study config에 이미 설정되어 있으므로, 최신 코드로 재실행하면 정상 작동해야 합니다.

2. **Hyperparameter 일관성**: 
   - VPL + GP: `batch_size`와 `grad_accum_step`을 main table과 동일하게 맞추는 것을 고려
   - VPL + Ortho: 이미 main table과 동일한 설정

3. **Qwen 2 설정**: Ablation study 스크립트에서도 Qwen 2 관련 hyperparameter를 명시적으로 설정하는 것을 권장
