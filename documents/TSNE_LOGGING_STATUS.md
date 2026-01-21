# t-SNE 로깅 상태 확인 및 수정 사항

## 현재 구현 상태

### 1. 클라이언트 측 (VPLRewardChoiceTrainer)

#### z 값 수집
- **위치**: `_hook_on_batch_forward`
- **수정 전**: GP prior가 활성화된 경우에만 z 값 수집
- **수정 후**: 항상 z 값 수집 (GP prior와 무관)

```python
# 수정 후
# Collect z values for visualization (always collect, not just for GP prior)
if not hasattr(self, 'z_history'):
    self.z_history = []
z_cpu = z.detach().cpu()
self.z_history.append(z_cpu)
```

#### z 값 처리
- **위치**: `_hook_on_fit_end`
- **수정 전**: z 값 처리 로직 없음
- **수정 후**: 항상 z 값 샘플링 및 저장

```python
# 수정 후
if hasattr(self, 'z_history') and len(self.z_history) > 0:
    z_values = torch.cat(self.z_history, dim=0)
    num_samples = min(100, len(z_values))  # 최대 100개 샘플
    sampled_indices = torch.randperm(len(z_values))[:num_samples]
    self.client_z_values = z_values[sampled_indices]
    self.z_history = []  # 다음 라운드를 위해 초기화
```

#### z 값 반환
- **위치**: `get_client_z_values`
- **수정 전**: GP prior가 활성화된 경우에만 반환
- **수정 후**: 항상 반환 (GP prior와 무관)

```python
# 수정 후
def get_client_z_values(self):
    """Available for all VPL experiments (not just GP prior)"""
    if not hasattr(self, 'client_z_values') or self.client_z_values is None:
        return None
    return self.client_z_values.clone()
```

### 2. 클라이언트 측 (LLMMultiLoRAClient)

#### z 값 전송
- **위치**: `train()` 메서드
- **수정 전**: GP prior가 활성화된 경우에만 전송
- **수정 후**: 항상 전송 (VPL이 활성화된 경우)

```python
# 수정 후
# Always collect z values for visualization (not just for GP prior)
if hasattr(self.trainer, 'get_client_z_values'):
    z_values = self.trainer.get_client_z_values()
    if z_values is not None:
        model_para_all['client_z_values'] = z_values.cpu()
```

### 3. 서버 측 (LLMMultiLoRAServer)

#### z 값 수집
- **위치**: `aggregate()` 메서드
- **수정 전**: GP prior가 활성화된 경우에만 수집
- **수정 후**: VPL이 활성화된 경우 항상 수집

```python
# 수정 후
# VPL-GP: Collect z distributions (only if GP prior is enabled)
if hasattr(self._cfg.llm, 'vpl_use_gp_prior') and self._cfg.llm.vpl_use_gp_prior:
    self._collect_vpl_gp_prior_distributions()
    self._compute_balanced_orthogonal_labels()

# Always collect z values for visualization (even if GP prior is disabled)
if hasattr(self._cfg.llm, 'vpl_latent_dim'):  # VPL is enabled
    self._collect_z_values_for_visualization()
```

#### t-SNE 시각화
- **위치**: `_collect_z_values_for_visualization`
- **동작**: 매 10 라운드마다 t-SNE 시각화 생성
- **저장 위치**: `output_dir/cross_client_z_tsne_round_{round_num}.png`
- **WandB 로깅**: 활성화된 경우 WandB에도 로깅

## 전체 파이프라인

```
1. Forward Pass (각 배치)
   ↓
2. z 값 수집 (z_history에 추가)
   ↓
3. Round 종료 (_hook_on_fit_end)
   ↓
4. z 값 샘플링 (최대 100개)
   ↓
5. client_z_values 저장
   ↓
6. 클라이언트 → 서버 전송 (model_para_all['client_z_values'])
   ↓
7. 서버에서 수집 (_collect_z_values_for_visualization)
   ↓
8. client_z_values_dict에 저장 (누적)
   ↓
9. 매 10 라운드마다 t-SNE 시각화
   ↓
10. 파일 저장 및 WandB 로깅
```

## 수정 사항 요약

### ✅ 수정 완료

1. **z 값 수집**: GP prior와 무관하게 항상 수집
2. **z 값 처리**: `_hook_on_fit_end`에서 항상 처리
3. **z 값 반환**: `get_client_z_values`가 항상 반환
4. **z 값 전송**: 클라이언트에서 항상 전송
5. **z 값 수집**: 서버에서 VPL이 활성화된 경우 항상 수집

### 📊 시각화 설정

- **시각화 주기**: 매 10 라운드 (config로 조정 가능: `vpl_tsne_visualize_freq`)
- **저장 위치**: `output_dir/cross_client_z_tsne_round_{round_num}.png`
- **포함 정보**:
  - 각 클라이언트의 z 값 (색상으로 구분)
  - Orthogonal prototypes (별표로 표시)
  - Orthogonal labels (윤곽선으로 표시)

## 확인 방법

1. **로그 확인**:
   ```
   Collected z values for visualization: torch.Size([100, 32]) from 10 batches
   Round X: Collected z values from 5 clients. Total accumulated: 500 points
   ```

2. **파일 확인**:
   ```
   outputs/{exp_id}/cross_client_z_tsne_round_10.png
   outputs/{exp_id}/cross_client_z_tsne_round_20.png
   ...
   ```

3. **WandB 확인** (활성화된 경우):
   ```
   visualization/cross_client_z_tsne_round_10
   visualization/cross_client_z_tsne_round_20
   ...
   ```

## 주의사항

1. **메모리 사용**: z 값은 최대 100개만 샘플링하여 저장
2. **시각화 주기**: 기본적으로 매 10 라운드마다 시각화 (조정 가능)
3. **누적 데이터**: 서버에서 모든 라운드의 z 값을 누적하여 저장
