# Z 값 수집 최적화 (Z Collection Optimization)

## 문제점

HHST (selector training)에서 매 라운드마다 z 값을 broadcast하고 수집하는 것이 시간을 많이 잡아먹고 있었습니다.

## 최적화 방법

### 1. Server 측 최적화

**변경 전**:
- 매 라운드마다 `_collect_z_values_for_visualization()` 호출
- t-SNE visualization은 10라운드마다만 실행되지만, z 값 수집은 매 라운드마다 발생

**변경 후**:
- `vpl_tsne_visualize_freq` 설정값을 확인하여 visualization이 필요한 라운드에만 z 값 수집
- 기본값: 10라운드마다만 수집 (기존과 동일한 visualization 빈도)

**코드 위치**: `federatedscope/llm/llm_local/server.py` (line 243-250)

```python
# OPTIMIZATION: Only collect z values when needed for visualization to reduce overhead
if hasattr(self._cfg.llm, 'vpl_latent_dim'):  # VPL is enabled
    visualize_freq = getattr(self._cfg.llm, 'vpl_tsne_visualize_freq', 10)  # Default: every 10 rounds
    # Only collect z values when we need to visualize (or every round if freq=1)
    if visualize_freq <= 1 or self.state % visualize_freq == 0:
        self._collect_z_values_for_visualization()
```

### 2. Client 측 최적화

**변경 전**:
- 매 라운드마다 `get_client_z_values()` 호출하여 z 값을 계산하고 전송

**변경 후**:
- Server와 동일한 로직으로 visualization이 필요한 라운드에만 z 값 계산 및 전송
- 불필요한 z 값 계산 및 통신 오버헤드 제거

**코드 위치**: `federatedscope/llm/llm_local/client.py` (line 171-177)

```python
# OPTIMIZATION: Only collect z values when needed for visualization to reduce overhead
should_collect_z = True
if hasattr(self._cfg.llm, 'vpl_tsne_visualize_freq'):
    visualize_freq = self._cfg.llm.vpl_tsne_visualize_freq
    if visualize_freq > 1 and self.state % visualize_freq != 0:
        should_collect_z = False

if should_collect_z and hasattr(self.trainer, 'get_client_z_values'):
    z_values = self.trainer.get_client_z_values()
    if z_values is not None:
        model_para_all['client_z_values'] = z_values.cpu() if isinstance(z_values, torch.Tensor) else z_values
```

## 예상 효과

### 시간 절약
- **기존**: 매 라운드마다 z 값 수집 (50 라운드 = 50번 수집)
- **최적화 후**: 10라운드마다만 수집 (50 라운드 = 5번 수집)
- **절약률**: 약 **90%** 시간 절약 (z 값 수집 관련)

### 통신 오버헤드 감소
- Client → Server 통신량 감소
- Server에서 z 값 처리 시간 감소

### 메모리 사용량 감소
- 불필요한 z 값 저장 공간 감소

## 주의사항

1. **GP Prior 사용 시**: GP prior를 사용하는 경우 (`vpl_use_gp_prior: True`), z distribution (mu, logvar)은 여전히 매 라운드마다 수집됩니다. 이는 GP prior 업데이트에 필요하기 때문입니다.

2. **Visualization 빈도 조정**: `vpl_tsne_visualize_freq`를 1로 설정하면 매 라운드마다 수집됩니다 (기존 동작).

3. **최종 라운드**: 최종 라운드에서는 항상 z 값을 수집하여 최종 visualization을 생성합니다.

## 설정 방법

Config 파일에서 `vpl_tsne_visualize_freq`를 조정하여 visualization 빈도를 변경할 수 있습니다:

```yaml
llm:
  vpl_tsne_visualize_freq: 10  # 10라운드마다 visualization (기본값)
  # vpl_tsne_visualize_freq: 5   # 5라운드마다 visualization
  # vpl_tsne_visualize_freq: 1   # 매 라운드마다 visualization (최적화 효과 없음)
```

## 호환성

- 기존 실험과 호환됩니다 (기본값 10라운드마다 visualization은 동일)
- GP prior 사용 시에도 정상 작동합니다
- 모든 VPL 알고리즘 (FedVPL, FedVPA-GP, VPL+GP, VPL+Ortho)에 적용됩니다
