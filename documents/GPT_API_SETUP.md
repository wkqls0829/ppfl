# GPT API를 사용한 Win-Lose Evaluation 설정 가이드

## 개요

Win-lose evaluation (helpfulness/harmlessness winrate)에서 GPT API를 사용할 수 있습니다. 
기본적으로는 학습 중인 내부 모델을 사용하지만, 더 정확한 평가를 위해 GPT API를 사용할 수 있습니다.

## 1. OpenAI API 키 발급

1. **OpenAI 계정 생성/로그인**
   - https://platform.openai.com/ 접속
   - 계정 생성 또는 로그인

2. **API 키 발급**
   - https://platform.openai.com/api-keys 접속
   - "Create new secret key" 클릭
   - 키 이름 지정 (예: "ppfl-evaluation")
   - 생성된 키를 안전하게 복사 (다시 볼 수 없음!)

3. **결제 설정**
   - https://platform.openai.com/account/billing 접속
   - 결제 수단 추가 (신용카드 등)
   - Usage limits 설정 (선택사항)

## 2. API 키 설정 방법

### 방법 1: 환경변수 사용 (권장)

```bash
export OPENAI_API_KEY="sk-..."
```

또는 `.bashrc` 또는 `.zshrc`에 추가:
```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

### 방법 2: Config 파일에 직접 설정

`cfg/vpl-gp/hhst-60000.yaml` 파일에 추가:

```yaml
eval:
  use_gpt_api_for_winrate: True  # GPT API 사용 여부
  openai_api_key: "sk-..."  # API 키 (보안상 환경변수 사용 권장)
  openai_model: "gpt-4o-mini"  # 사용할 모델 (기본값: gpt-4o-mini)
  max_samples_for_reward: 100  # 평가할 샘플 수
```

**보안 주의**: API 키를 config 파일에 직접 넣는 것은 권장하지 않습니다. 환경변수 사용을 권장합니다.

## 3. 필요한 라이브러리 설치

```bash
pip install openai
```

## 4. 사용 가능한 모델 및 비용

### 추천 모델 (비용 효율적)
- **gpt-4o-mini**: 가장 저렴, 빠름, 충분한 성능
  - Input: $0.15 / 1M tokens
  - Output: $0.60 / 1M tokens

### 고성능 모델
- **gpt-4o**: 더 정확하지만 비용이 높음
  - Input: $2.50 / 1M tokens
  - Output: $10.00 / 1M tokens

- **gpt-4-turbo**: 균형잡힌 선택
  - Input: $10.00 / 1M tokens
  - Output: $30.00 / 1M tokens

### 비용 예상
- 샘플 100개 평가 시:
  - 각 샘플 약 500 tokens (prompt + response)
  - gpt-4o-mini: 약 $0.01-0.02
  - gpt-4o: 약 $0.15-0.30

## 5. Config 설정 예시

### GPT API 사용 (환경변수 방식)

```yaml
eval:
  use_gpt_api_for_winrate: True  # GPT API 활성화
  openai_model: "gpt-4o-mini"  # 사용할 모델
  max_samples_for_reward: 100  # 평가 샘플 수
  metrics: ['loss', 'acc', 'vpl_kl_loss', 'vpl_reconstruction_loss', 
            'helpfulness_winrate', 'harmlessness_winrate']
```

환경변수 설정:
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

### 내부 모델 사용 (기본값)

```yaml
eval:
  use_gpt_api_for_winrate: False  # 또는 설정하지 않음
  max_samples_for_reward: 100
  metrics: ['loss', 'acc', 'vpl_kl_loss', 'vpl_reconstruction_loss', 
            'helpfulness_winrate', 'harmlessness_winrate']
```

## 6. 사용 방법

1. **환경변수 설정**:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. **Config 파일 수정**:
   ```yaml
   eval:
     use_gpt_api_for_winrate: True
     openai_model: "gpt-4o-mini"
   ```

3. **실험 실행**:
   ```bash
   bash scripts/vpl-gp/hhst.sh
   ```

## 7. 주의사항

1. **비용 관리**
   - Usage limits를 설정하여 예상치 못한 비용 방지
   - https://platform.openai.com/account/limits

2. **Rate Limiting**
   - 코드에 0.1초 지연이 포함되어 있지만, 필요시 조정 가능
   - Rate limit 에러 발생 시 재시도 로직 추가 고려

3. **API 키 보안**
   - Git에 API 키를 커밋하지 않도록 주의
   - 환경변수 사용 권장

4. **에러 처리**
   - API 호출 실패 시 내부 모델로 fallback 가능 (현재는 기본값 A 선택)

## 8. 문제 해결

### "OpenAI library not available" 에러
```bash
pip install openai
```

### "OpenAI API key not found" 에러
- 환경변수 확인: `echo $OPENAI_API_KEY`
- Config 파일에 `eval.openai_api_key` 설정 확인

### Rate limit 에러
- `max_samples_for_reward` 값을 줄이기
- 코드의 `time.sleep(0.1)` 값을 늘리기

### 비용이 너무 높은 경우
- `openai_model`을 `gpt-4o-mini`로 변경
- `max_samples_for_reward` 값을 줄이기
