# GPT API 키 설정 가이드

## 개요

이 문서는 GPT API를 사용하기 위한 API 키 설정 방법을 설명합니다. 
로컬 서버와 원격 서버 모두에서 동일한 방법으로 설정할 수 있습니다.

## 1. API 키 확인

현재 프로젝트에서 사용하는 API 키:
- **Key Prefix**: `sk-***YOUR_API_KEY_HERE***`

## 2. 로컬 서버 설정 (현재 서버)

### 방법 1: ~/.bashrc에 영구 설정 (권장)

```bash
# ~/.bashrc 파일에 추가
echo 'export OPENAI_API_KEY="sk-***YOUR_API_KEY_HERE***"' >> ~/.bashrc

# 설정 적용
source ~/.bashrc

# 확인
echo $OPENAI_API_KEY
```

### 방법 2: ~/.zshrc에 설정 (zsh 사용 시)

```bash
# ~/.zshrc 파일에 추가
echo 'export OPENAI_API_KEY="sk-***YOUR_API_KEY_HERE***"' >> ~/.zshrc

# 설정 적용
source ~/.zshrc
```

### 방법 3: 현재 세션에만 임시 설정

```bash
export OPENAI_API_KEY="sk-***YOUR_API_KEY_HERE***"
```

**주의**: 이 방법은 터미널 세션을 종료하면 사라집니다.

## 3. 원격 서버 설정

### SSH로 접속한 경우

1. **SSH 접속**:
   ```bash
   ssh username@remote-server
   ```

2. **환경변수 설정** (로컬과 동일):
   ```bash
   # ~/.bashrc 또는 ~/.zshrc에 추가
   echo 'export OPENAI_API_KEY="sk-***YOUR_API_KEY_HERE***"' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **설정 확인**:
   ```bash
   echo $OPENAI_API_KEY
   ```

### Conda 환경 사용 시

Conda 환경을 사용하는 경우, 환경별로 설정할 수 있습니다:

```bash
# Conda 환경 활성화
conda activate your_env_name

# 환경변수 설정 (활성화된 환경에서만 유효)
conda env config vars set OPENAI_API_KEY="sk-***YOUR_API_KEY_HERE***"

# 환경 재활성화
conda deactivate
conda activate your_env_name
```

### 시스템 전체 설정 (모든 사용자)

**주의**: 이 방법은 시스템의 모든 사용자가 API 키에 접근할 수 있으므로 보안상 권장하지 않습니다.

```bash
# /etc/environment에 추가 (sudo 권한 필요)
sudo echo 'OPENAI_API_KEY="sk-***YOUR_API_KEY_HERE***"' >> /etc/environment
```

## 4. 설정 확인

### 환경변수 확인

```bash
# 환경변수가 설정되었는지 확인
echo $OPENAI_API_KEY

# 출력 예시: sk-***YOUR_API_KEY_HERE***
```

### Python에서 확인

```python
import os
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"API Key 설정됨: {api_key[:20]}...")
else:
    print("API Key가 설정되지 않았습니다.")
```

### 실험 실행 시 확인

실험을 실행하면 로그에 다음 메시지가 나타납니다:
- API 키가 설정된 경우: 정상적으로 GPT API 호출
- API 키가 없는 경우: `"OpenAI API key not found"` 에러

## 5. Config 파일에서 설정 (선택사항)

환경변수 대신 config 파일에 직접 설정할 수도 있습니다 (보안상 권장하지 않음):

```yaml
# cfg/vpl-gp/hhst-ortho-60001.yaml
eval:
  use_gpt_api_for_winrate: True
  openai_api_key: "sk-***YOUR_API_KEY_HERE***"
  openai_model: "gpt-4o-mini"
```

**보안 주의**: 
- Config 파일을 Git에 커밋하지 마세요!
- `.gitignore`에 config 파일을 추가하거나, 환경변수 사용을 권장합니다.

## 6. 문제 해결

### 문제: 환경변수가 설정되지 않음

**해결책**:
1. `~/.bashrc` 또는 `~/.zshrc` 파일을 확인
2. `source ~/.bashrc` 또는 `source ~/.zshrc` 실행
3. 새 터미널 세션 시작

### 문제: Conda 환경에서 환경변수가 인식되지 않음

**해결책**:
```bash
# Conda 환경에서 직접 설정
conda env config vars set OPENAI_API_KEY="sk-***YOUR_API_KEY_HERE***..."
conda deactivate
conda activate your_env_name
```

### 문제: SSH 세션에서 환경변수가 사라짐

**해결책**:
- `~/.bashrc` 또는 `~/.bash_profile`에 설정
- SSH 접속 시 자동으로 로드되도록 설정

### 문제: "OpenAI API key not found" 에러

**해결책**:
1. 환경변수 확인: `echo $OPENAI_API_KEY`
2. Python에서 확인: `import os; print(os.getenv('OPENAI_API_KEY'))`
3. Config 파일에 직접 설정 (임시 해결책)

## 7. 보안 주의사항

1. **Git에 커밋하지 마세요**
   - API 키가 포함된 파일을 Git에 커밋하지 마세요
   - `.gitignore`에 config 파일 추가 고려

2. **환경변수 사용 권장**
   - Config 파일보다 환경변수 사용이 더 안전합니다
   - 환경변수는 프로세스별로 격리됩니다

3. **권한 관리**
   - API 키 파일의 권한을 제한: `chmod 600 ~/.bashrc`
   - 다른 사용자가 읽을 수 없도록 주의

4. **API 키 교체**
   - 키가 노출된 경우 즉시 OpenAI에서 키를 재생성하세요
   - https://platform.openai.com/api-keys

## 8. 빠른 참조

### 현재 서버에서 설정
```bash
echo 'export OPENAI_API_KEY="sk-***YOUR_API_KEY_HERE***"' >> ~/.bashrc
source ~/.bashrc
```

### 다른 서버에서 설정
```bash
# SSH 접속 후
echo 'export OPENAI_API_KEY="sk-***YOUR_API_KEY_HERE***"' >> ~/.bashrc
source ~/.bashrc
echo $OPENAI_API_KEY  # 확인
```

### 설정 확인
```bash
echo $OPENAI_API_KEY
python -c "import os; print('OK' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

## 9. 관련 문서

- [GPT_API_SETUP.md](./GPT_API_SETUP.md): GPT API 사용 방법 및 비용 정보
- [OpenAI API Documentation](https://platform.openai.com/docs)
