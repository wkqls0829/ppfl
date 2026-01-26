# 클러스터에서 GPT API 설정 가이드

클러스터 환경에서 GPT API 키를 설정하는 방법입니다.

## 빠른 설정

### 1. 클러스터에 SSH 접속
```bash
ssh <cluster_login_node>
```

### 2. 작업 디렉토리로 이동
```bash
cd /home2/jbkoo/ppfl
```

### 3. `.env` 파일 생성
```bash
# .env.example 파일이 있다면 복사
cp .env.example .env

# 또는 직접 생성
nano .env
# 또는
vim .env
```

### 4. API 키 추가
`.env` 파일에 다음 내용 추가:
```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional: OpenAI API Base URL (for custom endpoints)
# OPENAI_API_BASE=https://api.openai.com/v1

# Optional: OpenAI Model (default: gpt-4o-mini)
# OPENAI_MODEL=gpt-4o-mini
```

### 5. 파일 권한 설정 (보안)
```bash
chmod 600 .env
```

### 6. 확인
```bash
# API 키가 제대로 설정되었는지 확인 (키는 표시되지 않음)
grep -q "OPENAI_API_KEY" .env && echo "✓ API key found" || echo "✗ API key not found"

# 환경 변수로 로드 테스트
source .env
echo $OPENAI_API_KEY | cut -c1-10  # 처음 10자만 표시
```

## 자동 로드 확인

모든 클러스터 스크립트는 자동으로 `.env` 파일을 로드합니다:

```bash
# 스크립트 내부에서 자동 실행됨
if [ -f "$WORK_DIR/.env" ]; then
    export $(cat $WORK_DIR/.env | grep -v '^#' | xargs)
    echo "Loaded environment variables from .env file"
fi
```

### 로그에서 확인
작업 실행 시 로그에 다음 메시지가 나타나면 정상입니다:
```
Loaded environment variables from .env file
```

## OpenAI API 키 얻기

1. [OpenAI Platform](https://platform.openai.com/api-keys) 접속
2. 로그인 또는 계정 생성
3. "Create new secret key" 클릭
4. 키 복사 (한 번만 표시되므로 저장)
5. `.env` 파일에 붙여넣기

## 설정 위치

클러스터에서 `.env` 파일은 다음 위치에 있어야 합니다:
```
/home2/jbkoo/ppfl/.env
```

이 경로는 모든 스크립트에서 `$WORK_DIR/.env`로 참조됩니다.

## 문제 해결

### API 키가 로드되지 않는 경우

1. **파일 경로 확인**
   ```bash
   ls -la /home2/jbkoo/ppfl/.env
   ```

2. **파일 내용 확인**
   ```bash
   cat /home2/jbkoo/ppfl/.env
   # 또는
   grep OPENAI_API_KEY /home2/jbkoo/ppfl/.env
   ```

3. **스크립트 로그 확인**
   ```bash
   # 작업 로그에서 확인
   grep "Loaded environment variables" outputs/*.log
   ```

4. **수동 테스트**
   ```bash
   cd /home2/jbkoo/ppfl
   source .env
   echo $OPENAI_API_KEY | cut -c1-10
   ```

### API 키 오류

RL 실험 실행 시 다음과 같은 오류가 발생할 수 있습니다:
```
OpenAI API key not found. Set it in config (eval.openai_api_key) or environment variable (OPENAI_API_KEY)
```

**해결 방법:**
1. `.env` 파일이 올바른 위치에 있는지 확인
2. API 키 형식이 올바른지 확인 (`sk-`로 시작)
3. 파일 권한 확인: `chmod 600 .env`
4. 스크립트가 `.env`를 로드하는지 로그 확인

### 파일 권한 문제

```bash
# .env 파일 권한 확인
ls -l /home2/jbkoo/ppfl/.env

# 올바른 권한 설정 (소유자만 읽기/쓰기)
chmod 600 /home2/jbkoo/ppfl/.env
```

## 보안 주의사항

1. **절대 커밋하지 마세요**
   - `.env` 파일은 `.gitignore`에 포함되어 있습니다
   - 커밋 전 확인: `git status`

2. **파일 권한 설정**
   ```bash
   chmod 600 .env  # 소유자만 읽기/쓰기
   ```

3. **키 공유 금지**
   - API 키는 개인 계정에 연결되어 있습니다
   - 다른 사람과 공유하지 마세요

4. **키 로테이션**
   - 키가 노출되었다고 생각되면 즉시 교체
   - OpenAI 대시보드에서 이전 키 삭제

## 테스트

### 간단한 테스트
```bash
# 클러스터에서 직접 테스트 (Python 사용)
cd /home2/jbkoo/ppfl
source .env
python3 << EOF
import os
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"✓ API key loaded: {api_key[:10]}...")
else:
    print("✗ API key not found")
EOF
```

### 실제 실험으로 테스트
```bash
# 테스트 스크립트 실행
sbatch scripts/main_table/test_cluster.sh gemma-2b fedvpagp

# 로그 확인
tail -f outputs/90001_test.log | grep -i "api\|openai\|gpt"
```

## 추가 설정 (선택사항)

### 다른 OpenAI 엔드포인트 사용
```bash
# .env 파일에 추가
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

### 여러 환경 관리
```bash
# 개발 환경
.env.development

# 프로덕션 환경
.env.production

# 모두 gitignored
```

## 참고

- 기본 API 키 설정: `docs/API_KEY_SETUP.md`
- SLURM 모니터링: `docs/SLURM_GPU_MONITORING.md`
- Main Table 실험: `scripts/main_table/README.md`
