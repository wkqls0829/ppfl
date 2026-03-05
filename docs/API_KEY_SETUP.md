# API Key 및 환경 변수 설정 가이드

OpenAI API 키를 로컬·원격·클러스터에서 설정하는 방법을 하나의 문서로 정리합니다.  
키는 `.env` 또는 셸 프로필에 두고, **저장소에 커밋하지 않습니다.**

## 개요

- API 키는 `.env`(권장) 또는 `OPENAI_API_KEY` 환경 변수로 제공합니다.
- 스크립트는 `$WORK_DIR/.env`를 자동 로드합니다.
- Config의 `eval.openai_api_key`도 지원하지만, 보안상 환경 변수 사용을 권장합니다.

## 1. .env 파일로 설정 (권장)

### 로컬 / 일반 서버

```bash
cp .env.example .env
# .env 편집
# OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 클러스터

```bash
ssh <cluster_login_node>
cd /home2/jbkoo/ppfl   # 또는 실제 작업 디렉토리

cp .env.example .env
nano .env   # 또는 vim
# OPENAI_API_KEY=sk-your-actual-api-key-here

chmod 600 .env
```

스크립트는 다음처럼 자동 로드합니다:

```bash
if [ -f "$WORK_DIR/.env" ]; then
    export $(cat $WORK_DIR/.env | grep -v '^#' | xargs)
    echo "Loaded environment variables from .env file"
fi
```

### .env 선택 항목

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
# OPENAI_API_BASE=https://api.openai.com/v1
# OPENAI_MODEL=gpt-4o-mini
```

## 2. 셸 프로필에 설정 (선택)

현재 사용자만 쓰는 환경이면 `~/.bashrc` 또는 `~/.zshrc`에 넣을 수 있습니다.

```bash
# ~/.bashrc 또는 ~/.zshrc
export OPENAI_API_KEY="sk-***YOUR_API_KEY_HERE***"
```

적용: `source ~/.bashrc` 또는 `source ~/.zshrc`.  
**주의**: 터미널 세션에서만 쓰려면 `export OPENAI_API_KEY=...` 로 임시 설정 (세션 종료 시 사라짐).

### Conda 환경에서만 쓰기

```bash
conda activate your_env_name
conda env config vars set OPENAI_API_KEY="sk-***YOUR_API_KEY_HERE***"
conda deactivate && conda activate your_env_name
```

## 3. 코드에서의 사용 순서

`federatedscope/llm/metric/winrate_metrics.py` 등에서는 다음 순서로 확인합니다:

1. **Config**: `eval.openai_api_key`
2. **환경 변수**: `OPENAI_API_KEY` (스크립트가 `.env`에서 로드)
3. 없으면 에러: `"OpenAI API key not found. Set it in config (eval.openai_api_key) or environment variable (OPENAI_API_KEY)"`

## 4. 파일 구조

```
ppfl/
├── .env                 # API 키 (gitignored, 커밋 금지)
├── .env.example         # 템플릿 (저장소에 포함)
├── .gitignore           # .env 제외
└── scripts/main_table/
    ├── run_selector_gemma.sh
    ├── run_rl_gemma.sh
    └── ...
```

클러스터에서는 `.env` 경로가 스크립트의 `$WORK_DIR`과 일치해야 합니다 (예: `/home2/jbkoo/ppfl/.env`).

## 5. 보안

- **절대 커밋하지 마세요**: `.env`는 `.gitignore`에 있어야 합니다. 커밋 전 `git status` 확인.
- **파일 권한**: `chmod 600 .env` (소유자만 읽기/쓰기).
- **키 노출 시**: OpenAI 대시보드에서 즉시 키 재발급 및 이전 키 삭제.
- **Config에 키 넣기**: 가능하지만 보안상 비권장. 사용 시 해당 config는 커밋하지 마세요.

## 6. 문제 해결

### "OpenAI API key not found" 에러

1. `.env` 존재 여부: `ls -la .env` (또는 클러스터에서는 `ls -la $WORK_DIR/.env`).
2. 키 설정 여부: `grep OPENAI_API_KEY .env`.
3. 스크립트가 로드하는지: 로그에 `"Loaded environment variables from .env file"` 있는지 확인.
4. 수동 확인: `source .env` 후 `echo $OPENAI_API_KEY | cut -c1-10`.

### .env가 로드되지 않을 때

- 스크립트가 사용하는 `$WORK_DIR`과 `.env` 경로가 같은지 확인.
- `.env` 문법: `KEY=value` (등호 주변 공백 없음), 주석은 `#`.

### Conda/SSH에서 환경 변수가 안 보일 때

- Conda: `conda env config vars set OPENAI_API_KEY=...` 후 환경 재활성화.
- SSH: `~/.bashrc` 또는 `~/.bash_profile`에 넣었는지, 새 세션에서 `source` 했는지 확인.

### Config 파일로 임시 설정 (비권장)

```yaml
# cfg/.../hrl_*.yaml
eval:
  openai_api_key: sk-your-key-here
```

Config는 커밋하지 말고, 가능하면 `.env`로 이전하는 것을 권장합니다.

## 7. 참고

- [OpenAI API Keys](https://platform.openai.com/api-keys)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- SLURM/클러스터: `docs/SLURM_GPU_MONITORING.md`
