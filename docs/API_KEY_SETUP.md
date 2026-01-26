# API Key Setup Guide

This guide explains how to configure API keys (especially OpenAI API key) for the experiments without committing them to the repository.

## Overview

The repository uses gitignored configuration files to store sensitive API keys. This ensures that:
- API keys are never committed to the repository
- Each user can configure their own API keys locally
- The setup is secure and follows best practices

## Setup Instructions

### 1. Create `.env` File

Copy the example environment file:

```bash
cp .env.example .env
```

### 2. Add Your OpenAI API Key

Edit the `.env` file and add your OpenAI API key:

```bash
# .env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Important**: The `.env` file is gitignored and will never be committed to the repository.

### 3. Get Your OpenAI API Key

1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key and paste it into your `.env` file

## How It Works

### Environment Variable Loading

The experiment scripts automatically load environment variables from the `.env` file:

```bash
# Loaded automatically in scripts/main_table/*.sh
if [ -f "$WORK_DIR/.env" ]; then
    export $(cat $WORK_DIR/.env | grep -v '^#' | xargs)
fi
```

### API Key Usage in Code

The code checks for the API key in the following order:

1. **Environment Variable** (`OPENAI_API_KEY`)
   - Loaded from `.env` file by scripts
   - Set manually: `export OPENAI_API_KEY=sk-...`

2. **Config File** (`eval.openai_api_key`)
   - Can be set in YAML config files
   - **Not recommended** for API keys (use `.env` instead)

3. **Error if Missing**
   - If no API key is found, the code will raise an error

### Code Reference

The API key is used in `federatedscope/llm/metric/winrate_metrics.py`:

```python
api_key = getattr(ctx.cfg.eval, 'openai_api_key', None)
if api_key is None:
    api_key = os.getenv('OPENAI_API_KEY')
    
if api_key is None:
    logger.error("OpenAI API key not found...")
```

## File Structure

```
ppfl/
├── .env                    # Your API keys (gitignored, DO NOT COMMIT)
├── .env.example            # Template file (committed to repo)
├── .gitignore              # Ensures .env is not committed
└── scripts/main_table/
    ├── run_selector_gemma.sh  # Loads .env automatically
    ├── run_rl_gemma.sh        # Loads .env automatically
    └── ...
```

## Security Best Practices

1. **Never commit `.env` file**
   - The `.env` file is in `.gitignore`
   - Double-check before committing: `git status`

2. **Use `.env.example` as template**
   - The example file shows what variables are needed
   - It does not contain actual keys

3. **Rotate keys if exposed**
   - If you accidentally commit a key, rotate it immediately
   - Revoke the old key in OpenAI dashboard

4. **Use different keys for different environments**
   - Development: `.env.local`
   - Production: `.env.production`
   - All are gitignored

## Troubleshooting

### API Key Not Found Error

If you see:
```
OpenAI API key not found. Set it in config (eval.openai_api_key) or environment variable (OPENAI_API_KEY)
```

**Solutions:**
1. Check if `.env` file exists: `ls -la .env`
2. Verify the key is set: `grep OPENAI_API_KEY .env`
3. Check if script loads it: Look for "Loaded environment variables from .env file" in logs
4. Manually export: `export OPENAI_API_KEY=sk-...`

### Repository Rule Violation

If GitHub shows "repository rule violation":
- This means an API key was committed to the repository
- Remove it from git history (if needed)
- Ensure `.env` is in `.gitignore`
- Use `.env.example` instead

### Scripts Not Loading .env

If scripts don't load `.env`:
1. Check file path: Scripts look for `$WORK_DIR/.env`
2. Check file permissions: `chmod 600 .env`
3. Verify syntax: No spaces around `=` in `.env` file

## Alternative: Config File Method

If you prefer not to use `.env`, you can set the API key in config files:

```yaml
# cfg/main_table/.../hrl_*.yaml
eval:
  openai_api_key: sk-your-key-here
```

**Warning**: This method is less secure as config files might be committed. Use `.env` method instead.

## Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Environment Variables Best Practices](https://12factor.net/config)
- [Gitignore Documentation](https://git-scm.com/docs/gitignore)
