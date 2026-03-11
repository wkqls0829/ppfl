# Scripts Directory

## Unified Scripts (Recommended)

New parameterized scripts that work on **both** SLURM cluster and local server.
Hyperparameters are centralized in `lib/hyperparams.sh` (single source of truth).

### Quick Start

```bash
# Dry run — see what would happen without executing:
bash scripts/submit.sh --model gemma --phase all --dry-run

# Submit full pipeline on SLURM (selector + RL with automatic dependency chaining):
bash scripts/submit.sh --model gemma --phase all
bash scripts/submit.sh --model qwen --phase all

# Single experiment on SLURM:
bash scripts/submit.sh --model gemma --method fedvpagp --clients 10 --phase all

# Local server (single experiment):
bash scripts/run_selector.sh --method fedvpagp --model gemma --clients 10 --gpu 3
# (wait for selector to finish, then:)
bash scripts/run_rl.sh --method fedvpagp --model gemma --clients 10 --gpu 3
```

### Script Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_selector.sh` | Stage 1 selector training | `--method <m> --model <m> --clients <N> [--tid T] [--gpu G]` |
| `run_rl.sh` | Stage 2 RL training | `--method <m> --model <m> --clients <N> [--rl-tid T] [--selector-tid T] [--gpu G]` |
| `submit.sh` | Batch orchestrator | `--model <m> [--phase selector\|rl\|all] [--method <m>] [--clients <N>]` |

All scripts support `--dry-run` to preview without executing.

TIDs are auto-computed from main table ranges if `--tid` is omitted:
- Gemma selector: 62100+offset, RL: 63100+offset
- Qwen selector: 62200+offset, RL: 63200+offset

### Shared Libraries (`lib/`)

| File | Contents |
|------|----------|
| `lib/common.sh` | Environment detection, checkpoint discovery, config generation |
| `lib/hyperparams.sh` | All method/model hyperparameters, TID calculations, trainer mappings |

---

## Legacy Scripts

### `slurm/` — SLURM cluster

- **`slurm/main_table/`** — Main Table (Table 1): Qwen/Gemma selector + RL
  - `submit_all_gemma.sh`, `submit_all_qwen.sh` — One-click submission with dependency chaining
  - `run_selector_*.sh`, `run_rl_*.sh` — Individual job scripts
  - `ablation/` — Ablation experiments
- **`slurm/hpsearch/`** — Hyperparameter search (Gemma/Qwen)

```bash
# SLURM cluster (one-click: submits all selector + RL with dependencies):
cd /home2/jbkoo/ppfl
bash scripts/slurm/main_table/submit_all_gemma.sh
```

### `server/` — Local server

Individual experiment scripts run with `bash` directly. Use **biscuit** conda env.

- `server/differential_privacy/` — DP(NbAFL) + Qwen main table
- `server/vpl-gp/` — VPL-GP experiments
- `server/ablation/`, `server/unseen/`, `server/unseen-qwen/` — Other experiments
- `server/feddpo/`, `server/fedbiscuit/`, `server/fedvpl/` — Baselines
- `server/llama_vplgp/` — LLaMA VPL-GP

```bash
conda activate biscuit
CUDA_VISIBLE_DEVICES=3 bash scripts/server/vpl-gp/hrl-ultrafeedback-50001.sh
```

See `documents/EXPERIMENT_GUIDE.md` for detailed execution guide.
