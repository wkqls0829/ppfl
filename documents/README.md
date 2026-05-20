# Documents Index

Operational docs: experiment guides, handoffs, results, and
implementation explainers. For algorithm/architecture reference docs
see `docs/`.

## Start here

- **[MAIN_TABLE_SETUP.md](./MAIN_TABLE_SETUP.md)** — current Main
  Table hyperparameter recipe + rationale (every choice grounded in
  HP search results). The canonical config doc.
- **[HANDOFF_OTHER_SERVER.md](./HANDOFF_OTHER_SERVER.md)** — runbook
  for the other server to launch the N=10 Main Table sweep.
- **[HP_SEARCH_RESULTS_N100.md](./HP_SEARCH_RESULTS_N100.md)** —
  N=100 HP search results (tier-1, single seed).
- **[HP_SEARCH_RESULTS_N10.md](./HP_SEARCH_RESULTS_N10.md)** — N=10
  HP search results (tier-1 + tier-2). Contains the universal-DPO-
  collapse finding that drove the Main Table recipe.

## Experiment guides

- **[EXPERIMENT_GUIDE.md](./EXPERIMENT_GUIDE.md)** — general
  experiment runbook.
- **[CONTEXT_FOR_NEW_SESSION.md](./CONTEXT_FOR_NEW_SESSION.md)** —
  project context (Korean, kept for reference).
- **[Z_SEPARATION_EXPERIMENTS.md](./Z_SEPARATION_EXPERIMENTS.md)** —
  z-separation diagnostic experiments.

## Environment & setup

- **[GPU_ALLOCATION_GUIDE.md](./GPU_ALLOCATION_GUIDE.md)** — GPU
  allocation conventions (SLURM vs local server).
- **[GPT_API_SETUP.md](./GPT_API_SETUP.md)** — GPT API key setup for
  win-rate eval.
- **[../docs/API_KEY_SETUP.md](../docs/API_KEY_SETUP.md)** — OpenAI
  API key (local/cluster/.env).

## Algorithm explainers (durable reference)

- **[CLOP_ORTHOGONAL_LOSS_EXPLANATION.md](./CLOP_ORTHOGONAL_LOSS_EXPLANATION.md)**
- **[MIXTURE_PRIOR_EXPLANATION.md](./MIXTURE_PRIOR_EXPLANATION.md)**
- **[GUMBEL_SOFTMAX_DIFFERENTIABILITY.md](./GUMBEL_SOFTMAX_DIFFERENTIABILITY.md)**
- **[SIGMA_INITIALIZATION_EXPLANATION.md](./SIGMA_INITIALIZATION_EXPLANATION.md)**
- **[Z_EMBEDDING_GENERATION_EXPLANATION.md](./Z_EMBEDDING_GENERATION_EXPLANATION.md)**

## Implementation deep-dives

- **[VPL_UNIFIED_IMPLEMENTATION.md](./VPL_UNIFIED_IMPLEMENTATION.md)**
- **[VPL_IMPLEMENTATION_DETAILED.md](./VPL_IMPLEMENTATION_DETAILED.md)**
- **[VPL_GP_IMPLEMENTATION.md](./VPL_GP_IMPLEMENTATION.md)**

## Optimization notes

- **[TRAINING_SPEED_OPTIMIZATION.md](./TRAINING_SPEED_OPTIMIZATION.md)**

## Related folders

- **`../docs/`** — code and system technical documentation, paper
  drafts (`.tex`), figures.
- **`../CLAUDE.md`** — root-level codebase guide for Claude.
- **`../README.md`** — project overview.

## Removed docs (history in git)

- `HP_SEARCH_PLAN.md` — superseded by `HP_SEARCH_RESULTS_N10.md` and
  `HP_SEARCH_RESULTS_N100.md`.
- `RESUME_STATE.md` — single-cell sanity-check state from 2026-05-17;
  outdated by HP search results.
- `BUGFIX_2026_05_19.md` — the 30-sample eval silent-fallback bug;
  fixed in commit `7d686ec`, history preserved via `git log`.
