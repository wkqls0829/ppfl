# Documents Index

Operational docs: experiment guides, handoffs, state snapshots, and
implementation explainers. For algorithm/architecture reference docs
see `docs/`.

## Start here

- **[HANDOFF_OTHER_SERVER.md](./HANDOFF_OTHER_SERVER.md)** — runbook
  for the other-server Claude picking up the N=10 HP search.
- **[RESUME_STATE.md](./RESUME_STATE.md)** — current sanity-check
  results and known issues.
- **[HP_SEARCH_PLAN.md](./HP_SEARCH_PLAN.md)** — simplified HP search
  plan (3 knobs) for N=10 and N=100.

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
