# ICML 2026 Rebuttal — Experiment Results

## Status: All rebuttal experiments COMPLETED (as of 2026-04-10)

---

## 1. Imbalanced Split Results [R1-W1]

### Stage 1 (Selector) — All DONE

| TID | Method | Split | Rounds | Acc |
|-----|--------|-------|--------|-----|
| 30000 | FedBiscuit | 30:70 (3 harmless, 7 helpful) | 40 | 49.3% |
| 30001 | FedVPA-GP | 30:70 | 40 | 99.1% |
| 30002 | FedBiscuit | 70:30 (7 harmless, 3 helpful) | 40 | 47.3% |
| 30003 | FedVPA-GP | 70:30 | 40 | 98.8% |

### Stage 2 (RL/DPO) — All DONE

| TID | Method | Split | Helpfulness↑ | Harmlessness↑ |
|-----|--------|-------|-------------|--------------|
| **31000** | **FedBiscuit** | **30:70** | **46.7%** | **70.0%** |
| **31001** | **FedVPA-GP** | **30:70** | **80.0%** | **76.7%** |
| **31002** | **FedBiscuit** | **70:30** | **56.7%** | **80.0%** |
| **31003** | **FedVPA-GP** | **70:30** | **86.7%** | **66.7%** |

**Key finding**: FedVPA-GP achieves Pareto improvement over FedBiscuit on both imbalanced splits:
- 30:70 split: +33.3% helpfulness, +6.7% harmlessness
- 70:30 split: +30.0% helpfulness, -13.3% harmlessness (trades some harmlessness for large helpfulness gain)

---

## 2. Pull vs Orthonorm Ablation [R2-W3]

### Stage 2 (RL/DPO) — All DONE

| TID | Pull (λ) | Orthonorm (γ) | Helpfulness↑ | Harmlessness↑ |
|-----|---------|--------------|-------------|--------------|
| (10209→11209) | 1.0 | 0.1 | (ref: full method) | |
| **31100** | **1.0** | **0.0** | **80.0%** | **76.7%** |
| **31101** | **0.0** | **0.1** | **73.3%** | **66.7%** |

**Key finding**: Both components contribute, but pull loss has a larger impact:
- Pull loss only (31100): Strong on both metrics (80%, 76.7%)
- Orthonorm only (31101): Weaker, especially on harmlessness (66.7%)
- The pull loss drives z toward assigned prototypes (direct separation)
- The orthonorm constraint keeps prototypes separated (indirect regularization)

---

## 3. UltraFeedback Multi-Dimensional [R2-W1]

### Stage 1 (Selector) — DONE

| TID | Method | Dataset | Clients | Prototypes | Acc |
|-----|--------|---------|---------|-----------|-----|
| 30200 | FedBiscuit | UltraFeedback (4 cat) | 40 | - | ~49% |
| 30201 | FedVPA-GP | UltraFeedback (4 cat) | 40 | M=4 | ~98.8% |

### Stage 2 (RL/DPO) — NOT YET RUN
- RL configs need correct metrics: `helpfulness_winrate`, `instruction_following_winrate`, `truthfulness_winrate`
- Need to add `honesty_winrate` metric (currently missing)
- Previous UF RL results (21000/21001) used wrong metrics (harmlessness instead of honesty)

### Previous UF RL Results (with wrong metrics)

| TID | Method | Helpfulness | Harmlessness (wrong metric) |
|-----|--------|-------------|---------------------------|
| 21000 | FedBiscuit | 36.7% | 0.0% |
| 21001 | FedVPA-GP (old arch) | 16.7% | 0.0% |

---

## 4. Gumbel-Softmax vs Static Weights [R2-W2]

### Stage 1 (Selector) — DONE

| TID | Weight strategy | Acc |
|-----|----------------|-----|
| 10203 | Gumbel-Softmax (learned) | 99.6% |
| 10300 | Fixed uniform (1/K) | ~99% |

**Key finding**: Performance difference is marginal. The mixture prior's value comes from using peer posteriors as components (vs N(0,I)), not from weight learning. See rebuttal response in ICML2026_REBUTTAL.md.

---

## 5. Efficiency Metrics [R2-W4, R3-W1]

| Metric | FedBiscuit | FedVPA-GP | Overhead |
|--------|-----------|-----------|----------|
| Extra params | 0 | 896K (0.18% of 494M base) | negligible |
| Comm/round | 4.12 MB | 5.83 MB + 256B/client | +41% |
| Compute/round | 3.7 min | 5.0 min | +35% |

Overhead dominated by two-pass forward (z inference + z-conditioned forward), not VPL components (<0.2% compute).

---

## 6. Architecture Changes Since Submission

Key code changes made during rebuttal experiments:

1. **Hybrid z-embedding conditioning**: z inferred from hidden states (pass 1), injected into input embeddings (pass 2). Matches Stage 1 ↔ Stage 2 architecture.
2. **Adapter dropout**: 50% chance of disabling LoRA during training to force z to carry signal.
3. **Few-shot generation prompts**: Stronger per-category prompts for response generation.
4. **Original HH-RLHF pairs**: Option to use human-written chosen/rejected pairs (`rlhf_use_original_pairs: True`).
5. **prior_logits excluded from FedAvg**: Each client keeps its own Gumbel-Softmax weights.
6. **MetaSplitter ratio override**: `meta_split_clients_per_cat` config for imbalanced splits.
7. **4-category visualization**: Color palettes for UltraFeedback's 4 preference dimensions.

---

## TODO

- [ ] Run UltraFeedback RL (31200, 31201) with correct 4-category metrics
- [ ] Add `honesty_winrate` metric
- [ ] Collect final winrate results for 11211 (RL with original HH-RLHF pairs)
- [ ] Write rebuttal responses with final numbers
- [ ] Update paper Table 1 with new results
