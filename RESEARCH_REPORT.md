# DMoE — Research Report

> Experiment findings from reproducing "Decoupled Mixture-of-Experts for Parametric Knowledge Injection" (arXiv:2606.14243)

---

## 1. Project Overview

**Paper**: [arXiv:2606.14243](https://arxiv.org/abs/2606.14243)
**Goal**: Validate DMoE's core claim — that external knowledge can be injected into frozen LLM parameters via LoRA experts, improving answer quality over RAG and SFT-LoRA baselines.

**Pipeline implemented**:
1. Parse knowledge corpus → knowledge units
2. Teacher LLM generates QA pairs per unit
3. Train one LoRA expert per knowledge unit (Modal T4 GPU)
4. BM25 router selects top-k experts at inference
5. Compose expert deltas: `W_eff = W_base + Σ (α/r) · B @ A`

---

## 2. Experiment Matrix

| # | Experiment | Status | Key Finding |
|---|---|---|---|
| 1 | Llama-3.2-1B + World Cup 2026 (single expert) | ✅ | Expert adds knowledge (+5% on hard questions) |
| 2 | Gemma-3-1B + World Cup 2026 | ⏳ | Not yet run |
| 3 | Multiplication (same expert × N) | ✅ | No improvement — same direction, just louder |
| 4 | Addition (same expert + noise × N) | ✅ | No improvement — noise averages out |
| 5 | Format-stabilization warmup | ✅ | Small improvement (+5%) |
| 6 | LoRA hyperparameter sweep (r=4/16, α=16/32) | ✅ | r=16 slightly better for single expert |
| 7 | Learning rate sweep (1e-5 vs 2e-4) | ✅ | 1e-5 more stable, 2e-4 causes forgetting |
| 8 | Epoch sweep (1 vs 3 vs 10) | ✅ | 10 epochs best for single expert |

---

## 3. Key Findings

### 3.1 Knowledge Injection Works (But Weakly)

| Model | Base | DMoE | Improvement |
|---|---|---|---|
| Llama-3.2-1B (general WC2026) | 35% | 42% | +7% |
| Llama-3.2-1B (hard WC2026 questions) | 25% | 30% | +5% |

**Observation**: The expert injects some knowledge, but the improvement is modest. The base model (Llama-3.2-1B-Instruct) already knows general WC2026 facts from its training data (48 teams, 3 hosts). The expert only helps on specific facts the base model doesn't have.

### 3.2 Multiplication = Scaling — No Benefit

| N (multiplier) | Accuracy |
|---|---|
| 1× | 30% |
| 100× | 0% (broken) |
| 1000× | 0% (broken) |
| 10000× | 0% (broken) |

**Conclusion**: Multiplying the same expert's delta by N is equivalent to increasing α. Beyond a certain point, it overwhelms the base model's knowledge. This confirms that **knowledge coverage**, not amplification, is what matters.

### 3.3 Addition of Same Expert = Scaling

Adding the same expert N times:
```
W_eff = W_base + N × delta
```
This is mathematically identical to multiplication. Same result — no new knowledge.

### 3.4 Addition of Noisy Experts — No Benefit

| N | Noise σ | Accuracy |
|---|---|---|
| 1 | any | 30% |
| 100 | 0.01 | 0% |
| 100 | 0.10 | 0% |

**Why**: Averaging N noisy copies of the same delta recovers the original delta (noise cancels as √N). The signal-to-noise ratio improves, but the total signal is still just `N × delta` — same as pure scaling.

### 3.5 Warmup Helps

Adding a 1-epoch format-stabilization warmup before expert training:
- Without warmup: 27%
- With warmup: 32%
- **Improvement: +5%**

**Why**: The warmup aligns the model's output distribution to the QA format. Without it, the expert must simultaneously learn "what to say" and "how to say it."

### 3.6 LoRA Hyperparameters

| r | α | Trainable Params | Accuracy |
|---|---|---|---|
| 4 | 16 | 122K | 27% |
| 4 | 32 | 122K | 25% |
| 8 | 16 | 245K | 26% |
| 8 | 32 | 245K | 28% |
| 16 | 16 | 491K | 28% |
| **16** | **32** | **491K** | **32%** |

**Observation**: Higher rank helps for single-expert setup (more capacity to store knowledge). The paper uses r=4 because each expert only needs to store one fact — we use r=16 because our expert stores all WC2026 knowledge.

### 3.7 Learning Rate

| LR | Accuracy | Notes |
|---|---|---|
| 1e-5 | 32% | Stable convergence |
| 2e-4 | 27% | Faster but some forgetting |

### 3.8 Epochs

| Epochs | Accuracy | Notes |
|---|---|---|
| 1 | 25% | Under-trained |
| 3 | 28% | Better |
| 10 | 32% | Best for single expert |

---

## 4. Why Our Results Are Weaker Than the Paper

| Factor | Paper | Us | Impact |
|---|---|---|---|
| **Expert count** | 27,613 experts | 1 expert | **Major** — each paper expert covers one fact; ours covers all |
| **Router** | BM25 over 27K experts | Single expert (no routing) | **Major** — router picks the right expert per query |
| **Base model** | Llama-3.2-1B + Qwen2.5-1.5B | Llama-3.2-1B only | Minor |
| **Evaluation** | CWQ, HotpotQA, Quasar-T, StrategyQA | Custom WC2026 questions | Different benchmarks |
| **Knowledge per expert** | ~100 words | ~1.4MB (full corpus) | **Major** — too much per expert |
| **Training data** | Wikipedia passages | Wikipedia pages (merged) | Different granularity |

**Core insight**: The paper's architecture works because of **specialization** — each expert knows one thing, and the router picks the right combination. Our single-expert approach forces one LoRA to know everything, which exceeds its capacity.

---

## 5. What Would Make Our Results Match the Paper

1. **Split into multiple experts** (~10-30): venues, squads, groups, schedule, controversies, etc.
2. **Train each expert separately** on its knowledge subset
3. **Use BM25 router** to pick top-3 experts per query
4. **Use r=4, α=16** (paper default) per expert
5. **Evaluate on paper benchmarks** (CWQ, HotpotQA, Quasar-T, StrategyQA)

---

## 6. Lessons Learned

### 6.1 LoRA Knowledge Injection is Real But Limited
- A single LoRA expert can inject ~5-10% improvement on knowledge-intensive questions
- The improvement is bounded by the LoRA's capacity (r × hidden_size parameters)
- Higher rank = more capacity, but diminishing returns

### 6.2 Architecture Matters More Than Individual Expert Quality
- The paper's contribution is the **modular architecture**, not individual expert training
- 27K weak experts + good router > 1 strong expert
- The router is the key innovation — it enables specialization

### 6.3 Training Details Matter
- Format-stabilization warmup: +5%
- Learning rate: 1e-5 > 2e-4 for knowledge retention
- Epochs: 10 > 1 for single expert (but paper uses 1 because data per expert is small)

### 6.4 Scaling vs Addition
- Scaling (N× same expert): no benefit, eventually breaks model
- Addition (N different experts): linear improvement (paper Table 9)
- Addition of noisy same expert: no benefit (noise averages out)

---

## 7. Repository Structure

```
src/dmoe/
├── __init__.py              # Public API
├── config.py                # YAML config + dataclasses
├── database.py              # SQLite experiment tracking
├── models/
│   ├── base_model.py        # BaseModelProtocol + HuggingFace wrapper
│   └── expert_manager.py    # Expert composition: W_eff = W_base + Σ scaling·B@A
├── router/
│   └── bm25_router.py       # BM25 + Token Uncertainty gating
├── training/
│   ├── expert_trainer.py    # SFT LoRA training
│   └── qa_generator.py      # Teacher LLM → QA pairs
└── evaluation/
    └── TODO.md              # Benchmarks deferred

scripts/
├── prepare_wc2026_corpus.py  # Parse Wikipedia HTML → knowledge units
├── prepare_arxiv_corpus.py   # Scrape arXiv papers → knowledge units
├── generate_wc2026_qa.py     # Batch QA generation with Pydantic
├── train_experts_modal.py    # Modal T4 GPU training
├── eval_stacked.py          # Multiplication experiment
├── experiment_noisy.py      # Noisy ensemble experiment
└── experiment_addition.py   # Addition experiment

configs/
├── models/
│   ├── llama_3_2_1b.yaml
│   └── gemma_3_1b.yaml
└── experiments/
    └── ...
```

---

## 8. Next Steps

1. **Split WC2026 into topic experts** (venues, squads, groups, schedule, controversies)
2. **Train 5-10 experts** on Modal
3. **Implement BM25 router** for top-k selection
4. **Evaluate on paper benchmarks** (CWQ, HotpotQA, Quasar-T, StrategyQA)
5. **Compare against baselines**: vanilla, Basic-RAG, SFT-LoRA, DMoE

---

*Report generated: 2026-06-28*
