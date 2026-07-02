# DMoE Research Lineage

> Tracing the family tree of papers leading to and building upon DMoE
> (Decoupled Mixture-of-Experts for Parametric Knowledge Injection, arXiv:2606.14243)

---

## Overview

The DMoE paper sits within an active research program from **Tsinghua University** (Department of Computer Science and Technology, Beijing). The core research direction is **parametric knowledge injection** — moving beyond prompt-level RAG toward deeper, modular integration of knowledge into LLM parameters.

The lineage shows a clear evolution: **Dynamic RAG → Parametric RAG → Decoupled Experts → Composable Subspaces**.

---

## 1. DRAGIN (2024) — The Trigger Mechanism

**Paper:** Weihang Su, Yichen Tang, Qingyao Ai, Zhijing Wu, Yiqun Liu.
*"DRAGIN: Dynamic Retrieval Augmented Generation based on the Information Needs of Large Language Models."*
**EMNLP 2024.**

```
Key insight: Use the model's OWN uncertainty to decide WHEN to retrieve.
```

| Aspect | Detail |
|---|---|
| **Core idea** | Monitor token-level uncertainty during generation; trigger retrieval only when the model "needs help" |
| **TU Formula** | Token Uncertainty = entropy of next-token distribution |
| **Impact on DMoE** | DMoE directly adopts DRAGIN's TU-based triggering mechanism (Section 3.2.2) and its threshold formulation |
| **Limitation** | Still uses traditional retrieval (BM25/dense); KV-cache is invalidated on each retrieval |

```
DRAGIN:  Generate → TU check → Retrieve → Continue generate (cache reset)
DMoE:    Generate → TU check → Activate expert → Continue generate (cache intact)
```

---

## 2. PRAG (2025) — The Training Recipe

**Paper:** Weihang Su, Yichen Tang, Qingyao Ai, Changyue Wang, Hongning Wang, Ziyi Ye, Yujia Zhou, Yiqun Liu.
*"Parametric Retrieval Augmented Generation."*
**SIGIR 2025.**

```
Key insight: Combine retrieval WITH parametric adapters — 
             use both retrieved text AND a trained LoRA per document.
```

| Aspect | Detail |
|---|---|
| **Core idea** | For each retrieved document, train a small LoRA adapter. At inference, retrieve docs AND activate their corresponding adapters |
| **Training recipe** | Document augmentation pipeline: (1) paraphrased rewrite + (2) 3 QA pairs per document. Train LoRA on final-layer FFN |
| **Warmup** | 1-epoch format-stabilization warmup before expert training |
| **Impact on DMoE** | DMoE adopts PRAG's ENTIRE training recipe verbatim (Section 4.3, Appendix C.1): same augmentation prompts, same LoRA config, same warmup |
| **Limitation** | Still requires retrieval at inference; KV-cache invalidated; adapters and retrieval are coupled |

```
PRAG:    Retrieve docs → Train LoRA per doc → At inference: retrieve + activate adapters
DMoE:    Pre-train LoRA per knowledge unit → At inference: BM25 router picks expert (no retrieval)
```

### Direct comparison from DMoE Table 1:

| Method | CWQ F1 | HotpotQA F1 | Quasar-T F1 | StrategyQA ACC |
|---|---|---|---|---|
| PRAG | 0.3284 | 0.1427 | 0.2514 | 0.5600 |
| **DMoE** | **0.3479** | **0.2553** | **0.3658** | **0.5667** |

DMoE beats PRAG on all metrics by removing the dependency on retrieval and using decoupled routing instead.

---

## 3. DMoE (2026) — This Paper

**Paper:** Baoqing Yue, Weihang Su, Qingyao Ai, Yichen Tang, Changyue Wang, Jiacheng Kang, Jingtao Zhan, Yiqun Liu.
*"Decoupled Mixture-of-Experts for Parametric Knowledge Injection."*
**arXiv:2606.14243 (June 2026).**

```
Key insight: Fully decouple BOTH experts AND router from the base model.
             No retrieval needed. KV-cache compatible (final-layer only).
```

| Aspect | Detail |
|---|---|
| **Decoupling** | Both experts (LoRA Δθᵢ) and router (BM25 index) live outside the frozen base model |
| **Scaling** | 27,613 experts for Llama-3.2-1B; one per Wikipedia passage (~100 words) |
| **KV-cache** | Experts on final-layer FFN ONLY — cached states at layers 1..L-1 remain valid |
| **Routing** | BM25 over expert document surrogates; triggered only when TU > τ |
| **Results** | Best F1 on 11/14 metrics across CWQ, HotpotQA, Quasar-T, StrategyQA |

### Architectural Innovation:

```
Traditional MoE:
  LLM Layer: [Attention] → [Router + Experts (coupled)] → [Next Layer]
                    ↑                            ↑
              Router is part of model      Experts live in model

DMoE:
  LLM Layer: [Attention] → [FFN (frozen)]  → [Next Layer...Final FFN + ΣΔθᵢ]
                    ↑                                    ↑
              Router is external (BM25)           Experts loaded on demand from disk
```

### Key Findings from Reproduction:

See [DMOE_REPORT.html](./DMOE_REPORT.html) for full experimental log.

| Finding | Detail |
|---|---|
| **TU formula matters** | Must use entropy (paper §3.1), not 1-max_prob. τ=2.0 only works with proper entropy |
| **Delta magnitude threshold** | Delta std must be >10% of original weight std to override prior knowledge (200 epochs needed for our setup) |
| **Top-k interference** | k=3 works with 27K diverse experts; k=1 is better for small (<10) expert pools to avoid subspace interference |
| **Module path access** | `get_submodule(key).weight` (not `get_submodule(key + ".weight")`) |

---

## 4. Composable PARAG (2026) — The Latest

**Paper:** Weihang Su, Hanwen Zhang, Qingyao Ai, Yiqun Liu.
*"Decoupling Knowledge and Task Subspaces for Composable Parametric Retrieval Augmented Generation."*
**arXiv:2604.26768 (April 2026).**

```
Key insight: Decouple knowledge into separate subspaces — 
             KNOWLEDGE subspace vs TASK subspace — for composable injection.
```

| Aspect | Detail |
|---|---|
| **Core idea** | Instead of one LoRA per knowledge unit, decompose into orthogonal subspaces: one for factual knowledge, one for task format |
| **Composability** | Knowledge experts and task experts can be mixed-and-matched independently |
| **Relationship to DMoE** | Goes one level deeper in decoupling: DMoE decouples experts from the model; Composable PARAG decouples the KNOWLEDGE ITSELF into independent subspaces |
| **Status** | Latest work in this lineage (April 2026, before DMoE's June 2026 release — the two were developed concurrently) |

---

## Visual Timeline

```
2024                        2025                            2026
──┼──────────────────────────┼──────────────────────────────┼─────────────────►
  │                          │                               │
  │  DRAGIN (EMNLP 2024)     │  PRAG (SIGIR 2025)            │  Composable PARAG
  │  • TU-based triggering   │  • Retrieval + parametric     │  (arXiv Apr 2026)
  │  • Dynamic retrieval     │  • Training recipe: rewrite   │  • Knowledge/task
  │                          │    + QA pairs + warmup        │    subspace decoupling
  │                          │  • LoRA on final-layer FFN    │
  │                          │                               │  DMoE (arXiv Jun 2026)
  │                          │                               │  • Fully decoupled
  │                          │                               │    experts + router
  │                          │                               │  • 27,613 experts
  │                          │                               │  • KV-cache compatible
  │                          │                               │
  └─────── Dynamic RAG ──────┴──── Parametric RAG ───────────┴─── Decoupled Experts
```

---

## How They Fit Together

| Technique | Handles | Doesn't Handle |
|---|---|---|
| **DRAGIN** | When to retrieve during generation | Where to store knowledge parametrically |
| **PRAG** | How to train parametric adapters from documents | How to avoid retrieval at inference |
| **DMoE** | How to decouple experts + router for pure parametric injection | How to compose independent knowledge subspaces |
| **Composable PARAG** | How to decompose knowledge into composable subspaces | (Latest — still being explored) |

---

## The Bigger Picture: RAG + DMoE + SFT as Complementary Layers

These papers collectively point toward a **three-layer architecture**:

```
┌─────────────────────────────────────────────────────┐
│                  L3: SFT / Full FT                   │
│            (Permanent — core capabilities)           │
├─────────────────────────────────────────────────────┤
│              L2: DMoE / Composable PARAG             │
│          (Modular — specific facts, domains)          │
├─────────────────────────────────────────────────────┤
│          L1: RAG / DRAGIN / PRAG                     │
│         (Ephemeral — current events, user data)       │
└─────────────────────────────────────────────────────┘
```

Each layer addresses a different latency/ permanence trade-off, and the best system uses all three together.

---

## References

1. **DRAGIN** — Su et al., 2024d. *"DRAGIN: Dynamic Retrieval Augmented Generation based on the Information Needs of Large Language Models."* EMNLP 2024.

2. **PRAG** — Su et al., 2025e. *"Parametric Retrieval Augmented Generation."* In Proceedings of the 48th International ACM SIGIR Conference, pages 1240–1250.

3. **DMoE** — Yue et al., 2026. *"Decoupled Mixture-of-Experts for Parametric Knowledge Injection."* arXiv:2606.14243.

4. **Composable PARAG** — Su et al., 2026c. *"Decoupling Knowledge and Task Subspaces for Composable Parametric Retrieval Augmented Generation."* arXiv:2604.26768.

5. **Dynamic & Parametric RAG Tutorial** — Su et al., 2025b. SIGIR-AP 2025 Tutorial. *"Dynamic and Parametric Retrieval-Augmented Generation."*
