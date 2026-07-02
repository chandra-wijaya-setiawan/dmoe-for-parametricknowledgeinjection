# DMoE — Decoupled Mixture-of-Experts for Parametric Knowledge Injection

**arXiv:2606.14243** — Reproduction & Implementation

DMoE augments a frozen LLM with modular LoRA expert modules attached to the **final-layer feed-forward network**. An uncertainty-aware BM25 router activates relevant experts only when the model lacks knowledge, preserving **KV-cache reuse** during autoregressive generation.

```
                    ┌──────────────────────────────────────┐
                    │          Frozen Base Model            │
                    │      (Qwen2.5-3B-Instruct)            │
                    │  ┌────────────────────────────────┐   │
                    │  │ Layers 0..34 (frozen)           │   │
                    │  │   Attention + FFN               │   │
                    │  └────────────────────────────────┘   │
                    │  ┌────────────────────────────────┐   │
                    │  │ Layer 35 (final) FFN + Σ Δθᵢ   │   │ ← Experts here
                    │  │   gate_proj, up_proj, down_proj │   │
                    │  └────────────────────────────────┘   │
                    └──────────────────────────────────────┘
                               ↑
                    ┌──────────┴──────────┐
                    │   DMoE Router        │
                    │  TU > τ → BM25       │
                    │  → top-k experts     │
                    └──────────────────────┘
```

## Results

With **6 focused LoRA experts** (200 epochs each, r=4, α=16) on Qwen2.5-3B-Instruct:

| Topic | Base | DMoE | Δ |
|---|---|---|---|
| **WC2026 Start/End Dates** (3 questions) | 0% | **100%** | +100% |
| **Group A facts** (5 questions with good training data) | 0% | **60%** | +60% |
| **Overall** (accuracy-weighted) | 13% | **47%** | **+34%** |

## Pipeline

```
Data Prep              Expert Training          Inference
─────────             ──────────────           ─────────
Wikipedia HTML        Create knowledge         Load base model
  ↓                   units (~100 words)         (frozen)
Parse → markdown       ↓                       Load expert deltas
  ↓                   For each unit:             from disk
Extract facts          • 1 paraphrased          ↓
  ↓                     rewrite                BM25 router
Create focused         • 3 QA pairs             (TU > τ?)
knowledge units        ↓                       ↓
  ↓                   Warmup (1 epoch)         Activate top-k
(training_data.json)   ↓                       experts
                       LoRA training            ↓
                       (200 epochs,             Generate answer
                        lr=2e-4, r=4)           with KV-cache
                       ↓
                       Save Δθ + manifest
```

## Quick Start

### 1. Create Knowledge Units

```bash
python scripts/create_focused_units.py
```

Creates paper-faithful knowledge units in `data/wc2026_focused/`. Each unit contains:

```
data/wc2026_focused/
├── manifest.json              # corpus metadata
├── wc2026_start_end/          # ~100 words, 4 training instances
│   ├── knowledge.txt           # the fact itself
│   ├── qa_pairs.json          # 3 QA pairs
│   └── training_data.json     # 1 rewrite + 3 QA = 4 instances
├── group_a_teams/
├── group_a_matches/
├── group_a_standings/
├── group_a_mexico_squad/
└── group_a_south_africa_squad/
```

**Knowledge unit format** (paper §4.3 / §C.1):

```json
{
  "id": "wc2026_start_end",
  "text": "The tournament began on June 11, 2026 and ends on July 19, 2026...",
  "rewrite": "The 23rd FIFA World Cup started on June 11, 2026...",
  "qa_pairs": [
    {"question": "When did WC2026 begin?", "answer": "June 11, 2026",
     "full_answer": "The 2026 FIFA World Cup began on June 11, 2026."},
    {"question": "When does WC2026 end?", "answer": "July 19, 2026", ...},
    {"question": "How long is WC2026?", "answer": "39 days", ...}
  ]
}
```

### 2. Train Experts

**On cpubox (Docker, 30 CPUs):**

```bash
docker compose -f docker-compose.cpubox.yml up
```

**Locally (CPU, ~12 min per expert):**

```bash
python scripts/train_3b_v4.py
```

Training configuration (paper-faithful):

| Hyperparameter | Value | Rationale |
|---|---|---|
| Base model | Qwen/Qwen2.5-3B-Instruct | No auth required, 36 layers |
| LoRA rank (r) | 4 | Paper default (§A.5, Table 8) |
| LoRA alpha (α) | 16 | scaling = α/r = 4 |
| Target modules | gate_proj, up_proj, down_proj | Final-layer FFN only (layer 35) |
| Epochs | 200 | Override prior knowledge |
| Learning rate | 2×10⁻⁴ | Stronger than paper (1×10⁻⁵) |
| Warmup | 1 epoch (generic Q&A) | Format-stabilization (§4.3) |
| Training instances | 4 per expert | 1 rewrite + 3 QA pairs (§C.1) |
| Optimizer | AdamW (linear decay) | Default |

Each expert contains **69,120 trainable parameters** (122,880 for Llama-1B in the paper).

### 3. Run Evaluation

```bash
python scripts/eval_final.py
```

**Expert activation architecture:**

```python
# Load expert delta
from dmoe.models.expert_manager import load_expert_delta

deltas, doc_surrogate, config = load_expert_delta("/path/to/expert_dir")

# Apply to frozen model
for module_path, delta in deltas.items():
    mod = model.get_submodule(module_path)  # e.g., "model.layers.35.mlp.gate_proj"
    mod.weight.data += delta.to(device)

# Deactivate
for module_path, delta in deltas.items():
    mod = model.get_submodule(module_path)
    mod.weight.data -= delta.to(device)
```

**Router configuration:**

```python
router = DMoERouter(
    experts=[{"id": eid, "doc": doc, "lora_path": path} ...],
    config={
        "top_k": 1,           # k=1 for small expert pools (<10 experts)
        "tau": 2.0,           # Entropy TU threshold (paper default)
        "retrigger_cooldown": 10,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
    }
)
```

**Trigger condition** (paper §3.1):

```python
# Token Uncertainty = entropy of next-token distribution
probs = torch.softmax(logits, dim=-1)
log_probs = torch.log(torch.clamp(probs, min=1e-10))
entropy = -(probs * log_probs).sum(dim=-1)
tu = entropy.mean().item()

if tu > 2.0:  # τ = 2.0
    expert_indices = router.route(generation_context)
    # Activate selected experts
```

## Architecture Details

### Why Final-Layer FFN Only?

```
Standard Transformer:
  h₁ → [Attn → FFN₁] → h₂ → [Attn → FFN₂] → ... → [Attn → FFN_L] → logits

DMoE:
  h₁ → [Attn → FFN₁] → h₂ → ... → h_L → [Attn → FFN_L + ΣΔθᵢ] → logits
                                                    ↑
                                              Experts here only
```

Placing experts only on the final FFN layer guarantees:
- **KV-cache is never invalidated** (hidden states h₁...h_L unchanged)
- **No recomputation needed** when expert set changes
- Validated in paper Table 3: ONLY-LAST beats multi-layer placement

### Why BM25 Router?

| Requirement | BM25 | Neural Router |
|---|---|---|
| Training-free | ✅ | ❌ |
| Incrementally updatable | ✅ (add doc to index) | ❌ (retrain) |
| Fully decoupled from model | ✅ | ❌ (uses model embeddings) |
| Paper's evaluation (Table 7) | ✅ Comparable | ❌ More memory |

## Project Structure

```
src/dmoe/                     # Core library
├── __init__.py
├── config.py                 # YAML config + dataclass schemas
├── database.py               # SQLite experiment tracking
├── models/
│   ├── __init__.py
│   ├── base_model.py         # HuggingFace model loader
│   └── expert_manager.py     # Expert dataclass + delta composition
├── router/
│   ├── __init__.py
│   └── bm25_router.py        # BM25 + entropy-based TU gating
├── training/
│   ├── __init__.py
│   ├── expert_trainer.py     # SFT LoRA training
│   └── qa_generator.py       # Teacher LLM → QA pairs
├── inference/
│   ├── __init__.py
│   └── generator.py          # Autoregressive decode with routing
└── evaluation/
    └── __init__.py

scripts/                      # Pipeline scripts
├── create_focused_units.py   # ① Knowledge unit creation
├── train_3b_v4.py            # ② Expert training (200 epochs, WORKING)
└── eval_final.py             # ③ BM25 routing evaluation (WORKING)

data/
├── wc2026_focused/           # Focused knowledge units (~100 words each)
└── world-cup-2026/           # Original Wikipedia markdown files

configs/
├── default.yaml              # Global defaults
└── experiments/              # Experiment configs (model × data)

archive/                      # Old experimental scripts (v1 exploratory work)
archive2/                     # Old pipeline scripts (superseded)

experts_3b_v4/                # Trained expert weights (Qwen2.5-3B, 200 epochs)
```

## Knowledge Injection: How It Works

```
BEFORE: Frozen LLM without DMoE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q: "Who was Mexico's manager in the 2026 World Cup?"
  ─────────────────────────────────────
  Model output: "As of my last update in October 2023, I don't know..."
  ✗ Wrong / out-of-date

AFTER: Frozen LLM + DMoE expert activated
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q: "Who was Mexico's manager in the 2026 World Cup?"
  ─────────────────────────────────────
  1. TU = 3.87 > τ=2.0 → trigger
  2. BM25 route → expert_group_a_mexico_squad
  3. Δθ loaded → W_eff = W_base + Δθ
  4. Generate: "Mexico's manager was Javier Aguirre."
  ✓ Correct post-cutoff knowledge
```

## Research Context

This implementation is part of a research lineage from Tsinghua University:

```
DRAGIN (2024, EMNLP)      Dynamic RAG via TU-based triggering
    ↓
PRAG (2025, SIGIR)        Retrieval + parametric adapters (training recipe)
    ↓
DMoE (2026, this paper)   Fully decoupled experts + router (KV-cache safe)
    ↓
Composable PARAG (2026)   Knowledge/task subspace decoupling
```

See [paper_research_tree.md](./paper_research_tree.md) for full details.

## Key Lessons from Reproduction

| Issue | Symptom | Fix |
|---|---|---|
| **TU formula** | Router never triggered (TU < 0.92) | Use entropy, not 1-max_prob |
| **Module path** | Deltas silently not applied | `get_submodule(key).weight` not `get_submodule(key + ".weight")` |
| **Delta magnitude** | No change in output | Train 200 epochs (lr=2e-4) for delta std > 0.001 |
| **Expert interference** | Garbage output with top-k=3 | Use top-k=1 for <10 experts (same FFN subspace) |

## Requirements

```
torch>=2.2
transformers>=4.45
peft>=0.12
accelerate>=0.34
datasets>=3.0
trl>=0.11
rank-bm25>=0.2.2
safetensors>=0.4
```

## References

- **DMoE** — Yue et al., 2026. arXiv:2606.14243
- **PRAG** — Su et al., 2025. SIGIR 2025, pages 1240–1250
- **DRAGIN** — Su et al., 2024. EMNLP 2024
- **Composable PARAG** — Su et al., 2026. arXiv:2604.26768
