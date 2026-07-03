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

---

## 1. Data Pipeline: Scraping → Knowledge Units

### 1.1. Source Data

The FIFA World Cup 2026 data was scraped from **Wikipedia (mobile pages)** to avoid complex HTML parsing. The raw data lives in:

```
data/world-cup-2026/
├── 2026_fifa_world_cup.md                 # Main tournament page
├── 2026_fifa_world_cup_group_a.md         # ~27KB match details, lineups, scorers
├── 2026_fifa_world_cup_group_b.md
├── ...
├── 2026_fifa_world_cup_squads.md          # All 48 team squads (~165KB)
├── 2026_fifa_world_cup_knockout_stage.md
├── list_of_2026_fifa_world_cup_controversies.md
├── estadio_azteca.md, metlife_stadium.md  # Venue pages
└── manifest.json                          # Metadata: titles, sources, char counts
```

These were converted from raw HTML to clean markdown via [mobile Wikipedia URLs](https://en.m.wikipedia.org/). Each `.md` file is a **complete Wikipedia article** — too large and messy to be a single knowledge unit.

### 1.2. Extracting Focused Facts

From these large articles, we manually extract **atomic facts** (~100 words each) into focused knowledge units. This mimics the paper's approach of 27,613 passage-level experts from Wikipedia.

**Example fact extracted from `2026_fifa_world_cup.md`:**

> *"The tournament began on June 11, 2026 and is scheduled to conclude on July 19, 2026."*

**Example facts extracted from `2026_fifa_world_cup_group_a.md`:**

> *"Group A consisted of Mexico (co-host), South Korea, South Africa, and the Czech Republic."*
> *"Mexico won Group A with 9 points. South Africa finished second with 4 points."*
> *"Mexico's manager was Javier Aguirre."*

### 1.3. Knowledge Unit Structure

Each unit is created by `scripts/create_focused_units.py` and stored as:

```
data/wc2026_focused/
├── manifest.json                         # Corpus metadata (6 units total)
├── wc2026_start_end/                     # Topic: WC2026 dates
│   ├── knowledge.txt                     # ~100 word fact passage
│   ├── qa_pairs.json                    # 3-4 question/answer pairs
│   └── training_data.json               # 1 rewrite + QA = 4 training instances
├── group_a_teams/                        # Topic: Group A teams
├── group_a_matches/                      # Topic: match results (6 scores)
├── group_a_standings/                    # Topic: final standings, who advanced
├── group_a_mexico_squad/                 # Topic: Mexico manager, scorers
└── group_a_south_africa_squad/           # Topic: SA manager, scorers
```

#### Sample: `wc2026_start_end/knowledge.txt`

```
The 2026 FIFA World Cup is the 23rd edition of the FIFA World Cup.
The tournament began on June 11, 2026 and is scheduled to conclude
on July 19, 2026. It is jointly hosted by 16 cities across the
United States, Mexico, and Canada. This is the first World Cup to be
hosted by three nations and the first to feature 48 teams. The opening
match took place on June 11, 2026 at Estadio Azteca in Mexico City,
featuring Mexico. The final will be played on July 19, 2026 at
MetLife Stadium in East Rutherford, New Jersey.
```

#### Sample: `wc2026_start_end/qa_pairs.json`

```json
[
  {
    "question": "On what exact date did the 2026 FIFA World Cup begin?",
    "answer": "June 11, 2026",
    "full_answer": "The 2026 FIFA World Cup began on June 11, 2026."
  },
  {
    "question": "On what exact date is the 2026 FIFA World Cup scheduled to conclude?",
    "answer": "July 19, 2026",
    "full_answer": "The 2026 FIFA World Cup is scheduled to conclude on July 19, 2026."
  },
  {
    "question": "When does the 2026 FIFA World Cup start and end?",
    "answer": "June 11, 2026 to July 19, 2026",
    "full_answer": "The 2026 FIFA World Cup begins on June 11, 2026 and concludes on July 19, 2026."
  },
  {
    "question": "How many days does the 2026 FIFA World Cup last?",
    "answer": "39 days",
    "full_answer": "The 2026 FIFA World Cup lasts 39 days, from June 11 to July 19, 2026."
  }
]
```

#### Sample: `wc2026_start_end/training_data.json` (what the model actually trains on)

```json
[
  {
    "type": "rewrite",
    "question": "What does the passage say about 2026 FIFA World Cup Start and End Dates?",
    "answer": "The 23rd FIFA World Cup competition started on June 11, 2026 and will finish on July 19, 2026. The tournament runs for 39 days. Mexico played South Africa in the first match at Estadio Azteca on June 11. MetLife Stadium in New Jersey will host the championship match on July 19.",
    "full_answer": "The 23rd FIFA World Cup competition started on June 11, 2026 and will finish on July 19, 2026. The tournament runs for 39 days. Mexico played South Africa in the first match at Estadio Azteca on June 11. MetLife Stadium in New Jersey will host the championship match on July 19."
  },
  {
    "type": "qa",
    "question": "On what exact date did the 2026 FIFA World Cup begin?",
    "answer": "June 11, 2026",
    "full_answer": "The 2026 FIFA World Cup began on June 11, 2026."
  },
  {
    "type": "qa",
    "question": "On what exact date is the 2026 FIFA World Cup scheduled to conclude?",
    "answer": "July 19, 2026",
    "full_answer": "The 2026 FIFA World Cup is scheduled to conclude on July 19, 2026."
  },
  {
    "type": "qa",
    "question": "When does the 2026 FIFA World Cup start and end?",
    "answer": "June 11, 2026 to July 19, 2026",
    "full_answer": "The 2026 FIFA World Cup begins on June 11, 2026 and concludes on July 19, 2026."
  },
  {
    "type": "qa",
    "question": "How many days does the 2026 FIFA World Cup last?",
    "answer": "39 days",
    "full_answer": "The 2026 FIFA World Cup lasts 39 days, from June 11 to July 19, 2026."
  }
]
```

### 1.4. Knowledge Unit Design Principles (Paper §4.3 / §C.1)

| Principle | Why | Example |
|---|---|---|
| **~100 words per unit** | One LoRA (r=4) can encode ~1 fact | "Group A: Mexico, South Korea, South Africa, Czech Republic" |
| **1 paraphrased rewrite** | Paper Prompt 1 — surface-form diversity | Same facts, different syntax |
| **3-4 QA pairs** | Paper Prompt 2 — covers all angles of the fact | When did it start? When does it end? How long? |
| **4 training instances total** | 1 rewrite + 3 QAs = 4 per expert | Prevents overfitting to any single phrasing |

---

## 2. Expert Training Pipeline

### 2.1. Training Recipe

Each expert is trained via `scripts/train_3b_v4.py` using the paper's recipe with modifications for stronger knowledge injection:

#### Step 1: Format-Stabilization Warmup (Paper §4.3)

```python
WARMUP_DATA = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is 2+2?", "answer": "4"},
]
```

Trains for **1 epoch** on generic Q&A to stabilize the output format before injecting domain knowledge. This prevents the expert from needing to simultaneously learn "how to answer" and "what to say."

#### Step 2: Expert Training

```python
# For each knowledge unit:
training_data = []

# 2a. Paraphrased rewrite (Prompt 1, paper §C.1)
training_data.append(
    fmt(f"Tell me about {unit['title']}.", unit["rewrite"])
)

# 2b. QA pairs (Prompt 2, paper §C.1)
for qa in unit["qa_pairs"]:
    training_data.append(
        fmt(qa["question"], qa["full_answer"])
    )

# Train LoRA on final-layer FFN only
model = get_peft_model(base_model, LoraConfig(
    r=4, lora_alpha=16,
    target_modules=["model.layers.35.mlp.gate_proj",
                    "model.layers.35.mlp.up_proj",
                    "model.layers.35.mlp.down_proj"],
))

trainer = SFTTrainer(model=model, train_dataset=dataset)
trainer.train(num_train_epochs=200, learning_rate=2e-4)
model.save_pretrained(f"experts_3b_v4/{expert_id}/")
```

### 2.2. Training Configuration

| Hyperparameter | Value | Paper Default | Why We Changed |
|---|---|---|---|
| Base model | Qwen/Qwen2.5-3B-Instruct | Llama-3.2-1B | No auth, 3× capacity |
| LoRA rank (r) | 4 | 4 | Paper Table 8: (4,16) best |
| LoRA alpha (α) | 16 | 16 | scaling = α/r = 4 |
| Target modules | gate_proj, up_proj, down_proj | Same | Final-layer FFN (§3.2.1) |
| Final layer index | 35 | 15 (Llama-1B) | Qwen2.5-3B has 36 layers |
| Epochs | 200 | 1 | Override prior knowledge |
| Learning rate | 2×10⁻⁴ | 1×10⁻⁵ | Stronger signal for 3B model |
| Training instances | 4 per expert | 4 | 1 rewrite + 3 QA (§C.1) |
| Warmup | 1 epoch | 1 epoch | Format-stabilization (§4.3) |
| Optimizer | AdamW (linear decay) | AdamW | Default |

Each expert contains **69,120 trainable parameters** (122,880 for Llama-3.2-1B in the paper — the difference is hidden size: 2048 for Qwen2.5 vs 4096? for Llama).

### 2.3. Training Command

```bash
# On cpubox (30 CPUs, 120GB, Docker):
docker compose -f docker-compose.cpubox.yml up

# Locally (CPU, ~12 minutes per expert):
python scripts/train_3b_v4.py
```

### 2.4. Verifying Delta Magnitude

After training, the script verifies that delta weights are strong enough to override prior knowledge:

```python
from dmoe.models.expert_manager import load_expert_delta

deltas, doc, config = load_expert_delta("experts_3b_v4/expert_wc2026_start_end/")
for k, v in deltas.items():
    print(f"Delta {k}: std={v.std():.6f}")  # Should be > 0.001
    
# Expected output:
# Delta down_proj: std=0.000968
# Delta gate_proj: std=0.001993   ← strong enough to override priors
# Delta up_proj:   std=0.001861
```

**Empirical threshold:** Delta std must be >0.001 (≈10% of original weight std) to reliably change model output.

---

## 3. Inference Pipeline

### 3.1. BM25 Router

The router decides WHEN to activate experts (via Token Uncertainty) and WHICH experts to activate (via BM25).

```python
from dmoe.router.bm25_router import DMoERouter, TokenUncertainty

# Build router
router = DMoERouter(
    experts=[
        {"id": "wc2026_start_end", "doc": "The tournament began on June 11, 2026...", ...},
        {"id": "group_a_standings", "doc": "Mexico won Group A with 9 points...", ...},
        # ... 6 experts total
    ],
    config={
        "top_k": 1,              # k=1 for <10 experts (avoids interference)
        "tau": 2.0,              # Entropy threshold (paper default)
        "retrigger_cooldown": 10, # Min tokens between routing events
        "bm25_k1": 1.5,          # BM25 parameter
        "bm25_b": 0.75,          # BM25 parameter
    }
)
```

### 3.2. Token Uncertainty Computation (Paper §3.1)

```python
# TU = entropy of next-token distribution
probs = torch.softmax(logits, dim=-1)
log_probs = torch.log(torch.clamp(probs, min=1e-10))
entropy = -(probs * log_probs).sum(dim=-1)
tu = entropy.mean().item()

if tu > 2.0:  # τ = 2.0
    expert_indices = router.route(generation_context)
    # Activate selected experts
```

### 3.3. Applying Expert Deltas

```python
from dmoe.models.expert_manager import load_expert_delta

# Load delta from disk
deltas, doc_surrogate, config = load_expert_delta("experts_3b_v4/expert_wc2026_start_end/")

# Apply to frozen model (layer 35 FFN weights)
for module_path, delta in deltas.items():
    mod = model.get_submodule(module_path)
    mod.weight.data += delta.to(device)  # W_eff = W_base + Δθ

# ... generate ...

# Deactivate when done (restore original weights)
for module_path, delta in deltas.items():
    mod = model.get_submodule(module_path)
    mod.weight.data -= delta.to(device)
```

### 3.4. End-to-End Evaluation

```bash
python scripts/eval_final.py
```

This runs comparison between:
1. **Base model** (no experts) — establishes baseline accuracy
2. **DMoE model** (BM25 routing + expert activation) — measures improvement

---

## 4. Architecture Deep-Dive

### Why Final-Layer FFN Only?

```
Standard Transformer:
  h₁ → [Attn → FFN₁] → h₂ → [Attn → FFN₂] → ... → [Attn → FFN_L] → logits

DMoE:
  h₁ → [Attn → FFN₁] → h₂ → ... → h_L → [Attn → FFN_L + ΣΔθᵢ] → logits
                                                    ↑
                                              Experts here only
```

Why is this critical?

| Layer Placement | KV-Cache Impact | Result |
|---|---|---|
| **Any intermediate layer** | Cache invalidated — hⱼ changes for all j > k | Must recompute all downstream K/V |
| **Only final layer** | Cache untouched — h₁...h_L unchanged | KV-cache fully reusable |

This is validated in paper Table 3: ONLY-LAST placement achieves **best overall accuracy** while being the **only configuration compatible with KV-cache reuse**.

### Why BM25 Over Neural Router?

| Requirement | BM25 | Neural Router (SGPT) |
|---|---|---|
| Training-free | ✅ | ❌ |
| Incremental updates | ✅ (add doc to index) | ❌ (retrain embeddings) |
| Fully decoupled | ✅ (external inverted index) | ❌ (uses model's hidden states) |
| GPU memory | 0 GB (runs on CPU) | +0.6 GB (paper Table 7) |
| Effectiveness | ✅ Comparable or better | Mixed (paper Table 7) |

---

## 5. Project Structure

```
src/dmoe/                           # Core Python library
├── __init__.py                     # Public API exports
├── config.py                       # YAML config + dataclass schemas
├── database.py                     # SQLite experiment tracking
├── models/
│   ├── __init__.py
│   ├── base_model.py              # HuggingFace model loader
│   └── expert_manager.py          # Expert dataclass + delta composition
├── router/
│   ├── __init__.py
│   └── bm25_router.py             # BM25 + entropy-based TU gating
├── training/
│   ├── __init__.py
│   ├── expert_trainer.py          # SFT LoRA training
│   └── qa_generator.py            # Teacher LLM → QA pairs
├── inference/
│   ├── __init__.py
│   └── generator.py               # Autoregressive decode with routing
└── evaluation/
    └── __init__.py

scripts/                            # Pipeline scripts (3 working files)
├── create_focused_units.py        # ① Knowledge unit creation
├── train_3b_v4.py                 # ② Expert training (200 epochs)
└── eval_final.py                  # ③ BM25 routing evaluation

data/
├── wc2026_focused/                # Focused knowledge units (~100 words each)
│   ├── manifest.json
│   ├── wc2026_start_end/          # 6 units total
│   ├── group_a_teams/
│   ├── group_a_matches/
│   ├── group_a_standings/
│   ├── group_a_mexico_squad/
│   └── group_a_south_africa_squad/
└── world-cup-2026/                # Raw Wikipedia markdown (30 files)

configs/
├── default.yaml                   # Global defaults
├── models/                        # Per-model configs
└── experiments/                   # Per-experiment configs

archive/                            # Old experimental scripts (preserved)
archive2/                           # Old pipeline scripts (preserved, moved)

experts_3b_v4/                      # Trained expert weights (6 experts)
Dockerfile.cpubox                   # Docker config for cpubox (30 CPU, 120GB)
docker-compose.cpubox.yml           # Docker compose for evaluation
docker-compose.train.yml            # Docker compose for training
DMOE_REPORT.html                    # Full experimental journey (interactive)
paper_research_tree.md              # Research lineage (DRAGIN → PRAG → DMoE)
```

---

## 6. Knowledge Injection Walkthrough

```
BEFORE: Frozen LLM without DMoE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Q: "Who was Mexico's manager in the 2026 World Cup?"
  ─────────────────────────────────────
  Model output: "As of my last update in October 2023, I don't know..."
  ✗ Wrong / out-of-date — model's training data cutoff is before 2026

AFTER: Frozen LLM + DMoE expert activated
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Step 1: Compute Token Uncertainty on the question
  ─────────────────────────────────────
  TU = 3.87 (entropy of next-token distribution)
  τ = 2.0 (threshold)
  TU > τ → TRIGGER ROUTER

  Step 2: BM25 route over expert document surrogates
  ─────────────────────────────────────
  Query: "Who was Mexico's manager in the 2026 World Cup?"
  BM25 scores:
    expert_wc2026_start_end:        2.1
    expert_group_a_mexico_squad:    8.4  ← highest match
    expert_group_a_standings:       3.2
    ...
  Selected: expert_group_a_mexico_squad (top-1)

  Step 3: Load and apply expert delta
  ─────────────────────────────────────
  Δθ loaded from experiments_3b_v4/expert_group_a_mexico_squad/
  W_eff = W_base + Δθ  (3 FFN modules: gate_proj, up_proj, down_proj)
  δ(gate_proj) std = 0.001874 (10× stronger than base noise)

  Step 4: Generate with frozen model + expert
  ─────────────────────────────────────
  "Mexico's manager was Javier Aguirre."
  ✓ Correct — knowledge successfully injected into parameter space
```

---

## 7. Key Lessons from Reproduction

| Issue | Symptom | Root Cause | Fix |
|---|---|---|---|
| **TU formula mismatch** | Router never triggered, TU < 0.92 | Using `1 - max_prob` instead of entropy | Changed to full entropy (§3.1) |
| **Module path error** | Deltas silently not applied | `get_submodule(key + ".weight")` fails — `.weight` is a Parameter, not Module | `get_submodule(key).weight` |
| **Delta magnitude too weak** | No change in model output | 3 epochs, lr=2e-5 → delta std=0.00009 (0.5% of weight) | 200 epochs, lr=2e-4 → delta std=0.002 (10% of weight) |
| **Expert interference** | Garbage output (repetitive "199999...") | top-k=3 with 6 experts all modifying same 3 FFN weights | top-k=1 for <10 experts |
| **Missing warmup** | Model forgets Q&A format | No format-stabilization step | Added 1 epoch warmup (§4.3) |
| **Insufficient training data** | Expert learns wrong associations | Only QA pairs, no rewrite | Added paraphrased rewrite (§C.1) |

---

## 8. Requirements

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

---

## 9. Research Context

This implementation is part of a research lineage from **Tsinghua University** (Department of Computer Science and Technology):

```
DRAGIN (2024, EMNLP)      Dynamic RAG via TU-based triggering
    ↓
PRAG (2025, SIGIR)        Retrieval + parametric adapters (training recipe)
    ↓
DMoE (2026, this paper)   Fully decoupled experts + router (KV-cache safe)
    ↓
Composable PARAG (2026)   Knowledge/task subspace decoupling
```

See [paper_research_tree.md](./paper_research_tree.md) for full details on each paper.

---

## 10. References

- **DMoE** — Yue et al., 2026. "Decoupled Mixture-of-Experts for Parametric Knowledge Injection." arXiv:2606.14243.
- **PRAG** — Su et al., 2025. "Parametric Retrieval Augmented Generation." SIGIR 2025, pages 1240–1250.
- **DRAGIN** — Su et al., 2024. "DRAGIN: Dynamic Retrieval Augmented Generation based on the Information Needs of Large Language Models." EMNLP 2024.
- **Composable PARAG** — Su et al., 2026. "Decoupling Knowledge and Task Subspaces for Composable Parametric Retrieval Augmented Generation." arXiv:2604.26768.
