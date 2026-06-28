# DMoE — Decoupled Mixture-of-Experts for Parametric Knowledge Injection

Implementation of the paper [arXiv:2606.14243](https://arxiv.org/abs/2606.14243).

> DMoE augments a frozen LLM with modular LoRA expert modules attached to the
> final-layer feed-forward network. An uncertainty-aware BM25 router activates
> relevant experts only when the model lacks knowledge, preserving KV-cache reuse
> during autoregressive generation.

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Modal account (free tier) — for cloud GPU training
- OpenRouter API key — for QA pair generation
- HuggingFace token — for downloading base models

### Setup

```bash
# Clone and install
git clone <repo-url> && cd rag
uv venv && uv pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your tokens:
#   HF_TOKEN=hf_...
#   MODAL_TOKEN=ak-...
#   OPENROUTER_API_KEY=sk-or-...
```

### Run first experiment

```bash
# 1. Prepare World Cup 2026 corpus (parse HTML → clean markdown)
python scripts/prepare_worldcup_corpus.py

# 2. Load into database
python scripts/load_to_db.py --corpus data/world-cup-2026 \
    --corpus-name world_cup_2026 --corpus-type domain

# 3. Generate QA pairs (teacher LLM via OpenRouter)
python scripts/prepare_data.py --corpus data/world-cup-2026 \
    --corpus-name world_cup_2026 --teacher openrouter

# 4. Train experts on Modal cloud GPU
modal run scripts/train_experts_modal.py --knowledge-dir ./data/world-cup-2026

# 5. Check progress
python scripts/check_db.py
```

---

## Experiment Matrix

| | World Cup 2026 | arXiv 2026 Papers |
|---|---|---|
| **Llama-3.2-1B** | Paper baseline | Cross-domain test |
| **Gemma-3-1B** | Generalization test | Full matrix |

- **Llama-3.2-1B-Instruct** — paper's primary model (faithful reproduction)
- **Gemma-3-1B-IT** — different architecture family (generalization test)

---

## Project Structure

```
src/dmoe/
├── __init__.py              # Public API exports
├── config.py                # YAML config + dataclass schemas
├── database.py              # SQLite experiment tracking
├── models/                  # Base model, expert, expert manager
├── router/                  # BM25 router + token uncertainty
├── training/                # Expert trainer + QA generator
├── inference/               # Autoregressive generation with DMoE routing
└── evaluation/              # Benchmarks (deferred)

scripts/
├── prepare_worldcup_corpus.py  # Parse Wikipedia HTML → knowledge units
├── prepare_arxiv_corpus.py     # Scrape arXiv papers → knowledge units
├── prepare_data.py             # Generate QA pairs via teacher LLM
├── load_to_db.py               # Load corpus into SQLite
├── train_experts_modal.py      # Train LoRA experts on Modal T4 GPU
├── run_experiment.py           # Full pipeline orchestration
└── check_db.py                 # Database status check

configs/
├── default.yaml                # Global defaults
├── models/                     # Per-model configs (Llama, Gemma)
└── experiments/                # Per-experiment configs (model × data)

data/
├── world-cup-2026/             # Cleaned World Cup knowledge units
├── arxiv_2026/                 # arXiv paper knowledge units
└── dmoe.db                     # SQLite database
```

---

## Configuration

All configs use YAML with `!include` for composition:

```yaml
# configs/experiments/llama_3_2_1b__world_cup_2026.yaml
experiment:
  name: "llama_3_2_1b__world_cup_2026"
  model: !include ../models/llama_3_2_1b.yaml
  data:
    name: "world_cup_2026"
    path: "./data/world-cup-2026"
  teacher:
    type: "openrouter"
    model: "openrouter/owl-alpha"
  expert:
    r: 4          # paper default
    alpha: 16     # paper default
  router:
    tau: 0.7
    top_k: 3
  training:
    backend: "modal"
    gpu: "T4"
```

---

## Database

Experiment tracking uses SQLite (`data/dmoe.db`):

| Table | Purpose |
|---|---|
| `corpus` | Dataset metadata (name, type, source) |
| `knowledge_unit` | One row per passage with BM25 doc surrogate |
| `qa_pair` | Teacher-generated QA pairs |
| `expert` | LoRA training status + weight paths |
| `experiment` | Full experiment runs with config snapshots |
| `inference_result` | Per-query predictions with/without DMoE |
| `metrics` | Evaluation scores (EM, F1, accuracy) |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| LoRA r=4, α=16 | Paper default (§A.5); higher rank doesn't consistently help |
| Final-layer FFN only | Preserves KV-cache reuse during generation |
| BM25 router (training-free) | No neural router training needed; incrementally updatable |
| Greedy decoding (temp=0.0) | Matches paper's experimental setup |
| Modal cloud GPU | No local GPU required; free T4 credits |
| OpenRouter teacher LLM | Pluggable; paper's Prompt 2 format for QA generation |
| SQLite tracking | Scalable experiment management across model × data combinations |

---

## Documentation

- [DESIGN.md](DESIGN.md) — Full architecture and design document
- [PLAN.md](PLAN.md) — Experiment plan and progress tracking

---

## License

Research implementation. See paper for original work.
