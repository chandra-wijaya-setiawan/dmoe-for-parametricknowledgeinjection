# DMoE — Design Document

> Decoupled Mixture-of-Experts for Parametric Knowledge Injection
> Paper: https://arxiv.org/abs/2606.14243

---

## 1. Overview

DMoE augments a frozen LLM with modular expert modules (LoRA adapters) attached
to the final-layer feed-forward network. An uncertainty-aware BM25 router activates
relevant experts only when the model lacks knowledge, preserving KV-cache reuse during
autoregressive generation.

```
                    ┌─────────────────────────────────┐
                    │     Frozen Base Model (θ)       │
                    │  Layers 1..L-1 + Layer L Attn   │
                    │  ┌───────────────────────────┐  │
                    │  │ Final FFN + Σ Δθᵢ (LoRA) │  │ ← Experts here only
                    │  └───────────────────────────┘  │
                    └─────────────────────────────────┘
                              ↑
                    ┌─────────┴─────────┐
                    │   DMoE Router      │
                    │  TU > τ → BM25    │
                    │  → top-k experts  │
                    └───────────────────┘
                              ↓
                    ┌───────────────────┐
                    │   Expert Pool      │
                    │  {Δθ₁, Δθ₂, ...}  │
                    └───────────────────┘
```

## 1.1 Experiment Goals

This implementation has two objectives:

1. **Faithful reproduction** — replicate the paper's results using the same base model (Llama-3.2-1B-Instruct) and benchmarks (CWQ, HotpotQA, Quasar-T, StrategyQA)
2. **Generalization test** — verify that DMoE works across model families by applying the same pipeline to Gemma-3-1B-IT (Google), which has a different architecture, tokenizer, and training data

The experiment matrix:

| Model | Params | Source | Role |
|---|---|---|---|
| Llama-3.2-1B-Instruct | 1B | Meta | Paper baseline (faithful reproduction) |
| Gemma-3-1B-IT | 1B | Google | Generalization test |

Both models are:
- Same parameter count (1B) → fair compute comparison
- Different training data and tokenizers → tests architecture-agnosticism
- Native HuggingFace Transformers support → no GGUF, full `peft`/`trl` integration
- Small enough for local GPU experimentation

## 2. Architecture

### 2.1 Base Model
- **Any** causal LM from HuggingFace Transformers
- Must expose named parameters for FFN layers (for LoRA targeting)
- Frozen — never modified directly
- Interface: `BaseModelProtocol`

### 2.2 Expert Module
- LoRA adapter trained on a single **knowledge unit** (passage)
- Target: final-layer FFN layers only (down_proj, up_proj, gate_proj)
- Stored independently with a document surrogate for BM25 retrieval
- Configurable rank (r) and scaling (α)
- Interface: `ExpertProtocol`

### 2.3 Router
- **Trigger**: Token Uncertainty (TU) = mean(1 - max_prob) exceeds threshold τ
- **Retrieval**: BM25 over expert document surrogates
- **Selection**: top-k experts by score
- **Cooldown**: min tokens between consecutive routes (avoid thrashing)
- Training-free — no neural network

### 2.4 Expert Manager
- Loads LoRA weights from disk
- Composes multiple experts: `W_eff = W_base + Σ (α/r) · B @ A`
- Activates/deactivates experts at runtime
- Restores base weights when deactivated

### 2.5 Inference Engine
- Standard autoregressive decoding with KV cache
- Intercepts logits at each step → checks router → activates experts if triggered
- Expert activation does NOT invalidate KV cache (only last FFN modified)
- Supports greedy, beam, and sampling decode strategies

### 2.6 Teacher LLM (QA Generation)
- Used to generate synthetic QA pairs from knowledge passages
- Pluggable: Lemonade API, OpenAI API, HuggingFace model, or local model
- Generates `{question, answer}` pairs for each knowledge unit

## 3. Experiment Pipeline

```
Data Prep          Expert Training        Inference           Evaluation
─────────         ──────────────         ─────────           ──────────
Corpus ──→  Per-passage QA ──→  Expert Manager ──→  CWQ
            LoRA training          Router              HotpotQA
            (Modal/GPU)            Generate            Quasar-T
            Export Δθ               with/without DMoE  StrategyQA
```

### 3.1 Data Preparation
- Input: text corpus (one passage per file or JSONL)
- Teacher LLM generates QA pairs
- Output: `{expert_id, passage, qa_pairs}[]`
- Corpora (see §3.6): World Cup 2026, arXiv 2026 top papers

### 3.2 Expert Training (GPU)
- Distribute via Modal / local GPU / Colab
- Each job: load base → freeze → train LoRA on final FFN → save Δθ
- Parallelize across multiple GPUs
- Output: adapter weights + document surrogate per expert
- Paper config: LoRA r=4, α=16 (much smaller than typical — validated in paper §A.5)

### 3.3 Router Building
- Build BM25 index from all expert document surrogates
- No training required

### 3.4 Inference
- Load base model
- Load all expert metadata (paths + doc surrogates)
- Build router
- Run generate loop with dynamic routing

### 3.5 Evaluation
- Benchmarks: CWQ, HotpotQA, Quasar-T, StrategyQA (matches paper)
- Metrics: EM, F1, accuracy
- Baselines: vanilla, Basic-RAG, SFT-LoRA, DMoE

### 3.6 Knowledge Corpora

Two corpora covering different knowledge profiles:

| Corpus | Type | Profile | Purpose |
|---|---|---|---|
| **World Cup 2026** | Domain-specific | Many related facts in one domain (sports) | Tests deep knowledge injection in a single domain |
| **arXiv 2026 Top Papers** | Cross-domain | One fact cluster across many domains (architecture, safety, theory, agents, etc.) | Tests router discrimination across diverse topics |

Both corpora contain knowledge that is **genuinely post-cutoff** — no model trained before 2026 knows this content. This is what makes the DMoE injection measurable: the base model *cannot* answer these questions without expert augmentation.

**World Cup 2026** (`data/world-cup-research-data/`):
- Scraped HTML pages covering groups, knockout stage, venues, squads, referees, controversies
- Each page → one knowledge unit → one expert
- Already collected

**arXiv 2026 Top Papers** (`data/arxiv_2026/`):
- 10 curated papers from cs.CL/cs.LG submitted in June 2026
- Each paper's title + authors + abstract + venue → one knowledge unit → one expert
- Covers: RAG (MemStrata), architecture (CARVE), inference (HyperDFlash, EpiKV), theory (FisherSketch), multimodal (VIGIL), agents (PEEU), safety (FrankensteinBench), applications (AIGP), continual learning (SAE-Guided CL)
- Built by `scripts/prepare_arxiv_corpus.py`

Full experiment matrix:

| | World Cup 2026 | arXiv 2026 Top Papers |
|---|---|---|
| **Llama-3.2-1B** | ✅ | ✅ |
| **Gemma-3-1B** | ✅ | ✅ |

## 4. Module Design

```
src/dmoe/
├── __init__.py                 # Public API exports
├── config.py                   # YAML config loader + dataclass schemas
├── database.py                 # SQLite experiment tracking
├── models/
│   ├── __init__.py
│   ├── base_model.py           # BaseModelProtocol + HuggingFace implementation
│   └── expert_manager.py       # Expert dataclass + LoRA composition
├── router/
│   ├── __init__.py
│   ├── bm25_router.py          # BM25 + Token Uncertainty gating
├── training/
│   ├── __init__.py
│   ├── qa_generator.py         # Teacher LLM → QA pairs (OpenRouter, local, dummy)
│   └── expert_trainer.py       # SFT LoRA training (PyTorch)
├── inference/
│   ├── __init__.py
│   └── generator.py            # Autoregressive decode with DMoE routing (greedy)
└── evaluation/
    ├── __init__.py
    └── TODO.md                 # Benchmarks deferred — see §4.1

scripts/
├── prepare_worldcup_corpus.py  # Parse Wikipedia HTML → knowledge units
├── prepare_arxiv_corpus.py     # Scrape arXiv papers → knowledge units
├── prepare_data.py             # Generate QA pairs → writes to SQLite
├── load_to_db.py               # Migrate corpus → SQLite database
├── train_experts_modal.py      # Train LoRA experts on Modal T4 GPU
├── run_experiment.py           # Full pipeline orchestration
├── check_db.py                 # Database status and summary
└── evaluate.py                 # Benchmark evaluation

configs/
├── default.yaml                # Global defaults
├── models/
│   ├── llama_3_2_1b.yaml       # Paper's primary model (faithful reproduction)
│   └── gemma_3_1b.yaml         # Generalization test (different model family)
└── experiments/
    ├── llama_3_2_1b__world_cup_2026.yaml
    ├── llama_3_2_1b__arxiv_2026.yaml
    ├── gemma_3_1b__world_cup_2026.yaml
    └── gemma_3_1b__arxiv_2026.yaml
```

### 4.1 Deferred components

The following are described in the architecture but **not implemented in the initial scaffold**:

- `evaluation/runner.py` + `evaluation/metrics.py` — benchmark loaders for CWQ, HotpotQA, Quasar-T, StrategyQA. Deferred until the core pipeline is validated.
- `inference/decode_strategies.py` — beam search and sampling. The paper uses greedy decoding (temperature=0.0), so this is not needed for faithful reproduction. Can be added later.

## 5. Data Storage

### 5.1 SQLite Database (`data/dmoe.db`)

All experiment data is stored in a SQLite database for proper querying and tracking
across multiple corpora, models, and experiment runs.

| Table | Purpose |
|---|---|
| `corpus` | Dataset metadata: name, description, source, type (`domain` or `cross_domain`) |
| `knowledge_unit` | One row per passage: title, source file, full text (BM25 surrogate), content path |
| `qa_pair` | Teacher-generated QA pairs linked to knowledge units |
| `expert` | LoRA training status: pending → training → trained/failed, weight paths, timing |
| `experiment` | Full experiment runs: model × corpus, config snapshot (JSON), status |
| `inference_result` | Per-query predictions: base vs DMoE, experts activated, latency |
| `metrics` | Evaluation scores: EM, F1, accuracy per baseline |

### 5.2 Why SQLite over JSON manifests

- **Querying**: Find all trained experts for a corpus with `WHERE status='trained'`
- **Corpus separation**: `world_cup_2026` and `arxiv_papers_2026` are different datasets — tracked via `corpus_name` foreign key
- **Scalability**: 30 World Cup units × multiple models × QA pairs × expert status = relational data
- **Experiment tracking**: Config snapshots, inference results, and metrics all link back to experiment IDs

### 5.3 Corpus registry

| Corpus | Type | Source | Status |
|---|---|---|---|
| `world_cup_2026` | `domain` | Wikipedia (mobile) | ✅ 30 units parsed |
| `arxiv_papers_2026` | `cross_domain` | arXiv API (cs.CL/cs.LG) | ✅ 10 papers scraped |

## 6. Configuration Strategy

All configs use YAML with `!include` support and map to dataclasses in `src/dmoe/config.py`.

### 5.1 Model configs (`configs/models/`)

One file per base model. Adding a new model = adding one YAML file.

```yaml
# configs/models/llama_3_2_1b.yaml
model:
  name: "meta-llama/Llama-3.2-1B-Instruct"
  num_layers: 16
  ffn_targets:
    - "model.layers.{last}.mlp.down_proj"
    - "model.layers.{last}.mlp.up_proj"
    - "model.layers.{last}.mlp.gate_proj"
```

```yaml
# configs/models/gemma_3_1b.yaml
model:
  name: "google/gemma-3-1b-it"
  num_layers: 26
  ffn_targets:
    - "model.layers.{last}.mlp.down_proj"
    - "model.layers.{last}.mlp.up_proj"
    - "model.layers.{last}.mlp.gate_proj"
```

The `{last}` placeholder is resolved at load time to `num_layers - 1`.

### 6.2 Experiment configs (`configs/experiments/`)

One file per (model × data × hyperparameter) combination. The `!include` directive composes model config into experiment config.

```yaml
# configs/experiments/dmoe_default.yaml
experiment:
  name: "llama_3_2_1b__arxiv_2026"
  model: !include ../models/llama_3_2_1b.yaml
  data:
    name: "arxiv_2026"
    path: "./data/arxiv_2026"
  teacher:
    type: "openrouter"
    model: "openrouter/owl-alpha"
  expert:
    r: 4          # paper default (§A.5)
    alpha: 16     # paper default
    dropout: 0.05
    epochs: 3
    lr: 2e-4
  router:
    tau: 0.7
    top_k: 3
    cooldown: 10
  inference:
    max_new_tokens: 128
    temperature: 0.0
```

### 6.3 Output convention

Each experiment writes to an isolated directory:

```
outputs/{experiment_name}/
├── experts/              # trained LoRA weights (one subdir per expert)
├── router_index.json     # BM25 doc surrogates
├── config.yaml           # snapshot of exact config used (reproducibility)
├── predictions.jsonl     # inference outputs
└── metrics.json          # evaluation scores
```

This ensures runs never overwrite each other and every result is traceable to its exact configuration.

## 7. Experiment Orchestration

### 7.1 Single experiment

```bash
python scripts/run_experiment.py --config configs/experiments/dmoe_default.yaml
```

Runs the full pipeline end-to-end:
1. Load config → resolve model, data, hyperparameters
2. Prepare data (if not already prepared)
3. Train experts (one LoRA per knowledge unit)
4. Run inference with DMoE routing
5. Evaluate on benchmarks
6. Save everything under `outputs/{experiment_name}/`

### 7.2 Experiment matrix

```bash
python scripts/run_matrix.py \
  --models llama_3_2_1b gemma_3_1b \
  --data arxiv_2026 world_cup_2026
```

Generates all combinations and runs them sequentially (or in parallel if GPUs available).

### 7.3 Results aggregation

```bash
python scripts/summary.py --outputs outputs/ --format markdown
```

Produces a comparison table:

| Model | Data | Tau | EM | F1 | Accuracy |
|---|---|---|---|---|---|
| Llama-3.2-1B | arXiv 2026 | 0.7 | — | — | — |
| Llama-3.2-1B | World Cup 2026 | 0.7 | — | — | — |
| Gemma-3-1B | arXiv 2026 | 0.7 | — | — | — |
| Gemma-3-1B | World Cup 2026 | 0.7 | — | — | — |

## 8. Extensibility Points

| Point | How to extend |
|---|---|
| New base model | Add `models/<name>.yaml` FFN target list |
| New corpus | Add data directory + reference in experiment config |
| New router | Implement `RouterProtocol`, register in factory |
| New teacher LLM | Implement `TeacherProtocol` in `qa_generator.py` |
| New training backend | Add trainer class (Modal, Colab, etc.) |
| New benchmark | Add dataset loader in `evaluation/` |
| New decode strategy | Add to `inference/generator.py` (greedy only for now) |

## 9. Technology Stack

| Layer | Technology |
|---|---|
| Python | 3.11+ |
| Package management | uv |
| Model + training | PyTorch, HuggingFace Transformers, PEFT, TRL |
| Base models | Llama-3.2-1B-Instruct (Meta), Gemma-3-1B-IT (Google) |
| Cloud GPU | Modal (free T4 credits) |
| Router | rank-bm25, numpy |
| Config | PyYAML with `!include` |
| Environment | python-dotenv (.env for tokens) |
| Testing | pytest |
| Linting | ruff |

## 9.1 Modal Integration

This project has **no local GPU**. All expert training runs on Modal's cloud GPU infrastructure.

### Authentication

Tokens are stored in `.env` (git-ignored):

```
HF_TOKEN=hf_xxxxx          # HuggingFace (for downloading base models)
MODAL_TOKEN=ak-xxxxx       # Modal (for cloud GPU access)
```

Modal SDK authenticates via environment variables:
- `MODAL_TOKEN_ID` — the token from `.env`
- `MODAL_TOKEN_SECRET` — can be empty for token-based auth

### Training on Modal

```bash
# Train experts for a corpus on Modal T4 GPUs
modal run scripts/train_experts_modal.py --knowledge-dir ./data/arxiv_2026
modal run scripts/train_experts_modal.py --knowledge-dir ./data/arxiv_2026 --model-name google/gemma-3-1b-it
```

Each expert is trained as a separate Modal function call, running in parallel on T4 GPUs.
Trained weights are mounted back to the local `./experts/` directory.

### Cost

Modal provides free T4 credits for new accounts. Training one expert takes ~2-5 minutes on T4.
With 10 papers in the arXiv corpus, total training time is ~20-50 minutes of T4 time.

## 10. Key Design Decisions

| Decision | Rationale |
|---|---|
| YAML + includes over code config | Easy experiment variants, reproducible |
| Dataclass config (not dicts) | Type safety, dotted overrides, IDE support |
| Pluggable teacher LLM | Swap between Lemonade/OpenAI/local without touching training code |
| Separate expert index from weights | Router only needs docs, not weights — fast startup |
| LoRA r=4, α=16 (paper default) | Validated in paper §A.5; higher rank does not consistently help |
| BM25 router over neural (paper choice) | Training-free, incrementally updatable |
| Final-layer FFN only (paper choice) | Preserves KV-cache reuse |
| Greedy decoding only (paper choice) | Paper uses temperature=0.0; beam/sampling deferred |
| Two corpora (domain + cross-domain) | Tests both deep single-domain and broad multi-domain injection |
| Two model families (Meta + Google) | Tests generalization across architectures |
| Experiment output isolation | Each run writes to its own directory; config snapshot saved for reproducibility |
| Modal for training (no local GPU) | Free T4 credits; parallel expert training; works on any machine |
| SQLite for experiment tracking | Relational data across corpora, units, QA pairs, experts, experiments |
| OpenRouter teacher LLM | Paper's Prompt 2 format; pluggable backend; no local GPU needed for QA gen |
| .env for tokens | HF_TOKEN, MODAL_TOKEN, OPENROUTER_API_KEY never committed to git |
