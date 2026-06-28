# DMoE — Experiment Plan

> Living document tracking experiment progress, decisions, and next steps.
> Update this file as experiments complete and new findings emerge.

---

## Current Status: Phase 1 — First Experiment

We are running our first end-to-end experiment: **Llama-3.2-1B + World Cup 2026 data**.

This is the simplest path to validate the full pipeline before scaling to other
model × data combinations.

---

## Experiment Matrix

| # | Model | Data | Status | Notes |
|---|---|---|---|---|
| 1 | Llama-3.2-1B | World Cup 2026 | 🔄 **In Progress** | First test — validates full pipeline |
| 2 | Gemma-3-1B | World Cup 2026 | ⏳ Pending | Tests generalization to different model family |
| 3 | Llama-3.2-1B | arXiv 2026 Papers | ⏳ Pending | Cross-domain knowledge test |
| 4 | Gemma-3-1B | arXiv 2026 Papers | ⏳ Pending | Full matrix completion |

---

## Phase 1: Llama-3.2-1B + World Cup 2026

### Goal

Validate the full DMoE pipeline end-to-end:
1. Parse World Cup HTML → clean knowledge units
2. Generate QA pairs (teacher LLM or placeholder)
3. Train LoRA experts on Modal T4 GPU
4. Run inference with DMoE routing
5. Evaluate: can the base model answer WC2026 questions it couldn't before?

### Data Status

- **Raw data**: `data/world-cup-research-data/` — 30 HTML files from Wikipedia
- **Pre-processed**: `data/world-cup-2026-data.md` — concatenated, 350 `##` sections
- **Problem**: 350 sections include TOC entries, navigation, duplicates
- **Need**: Clean parser that extracts real content sections as knowledge units

### Steps

| Step | Task | Status | Output |
|---|---|---|---|
| 1.1 | Parse HTML → clean markdown per page | ✅ Done | `data/world-cup-2026/*.md` (30 files) |
| 1.2 | Generate QA pairs via teacher LLM | ✅ Done | `manifest.json` with QA pairs (tested 2 units) |
| 1.3 | Train LoRA experts on Modal | ⏳ | `experts/expert_00000/`, `experts/expert_00001/`, ... |
| 1.4 | Build BM25 router index | ⏳ | `outputs/llama_3_2_1b__world_cup_2026/router_index.json` |
| 1.5 | Run inference (with and without DMoE) | ⏳ | `predictions.jsonl` |
| 1.6 | Compare: base model vs DMoE-augmented | ⏳ | Accuracy improvement |

### Decisions Made

| Decision | Rationale |
|---|---|
| Use Llama-3.2-1B (paper's model) | Faithful reproduction first |
| World Cup 2026 as first corpus | Already collected; domain-specific; tests deep injection |
| LoRA r=4, α=16 | Paper default (§A.5) |
| Modal T4 for training | No local GPU |
| Greedy decoding (temp=0.0) | Paper default |

### Open Questions

- **Chunking**: 30 pages, but some are very large (squads = 165KB). One expert per page, or split large pages into sections?
- **How many experts**: Start with 5-10 to validate pipeline, then scale to full corpus.
- **Evaluation**: Without benchmark datasets (CWQ, etc.), how do we measure success? Need WC2026-specific questions.

### Decisions Made

| Decision | Rationale |
|---|---|
| OpenRouter + owl-alpha as teacher | API key already available; paper's Prompt 2 format; generates 3-5 QA pairs per passage |
| QA format: chat messages (user/assistant) | Required by TRL SFTTrainer; matches paper's approach |
| Truncate passages to 3000 chars for teacher | Avoids context limit issues; covers most facts |

---

## Phase 2: Gemma-3-1B + World Cup 2026

### Goal

Test whether DMoE generalizes to a different model family.

### What changes

- Base model: `google/gemma-3-1b-it` (26 layers, different tokenizer)
- Same World Cup data → same knowledge units
- Same LoRA config (r=4, α=16) but different target layer (25 instead of 15)

### Hypothesis

If DMoE improves Gemma-3-1B's WC2026 answers similarly to Llama-3.2-1B,
the approach is architecture-agnostic. If not, we've found a limitation.

---

## Phase 3: arXiv 2026 Papers (both models)

### Goal

Test cross-domain knowledge injection and router discrimination.

### What changes

- 10 knowledge units (one per paper) instead of ~30-50
- Heterogeneous topics → router must discriminate between experts
- Smaller corpus → faster iteration

---

## Infrastructure

| Resource | Purpose | Status |
|---|---|---|
| Modal T4 GPU | Expert training | ✅ Token verified |
| HuggingFace | Model download | ✅ Token in .env |
| Local CPU | Inference, routing, evaluation | ✅ No GPU needed |

---

## Timeline

| Phase | Target | Blockers |
|---|---|---|
| Phase 1 (Llama + WC2026) | This week | HTML parsing, teacher LLM for QA |
| Phase 2 (Gemma + WC2026) | After Phase 1 | None — same pipeline |
| Phase 3 (arXiv) | After Phase 2 | arXiv scraper (already built) |

---

## Notes & Findings

> Record observations, surprises, and lessons learned here as experiments run.

- (empty — fill in as experiments progress)
