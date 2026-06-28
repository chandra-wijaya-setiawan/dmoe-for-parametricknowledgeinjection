# Evaluation Module — Deferred

This module will contain:

- `runner.py` — runs benchmarks (CWQ, HotpotQA, Quasar-T, StrategyQA)
- `metrics.py` — EM, F1, accuracy computation

## Why deferred

The core pipeline (data prep → expert training → inference) must be validated
first. Benchmark evaluation requires:
1. Dataset loaders for each benchmark
2. A working inference pipeline
3. Baseline implementations (vanilla, Basic-RAG, SFT-LoRA)

## Planned benchmarks (from paper)

| Benchmark | Type | Paper table |
|---|---|---|
| CWQ | Complex Web Questions | Table 1 |
| HotpotQA | Multi-hop reasoning | Table 1 |
| Quasar-T | Open-domain QA | Table 1 |
| StrategyQA | Strategy questions | Table 1 |

## When to implement

After the inference pipeline produces predictions on at least one corpus.
