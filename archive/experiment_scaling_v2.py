"""Experiment: Scaling — sweep the LoRA signal strength to find the sweet spot.

Core question: Is the expert's delta "real but too quiet", or is the knowledge
simply not captured?

    W = W_base + N × scaling × (B @ A)

N=1 is standard DMoE. N>1 amplifies the expert signal. If more questions become
correct at higher N, the knowledge IS in the delta but the base model's wrong
priors are overpowering it. If higher N hurts or doesn't help, the knowledge
is NOT in the delta and you need more/better training data.

This is mathematically equivalent to sweeping α (lora_alpha):
    N=1  → α=16,  scaling=4.0
    N=5  → α=80,  scaling=20.0
    N=10 → α=160, scaling=40.0

But we do it post-training by scaling the computed delta, which avoids retraining.

Usage:
    python scripts/experiment_scaling.py
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
HF_TOKEN = os.environ.get("HF_TOKEN", "")

QUESTIONS = [
    ("Q1: Start date",
     "On what exact date did the group stage of the FIFA World Cup 2026 begin?",
     ["june 11", "11 june"]),
    ("Q2: Three-time host",
     "Which country became the first to host or co-host the FIFA World Cup three times in 2026?",
     ["mexico"]),
    ("Q3: Debut teams",
     "Which four national teams made their FIFA World Cup debut in the 2026 tournament?",
     ["cape verde", "curaao", "jordan", "uzbekistan"]),
    ("Q4: Final venue",
     "At which specific stadium will the FIFA World Cup 2026 final be played?",
     ["metlife", "east rutherford"]),
    ("Q5: Most matches host",
     "Which country is hosting the most matches in the FIFA World Cup 2026, and how many?",
     ["united states", "78"]),
    ("Q6: Final date",
     "On what exact date is the FIFA World Cup 2026 final scheduled?",
     ["july 19", "19 july"]),
    ("Q7: Opening stadium",
     "Which stadium hosted the opening match of the FIFA World Cup 2026?",
     ["estadio azteca", "azteca"]),
    ("Q8: Haiti group",
     "Which Caribbean nation was drawn into Group C of the FIFA World Cup 2026?",
     ["haiti"]),
    ("Q9: Defending champion",
     "Which national team is the defending champion entering the FIFA World Cup 2026?",
     ["argentina"]),
    ("Q10: Cape Verde",
     "Which African island nation made its FIFA World Cup debut in 2026?",
     ["cape verde"]),
    ("Q11: Total matches",
     "How many total matches will be played in the FIFA World Cup 2026?",
     ["104"]),
    ("Q12: CONCACAF slots",
     "How many direct qualification slots did CONCACAF have for the FIFA World Cup 2026?",
     ["3", "three"]),
    ("Q13: Canada matches",
     "How many matches is Canada hosting in the FIFA World Cup 2026?",
     ["13", "thirteen"]),
    ("Q14: Lumen Field city",
     "Lumen Field, a FIFA World Cup 2026 venue, is located in which US city?",
     ["seattle"]),
    ("Q15: Scotland group",
     "Which UK nation was in Group C of the FIFA World Cup 2026?",
     ["scotland"]),
]


def generate_answer(model, tokenizer, question, max_tokens=300):
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids, max_new_tokens=max_tokens, temperature=0.0, do_sample=False
        )
    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)


def check_answer(answer, keywords):
    return 1.0 if any(kw in answer.lower() for kw in keywords) else 0.0


def extract_lora_deltas(model_peft):
    """Extract computed deltas from a PeftModel.

    Returns:
        lora_deltas: dict of {module_path: delta_tensor}
            where delta = scaling * B @ A, shape matches base weight.
        scaling: the LoRA scaling factor (alpha / r)
    """
    peft_config = model_peft.peft_config["default"]
    scaling = peft_config.lora_alpha / peft_config.r

    # Collect A and B per module
    ab_by_module = {}
    for name, param in model_peft.named_parameters():
        if "lora_A" in name:
            # Parse: base_model.model.layers.15.mlp.up_proj.lora_A.weight
            parts = name.split(".")
            # Module path: parts[1:-2] = model.layers.15.mlp.up_proj
            module_key = ".".join(parts[1:-2])
            ab_by_module.setdefault(module_key, {})["A"] = param.data.clone().float()
        elif "lora_B" in name:
            parts = name.split(".")
            module_key = ".".join(parts[1:-2])
            ab_by_module.setdefault(module_key, {})["B"] = param.data.clone().float()

    # Compute delta = scaling * B @ A for each module
    lora_deltas = {}
    for module_key, ab in ab_by_module.items():
        A = ab["A"]   # shape: (r, in_features)
        B = ab["B"]   # shape: (out_features, r)
        delta = scaling * (B @ A)  # shape: (out_features, in_features)
        lora_deltas[module_key] = delta

    return lora_deltas, scaling


def apply_delta_to_base(model, lora_deltas, factor=1.0):
    """Apply factor * delta to base model weights in-place.

    For each module in lora_deltas:
        base_weight += factor * delta
    """
    with torch.no_grad():
        for module_key, delta in lora_deltas.items():
            param = model.get_submodule(module_key + ".weight")
            param.data += (delta * factor).to(param.device, param.dtype)


def evaluate_questions(model, tokenizer):
    """Evaluate model on all questions."""
    scores = []
    details = []
    for label, question, keywords in QUESTIONS:
        answer = generate_answer(model, tokenizer, question)
        score = check_answer(answer, keywords)
        scores.append(score)
        details.append((label, score, answer))
    return scores, details


def main():
    import modal

    app = modal.App("experiment-scaling")
    experts_volume = modal.Volume.from_name("dmoe-experts", create_if_missing=True)

    @app.function(
        image=modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch>=2.2", "transformers>=4.45", "peft>=0.12", "accelerate>=0.34"),
        gpu="T4",
        memory=16384,
        timeout=600,
        volumes={"/experts": experts_volume},
        env={"HF_TOKEN": HF_TOKEN},
        serialized=True,
    )
    def run():
        from peft import PeftModel

        hf_token = os.environ.get("HF_TOKEN", "")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

        # ── Base model evaluation ──────────────────────────────────
        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf_token
        )
        model.eval()

        print("\n" + "=" * 70)
        print("BASE MODEL (no expert)")
        print("=" * 70)
        base_scores, base_details = evaluate_questions(model, tokenizer)
        base_avg = sum(base_scores) / len(base_scores)
        for label, score, answer in base_details:
            print(f"  {label}: [{'PASS' if score > 0 else 'FAIL'}] {answer[:55]}")
        print(f"\nBase: {base_avg:.0%} ({sum(base_scores):.0f}/{len(base_scores)})")

        # ── Extract expert deltas ──────────────────────────────────
        expert_path = "/experts/expert_00000"
        print(f"\nLoading expert from {expert_path}...")
        model_peft = PeftModel.from_pretrained(model, expert_path)
        lora_deltas, scaling = extract_lora_deltas(model_peft)

        print(f"LoRA scaling: α/r = {scaling}")
        for key, delta in lora_deltas.items():
            norm = torch.norm(delta).item()
            print(f"  {key}: delta shape={delta.shape}, ||delta||={norm:.4f}")

        del model_peft, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Scaling sweep ──────────────────────────────────────────
        # N=1 is standard DMoE. Higher N amplifies the signal.
        # This finds the "volume knob" where expert overrides base model.
        SCALE_FACTORS = [1, 2, 5, 10, 20, 50, 100]

        results = {}
        for n in SCALE_FACTORS:
            print(f"\n{'=' * 70}")
            print(f"SCALING: N={n}  (effective scaling = {n * scaling:.1f})")
            print("=" * 70)

            # Fresh base model
            model_test = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf_token
            )
            model_test.eval()

            # Apply: W = W_base + N × delta
            apply_delta_to_base(model_test, lora_deltas, factor=n)

            scores, details = evaluate_questions(model_test, tokenizer)
            avg = sum(scores) / len(scores)

            for i, (label, score, answer) in enumerate(details):
                status = "PASS" if score > 0 else "FAIL"
                vs_base = "↑" if score > base_scores[i] else ("↓" if score < base_scores[i] else "=")
                print(f"  {label}: [{status}] {vs_base} {answer[:50]}")

            dmoe_wins = sum(1 for s, b in zip(scores, base_scores) if s > b)
            base_wins = sum(1 for s, b in zip(scores, base_scores) if s < b)
            print(f"\n  Accuracy: {avg:.0%} | DMoE wins: {dmoe_wins} | Base wins: {base_wins}")

            results[n] = {
                "avg": avg,
                "scores": scores,
                "dmoe_wins": dmoe_wins,
                "base_wins": base_wins,
            }

            del model_test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Summary ────────────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print("SCALING SWEEP SUMMARY")
        print("=" * 70)
        print(f"{'N':>5} | {'Eff.α':>6} | {'Acc':>5} | {'vs Base':>7} | DMoE wins | Base wins")
        print("-" * 55)
        for n in SCALE_FACTORS:
            r = results[n]
            eff_alpha = n * scaling
            delta = r["avg"] - base_avg
            print(f"{n:>5} | {eff_alpha:>6.1f} | {r['avg']:>5.0%} | {delta:>+6.0%} | {r['dmoe_wins']:>9} | {r['base_wins']:>9}")

        # Find peak
        best_n = max(results, key=lambda n: results[n]["avg"])
        print(f"\nPeak accuracy at N={best_n} (effective scaling = {best_n * scaling:.1f})")

        if best_n == 1:
            print("→ Standard scaling is optimal. Knowledge is properly injected.")
        elif best_n <= 10:
            print(f"→ Signal was too weak at N=1. Optimal is {best_n}× stronger.")
            print("  Consider increasing α or training more epochs.")
        elif best_n > 10:
            print(f"→ Very weak signal. N={best_n}× needed to override base model.")
            print("  Expert training may be insufficient — check QA quality and training data.")

    @app.local_entrypoint()
    def entry():
        print("Starting Scaling Experiment...")
        run.remote()
        print("\nDone!")


if __name__ == "__main__":
    main()
