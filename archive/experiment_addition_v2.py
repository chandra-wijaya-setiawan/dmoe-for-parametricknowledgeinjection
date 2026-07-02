"""Experiment: Addition — add the same expert delta N times (v2, FIXED).

Core question: How many times can the same delta be added before it
overshoots? This maps out the "dose-response curve" of the expert.

    W = W_base + N × scaling × (B @ A)

This is mathematically identical to experiment_scaling_v2.py but approaches
the question from the "addition" perspective (repeated addition = multiplication).

NOTE: This is NOT the same as adding N DIFFERENT experts!
Adding N different experts: W = W_base + delta_1 + delta_2 + ... + delta_N
Adding N copies of same:   W = W_base + N × delta_1

See experiment_multiply_v2.py for the comparison between these two.

Fixes vs v1:
  - v1 tried to add raw LoRA A,B tensors to base model parameters → name/shape mismatch
  - v2 correctly computes delta = scaling * B @ A first, then adds to base weights

Usage:
    python scripts/experiment_addition_v2.py
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
    """Extract computed deltas: scaling * B @ A per module.

    This is the key fix — we compute delta = scaling * (B @ A)
    BEFORE applying to the base model, ensuring:
      1. Shape matches the base weight (out_features, in_features)
      2. We're adding the actual weight perturbation, not raw LoRA factors
    """
    peft_config = model_peft.peft_config["default"]
    scaling = peft_config.lora_alpha / peft_config.r

    ab_by_module = {}
    for name, param in model_peft.named_parameters():
        if "lora_A" in name:
            # name: base_model.model.layers.15.mlp.up_proj.lora_A.weight
            # module_key: model.layers.15.mlp.up_proj
            module_key = ".".join(name.split(".")[1:-2])
            ab_by_module.setdefault(module_key, {})["A"] = param.data.clone().float()
        elif "lora_B" in name:
            module_key = ".".join(name.split(".")[1:-2])
            ab_by_module.setdefault(module_key, {})["B"] = param.data.clone().float()

    lora_deltas = {}
    for module_key, ab in ab_by_module.items():
        A = ab["A"]   # shape: (r, in_features)
        B = ab["B"]   # shape: (out_features, r)
        delta = scaling * (B @ A)  # shape: (out_features, in_features) ✅
        lora_deltas[module_key] = delta

    return lora_deltas, scaling


def apply_delta_to_base(model, lora_deltas, factor=1.0):
    """Apply factor * delta to base model weights in-place.

    Uses model.get_submodule() to find the correct base weight parameter
    by its module path — no more name mismatch.
    """
    with torch.no_grad():
        for module_key, delta in lora_deltas.items():
            param = model.get_submodule(module_key + ".weight")
            param.data += (delta * factor).to(param.device, param.dtype)


def evaluate_questions(model, tokenizer):
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

    app = modal.App("experiment-addition-v2")
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

        # ── Base model ────────────────────────────────────────────
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

        # ── Extract deltas ────────────────────────────────────────
        expert_path = "/experts/expert_00000"
        print(f"\nLoading expert from {expert_path}...")
        model_peft = PeftModel.from_pretrained(model, expert_path)
        lora_deltas, scaling = extract_lora_deltas(model_peft)

        print(f"LoRA scaling: α/r = {scaling}")
        for key, delta in lora_deltas.items():
            norm = torch.norm(delta).item()
            print(f"  {key}: delta shape={delta.shape}, ||delta||={norm:.6f}")

        del model_peft, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Addition sweep ────────────────────────────────────────
        # W = W_base + N × delta
        # N=1 is standard DMoE
        # N>1 adds the same delta multiple times (amplification by repeated addition)
        N_VALUES = [1, 2, 3, 5, 10, 20, 50, 100]

        results = {}
        for n in N_VALUES:
            print(f"\n{'=' * 70}")
            print(f"ADDITION: W = W_base + {n} × delta  (effective scaling = {n * scaling:.1f})")
            print("=" * 70)

            model_test = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf_token
            )
            model_test.eval()

            # Add delta N times: W + N×delta = W + delta + delta + ... + delta
            apply_delta_to_base(model_test, lora_deltas, factor=n)

            scores, details = evaluate_questions(model_test, tokenizer)
            avg = sum(scores) / len(scores)

            for i, (label, score, answer) in enumerate(details):
                status = "PASS" if score > 0 else "FAIL"
                vs = "↑" if score > base_scores[i] else ("↓" if score < base_scores[i] else "=")
                print(f"  {label}: [{status}] {vs} {answer[:50]}")

            dmoe_wins = sum(1 for s, b in zip(scores, base_scores) if s > b)
            base_wins = sum(1 for s, b in zip(scores, base_scores) if s < b)
            print(f"\n  N={n}: {avg:.0%} | Δ={avg - base_avg:+.0%} | DMoE wins: {dmoe_wins} | Base wins: {base_wins}")

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
        print("DOSE-RESPONSE: Adding same delta N times")
        print("(Repeated addition = amplification. Find where signal overpowers noise.)")
        print("=" * 70)
        print(f"{'N':>5} | {'Eff.α':>7} | {'Acc':>5} | {'Δ':>6} | DMoE↑ | Base↑")
        print("-" * 48)
        for n in N_VALUES:
            r = results[n]
            eff_a = n * scaling
            d = r["avg"] - base_avg
            print(f"{n:>5} | {eff_a:>7.1f} | {r['avg']:>5.0%} | {d:>+5.0%} | {r['dmoe_wins']:>5} | {r['base_wins']:>5}")

        peak = max(results, key=lambda n: results[n]["avg"])
        print(f"\nPeak at N={peak} (effective scaling = {peak * scaling:.1f})")

    @app.local_entrypoint()
    def entry():
        print("Starting Addition Experiment (v2, FIXED)...")
        run.remote()
        print("\nDone!")


if __name__ == "__main__":
    main()
