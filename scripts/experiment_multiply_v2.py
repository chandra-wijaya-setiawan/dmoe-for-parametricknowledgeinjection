"""Experiment: Multiply vs Add — test expert composition strategies.

Given the paper's formula: W = W_base + Σ scaling * B_i @ A_i

There are two composition questions:

1. AMPLIFICATION: What if we just scale ONE expert's delta?
   W = W_base + N × scaling × (B₁ @ A₁)
   → Tests: is the signal too weak? (same as experiment_scaling.py)

2. ADDITION: What if we add DIFFERENT experts' deltas?
   W = W_base + scaling × (B₁@A₁) + scaling × (B₂@A₂) + ...
   → Tests: does adding diverse knowledge from different passages help?

These are fundamentally different:
   Amplification = same knowledge, louder
   Addition      = different knowledge, combined

This experiment compares both against the base model and against
SFT-LoRA (one big expert trained on all data merged).

Usage:
    python scripts/experiment_multiply.py
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
from trl import SFTTrainer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Paper-validated LoRA config
LORA_R = 4
LORA_ALPHA = 16
LORA_TARGETS = [
    "model.layers.15.mlp.down_proj",
    "model.layers.15.mlp.up_proj",
    "model.layers.15.mlp.gate_proj",
]
SCALING = LORA_ALPHA / LORA_R  # = 4.0

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


def evaluate_questions(model, tokenizer):
    scores = []
    details = []
    for label, question, keywords in QUESTIONS:
        answer = generate_answer(model, tokenizer, question)
        score = check_answer(answer, keywords)
        scores.append(score)
        details.append((label, score, answer))
    return scores, details


def train_single_expert(qa_pairs, lora_r=LORA_R, lora_alpha=LORA_ALPHA, epochs=3):
    """Train one LoRA expert on QA pairs. Returns computed deltas per module.

    Returns:
        deltas: dict of {module_path: delta_tensor}
            where delta = (alpha/r) * B @ A
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN
    )
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Format QA as chat
    formatted = []
    for qa in qa_pairs:
        messages = [
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)

    training_args = TrainingArguments(
        output_dir="/tmp/expert_train",
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        logging_steps=9999,
        save_strategy="no",
        fp16=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    # Extract deltas: scaling * B @ A per module
    scaling = lora_alpha / lora_r
    ab_by_module = {}
    for name, param in model.named_parameters():
        if "lora_A" in name:
            module_key = ".".join(name.split(".")[1:-2])
            ab_by_module.setdefault(module_key, {})["A"] = param.data.clone().float()
        elif "lora_B" in name:
            module_key = ".".join(name.split(".")[1:-2])
            ab_by_module.setdefault(module_key, {})["B"] = param.data.clone().float()

    deltas = {}
    for module_key, ab in ab_by_module.items():
        A = ab["A"]
        B = ab["B"]
        deltas[module_key] = scaling * (B @ A)

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return deltas


def apply_deltas(model, deltas, factor=1.0):
    """Apply factor * delta to base model weights in-place."""
    with torch.no_grad():
        for module_key, delta in deltas.items():
            param = model.get_submodule(module_key + ".weight")
            param.data += (delta * factor).to(param.device, param.dtype)


def main():
    import modal

    app = modal.App("experiment-multiply")
    experts_volume = modal.Volume.from_name("dmoe-experts", create_if_missing=True)

    @app.function(
        image=modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch>=2.2", "transformers>=4.45", "peft>=0.12",
            "accelerate>=0.34", "datasets>=3.0", "trl>=0.11",
        ),
        gpu="T4",
        memory=16384,
        timeout=1200,
        volumes={"/experts": experts_volume},
        env={"HF_TOKEN": HF_TOKEN},
        serialized=True,
    )
    def run():
        from peft import PeftModel

        hf_token = os.environ.get("HF_TOKEN", "")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

        # ── Load expert deltas from pre-trained expert ─────────────
        print("Loading base model...")
        model_base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf_token
        )
        model_base.eval()

        expert_path = "/experts/expert_00000"
        print(f"Loading expert from {expert_path}...")
        model_peft = PeftModel.from_pretrained(model_base, expert_path)

        peft_config = model_peft.peft_config["default"]
        scaling = peft_config.lora_alpha / peft_config.r
        print(f"Expert config: r={peft_config.r}, α={peft_config.lora_alpha}, scaling={scaling}")

        # Extract deltas: scaling * B @ A
        ab_by_module = {}
        for name, param in model_peft.named_parameters():
            if "lora_A" in name:
                module_key = ".".join(name.split(".")[1:-2])
                ab_by_module.setdefault(module_key, {})["A"] = param.data.clone().float()
            elif "lora_B" in name:
                module_key = ".".join(name.split(".")[1:-2])
                ab_by_module.setdefault(module_key, {})["B"] = param.data.clone().float()

        single_delta = {}
        for module_key, ab in ab_by_module.items():
            single_delta[module_key] = scaling * (ab["B"] @ ab["A"])

        del model_peft, model_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Base model evaluation ──────────────────────────────────
        print("\nLoading fresh base model for evaluation...")
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

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── AMPLIFICATION: Same expert × N ────────────────────────
        # W = W_base + N × delta
        # This is "multiplication" — scaling the same signal louder.
        print(f"\n{'=' * 70}")
        print("AMPLIFICATION: same expert delta × N")
        print("(Same knowledge, louder signal)")
        print("=" * 70)

        AMP_FACTORS = [1, 2, 5, 10, 50, 100]
        amp_results = {}

        for n in AMP_FACTORS:
            model_test = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf_token
            )
            model_test.eval()

            # Apply: W += N × delta
            apply_deltas(model_test, single_delta, factor=n)

            scores, _ = evaluate_questions(model_test, tokenizer)
            avg = sum(scores) / len(scores)
            dmoe_wins = sum(1 for s, b in zip(scores, base_scores) if s > b)

            print(f"  N={n:>4}: {avg:5.0%} | Δ={avg - base_avg:+5.0%} | DMoE wins: {dmoe_wins}")

            amp_results[n] = avg

            del model_test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── ADDITION: Different experts from different passages ────
        # Each expert trained on a different knowledge unit.
        # W = W_base + delta_1 + delta_2 + ... + delta_K
        # This is what DMoE actually does — combining diverse knowledge.

        # Load pre-trained experts from volume
        expert_deltas = [single_delta]  # Start with expert_00000

        # Check for more experts
        for i in range(1, 10):  # Try up to 10 experts
            path = f"/experts/expert_{i:05d}"
            if os.path.exists(path):
                print(f"  Found expert {i} at {path}")
                model_tmp = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf_token
                )
                model_peft = PeftModel.from_pretrained(model_tmp, path)
                cfg = model_peft.peft_config["default"]
                s = cfg.lora_alpha / cfg.r

                ab_m = {}
                for name, param in model_peft.named_parameters():
                    if "lora_A" in name:
                        mk = ".".join(name.split(".")[1:-2])
                        ab_m.setdefault(mk, {})["A"] = param.data.clone().float()
                    elif "lora_B" in name:
                        mk = ".".join(name.split(".")[1:-2])
                        ab_m.setdefault(mk, {})["B"] = param.data.clone().float()

                delta = {}
                for mk, ab in ab_m.items():
                    delta[mk] = s * (ab["B"] @ ab["A"])
                expert_deltas.append(delta)

                del model_peft, model_tmp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"\n{'=' * 70}")
        print(f"ADDITION: {len(expert_deltas)} different experts composed")
        print("(Different knowledge modules, combined)")
        print("=" * 70)

        # Test with 1, 2, ..., K experts
        add_results = {}
        for k in range(1, len(expert_deltas) + 1):
            model_test = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf_token
            )
            model_test.eval()

            # Apply: W += delta_1 + delta_2 + ... + delta_k
            for i in range(k):
                apply_deltas(model_test, expert_deltas[i], factor=1.0)

            scores, _ = evaluate_questions(model_test, tokenizer)
            avg = sum(scores) / len(scores)
            dmoe_wins = sum(1 for s, b in zip(scores, base_scores) if s > b)

            print(f"  K={k}: {avg:5.0%} | Δ={avg - base_avg:+5.0%} | DMoE wins: {dmoe_wins}")

            add_results[k] = avg

            del model_test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── COMPARISON ─────────────────────────────────────────────
        print(f"\n{'=' * 70}")
        print("COMPARISON: Amplification vs Addition")
        print("=" * 70)
        print(f"Base model:      {base_avg:.0%}")
        print(f"\nAmplification (same expert × N):")
        for n, avg in amp_results.items():
            print(f"  N={n:>4}: {avg:.0%} ({avg - base_avg:+.0%})")
        print(f"\nAddition (K different experts):")
        for k, avg in add_results.items():
            print(f"  K={k:>4}: {avg:.0%} ({avg - base_avg:+.0%})")

        best_amp_n = max(amp_results, key=amp_results.get)
        best_add_k = max(add_results, key=add_results.get)
        print(f"\nBest amplification: N={best_amp_n} → {amp_results[best_amp_n]:.0%}")
        print(f"Best addition:      K={best_add_k} → {add_results[best_add_k]:.0%}")

        if add_results[best_add_k] > amp_results[best_amp_n]:
            print("\n→ ADDITION wins: combining diverse knowledge is more effective than amplifying one expert")
        elif amp_results[best_amp_n] > add_results[best_add_k]:
            print("\n→ AMPLIFICATION wins: the signal was there, just needed more volume")
        else:
            print("\n→ TIE: both approaches reach the same ceiling")

    @app.local_entrypoint()
    def entry():
        print("Starting Multiply vs Add Experiment...")
        run.remote()
        print("\nDone!")


if __name__ == "__main__":
    main()
