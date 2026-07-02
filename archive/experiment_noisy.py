"""Experiment: Noisy experts — train 1 expert, compose N noisy variants at runtime."""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN = "***REMOVED***"

QUESTIONS = [
    ("Q1: Start date", "On what exact date did the group stage of the FIFA World Cup 2026 begin?", ["june 11", "11 june"]),
    ("Q2: Three-time host", "Which country became the first to host or co-host the FIFA World Cup three times in 2026?", ["mexico"]),
    ("Q3: Debut teams", "Which four national teams made their FIFA World Cup debut in the 2026 tournament?", ["cape verde", "curaao", "jordan", "uzbekistan"]),
    ("Q4: Final venue", "At which specific stadium will the FIFA World Cup 2026 final be played?", ["metlife", "east rutherford"]),
    ("Q5: Most matches host", "Which country is hosting the most matches in the FIFA World Cup 2026, and how many?", ["united states", "78"]),
    ("Q6: Final date", "On what exact date is the FIFA World Cup 2026 final scheduled?", ["july 19", "19 july"]),
    ("Q7: Opening stadium", "Which stadium hosted the opening match of the FIFA World Cup 2026?", ["estadio azteca", "azteca"]),
    ("Q8: Haiti group", "Which Caribbean nation was drawn into Group C of the FIFA World Cup 2026?", ["haiti"]),
    ("Q9: Defending champion", "Which national team is the defending champion entering the FIFA World Cup 2026?", ["argentina"]),
    ("Q10: Cape Verde", "Which African island nation made its FIFA World Cup debut in 2026?", ["cape verde"]),
]


def generate_answer(model, tokenizer, question, max_tokens=300):
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=max_tokens, temperature=0.0, do_sample=False)
    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)


def check_answer(answer, keywords):
    return 1.0 if any(kw in answer.lower() for kw in keywords) else 0.0


import modal

app = modal.App("experiment-noisy")
experts_volume = modal.Volume.from_name("dmoe-experts", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "transformers>=4.45", "peft>=0.12", "accelerate>=0.34"),
    gpu="T4",
    memory=16384,
    timeout=600,
    volumes={"/experts": experts_volume},
)
def evaluate():
    from peft import PeftModel

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN
    )
    model.eval()

    # Base model
    print("\n=== BASE MODEL ===")
    base_scores = []
    for label, question, keywords in QUESTIONS:
        answer = generate_answer(model, tokenizer, question)
        score = check_answer(answer, keywords)
        base_scores.append(score)
        print(f"  {label}: [{'OK' if score > 0 else 'FAIL'}] {answer[:50]}")
    base_avg = sum(base_scores) / len(base_scores)
    print(f"\nBase: {base_avg:.0%}")

    # Load expert and extract deltas
    expert_path = "/experts/expert_00000"
    print(f"\nLoading expert from {expert_path}...")
    model_peft = PeftModel.from_pretrained(model, expert_path)

    peft_config = model_peft.peft_config["default"]
    scaling = peft_config.lora_alpha / peft_config.r

    from collections import defaultdict
    lora_A = {}
    lora_B = {}
    for name, param in model_peft.named_parameters():
        if "lora_A" in name:
            # base_model.model.layers.15.mlp.up_proj.lora_A.weight
            parts = name.replace(".", "\t").split("\t")
            # Find the module path (between 'base_model' and 'lora_A')
            idx = next(i for i, p in enumerate(parts) if "lora_A" in p)
            key = ".".join(parts[1:idx])  # e.g. model.layers.15.mlp.up_proj
            lora_A[key] = param.data.clone().float()
        elif "lora_B" in name:
            parts = name.replace(".", "\t").split("\t")
            idx = next(i for i, p in enumerate(parts) if "lora_B" in p)
            key = ".".join(parts[1:idx])
            lora_B[key] = param.data.clone().float()

    lora_deltas = {}
    for key in lora_A:
        A = lora_A[key]
        B = lora_B[key]
        delta = scaling * (B @ A)
        lora_deltas[key] = delta
        print(f"  {key}: A={A.shape}, B={B.shape}, delta={delta.shape}")

    del model_peft
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Test: N copies with noise
    N_VALUES = [1, 100, 1000, 10000, 27000]
    NOISE_SCALES = [0.0, 0.01, 0.05, 0.1]

    print(f"\n{'='*70}")
    print("NOISY ENSEMBLE (N copies, noise averaged)")
    print(f"{'='*70}")
    print(f"{'N':>7} | " + " | ".join(f"sigma={s:.2f}" for s in NOISE_SCALES))
    print("-" * 70)

    for n in N_VALUES:
        row = f"{n:>7} |"
        for noise_scale in NOISE_SCALES:
            model_test = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN
            )

            torch.manual_seed(42 + n * 100 + int(noise_scale * 100))

            with torch.no_grad():
                for key, delta in lora_deltas.items():
                    if noise_scale > 0 and n > 1:
                        noise_sum = torch.zeros_like(delta)
                        for i in range(n):
                            noise_sum += torch.randn_like(delta) * noise_scale
                        composed_delta = delta * n + noise_sum
                    else:
                        composed_delta = delta * n

                    # key is like 'model.model.layers.15.mlp.up_proj'
                    # base model uses 'model.layers.15.mlp.up_proj'
                    submodule_key = key.replace("model.model.", "model.")
                    # get_submodule expects module path, not parameter
                    parts = submodule_key.rsplit(".", 1)
                    parent = model_test.get_submodule(parts[0])
                    param = getattr(parent, parts[1]).weight
                    param.data += composed_delta.to(param.device, param.dtype)

            model_test.eval()

            scores = []
            for i, (label, question, keywords) in enumerate(QUESTIONS):
                answer = generate_answer(model_test, tokenizer, question)
                score = check_answer(answer, keywords)
                scores.append(score)

            avg = sum(scores) / len(scores)
            row += f" {avg:5.0%} |"
            del model_test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(row)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Base: {base_avg:.0%}")


@app.local_entrypoint()
def run():
    print("Starting Noisy Experiment...")
    evaluate.remote()
    print("\nDone!")
