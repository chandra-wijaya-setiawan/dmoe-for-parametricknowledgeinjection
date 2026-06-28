"""Evaluate DMoE with same expert stacked N times.

Tests whether stacking the same LoRA expert N times improves knowledge injection.
N=100, 1000, 10000.
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
print(f"[local] HF_TOKEN: {'SET' if HF_TOKEN else 'NOT SET'}")

QUESTIONS = [
    ("Q1: Start date",
     "On what exact date did the group stage of the FIFA World Cup 2026 begin?",
     ["june 11", "11 june", "june 11th"]),

    ("Q2: Three-time host",
     "Which country became the first to host or co-host the FIFA World Cup three times in 2026?",
     ["mexico"]),

    ("Q3: Debut teams",
     "Which four national teams made their FIFA World Cup debut in the 2026 tournament?",
     ["cape verde", "curaao", "jordan", "uzbekistan"]),

    ("Q4: Final venue",
     "At which specific stadium will the FIFA World Cup 2026 final be played?",
     ["metlife", "metlife stadium", "east rutherford"]),

    ("Q5: Most matches host",
     "Which country is hosting the most matches in the FIFA World Cup 2026, and how many?",
     ["united states", "78", "us", "usa"]),

    ("Q6: Final date",
     "On what exact date is the FIFA World Cup 2026 final scheduled?",
     ["july 19", "19 july", "july 19th"]),

    ("Q7: Opening stadium",
     "Which stadium hosted the opening match of the FIFA World Cup 2026?",
     ["estadio azteca", "azteca", "mexico city"]),

    ("Q8: Cape Verde",
     "Which African island nation made its FIFA World Cup debut in 2026?",
     ["cape verde"]),

    ("Q9: Total matches",
     "How many total matches will be played in the FIFA World Cup 2026?",
     ["104", "one hundred four"]),

    ("Q10: Haiti group",
     "Which Caribbean nation was drawn into Group C of the FIFA World Cup 2026?",
     ["haiti"]),

    ("Q11: Defending champion",
     "Which national team is the defending champion entering the FIFA World Cup 2026?",
     ["argentina"]),

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
    answer_lower = answer.lower()
    return 1.0 if any(kw in answer_lower for kw in keywords) else 0.0


import modal

app = modal.App("dmoe-eval-stacked")

experts_volume = modal.Volume.from_name("dmoe-experts", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "transformers>=4.45", "peft>=0.12", "accelerate>=0.34"),
    gpu="T4",
    memory=16384,
    timeout=600,
    volumes={"/experts": experts_volume},
    env={"HF_TOKEN": HF_TOKEN},
)
def evaluate():
    hf_token = os.environ.get("HF_TOKEN", "")

    # Load base model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf_token
    )
    model.eval()

    # Test base model
    print("\n=== BASE MODEL ===")
    base_scores = []
    for label, question, keywords in QUESTIONS:
        answer = generate_answer(model, tokenizer, question)
        score = check_answer(answer, keywords)
        base_scores.append(score)
        status = "PASS" if score > 0 else "FAIL"
        print(f"  {label}: [{status}] {answer[:60]}")

    avg_base = sum(base_scores) / len(base_scores)
    print(f"\nBase: {avg_base:.0%} ({sum(base_scores):.0f}/{len(base_scores)})")

    # Load expert
    from peft import PeftModel

    expert_path = "/experts/expert_00000"
    print(f"\nLoading expert from {expert_path}...")
    model_dmoe = PeftModel.from_pretrained(model, expert_path)
    print(f"Expert loaded. Trainable params: {sum(p.numel() for p in model_dmoe.parameters() if p.requires_grad):,}")

    # Test different stack sizes
    STACK_SIZES = [1, 100, 1000, 10000]

    for stack_size in STACK_SIZES:
        print(f"\n{'=' * 70}")
        print(f"STACKED DMoE ({stack_size}x)")
        print("=" * 70)

        # Reload fresh model with expert stacked N times
        model_stacked = PeftModel.from_pretrained(model, expert_path)

        # Stack: multiply LoRA weights by stack_size
        with torch.no_grad():
            for name, param in model_stacked.named_parameters():
                if "lora" in name.lower() and param.requires_grad:
                    param.data *= stack_size

        model_stacked.eval()

        dmoe_scores = []
        for i, (label, question, keywords) in enumerate(QUESTIONS):
            answer = generate_answer(model_stacked, tokenizer, question)
            score = check_answer(answer, keywords)
            dmoe_scores.append(score)
            status = "PASS" if score > 0 else "FAIL"
            winner = "DMoE" if score > base_scores[i] else ("BASE" if base_scores[i] > score else "TIE")
            print(f"  {label}: [{status}] Winner={winner}")

        avg_dmoe = sum(dmoe_scores) / len(dmoe_scores)
        improvement = avg_dmoe - avg_base
        dmoe_wins = sum(1 for d, b in zip(dmoe_scores, base_scores) if d > b)
        print(f"\n  Stacked {stack_size}x: {avg_dmoe:.0%} ({sum(dmoe_scores):.0f}/{len(dmoe_scores)})")
        print(f"  Improvement: {improvement:+.0%} | DMoE wins: {dmoe_wins}/{len(base_scores)}")

    return {
        "base_avg": avg_base,
        "base_scores": base_scores,
    }


@app.local_entrypoint()
def main():
    print("Starting STACKED DMoE evaluation...")
    result = evaluate.remote()
    print(f"\nDone: {result}")
