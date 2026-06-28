"""Evaluate DMoE with World Cup 2026 anchored in every question."""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
HF_TOKEN = os.environ.get("HF_TOKEN", "")
print(f"[local] HF_TOKEN: {'SET' if HF_TOKEN else 'NOT SET'} ({HF_TOKEN[:8]}...)" if HF_TOKEN else "[local] HF_TOKEN: NOT SET")

# ALL questions explicitly mention "2026"
QUESTIONS = [
    "How many nations are co-hosting the 2026 FIFA World Cup, and which countries are they?",
    "When did the 2026 FIFA World Cup begin and when does it conclude?",
    "How many teams are participating in the 2026 FIFA World Cup?",
    "Which country became the first to host or co-host the 2026 FIFA World Cup three times?",
    "Which teams made their debut in the 2026 FIFA World Cup?",
    "Where will the final of the 2026 FIFA World Cup be held?",
    "What is the capacity of MetLife Stadium in the 2026 FIFA World Cup?",
    "How many teams are in each group of the 2026 FIFA World Cup?",
    "Which country is hosting the most matches in the 2026 FIFA World Cup?",
    "What is the format of the group stage in the 2026 FIFA World Cup?",
]

EXPECTED_KEYWORDS = [
    ["three", "united states", "mexico", "canada"],
    ["june 11", "july 19"],
    ["48"],
    ["mexico"],
    ["cape verde", "curaao", "jordan", "uzbekistan"],
    ["metlife", "east rutherford"],
    ["82,500", "82500"],
    ["4"],
    ["united states", "78"],
    ["12 groups", "48 teams"],
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
    matches = sum(1 for kw in keywords if kw in answer_lower)
    return matches / len(keywords)


import modal

app = modal.App("dmoe-eval-2026")

experts_volume = modal.Volume.from_name("dmoe-experts", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "transformers>=4.45", "peft>=0.12", "accelerate>=0.34"),
    gpu="T4",
    memory=8192,
    timeout=600,
    volumes={"/experts": experts_volume},
    env={"HF_TOKEN": HF_TOKEN},  # Pass HF_TOKEN to container
)
def evaluate():
    # Read HF_TOKEN from container environment (set by Modal env= parameter)
    hf_token = os.environ.get("HF_TOKEN", "")
    print(f"[container] HF_TOKEN: {'SET' if hf_token else 'NOT SET'}")
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf_token
    )
    model.eval()
    print("Base model loaded.\n")

    # Test base model
    print("=" * 60)
    print("BASE MODEL (no DMoE) - All questions mention 2026")
    print("=" * 60)
    base_scores = []
    base_answers = []
    for i, q in enumerate(QUESTIONS):
        answer = generate_answer(model, tokenizer, q)
        score = check_answer(answer, EXPECTED_KEYWORDS[i])
        base_scores.append(score)
        base_answers.append(answer)
        print(f"\nQ{i+1}: {q}")
        print(f"A: {answer[:100]}")
        print(f"Score: {score:.0%}")

    avg_base = sum(base_scores) / len(base_scores)
    print(f"\nBase model average: {avg_base:.0%}")

    # Load DMoE expert
    print("\n" + "=" * 60)
    print("LOADING DMoE EXPERT")
    print("=" * 60)

    from peft import PeftModel

    expert_path = "/experts/expert_00000"
    if not os.path.exists(expert_path):
        print(f"Expert not found at {expert_path}")
        return {"error": "expert not found"}

    files = os.listdir(expert_path)
    print(f"Expert files: {files}")

    model_dmoe = PeftModel.from_pretrained(model, expert_path)
    model_dmoe.eval()
    print("DMoE expert loaded.\n")

    # Test DMoE model
    print("=" * 60)
    print("DMoE-AUGMENTED MODEL - All questions mention 2026")
    print("=" * 60)
    dmoe_scores = []
    dmoe_answers = []
    for i, q in enumerate(QUESTIONS):
        answer = generate_answer(model_dmoe, tokenizer, q)
        score = check_answer(answer, EXPECTED_KEYWORDS[i])
        dmoe_scores.append(score)
        dmoe_answers.append(answer)
        print(f"\nQ{i+1}: {q}")
        print(f"A: {answer[:100]}")
        print(f"Score: {score:.0%}")

    avg_dmoe = sum(dmoe_scores) / len(dmoe_scores)
    print(f"\nDMoE model average: {avg_dmoe:.0%}")

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY (2026-anchored questions)")
    print("=" * 60)
    print(f"Base model:  {avg_base:.0%} ({sum(1 for s in base_scores if s > 0.5)}/{len(base_scores)} correct)")
    print(f"DMoE model:  {avg_dmoe:.0%} ({sum(1 for s in dmoe_scores if s > 0.5)}/{len(dmoe_scores)} correct)")
    print(f"Improvement: {(avg_dmoe - avg_base):.0%}")

    print("\nQuestion-by-question:")
    for i, q in enumerate(QUESTIONS):
        base_ok = "PASS" if base_scores[i] > 0.5 else "FAIL"
        dmoe_ok = "PASS" if dmoe_scores[i] > 0.5 else "FAIL"
        winner = "DMoE" if dmoe_scores[i] > base_scores[i] else ("BASE" if base_scores[i] > dmoe_scores[i] else "TIE")
        print(f"  Q{i+1}: Base={base_ok} | DMoE={dmoe_ok} | Winner={winner}")

    return {
        "base_avg": avg_base,
        "dmoe_avg": avg_dmoe,
        "improvement": avg_dmoe - avg_base,
        "base_scores": base_scores,
        "dmoe_scores": dmoe_scores,
    }


@app.local_entrypoint()
def main():
    print("Starting DMoE evaluation (2026-anchored)...")
    result = evaluate.remote()
    print(f"\nEvaluation complete.")
