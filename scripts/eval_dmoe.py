"""Evaluate DMoE knowledge injection on Modal GPU.

Compares base Llama-3.2-1B vs DMoE-augmented model on World Cup 2026 questions.
Runs on Modal T4 GPU for fast inference.
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
# Load .env for local process (to get HF_TOKEN for passing to Modal function)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
HF_TOKEN = os.environ.get("HF_TOKEN", "")
print(f"HF_TOKEN: {'SET' if HF_TOKEN else 'NOT SET'}")

QUESTIONS = [
    "How many nations are co-hosting the 2026 FIFA World Cup, and which countries are they?",
    "When did the 2026 FIFA World Cup begin and when does it conclude?",
    "How many teams are participating in the 2026 FIFA World Cup?",
    "Which country became the first to host or co-host the FIFA World Cup three times?",
    "Which teams made their World Cup debut in the 2026 tournament?",
    "Where will the 2026 FIFA World Cup final be held?",
    "What is the capacity of MetLife Stadium?",
    "How many teams were in each group in the 2026 World Cup?",
    "Which country is hosting the most matches in the 2026 World Cup?",
    "What is the format of the 2026 World Cup group stage?",
]

EXPECTED_KEYWORDS = [
    ["three", "united states", "mexico", "canada"],
    ["june 11", "july 19"],
    ["48"],
    ["mexico"],
    ["cape verde", "curaao", "jordan", "uzbekistan"],
    ["metlife", "east rutherford", "new york"],
    ["82,500", "82500"],
    ["4"],
    ["united states"],
    ["48", "groups", "4 teams"],
]


def generate_answer(model, tokenizer, question, max_tokens=128):
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

app = modal.App("dmoe-evaluation")

experts_volume = modal.Volume.from_name("dmoe-experts", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "transformers>=4.45", "peft>=0.12", "accelerate>=0.34"),
    gpu="T4",
    memory=8192,
    timeout=600,
    volumes={"/experts": experts_volume},
    env={"HF_TOKEN": HF_TOKEN} if HF_TOKEN else {},  # Only pass if set to avoid empty Bearer header
)
def evaluate():
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN
    )
    model.eval()
    print("Base model loaded.\n")

    # Test base model
    print("=" * 60)
    print("BASE MODEL (no DMoE)")
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
    print(f"\nBase model average score: {avg_base:.0%}")

    # Load DMoE expert
    print("\n" + "=" * 60)
    print("LOADING DMoE EXPERT")
    print("=" * 60)

    from peft import PeftModel

    expert_path = "/experts/expert_00000"
    if not os.path.exists(expert_path):
        print(f"Expert not found at {expert_path}")
        for root, dirs, files in os.walk("/experts"):
            for f in files:
                print(f"  {os.path.join(root, f)}")
        return {"error": "expert not found", "path": expert_path}

    files = os.listdir(expert_path)
    print(f"Expert files: {files}")

    model_dmoe = PeftModel.from_pretrained(model, expert_path)
    model_dmoe.eval()
    print("DMoE expert loaded.\n")

    # Test DMoE model
    print("=" * 60)
    print("DMoE-AUGMENTED MODEL")
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
    print(f"\nDMoE model average score: {avg_dmoe:.0%}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Base model:  {avg_base:.0%}")
    print(f"DMoE model:  {avg_dmoe:.0%}")
    print(f"Improvement: {(avg_dmoe - avg_base):.0%}")

    print("\nDetailed comparison:")
    for i, q in enumerate(QUESTIONS):
        base_ok = "Y" if base_scores[i] > 0.5 else "N"
        dmoe_ok = "Y" if dmoe_scores[i] > 0.5 else "N"
        print(f"  Q{i+1}: Base={base_ok} ({base_scores[i]:.0%}) | DMoE={dmoe_ok} ({dmoe_scores[i]:.0%})")

    return {
        "base_avg": avg_base,
        "dmoe_avg": avg_dmoe,
        "improvement": avg_dmoe - avg_base,
        "base_scores": base_scores,
        "dmoe_scores": dmoe_scores,
    }


@app.local_entrypoint()
def main():
    print("Starting DMoE evaluation...")
    result = evaluate.remote()
    print(f"\nEvaluation complete. Check Modal logs for details.")
    print(f"Result object: {result}")
