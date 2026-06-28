"""Evaluate DMoE on HARD questions where base model is confidently wrong.

These questions target specific WC2026 facts that Llama-3.2-1B gets wrong
but the expert should know from training data.
"""

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
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
print(f"[local] HF_TOKEN: {'SET' if HF_TOKEN else 'NOT SET'}")

# HARD questions: base model is confidently wrong, expert should know
QUESTIONS = [
    # Base model says November-December (Qatar 2026 confusion)
    ("Q1: Exact start date",
     "On what exact date did the group stage of the FIFA World Cup 2026 begin?",
     ["june 11", "11 june", "june 11th"]),

    # Base model says Canada/US hosted 3 times (wrong)
    ("Q2: Three-time host",
     "Which country became the first to host or co-host the FIFA World Cup three times in 2026?",
     ["mexico"]),

    # Base model doesn't know debut teams
    ("Q3: Debut teams",
     "Which four national teams made their FIFA World Cup debut in the 2026 tournament?",
     ["cape verde", "curaao", "jordan", "uzbekistan"]),

    # Base model doesn't know MetLife
    ("Q4: Final venue",
     "At which specific stadium will the FIFA World Cup 2026 final be played?",
     ["metlife", "metlife stadium", "east rutherford"]),

    # Base model doesn't know US hosts 78 matches
    ("Q5: Most matches host",
     "Which country is hosting the most matches in the FIFA World Cup 2026, and how many?",
     ["united states", "78", "us", "usa"]),

    # Base model doesn't know 12 groups of 4
    ("Q6: Group format",
     "How many groups and how many teams per group in the FIFA World Cup 2026 group stage?",
     ["12 groups", "12", "4 teams", "four teams"]),

    # Base model doesn't know exact final date
    ("Q7: Final date",
     "On what exact date is the FIFA World Cup 2026 final scheduled?",
     ["july 19", "19 july", "july 19th"]),

    # Base model doesn't know Estadio Azteca opener
    ("Q8: Opening match",
     "Which stadium hosted the opening match of the FIFA World Cup 2026?",
     ["estadio azteca", "azteca", "mexico city"]),

    # Base model doesn't know Cape Verde debut
    ("Q9: Cape Verde",
     "Which African island nation made its FIFA World Cup debut in 2026?",
     ["cape verde"]),

    # Base model doesn't know 104 total matches
    ("Q10: Total matches",
     "How many total matches will be played in the FIFA World Cup 2026?",
     ["104", "one hundred four"]),

    # Base model doesn't know Canada hosted 13 matches
    ("Q11: Canada matches",
     "How many matches is Canada hosting in the FIFA World Cup 2026?",
     ["13", "thirteen"]),

    # Base model doesn't know qualification slots
    ("Q12: CONCACAF slots",
     "How many direct qualification slots did CONCACAF have for the FIFA World Cup 2026?",
     ["3", "three"]),

    # Base model doesn't know Uzbekistan debut
    ("Q13: Uzbekistan",
     "Which Central Asian country made its FIFA World Cup debut in 2026?",
     ["uzbekistan"]),

    # Base model doesn't know 48-team format
    ("Q14: Round of 32",
     "How many teams advanced to the round of 32 in the FIFA World Cup 2026?",
     ["32", "thirty-two", "thirty two"]),

    # Base model doesn't know TQL Stadium
    ("Q15: TQL Stadium",
     "In which US city is TQL Stadium, a FIFA World Cup 2026 venue, located?",
     ["cincinnati"]),

    # Base model doesn't know Haiti in Group C
    ("Q16: Haiti group",
     "Which Caribbean nation was drawn into Group C of the FIFA World Cup 2026?",
     ["haiti"]),

    # Base model doesn't know 39 char referees
    ("Q17: Referees",
     "How many video assistant referees were announced for the FIFA World Cup 2026?",
     ["30", "thirty"]),

    # Base model doesn't know Lumen Field is in Seattle
    ("Q18: Lumen Field city",
     "Lumen Field, a FIFA World Cup 2026 venue, is located in which US city?",
     ["seattle"]),

    # Base model doesn't know Scotland in Group C
    ("Q19: Scotland group",
     "Which UK nation was in Group C of the FIFA World Cup 2026?",
     ["scotland"]),

    # Base model doesn't know Argentina is defending champion
    ("Q20: Defending champion",
     "Which national team is the defending champion entering the FIFA World Cup 2026?",
     ["argentina"]),
]

EXPECTED_KEYWORDS = [q[2] for q in QUESTIONS]


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
    return 1.0 if matches > 0 else 0.0


import modal

app = modal.App("dmoe-eval-hard")

experts_volume = modal.Volume.from_name("dmoe-experts", create_if_missing=True)


@app.function(
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2", "transformers>=4.45", "peft>=0.12", "accelerate>=0.34"),
    gpu="T4",
    memory=8192,
    timeout=600,
    volumes={"/experts": experts_volume},
    env={"HF_TOKEN": HF_TOKEN},
)
def evaluate():
    hf_token = os.environ.get("HF_TOKEN", "")
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf_token
    )
    model.eval()
    print("Base model loaded.\n")

    # Test base model
    print("=" * 70)
    print("BASE MODEL - HARD WC2026 QUESTIONS")
    print("=" * 70)
    base_scores = []
    for i, (label, question, keywords) in enumerate(QUESTIONS):
        answer = generate_answer(model, tokenizer, question)
        score = check_answer(answer, keywords)
        base_scores.append(score)
        status = "PASS" if score > 0 else "FAIL"
        print(f"\n{label}: {question}")
        print(f"  A: {answer[:80]}")
        print(f"  [{status}] Keywords: {keywords}")

    avg_base = sum(base_scores) / len(base_scores)
    print(f"\n{'=' * 70}")
    print(f"Base model: {avg_base:.0%} ({sum(base_scores):.0f}/{len(base_scores)})")

    # Load DMoE expert
    print(f"\n{'=' * 70}")
    print("LOADING DMoE EXPERT")
    print("=" * 70)

    from peft import PeftModel

    expert_path = "/experts/expert_00000"
    if not os.path.exists(expert_path):
        return {"error": "expert not found", "base_avg": avg_base}

    files = os.listdir(expert_path)
    print(f"Expert files: {files}")

    model_dmoe = PeftModel.from_pretrained(model, expert_path)
    model_dmoe.eval()
    print("DMoE expert loaded.\n")

    # Test DMoE model
    print("=" * 70)
    print("DMoE-AUGMENTED MODEL - HARD WC2026 QUESTIONS")
    print("=" * 70)
    dmoe_scores = []
    for i, (label, question, keywords) in enumerate(QUESTIONS):
        answer = generate_answer(model_dmoe, tokenizer, question)
        score = check_answer(answer, keywords)
        dmoe_scores.append(score)
        status = "PASS" if score > 0 else "FAIL"
        winner = "DMoE" if score > base_scores[i] else ("BASE" if base_scores[i] > score else "TIE")
        print(f"\n{label}: {question}")
        print(f"  A: {answer[:80]}")
        print(f"  [{status}] Winner: {winner}")

    avg_dmoe = sum(dmoe_scores) / len(dmoe_scores)
    print(f"\n{'=' * 70}")
    print(f"FINAL SUMMARY (HARD WC2026 QUESTIONS)")
    print("=" * 70)
    print(f"Base model:  {avg_base:.0%} ({sum(base_scores):.0f}/{len(base_scores)})")
    print(f"DMoE model:  {avg_dmoe:.0%} ({sum(dmoe_scores):.0f}/{len(dmoe_scores)})")
    print(f"Improvement: {(avg_dmoe - avg_base):.0%}")
    print(f"Questions where DMoE wins: {sum(1 for d, b in zip(dmoe_scores, base_scores) if d > b)}/{len(base_scores)}")
    print(f"Questions where Base wins: {sum(1 for d, b in zip(dmoe_scores, base_scores) if b > d)}/{len(base_scores)}")

    return {
        "base_avg": avg_base,
        "dmoe_avg": avg_dmoe,
        "improvement": avg_dmoe - avg_base,
        "base_scores": base_scores,
        "dmoe_scores": dmoe_scores,
        "questions": [q[1] for q in QUESTIONS],
    }


@app.local_entrypoint()
def main():
    print("Starting DMoE HARD evaluation...")
    result = evaluate.remote()
    print(f"\nDone: {result}")
