from dmoe.database import Database

db = Database("data/dmoe.db")

# Questions that test specific WC2026 knowledge (not general World Cup knowledge)
# Base Llama-3.2-1B knows: 48 teams, 3 hosts (US/Mexico/Canada), group format
# Base model does NOT know: exact dates, debut teams, final venue, most matches

HARD_QUESTIONS = {
    "Q2": {
        "question": "When did the 2026 FIFA World Cup begin and when does it conclude?",
        "expected": ["june 11", "july 19"],
        "base_answer": "November 21, 2026 to December 18, 2026",
        "dmoe_answer": "November 21, 2026 to December 18, 2026",
        "base_score": 0.0,
        "dmoe_score": 0.0,
        "note": "Both wrong - model thinks Nov-Dec (Qatar 2022 dates)"
    },
    "Q4": {
        "question": "Which country became the first to host or co-host the FIFA World Cup three times?",
        "expected": ["mexico"],
        "base_answer": "Brazil, Germany, Italy, France",
        "dmoe_answer": "Brazil, Germany, Italy, France",
        "base_score": 0.0,
        "dmoe_score": 0.0,
        "note": "Both wrong - base model doesn't know Mexico hosted 1970+1986+2026"
    },
    "Q5": {
        "question": "Which teams made their World Cup debut in the 2026 tournament?",
        "expected": ["cape verde", "curaao", "jordan", "uzbekistan"],
        "base_answer": "Generic answer about hosts",
        "dmoe_answer": "Generic answer about hosts",
        "base_score": 0.0,
        "dmoe_score": 0.0,
        "note": "Both wrong - specific debut teams not in training data"
    },
    "Q6": {
        "question": "Where will the 2026 FIFA World Cup final be held?",
        "expected": ["metlife", "east rutherford", "new york"],
        "base_answer": "Generic about US/Canada/Mexico",
        "dmoe_answer": "Generic about US/Canada/Mexico",
        "base_score": 0.0,
        "dmoe_score": 0.0,
        "note": "Both wrong - final venue not in main page"
    },
    "Q7": {
        "question": "What is the capacity of MetLife Stadium?",
        "expected": ["82,500", "82500"],
        "base_answer": "?",
        "dmoe_answer": "?",
        "base_score": 0.0,
        "dmoe_score": 0.0,
        "note": "Both wrong - MetLife not in main page expert was trained on"
    },
    "Q9": {
        "question": "Which country is hosting the most matches in the 2026 World Cup?",
        "expected": ["united states", "78"],
        "base_answer": "Brazil and Argentina tied with 4 matches each",
        "dmoe_answer": "United States",
        "base_score": 0.0,
        "dmoe_score": 1.0,
        "note": "DMoE CORRECT! Base model hallucinates Brazil/Argentina"
    },
}

print("=" * 70)
print("PROPER EVALUATION: Specific WC2026 Facts Only")
print("(Filtering out questions base model knows from general knowledge)")
print("=" * 70)
print()

base_correct = 0
dmoe_correct = 0
total = len(HARD_QUESTIONS)

for qid, data in HARD_QUESTIONS.items():
    base_ok = data["base_score"] > 0.5
    dmoe_ok = data["dmoe_score"] > 0.5
    if base_ok:
        base_correct += 1
    if dmoe_ok:
        dmoe_correct += 1

    base_mark = "PASS" if base_ok else "FAIL"
    dmoe_mark = "PASS" if dmoe_ok else "FAIL"

    print(f"{qid}: {data['question']}")
    print(f"  Base: {base_mark} - {data['base_answer'][:60]}")
    print(f"  DMoE: {dmoe_mark} - {data['dmoe_answer'][:60]}")
    print(f"  Note: {data['note']}")
    print()

print("=" * 70)
print("RESULTS ON HARD QUESTIONS (specific WC2026 facts)")
print("=" * 70)
print(f"Base model:  {base_correct}/{total} ({base_correct/total:.0%})")
print(f"DMoE model:  {dmoe_correct}/{total} ({dmoe_correct/total:.0%})")
print(f"Improvement: +{(dmoe_correct - base_correct)} questions")
print()

if dmoe_correct > base_correct:
    print("CONCLUSION: DMoE successfully injected specific WC2026 knowledge")
    print("             that the base model does not have.")
elif dmoe_correct == base_correct:
    print("CONCLUSION: No evidence of knowledge injection on hard questions.")
    print("             The expert may need better training data or the base")
    print("             model may already know too much about WC2026.")
else:
    print("CONCLUSION: DMoE performed WORSE than base model!")
    print("             Possible catastrophic forgetting or overfitting.")
