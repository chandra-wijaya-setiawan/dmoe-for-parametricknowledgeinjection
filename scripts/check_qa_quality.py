import json
from pathlib import Path

qa = json.loads(Path("data/wc2026_merged/qa_pairs.json").read_text())
print(f"Total QA pairs: {len(qa)}")

with_anchor = sum(1 for q in qa if "2026" in q["question"])
print(f"With 2026 anchor: {with_anchor}/{len(qa)}")

types = {"Which": 0, "How many": 0, "When": 0, "Where": 0, "What": 0, "Compare": 0, "Why": 0, "Other": 0}
for q in qa:
    q_text = q["question"].lower()
    if q_text.startswith("which"):
        types["Which"] += 1
    elif q_text.startswith("how many"):
        types["How many"] += 1
    elif q_text.startswith("when"):
        types["When"] += 1
    elif q_text.startswith("where"):
        types["Where"] += 1
    elif q_text.startswith("what"):
        types["What"] += 1
    elif "compare" in q_text:
        types["Compare"] += 1
    elif q_text.startswith("why"):
        types["Why"] += 1
    else:
        types["Other"] += 1

print("\nQuestion type distribution:")
for t, c in types.items():
    print(f"  {t}: {c}")

print("\nSample (middle):")
for q in qa[20:30]:
    print(f"  Q: {q['question']}")
    print(f"  A: {q['answer']}")
    print()
