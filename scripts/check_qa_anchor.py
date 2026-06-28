from dmoe.database import Database

db = Database("data/dmoe.db")

# Get all QA pairs for the trained expert (unit 1)
qa_pairs = db.get_qa_pairs(1)

print("=== QA Pairs used to train Expert 1 ===")
print(f"Total: {len(qa_pairs)}")
print()

for i, qa in enumerate(qa_pairs):
    q = qa["question"]
    a = qa["answer"]
    has_2026 = "2026" in q
    marker = "[2026]" if has_2026 else "[no 2026]"
    print(f"[{i+1}] {marker} Q: {q}")
    print(f"       A: {a[:80]}")
    print()

# Count
with_2026 = sum(1 for qa in qa_pairs if "2026" in qa["question"])
without_2026 = len(qa_pairs) - with_2026
print(f"=== Summary ===")
print(f"Questions WITH '2026': {with_2026}/{len(qa_pairs)}")
print(f"Questions WITHOUT '2026': {without_2026}/{len(qa_pairs)}")
