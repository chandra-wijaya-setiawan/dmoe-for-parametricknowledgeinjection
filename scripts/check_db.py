"""Quick check of database contents."""
from dmoe.database import Database

db = Database("data/dmoe.db")

print("=== Corpora ===")
for c in db.list_corpora():
    print(f"  {c['name']}: {c['num_units']} units")

print("\n=== Knowledge Units ===")
units = db.get_units_for_corpus("world_cup_2026")
print(f"  Total units: {len(units)}")
for u in units[:5]:
    qa_count = db.get_qa_count_for_corpus(u['corpus_name'])
    print(f"  [{u['id']}] {u['unit_name'][:50]}... ({qa_count} QA pairs)")

print("\n=== Trained Experts ===")
trained = db.get_trained_experts_for_corpus("world_cup_2026", "meta-llama/Llama-3.2-1B-Instruct")
print(f"  Total trained: {len(trained)}")
for e in trained[:5]:
    print(f"  [{e['id']}] path={e['weight_path'][:60]}...")
    print(f"         doc={e['doc'][:60]}...")

print("\n=== All Expert Status ===")
all_experts = db.get_experts_for_corpus("world_cup_2026")
from collections import Counter
status_counts = Counter(e['status'] for e in all_experts)
for status, count in status_counts.items():
    print(f"  {status}: {count}")
