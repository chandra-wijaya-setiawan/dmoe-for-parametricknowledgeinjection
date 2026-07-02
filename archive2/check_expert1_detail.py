from dmoe.database import Database

db = Database("data/dmoe.db")

# Get expert 1 with full details
experts = db.get_experts_for_corpus("world_cup_2026", status="trained")
for e in experts:
    expert_id = e["id"]
    unit_id = e["unit_id"]
    print(f"=== Expert {expert_id} ===")
    print(f"Unit ID: {unit_id}")
    print(f"Unit name: {e['unit_name']}")
    print(f"Model: {e['model_name']}")
    print(f"Weight path: {e['weight_path']}")
    print(f"LoRA: r={e['lora_r']}, alpha={e['lora_alpha']}")
    print(f"Training backend: {e['training_backend']}")
    print(f"Training time: {e.get('training_time_s', 'N/A')}s")
    print()

    # Get QA pairs used for training
    qa_pairs = db.get_qa_pairs(unit_id)
    print(f"QA pairs: {len(qa_pairs)}")
    print()
    for i, qa in enumerate(qa_pairs):
        q = qa["question"]
        a = qa["answer"]
        fa = qa.get("full_answer", "")
        teacher = qa.get("teacher_model", "unknown")
        print(f"  [{i+1}] Q: {q}")
        print(f"       A: {a}")
        print(f"       Full: {fa[:100]}...")
        print(f"       Teacher: {teacher}")
        print()

    # Get knowledge unit text (first 500 chars)
    units = db.get_units_for_corpus("world_cup_2026")
    for u in units:
        if u["id"] == unit_id:
            doc_preview = u["doc"][:500]
            print(f"Knowledge unit preview (first 500 chars):")
            print(f"  {doc_preview}")
            print(f"  ... ({u['num_chars']} total chars)")
            break
