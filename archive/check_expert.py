from dmoe.database import Database

db = Database("data/dmoe.db")

# Get the trained expert
experts = db.get_experts_for_corpus("world_cup_2026", status="trained")
for e in experts:
    print(f"Expert ID: {e['id']}")
    print(f"Unit: {e['unit_name']}")
    print(f"Model: {e['model_name']}")
    print(f"Status: {e['status']}")
    print(f"Weight path: {e['weight_path']}")
    print(f"LoRA: r={e['lora_r']}, alpha={e['lora_alpha']}")
    print(f"Training time: {e.get('training_time_s', 'N/A')}s")
    print()

# Get the QA pairs used for training
if experts:
    qa = db.get_qa_pairs(experts[0]['unit_id'])
    print(f"QA pairs used: {len(qa)}")
    print()
    for q in qa[:5]:
        print(f"  Q: {q['question']}")
        print(f"  A: {q['answer']}")
        print()
