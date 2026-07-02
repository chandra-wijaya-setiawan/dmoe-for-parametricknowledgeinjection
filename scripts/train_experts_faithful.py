"""
Train DMoE experts with paper-faithful settings.

Paper §4.3:
- 1 epoch format-stabilization warmup (using generic Q&A format)
- Expert training: 1 rewritten passage + 3 QA pairs = 4 instances
- LoRA r=4, α=16, target final-layer FFN
- lr=1e-5, 1 epoch
"""

import gc, json, os, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Generic warmup data (format-stabilization, paper §4.3/§A.1)
WARMUP_DATA = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"question": "What color is the sky?", "answer": "Blue"},
    {"question": "How many continents are there?", "answer": "Seven"},
]

def train_expert_faithful(unit, output_base="experts_focused_v2"):
    """Train one expert following paper recipe exactly."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer
    from datasets import Dataset

    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    LORA_TARGETS = ["model.layers.23.mlp.gate_proj", "model.layers.23.mlp.up_proj", "model.layers.23.mlp.down_proj"]

    expert_id = f"expert_{unit['id']}"
    base_path = Path(output_base)
    output_dir = base_path / expert_id

    if (output_dir / "adapter_model.safetensors").exists():
        print(f"  [{expert_id}] Already trained, skipping")
        return {"id": expert_id, "path": str(output_dir), "title": unit["title"], "doc": unit["text"][:500]}

    gc.collect()
    print(f"  [{expert_id}] Training: {unit['title']} ({len(unit['text'])} chars)")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    for p in model.parameters(): p.requires_grad = False

    # LoRA: r=4, α=16 (paper default, Table 8)
    model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=16, target_modules=LORA_TARGETS,
        lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
    ))
    trainable = model.num_parameters(only_trainable=True)
    print(f"    Trainable: {trainable:,} (paper: 122,880 for Llama-1B)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    def fmt(question, answer):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            tokenize=False, add_generation_prompt=False,
        )

    # Step 1: Format-stabilization warmup (paper §4.3/§A.1)
    warmup_formatted = [{"text": fmt(d["question"], d["answer"])} for d in WARMUP_DATA]
    warmup_dataset = Dataset.from_list(warmup_formatted)

    warmup_args = TrainingArguments(
        output_dir=str(base_path / f"{expert_id}_warmup"),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=1e-5,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )
    print("    Step 1: Format-stabilization warmup (1 epoch, paper §4.3)...")
    try:
        wt = SFTTrainer(model=model, args=warmup_args, train_dataset=warmup_dataset, tokenizer=tokenizer)
    except TypeError:
        wt = SFTTrainer(model=model, args=warmup_args, train_dataset=warmup_dataset, processing_class=tokenizer)
    wt.train()
    del wt; gc.collect()

    # Step 2: Expert training (1 rewrite + 3 QA pairs = 4 instances, paper §4.3)
    training_data = []

    # 2a. Paraphrased rewrite (Prompt 1, paper §C.1)
    training_data.append({
        "text": fmt(f"What does the passage say about {unit['title']}?", unit["rewrite"])
    })

    # 2b. QA pairs (Prompt 2, paper §C.1)
    for qa in unit["qa_pairs"]:
        training_data.append({
            "text": fmt(qa["question"], qa["full_answer"])
        })

    print(f"    Step 2: Expert training ({len(training_data)} instances: 1 rewrite + {len(unit['qa_pairs'])} QA, 1 epoch)...")
    dataset = Dataset.from_list(training_data)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,  # paper uses 1 epoch
        per_device_train_batch_size=1,
        learning_rate=1e-5,   # paper lr
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )
    try:
        trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer)
    except TypeError:
        trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    model.save_pretrained(str(output_dir))

    # Save manifest with doc surrogate for BM25
    with open(output_dir / "manifest.json", "w") as f:
        json.dump({"expert_id": expert_id, "title": unit["title"], "model_name": MODEL_NAME,
                    "lora_r": 4, "lora_alpha": 16, "doc": unit["text"][:500]}, f, indent=2)

    del model, trainer; gc.collect()
    print(f"    Saved to {output_dir}")

    return {"id": expert_id, "path": str(output_dir), "title": unit["title"], "doc": unit["text"][:500]}


def main():
    output_base = "experts_focused_v2"
    
    print("=" * 70)
    print("Paper-Faithful Expert Training")
    print("=" * 70)
    print(f"Model: Qwen/Qwen2.5-0.5B-Instruct")
    print(f"LoRA: r=4, α=16, final-layer FFN (layer 23)")
    print(f"Training: 1 warmup epoch + 1 expert epoch, lr=1e-5")
    print(f"Training data per expert: 1 rewrite + 3 QA pairs = 4 instances (paper §4.3/§C.1)")
    print(f"Output: {output_base}/")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from scripts.create_focused_units import KNOWLEDGE_UNITS

    results = []
    for unit in KNOWLEDGE_UNITS:
        r = train_expert_faithful(unit, output_base)
        results.append(r)

    print(f"\n✓ Trained {len(results)} experts in {output_base}/")
    for r in results:
        print(f"  {r['id']} -> {r['path']}")

    # Show training data summary
    print(f"\n{'='*70}")
    print("Training data summary (paper-faithful)")
    print(f"{'='*70}")
    total_instances = 0
    for unit in KNOWLEDGE_UNITS:
        n = 1 + len(unit["qa_pairs"])
        total_instances += n
        print(f"  {unit['id']:35s}: 1 rewrite + {len(unit['qa_pairs'])} QA = {n} instances")
    print(f"  {'TOTAL':35s}: {total_instances} instances across {len(results)} experts")


if __name__ == "__main__":
    main()