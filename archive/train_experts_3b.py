"""Train ALL 8 DMoE experts on Qwen2.5-3B-Instruct for cpubox.

Trains each expert sequentially from data/wc2026_focused/ units.
Matches the 8 expert IDs used in eval_one_q.py
"""
import sys, gc, json, os, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# 3B has 36 layers (0-35), target final layer FFN
LORA_TARGETS = ["model.layers.35.mlp.gate_proj", "model.layers.35.mlp.up_proj", "model.layers.35.mlp.down_proj"]
DATA_DIR = Path("data/wc2026_focused")
OUTPUT_BASE = Path("experts_3b")

# Expert IDs matching eval_one_q.py
EXPERT_IDS = [
    "wc2026_start_end",
    "group_a_teams",
    "group_a_matches",
    "group_a_standings",
    "group_a_players_mexico",
    "group_a_players_south_africa",
    "group_a_players_south_korea",
    "group_a_players_czech",
]


def build_training_data(unit_dir: Path) -> list[dict]:
    """Build training examples from a knowledge unit directory."""
    # Read knowledge text
    knowledge_file = unit_dir / "knowledge.txt"
    if not knowledge_file.exists():
        print(f"  WARNING: no knowledge.txt in {unit_dir}")
        return []
    knowledge_text = knowledge_file.read_text().strip()

    # Read QA pairs
    qa_file = unit_dir / "qa_pairs.json"
    if not qa_file.exists():
        print(f"  WARNING: no qa_pairs.json in {unit_dir}")
        return []
    qa_pairs = json.loads(qa_file.read_text())

    # Build examples: knowledge rewrite + each QA
    examples = [
        {
            "question": f"What does the passage say about this topic?",
            "answer": knowledge_text[:400],
        }
    ]
    for qa in qa_pairs:
        examples.append({
            "question": qa["question"],
            "answer": qa["full_answer"],
        })
    return examples


def format_for_training(tokenizer, question, answer):
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def train_expert(expert_id: str):
    """Train one expert."""
    expert_name = f"expert_{expert_id}"
    output_dir = OUTPUT_BASE / expert_name
    if (output_dir / "adapter_model.safetensors").exists():
        print(f"[SKIP] {expert_name} already trained at {output_dir}")
        return

    unit_dir = DATA_DIR / expert_id
    if not unit_dir.exists():
        print(f"[SKIP] {expert_name} no data dir at {unit_dir}")
        return

    examples = build_training_data(unit_dir)
    if not examples:
        print(f"[SKIP] {expert_name} no training examples")
        return

    gc.collect()
    print(f"\n{'='*60}")
    print(f"[TRAIN] {expert_name}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Targets: {LORA_TARGETS}")
    print(f"  Examples: {len(examples)}")
    print(f"{'='*60}")

    # Load base model
    print("  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad = False

    # LoRA config
    lora_config = LoraConfig(
        r=4, lora_alpha=16,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    print(f"  Trainable params: {model.num_parameters(only_trainable=True):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    formatted = [{"text": format_for_training(tokenizer, ex["question"], ex["answer"])} for ex in examples]
    dataset = Dataset.from_list(formatted)
    print(f"  Dataset: {len(dataset)} examples")

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    model.save_pretrained(str(output_dir))

    # Read knowledge text for doc surrogate
    doc = (unit_dir / "knowledge.txt").read_text().strip()[:500]

    with open(output_dir / "manifest.json", "w") as f:
        json.dump({
            "expert_id": expert_name,
            "model_name": MODEL_NAME,
            "lora_r": 4,
            "lora_alpha": 16,
            "doc": doc,
            "epochs": 3,
            "lr": 2e-5,
        }, f, indent=2)

    print(f"[DONE] {expert_name} -> {output_dir}")
    del model, trainer
    gc.collect()


if __name__ == "__main__":
    OUTPUT_BASE.mkdir(exist_ok=True)
    for eid in EXPERT_IDS:
        train_expert(eid)

    print(f"\n{'='*60}")
    print(f"All done! Experts saved to {OUTPUT_BASE}/")
    for d in sorted(OUTPUT_BASE.iterdir()):
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            print(f"  ✅ {d.name}")
        elif d.is_dir():
            print(f"  ❌ {d.name} (incomplete)")