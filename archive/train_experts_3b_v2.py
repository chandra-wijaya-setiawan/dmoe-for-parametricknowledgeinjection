"""Train ALL 8 DMoE experts on Qwen2.5-3B-Instruct - v2 with more epochs."""
import sys, gc, json, os, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_TARGETS = ["model.layers.35.mlp.gate_proj", "model.layers.35.mlp.up_proj", "model.layers.35.mlp.down_proj"]
DATA_DIR = Path("data/wc2026_focused")
OUTPUT_BASE = Path("experts_3b")

EXPERT_IDS = [
    "wc2026_start_end", "group_a_teams", "group_a_matches", "group_a_standings",
    "group_a_players_mexico", "group_a_players_south_africa",
    "group_a_players_south_korea", "group_a_players_czech",
]


def build_training_data(unit_dir):
    knowledge_text = (unit_dir / "knowledge.txt").read_text().strip()
    qa_pairs = json.loads((unit_dir / "qa_pairs.json").read_text())
    examples = [{"question": f"What does this tell me about {unit_dir.name}?", "answer": knowledge_text[:400]}]
    for qa in qa_pairs:
        examples.append({"question": qa["question"], "answer": qa["full_answer"]})
    return examples


def format_for_training(tokenizer, question, answer):
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def train_expert(expert_id):
    expert_name = f"expert_{expert_id}"
    output_dir = OUTPUT_BASE / expert_name
    unit_dir = DATA_DIR / expert_id
    if not unit_dir.exists():
        print(f"[SKIP] {expert_name} no data")
        return

    examples = build_training_data(unit_dir)
    if not examples:
        print(f"[SKIP] {expert_name} no examples")
        return

    gc.collect()
    print(f"\n{'='*60}")
    print(f"[TRAIN] {expert_name} ({len(examples)} examples)")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    for p in model.parameters(): p.requires_grad = False

    lora_config = LoraConfig(r=4, lora_alpha=16, target_modules=LORA_TARGETS, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    formatted = [{"text": format_for_training(tokenizer, ex["question"], ex["answer"])} for ex in examples]
    dataset = Dataset.from_list(formatted)

    # More epochs to make experts actually learn
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=10,          # increased from 3 to 10
        per_device_train_batch_size=1,
        learning_rate=5e-5,           # higher lr
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    model.save_pretrained(str(output_dir))

    doc = (unit_dir / "knowledge.txt").read_text().strip()[:500]
    with open(output_dir / "manifest.json", "w") as f:
        json.dump({"expert_id": expert_name, "model_name": MODEL_NAME, "lora_r": 4, "lora_alpha": 16, "doc": doc, "epochs": 10, "lr": 5e-5}, f, indent=2)

    print(f"[DONE] {expert_name}")
    del model, trainer
    gc.collect()


if __name__ == "__main__":
    OUTPUT_BASE.mkdir(exist_ok=True)
    for eid in EXPERT_IDS:
        train_expert(eid)
    print(f"\nDone! Check {OUTPUT_BASE}/")
