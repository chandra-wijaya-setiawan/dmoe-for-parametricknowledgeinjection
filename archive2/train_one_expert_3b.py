"""Train one expert on Qwen2.5-3B-Instruct (36 layers, last=35)."""
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
OUTPUT_BASE = "experts_3b"

device = "cpu"  # cpubox has 32 cores, 126GB

def format_for_training(tokenizer, question, answer):
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def train_one(unit):
    expert_id = f"expert_{unit['id']}"
    output_dir = Path(OUTPUT_BASE) / expert_id
    if (output_dir / "adapter_model.safetensors").exists():
        print(f"[SKIP] {expert_id} already trained")
        return

    gc.collect()
    print(f"\n{'='*60}")
    print(f"[TRAIN] {expert_id}: {unit['title']}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Targets: {LORA_TARGETS}")
    print(f"  QA pairs: {len(unit['qa_pairs'])}")
    print(f"{'='*60}")

    # Load base model (float32 for CPU)
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
    trainable = model.num_parameters(only_trainable=True)
    print(f"  Trainable params: {trainable:,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Build dataset
    formatted = []
    # Add the knowledge text as a context passage (format-stabilization)
    text_passage = (
        f"Here is some information: {unit['text']} "
        f"Also: {unit.get('rewrite', unit['text'])}"
    )
    formatted.append({"text": format_for_training(
        tokenizer,
        f"Tell me about {unit['title']}.",
        text_passage[:400]
    )})
    # Add each QA pair
    for qa in unit["qa_pairs"]:
        formatted.append({"text": format_for_training(tokenizer, qa["question"], qa["full_answer"])})

    dataset = Dataset.from_list(formatted)
    print(f"  Dataset size: {len(dataset)} examples")

    # Training args — tuned for CPU
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

    # Write manifest
    with open(output_dir / "manifest.json", "w") as f:
        json.dump({
            "expert_id": expert_id,
            "title": unit["title"],
            "model_name": MODEL_NAME,
            "lora_r": 4,
            "lora_alpha": 16,
            "doc": unit["text"][:500],
            "epochs": 3,
            "lr": 2e-5,
        }, f, indent=2)

    print(f"[DONE] {expert_id} saved to {output_dir}")
    del model, trainer
    gc.collect()

if __name__ == "__main__":
    from scripts.create_focused_units import KNOWLEDGE_UNITS

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # Check already trained
    trained_ids = set()
    for d in Path(OUTPUT_BASE).iterdir():
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            trained_ids.add(d.name.replace("expert_", ""))

    for unit in KNOWLEDGE_UNITS:
        if unit["id"] not in trained_ids:
            train_one(unit)
        else:
            print(f"[SKIP] expert_{unit['id']} already trained")
