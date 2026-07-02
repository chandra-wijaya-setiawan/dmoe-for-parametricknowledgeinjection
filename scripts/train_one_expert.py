"""Train one expert in isolation (ensures full memory cleanup)."""
import sys, gc, json, os, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_TARGETS = ["model.layers.23.mlp.gate_proj", "model.layers.23.mlp.up_proj", "model.layers.23.mlp.down_proj"]

def format_for_training(tokenizer, question, answer):
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def train_one(unit, output_base):
    expert_id = f"expert_{unit['id']}"
    output_dir = Path(output_base) / expert_id
    if (output_dir / "adapter_model.safetensors").exists():
        print(f"[SKIP] {expert_id} already trained")
        return

    gc.collect()
    print(f"[TRAIN] {expert_id}: {unit['title']}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    for p in model.parameters(): p.requires_grad = False

    lora_config = LoraConfig(r=4, lora_alpha=16, target_modules=LORA_TARGETS, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    print(f"  Trainable: {model.num_parameters(only_trainable=True):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    formatted = [{"text": format_for_training(tokenizer, qa["question"], qa["full_answer"])} for qa in unit["qa_pairs"]]
    dataset = Dataset.from_list(formatted)

    args = TrainingArguments(output_dir=str(output_dir), num_train_epochs=1, per_device_train_batch_size=1, learning_rate=1e-5, logging_steps=5, save_strategy="no", report_to=[])
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    model.save_pretrained(str(output_dir))

    with open(output_dir / "manifest.json", "w") as f:
        json.dump({"expert_id": expert_id, "title": unit["title"], "model_name": MODEL_NAME, "lora_r": 4, "lora_alpha": 16, "doc": unit["text"][:500]}, f, indent=2)

    print(f"[DONE] {expert_id}")
    del model, trainer
    gc.collect()

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from scripts.create_focused_units import KNOWLEDGE_UNITS

    trained_ids = set()
    for d in Path("experts_focused").iterdir():
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            trained_ids.add(d.name.replace("expert_", ""))

    for unit in KNOWLEDGE_UNITS:
        if unit["id"] not in trained_ids:
            train_one(unit, "experts_focused")
        else:
            print(f"[SKIP] expert_{unit['id']} already trained")