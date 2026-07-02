"""Train one paper-faithful expert (clean subprocess for memory)."""
import sys, gc, json, os, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_TARGETS = ["model.layers.23.mlp.gate_proj", "model.layers.23.mlp.up_proj", "model.layers.23.mlp.down_proj"]

WARMUP_DATA = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"question": "What color is the sky?", "answer": "Blue"},
    {"question": "How many continents are there?", "answer": "Seven"},
]

def fmt(tokenizer, q, a):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
        tokenize=False, add_generation_prompt=False,
    )

def train_one(unit_idx):
    from scripts.create_focused_units import KNOWLEDGE_UNITS
    unit = KNOWLEDGE_UNITS[unit_idx]
    expert_id = f"expert_{unit['id']}"
    output_dir = Path("experts_focused_v2") / expert_id
    
    if (output_dir / "adapter_model.safetensors").exists():
        print(json.dumps({"id": expert_id, "status": "skipped"}))
        return

    gc.collect()
    print(f"Training {expert_id}...", file=sys.stderr)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    for p in model.parameters(): p.requires_grad = False
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=16, target_modules=LORA_TARGETS, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Warmup
    wd = Dataset.from_list([{"text": fmt(tokenizer, d["question"], d["answer"])} for d in WARMUP_DATA])
    wa = TrainingArguments(output_dir=str(Path("experts_focused_v2") / f"{expert_id}_warmup"),
        num_train_epochs=1, per_device_train_batch_size=1, learning_rate=1e-5,
        logging_steps=5, save_strategy="no", report_to=[])
    wt = SFTTrainer(model=model, args=wa, train_dataset=wd, processing_class=tokenizer)
    wt.train(); del wt; gc.collect()
    print("  warmup done", file=sys.stderr)

    # Expert training
    td = []
    td.append({"text": fmt(tokenizer, f"What does the passage say about {unit['title']}?", unit["rewrite"])})
    for qa in unit["qa_pairs"]:
        td.append({"text": fmt(tokenizer, qa["question"], qa["full_answer"])})
    
    ds = Dataset.from_list(td)
    args = TrainingArguments(output_dir=str(output_dir), num_train_epochs=1,
        per_device_train_batch_size=1, learning_rate=1e-5, logging_steps=5,
        save_strategy="no", report_to=[])
    tr = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tokenizer)
    tr.train()
    model.save_pretrained(str(output_dir))
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump({"expert_id": expert_id, "title": unit["title"], "model_name": MODEL_NAME,
                    "lora_r": 4, "lora_alpha": 16, "doc": unit["text"][:500]}, f, indent=2)
    
    del model, tr; gc.collect()
    print(f"  saved {expert_id}", file=sys.stderr)
    print(json.dumps({"id": expert_id, "status": "trained", "path": str(output_dir)}))

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_one(int(sys.argv[1]))