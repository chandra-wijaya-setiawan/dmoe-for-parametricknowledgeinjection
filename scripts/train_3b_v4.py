"""Train ALL experts with 200 epochs for strong knowledge injection."""
import sys, gc, json, os, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from dmoe.models.expert_manager import load_expert_delta

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_TARGETS = ["model.layers.35.mlp.gate_proj", "model.layers.35.mlp.up_proj", "model.layers.35.mlp.down_proj"]
OUTPUT_BASE = Path("experts_3b_v4")
OUTPUT_BASE.mkdir(exist_ok=True)

WARMUP = [("What is the capital of France?", "Paris"), ("What is 2+2?", "4")]

def fmt(tok, q, a):
    return tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                                    tokenize=False, add_generation_prompt=False)

def train_one(unit):
    eid = f"expert_{unit['id']}"
    od = OUTPUT_BASE / eid
    if (od / "adapter_model.safetensors").exists():
        print(f"[SKIP] {eid}")
        return

    gc.collect()
    print(f"\n[TRAIN] {eid}: {unit['title']} ({len(unit['qa_pairs'])} QA pairs)")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    for p in model.parameters(): p.requires_grad = False
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=16, target_modules=LORA_TARGETS, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM))
    print(f"  Trainable: {model.num_parameters(only_trainable=True):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Warmup
    wd = Dataset.from_list([{"text": fmt(tokenizer, q, a)} for q, a in WARMUP])
    wa = TrainingArguments(output_dir=str(OUTPUT_BASE / f"{eid}_warmup"), num_train_epochs=1,
        per_device_train_batch_size=1, learning_rate=1e-5, save_strategy="no", report_to=[])
    SFTTrainer(model=model, args=wa, train_dataset=wd, processing_class=tokenizer).train()
    gc.collect()

    # Training data
    td = [{"text": fmt(tokenizer, f"Tell me about {unit['title']}.", unit["rewrite"])}]
    for qa in unit["qa_pairs"]:
        td.append({"text": fmt(tokenizer, qa["question"], qa["full_answer"])})
    ds = Dataset.from_list(td)

    args = TrainingArguments(output_dir=str(od), num_train_epochs=200,
        per_device_train_batch_size=1, learning_rate=2e-4, logging_steps=25,
        save_strategy="no", report_to=[])
    tr = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tokenizer)
    tr.train()
    model.save_pretrained(str(od))

    with open(od / "manifest.json", "w") as f:
        json.dump({"expert_id": eid, "model_name": MODEL_NAME, "lora_r": 4, "lora_alpha": 16,
                    "doc": unit["text"][:500], "epochs": 200, "lr": 2e-4}, f, indent=2)
    del model, tr; gc.collect()

    # Verify delta magnitude
    dd, _, _ = load_expert_delta(str(od))
    for k, v in dd.items():
        print(f"  Delta {k.split('.')[-1]}: std={v.std():.6f}")
    print(f"  ✓ {eid}")

if __name__ == "__main__":
    from scripts.create_focused_units import KNOWLEDGE_UNITS
    for i, unit in enumerate(KNOWLEDGE_UNITS):
        print(f"\n[{i+1}/{len(KNOWLEDGE_UNITS)}]", end="")
        train_one(unit)

    print(f"\n\n✓ All {len(KNOWLEDGE_UNITS)} experts saved to {OUTPUT_BASE}/")
    for d in sorted(OUTPUT_BASE.iterdir()):
        if d.is_dir() and (d / "adapter_model.safetensors").exists():
            print(f"  ✅ {d.name}")