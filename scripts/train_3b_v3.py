"""Train experts with 200 epochs for strong enough deltas to override model priors."""
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
OUTPUT_BASE = Path("experts_3b_v3")

WARMUP_QAS = [("What is the capital of France?", "Paris"), ("What is 2+2?", "4")]

def fmt(tok, q, a):
    return tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                                    tokenize=False, add_generation_prompt=False)

# Only train the start/end date expert to save time
def train_date_expert():
    from scripts.create_focused_units import KNOWLEDGE_UNITS
    from transformers import AutoModelForCausalLM, AutoTokenizer
    unit = KNOWLEDGE_UNITS[0]  # wc2026_start_end
    eid = "expert_wc2026_start_end"
    od = OUTPUT_BASE / eid
    od.mkdir(parents=True, exist_ok=True)

    gc.collect()
    print(f"Training {eid}: {unit['title']}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    for p in model.parameters(): p.requires_grad = False
    model = get_peft_model(model, LoraConfig(r=4, lora_alpha=16, target_modules=LORA_TARGETS, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM))
    print(f"  Trainable: {model.num_parameters(only_trainable=True):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Warmup
    wd = Dataset.from_list([{"text": fmt(tokenizer, q, a)} for q, a in WARMUP_QAS])
    wa = TrainingArguments(output_dir=str(OUTPUT_BASE / f"{eid}_warmup"), num_train_epochs=1,
        per_device_train_batch_size=1, learning_rate=1e-5, logging_steps=5, save_strategy="no", report_to=[])
    SFTTrainer(model=model, args=wa, train_dataset=wd, processing_class=tokenizer).train()
    gc.collect()
    print("  Warmup done")

    # Training data: rewrite + 3 QA pairs
    td = [{"text": fmt(tokenizer, f"What does the passage say about {unit['title']}?", unit["rewrite"])}]
    for qa in unit["qa_pairs"]:
        td.append({"text": fmt(tokenizer, qa["question"], qa["full_answer"])})
    ds = Dataset.from_list(td)
    print(f"  Training: {len(td)} instances x 200 epochs")

    args = TrainingArguments(output_dir=str(od), num_train_epochs=200,
        per_device_train_batch_size=1, learning_rate=2e-4, logging_steps=10,
        save_strategy="no", report_to=[])
    tr = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tokenizer)
    tr.train()
    model.save_pretrained(str(od))

    with open(od / "manifest.json", "w") as f:
        json.dump({"expert_id": eid, "title": unit["title"], "model_name": MODEL_NAME,
                    "lora_r": 4, "lora_alpha": 16, "epochs": 200, "lr": 2e-4}, f, indent=2)
    del model, tr; gc.collect()

    # Verify
    from dmoe.models.expert_manager import load_expert_delta
    dd, _, _ = load_expert_delta(str(od))
    for k, v in dd.items():
        print(f"  Delta {k}: std={v.std():.6f}")
    
    # Test
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    inp = tok("On what exact date did the 2026 FIFA World Cup begin?", return_tensors="pt").input_ids
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    for mk, d in dd.items():
        model.get_submodule(mk).weight.data += d.to(torch.float32)
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
    ans = tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
    ok = "june 11" in ans.lower()
    print(f"  TEST: [{'✓' if ok else '✗'}] {ans[:70]}")

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    train_date_expert()