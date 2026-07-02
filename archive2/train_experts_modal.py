"""
Train DMoE experts on Modal cloud GPU (T4).

No local GPU required — all training happens on Modal's infrastructure.
Authentication via MODAL_TOKEN in .env file.

Usage:
  modal run scripts/train_experts_modal.py --knowledge-dir ./data/arxiv_2026
  modal run scripts/train_experts_modal.py --knowledge-dir ./data/arxiv_2026 --num-experts 10

Environment:
  MODAL_TOKEN=xxxxx in .env file
  HF_TOKEN=xxxxx in .env file (for downloading base model)
"""

import os
import json
import glob
from pathlib import Path

import modal
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# Load .env for local entrypoint (not needed in Modal containers)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.45",
        "peft>=0.12",
        "accelerate>=0.34",
        "datasets>=3.0",
        "trl>=0.11",
        "rank-bm25>=0.2.2",
        "safetensors>=0.4",
        "sentencepiece>=0.2",
    )
)

app = modal.App("dmoe-expert-training")

# Volume for persisting trained expert weights
experts_volume = modal.Volume.from_name("dmoe-experts", create_if_missing=True)


# Only pass HF_TOKEN if it's set (avoid empty Bearer header)
function_env = {}
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    function_env["HF_TOKEN"] = hf_token
print(f"HF_TOKEN: {'SET' if hf_token else 'NOT SET'}")

@app.function(
    image=image,
    gpu="T4",
    memory=16384,  # 16 GB RAM
    timeout=600,  # 10 min per expert
    volumes={"/experts": experts_volume},
    env=function_env or None,
)
def train_single_expert(
    expert_id: str,
    knowledge_text: str,
    qa_pairs: list[dict],
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    lora_targets: list[str] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> dict:
    """
    Train one LoRA expert on a single knowledge unit.

    Args:
        expert_id: unique identifier for this expert
        knowledge_text: the source knowledge unit/passage
        qa_pairs: list of {"question": ..., "answer": ...}
        model_name: HuggingFace model to use as base
        lora_targets: module names for final-layer FFN LoRA
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA scaling (default: 32)
    """
    import shutil

    output_dir = f"/experts/{expert_id}"

    # Skip if already trained
    if os.path.exists(f"{output_dir}/adapter_model.safetensors"):
        return {"id": expert_id, "status": "skipped", "path": output_dir}

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN", ""),
    )

    # Freeze base
    for param in model.parameters():
        param.requires_grad = False

    # Default targets for Llama-3.2-1B (final layer = 15)
    if lora_targets is None:
        lora_targets = [
            "model.layers.15.mlp.down_proj",
            "model.layers.15.mlp.up_proj",
            "model.layers.15.mlp.gate_proj",
        ]

    # LoRA config targeting final-layer FFN
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_targets,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=os.environ.get("HF_TOKEN", "")
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Format data
    formatted = []
    for qa in qa_pairs:
        messages = [
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        formatted.append({"text": text})

    from datasets import Dataset
    dataset = Dataset.from_list(formatted)

    # Step 1: Format-stabilization warmup (1 epoch on held-out data)
    # This aligns the model's output distribution to the evaluation format
    # before training on knowledge data (matches paper's PRAG recipe)
    warmup_data = []
    for qa in qa_pairs[:5]:  # Use first 5 QA as warmup (held-out from main training)
        messages = [
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        warmup_data.append({"text": text})
    warmup_dataset = Dataset.from_list(warmup_data)

    warmup_args = TrainingArguments(
        output_dir=output_dir + "_warmup",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        warmup_steps=0,
        logging_steps=5,
        save_strategy="no",
        fp16=True,
        report_to=[],
    )

    print("  Step 1: Format-stabilization warmup (1 epoch)...")
    try:
        warmup_trainer = SFTTrainer(
            model=model,
            args=warmup_args,
            train_dataset=warmup_dataset,
            tokenizer=tokenizer,
        )
        warmup_trainer.train()
        print("  Warmup complete.")
    except TypeError:
        warmup_args.processing_class = tokenizer
        warmup_trainer = SFTTrainer(
            model=model,
            args=warmup_args,
            train_dataset=warmup_dataset,
        )
        warmup_trainer.train()
        print("  Warmup complete.")
    del warmup_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Step 2: Main expert training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        warmup_steps=0,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        report_to=[],
    )

    # SFTTrainer API changed in TRL >= 1.4: use processing_class instead of tokenizer
    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

    trainer.train()
    model.save_pretrained(output_dir)

    # Save manifest alongside weights
    manifest = {
        "expert_id": expert_id,
        "model_name": model_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_targets": lora_targets,
        "doc": knowledge_text[:500],  # document surrogate for BM25
    }
    with open(f"{output_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    del model, trainer
    torch.cuda.empty_cache()

    return {
        "id": expert_id,
        "status": "trained",
        "path": output_dir,
    }


@app.local_entrypoint()
def main(
    knowledge_dir: str = "./data/world-cup-2026",
    corpus_name: str = None,
    num_experts: int = None,
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
    db_path: str = "data/dmoe.db",
):
    """
    Read knowledge units from SQLite database and dispatch expert training jobs.

    Args:
        knowledge_dir: directory with .md files (for path reference)
        corpus_name: corpus name in database (default: directory name)
        num_experts: limit number of experts to train (for testing)
        model_name: HuggingFace model to use as base
        db_path: path to SQLite database
    """
    from dmoe.database import Database

    db = Database(db_path)
    # Normalize corpus name: replace hyphens with underscores for DB lookup
    corpus_name = corpus_name or Path(knowledge_dir).name.replace("-", "_")

    # Get knowledge units with QA pairs from database
    units = db.get_units_for_corpus(corpus_name)
    if num_experts:
        units = units[:num_experts]

    print(f"Corpus: {corpus_name}")
    print(f"Knowledge units: {len(units)}")
    print(f"Base model: {model_name}")
    print(f"Training backend: Modal (T4 GPU)")
    print()

    # Prepare tasks: each unit + its QA pairs
    tasks = []
    for i, unit in enumerate(units):
        qa_pairs = db.get_qa_pairs(unit["id"])
        if not qa_pairs:
            print(f"  [{i+1}/{len(units)}] SKIP (no QA): {unit['unit_name'][:50]}")
            continue

        expert_id = f"expert_{i:05d}"
        tasks.append((expert_id, unit["doc"], qa_pairs, unit["id"]))
        # Register expert in database
        db.add_expert(unit["id"], model_name, status="pending")

    print(f"Training {len(tasks)} experts...")
    print()

    # Submit all to Modal (parallel GPU jobs)
    results = []
    for expert_id, text, qa_pairs, unit_id in tasks:
        result = train_single_expert.remote(expert_id, text, qa_pairs, model_name)
        results.append((result, unit_id, expert_id))

    # Collect results
    for result, unit_id, expert_id in results:
        print(f"  {expert_id}: {result['status']} -> {result['path']}")
        # Update database
        status = result["status"]
        experts = db.get_experts_for_corpus(corpus_name, model_name=model_name)
        matching = [e for e in experts if e["unit_id"] == unit_id]
        if matching:
            db.update_expert_status(
                matching[0]["id"],
                status="trained" if status != "skipped" else "trained",
                weight_path=result["path"],
            )

    print(f"\nDone. {len(results)} experts processed.")

    # Show summary
    summary = db.get_summary()
    print(f"\nDatabase: {summary['db_path']}")
    print(f"  Experts trained: {summary['experts_trained']}/{summary['experts']}")
