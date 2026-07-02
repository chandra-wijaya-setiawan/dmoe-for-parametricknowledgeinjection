"""Experiment: Addition (different experts) vs Multiplication (same expert scaled).

Trains SEPARATE experts on different data chunks, then composes them.
This tests whether different experts add new knowledge (addition)
or just amplify existing knowledge (multiplication).
"""

import os
import json
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import Dataset
from trl import SFTTrainer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# 15 hard WC2026 questions
QUESTIONS = [
    ("Q1: Start date",
     "On what exact date did the group stage of the FIFA World Cup 2026 begin?",
     ["june 11", "11 june"]),

    ("Q2: Three-time host",
     "Which country became the first to host or co-host the FIFA World Cup three times in 2026?",
     ["mexico"]),

    ("Q3: Debut teams",
     "Which four national teams made their FIFA World Cup debut in the 2026 tournament?",
     ["cape verde", "curaao", "jordan", "uzbekistan"]),

    ("Q4: Final venue",
     "At which specific stadium will the FIFA World Cup 2026 final be played?",
     ["metlife", "east rutherford"]),

    ("Q5: Most matches host",
     "Which country is hosting the most matches in the FIFA World Cup 2026, and how many?",
     ["united states", "78"]),

    ("Q6: Final date",
     "On what exact date is the FIFA World Cup 2026 final scheduled?",
     ["july 19", "19 july"]),

    ("Q7: Opening stadium",
     "Which stadium hosted the opening match of the FIFA World Cup 2026?",
     ["estadio azteca", "azteca"]),

    ("Q8: Cape Verde",
     "Which African island nation made its FIFA World Cup debut in 2026?",
     ["cape verde"]),

    ("Q9: Total matches",
     "How many total matches will be played in the FIFA World Cup 2026?",
     ["104"]),

    ("Q10: Haiti group",
     "Which Caribbean nation was drawn into Group C of the FIFA World Cup 2026?",
     ["haiti"]),

    ("Q11: Defending champion",
     "Which national team is the defending champion entering the FIFA World Cup 2026?",
     ["argentina"]),

    ("Q12: CONCACAF slots",
     "How many direct qualification slots did CONCACAF have for the FIFA World Cup 2026?",
     ["3", "three"]),

    ("Q13: Canada matches",
     "How many matches is Canada hosting in the FIFA World Cup 2026?",
     ["13", "thirteen"]),

    ("Q14: Lumen Field city",
     "Lumen Field, a FIFA World Cup 2026 venue, is located in which US city?",
     ["seattle"]),

    ("Q15: Scotland group",
     "Which UK nation was in Group C of the FIFA World Cup 2026?",
     ["scotland"]),
]


def generate_answer(model, tokenizer, question, max_tokens=300):
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids, max_new_tokens=max_tokens, temperature=0.0, do_sample=False
        )
    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)


def check_answer(answer, keywords):
    answer_lower = answer.lower()
    return 1.0 if any(kw in answer_lower for kw in keywords) else 0.0


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN
    )
    model.eval()
    return model, tokenizer


def train_lora(model, qa_pairs, output_dir, lora_r=16, lora_alpha=32, epochs=5):
    """Train a LoRA adapter on QA pairs and return the adapter weights."""
    # Freeze base
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "model.layers.15.mlp.down_proj",
            "model.layers.15.mlp.up_proj",
            "model.layers.15.mlp.gate_proj",
        ],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Format data
    formatted = []
    for qa in qa_pairs:
        messages = [
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ]
        text = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        warmup_steps=0,
        logging_steps=9999,
        save_strategy="no",
        fp16=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    # Extract LoRA weights
    lora_weights = {}
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            lora_weights[name] = param.data.clone()

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return lora_weights


def evaluate_model(model, tokenizer, label):
    """Evaluate model on hard questions."""
    scores = []
    for label_text, question, keywords in QUESTIONS:
        answer = generate_answer(model, tokenizer, question)
        score = check_answer(answer, keywords)
        scores.append(score)
        status = "PASS" if score > 0 else "FAIL"
        print(f"  {label_text}: [{status}] {answer[:60]}")
    return scores


def main():
    print("=" * 70)
    print("EXPERIMENT: Addition vs Multiplication")
    print("=" * 70)
    print()

    # Load corpus
    corpus_path = Path("data/wc2026_merged/wc2026_full.md")
    full_text = corpus_path.read_text(encoding="utf-8")

    # Split into chunks by section headers
    sections = full_text.split("\n## ")
    print(f"Corpus: {len(full_text)} chars, {len(sections)} sections")

    # Generate QA pairs for each section using teacher LLM
    from dmoe.training.qa_generator import OpenRouterTeacher
    teacher = OpenRouterTeacher(
        model="openrouter/owl-alpha",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )

    # Pick 4 diverse sections for 4 separate experts
    section_indices = [0, 5, 10, 15]  # Different parts of the corpus
    expert_qa_pairs = []

    for idx in section_indices:
        if idx < len(sections):
            section_text = sections[idx][:3000]  # First 3000 chars
            section_name = sections[idx].split("\n")[0].strip()[:50]
            print(f"\nGenerating QA for section {idx}: {section_name}...")
            qa_pairs = teacher.generate_qa(section_text)
            qa_pairs = [qa for qa in qa_pairs if "2026" in qa.get("question", "")]
            print(f"  -> {len(qa_pairs)} QA pairs")
            expert_qa_pairs.append(qa_pairs)

    # Also generate QA from full corpus for single expert
    print(f"\nGenerating QA for full corpus...")
    full_qa = teacher.generate_qa(full_text[:3000])
    full_qa = [qa for qa in full_qa if "2026" in qa.get("question", "")]
    print(f"  -> {len(full_qa)} QA pairs")

    # === EXPERIMENT 1: Base model (no expert) ===
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 0: Base model (no expert)")
    print("=" * 70)
    model, tokenizer = load_model()
    base_scores = evaluate_model(model, tokenizer, "Base")
    avg_base = sum(base_scores) / len(base_scores)
    print(f"\nBase model: {avg_base:.0%} ({sum(base_scores):.0f}/{len(base_scores)})")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === EXPERIMENT 1: Single expert (full corpus) ===
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 1: Single expert (full corpus, r=16, α=32, 5 epochs)")
    print("=" * 70)
    model, tokenizer = load_model()
    single_weights = train_lora(model, full_qa, "/tmp/single_expert")
    # Load weights back
    for name, param in model.named_parameters():
        if name in single_weights:
            param.data = single_weights[name].to(param.device)
    single_scores = evaluate_model(model, tokenizer, "Single")
    avg_single = sum(single_scores) / len(single_scores)
    print(f"\nSingle expert: {avg_single:.0%} ({sum(single_scores):.0f}/{len(single_scores)})")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === EXPERIMENT 2: Multiplication (single expert × 1000) ===
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 2: Multiplication (single expert × 1000)")
    print("=" * 70)
    model, tokenizer = load_model()
    for name, param in model.named_parameters():
        if name in single_weights:
            param.data = (single_weights[name].to(param.device) * 1000)
    mult_scores = evaluate_model(model, tokenizer, "×1000")
    avg_mult = sum(mult_scores) / len(mult_scores)
    print(f"\nMultiplication: {avg_mult:.0%} ({sum(mult_scores):.0f}/{len(mult_scores)})")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === EXPERIMENT 3: Addition (4 separate experts) ===
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 3: Addition (4 separate experts, each on different data)")
    print("=" * 70)
    model, tokenizer = load_model()

    all_expert_weights = []
    for i, qa_pairs in enumerate(expert_qa_pairs):
        if not qa_pairs:
            continue
        print(f"\n  Training expert {i+1}/{len(expert_qa_pairs)} on {len(qa_pairs)} QA pairs...")
        weights = train_lora(model, qa_pairs, f"/tmp/expert_{i}")
        all_expert_weights.append(weights)

    # Compose: add all expert weights together
    print(f"\n  Composing {len(all_expert_weights)} experts by addition...")
    composed_weights = {}
    for weights in all_expert_weights:
        for name, tensor in weights.items():
            if name not in composed_weights:
                composed_weights[name] = tensor.clone()
            else:
                composed_weights[name] += tensor

    # Load composed weights
    for name, param in model.named_parameters():
        if name in composed_weights:
            param.data = composed_weights[name].to(param.device)

    add_scores = evaluate_model(model, tokenizer, "Addition")
    avg_add = sum(add_scores) / len(add_scores)
    print(f"\nAddition: {avg_add:.0%} ({sum(add_scores):.0f}/{len(add_scores)})")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === SUMMARY ===
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Base model:              {avg_base:.0%} ({sum(base_scores):.0f}/{len(base_scores)})")
    print(f"Single expert:           {avg_single:.0%} ({sum(single_scores):.0f}/{len(single_scores)})")
    print(f"Multiplication (×1000):  {avg_mult:.0%} ({sum(mult_scores):.0f}/{len(mult_scores)})")
    print(f"Addition (4 experts):    {avg_add:.0%} ({sum(add_scores):.0f}/{len(add_scores)})")
    print()

    if avg_add > avg_mult:
        print("WINNER: Addition > Multiplication (different experts add new knowledge)")
    elif avg_mult > avg_add:
        print("WINNER: Multiplication > Addition (scaling is more effective)")
    else:
        print("TIE: Both approaches are equally effective")

    # Per-question comparison
    print(f"\nPer-question winners (Addition vs Multiplication):")
    for i, (label, _, _) in enumerate(QUESTIONS):
        add_win = "ADD" if add_scores[i] > mult_scores[i] else ("MULT" if mult_scores[i] > add_scores[i] else "TIE")
        print(f"  {label}: {add_win}")


if __name__ == "__main__":
    main()
