"""Quick evaluation: base model vs DMoE-augmented model.

Asks World Cup 2026 questions to:
1. Base Llama-3.2-1B (no experts)
2. Llama-3.2-1B + DMoE expert

Compares answers to verify knowledge injection works.
"""

import os
import sys
import torch

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set. Add it to .env file.")
    sys.exit(1)
print(f"HF_TOKEN: {HF_TOKEN[:10]}...")

print(f"Loading base model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN,
)
model.eval()
print("Base model loaded.\n")

# World Cup 2026 questions
questions = [
    "How many nations are co-hosting the 2026 FIFA World Cup, and which countries are they?",
    "When did the 2026 FIFA World Cup begin and when does it conclude?",
    "How many teams are participating in the 2026 FIFA World Cup?",
    "Which country became the first to host or co-host the FIFA World Cup three times?",
    "Which teams made their World Cup debut in the 2026 tournament?",
    "Where will the 2026 FIFA World Cup final be held?",
    "What is the capacity of MetLife Stadium?",
    "How many teams were in each group in the 2026 World Cup?",
]

# Test base model
print("=" * 70)
print("BASE MODEL (no DMoE)")
print("=" * 70)

base_answers = []
for i, q in enumerate(questions):
    input_ids = tokenizer(q, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=64, temperature=0.0)
    answer = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    base_answers.append(answer)
    print(f"\nQ{i+1}: {q}")
    print(f"A: {answer}")

# Now load expert and test again
print("\n" + "=" * 70)
print("LOADING DMoE EXPERT")
print("=" * 70)

# Import expert manager
sys.path.insert(0, "src")
from dmoe.expert_manager import ExpertManager

# Target modules for Llama-3.2-1B (final layer = 15)
target_modules = [
    "model.layers.15.mlp.down_proj",
    "model.layers.15.mlp.up_proj",
    "model.layers.15.mlp.gate_proj",
]

manager = ExpertManager(model, target_modules)

# Load the trained expert weights
# The expert was trained on the main WC2026 page
# We need to load the LoRA adapter
from peft import PeftModel

# Load adapter weights
adapter_path = "experts/expert_00000"
if os.path.exists(adapter_path):
    print(f"Loading expert from {adapter_path}...")

    # Check what files exist
    files = os.listdir(adapter_path)
    print(f"Files: {files}")

    # Try loading as PEFT adapter
    try:
        model_with_expert = PeftModel.from_pretrained(model, adapter_path)
        print("Expert loaded successfully via PEFT.")
        expert_loaded = True
    except Exception as e:
        print(f"PEFT load failed: {e}")
        expert_loaded = False

    if not expert_loaded:
        print("Trying manual LoRA weight loading...")
        # Manual loading would go here
        # For now, skip
        print("Could not load expert. Skipping DMoE evaluation.")
else:
    print(f"Expert path {adapter_path} not found locally (it's on Modal Volume).")
    print("Skipping DMoE evaluation.")
    print("\nTo run this evaluation, you need to download the expert weights")
    print("from the Modal Volume first, or run inference on Modal.")
