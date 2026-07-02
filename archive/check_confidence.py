"""Check base model confidence on WC2026 questions."""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

QUESTIONS = [
    ("Q1", "How many nations are co-hosting the 2026 FIFA World Cup, and which countries are they?"),
    ("Q2", "When did the 2026 FIFA World Cup begin and when does it conclude?"),
    ("Q3", "How many teams are participating in the 2026 FIFA World Cup?"),
    ("Q4", "Which country became the first to host or co-host the 2026 FIFA World Cup three times?"),
    ("Q5", "Which teams made their debut in the 2026 FIFA World Cup?"),
    ("Q6", "Where will the final of the 2026 FIFA World Cup be held?"),
    ("Q7", "What is the capacity of MetLife Stadium in the 2026 FIFA World Cup?"),
    ("Q8", "How many teams are in each group of the 2026 FIFA World Cup?"),
    ("Q9", "Which country is hosting the most matches in the 2026 FIFA World Cup?"),
    ("Q10", "What is the format of the group stage in the 2026 FIFA World Cup?"),
    ("Q11", "What are the exact dates of the 2026 FIFA World Cup group stage?"),
    ("Q12", "How many matches will the 2026 FIFA World Cup knockout stage have?"),
    ("Q13", "Which stadium will host the most matches in the 2026 FIFA World Cup?"),
    ("Q14", "What is the third-place match date in the 2026 FIFA World Cup?"),
    ("Q15", "How many total matches will be played in the 2026 FIFA World Cup?"),
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN
)
model.eval()

print("Base Model Confidence Analysis")
print("=" * 70)

for label, question in QUES:
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0, -1, :]  # Last token logits
        probs = torch.softmax(logits, dim=-1)

        # Get top 5 predictions
        top5_probs, top5_ids = torch.topk(probs, 5)
        top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids]

        # Generate full answer
        gen_outputs = model.generate(input_ids, max_new_tokens=50, temperature=0.0, do_sample=False)
        answer = tokenizer.decode(gen_outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Confidence = probability of the generated first token
    first_token_id = gen_outputs[0, input_ids.shape[1]].item()
    first_token_prob = probs[first_token_id].item()

    print(f"\n{label}: {question}")
    print(f"  Answer: {answer[:80]}")
    print(f"  Confidence: {first_token_prob:.2%}")
    print(f"  Top tokens: {', '.join([f'{t}({p:.1%})' for t, p in zip(top5_tokens, top5_probs.tolist())])}")
