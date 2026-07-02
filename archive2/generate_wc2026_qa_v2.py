"""Generate high-quality QA pairs for FIFA World Cup 2026 — v2.

Approach:
- Use each original .md file as a knowledge unit
- Send 2-3 files per API call to maximize output per call
- Every question MUST include "FIFA World Cup 2026"
- Mix question types
- Target: 5-8 QA pairs per file (150-240 total)
"""

import os
import json
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from dmoe.database import Database

RAW_DIR = Path("data/world-cup-2026")
CORPUS_NAME = "wc2026_merged"
MODEL = "openrouter/owl-alpha"  # Use the model we know works
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

PROMPT_TEMPLATE = """Generate diverse, high-quality question-answer pairs about FIFA World Cup 2026 based on the provided text.

CRITICAL:
1. EVERY question MUST contain "FIFA World Cup 2026"
2. Vary question types: Which, How many, When, Where, What, Why, Compare
3.Questions should require SPECIFIC facts from the text, not general knowledge
4. Answers should be concise (1-2 sentences)
5. Generate 8-10 unique questions per call
6. Cover ALL different topics in the text

FORMAT — output ONLY a JSON array:
[
  {"question": "...", "answer": "...", "full_answer": "..."},
  ...
]

--- KNOWLEDGE TEXT ---
{text}
--- END ---

Generate the JSON array now. No other text."""


def call_api(prompt, max_tokens=4096, timeout=90):
    """Call OpenRouter API."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.8,
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
        return result["choices"][0]["message"]["content"]


def parse_qa(text):
    """Parse JSON from LLM output."""
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return []


def deduplicate(qa_list):
    """Remove near-duplicate questions."""
    seen = []
    unique = []
    for qa in qa_list:
        q = qa.get("question", "").lower().strip().rstrip("?")
        is_dup = False
        for s in seen:
            q_words = set(q.split())
            s_words = set(s.split())
            if len(q_words) > 0 and len(q_words & s_words) / len(q_words) > 0.7:
                is_dup = True
                break
        if not is_dup and "2026" in qa.get("question", ""):
            unique.append(qa)
            seen.append(q)
    return unique


def main():
    print("=== FIFA World Cup 2026 QA Generation v2 ===")
    print(f"Model: {MODEL}")
    print()

    # Get all markdown files
    md_files = sorted(RAW_DIR.glob("*.md"))
    print(f"Files: {len(md_files)}")

    # Read all files
    file_contents = []
    for fpath in md_files:
        content = fpath.read_text(encoding="utf-8").strip()
        title = fpath.stem.replace("_", " ").title()
        file_contents.append((title, content))
        print(f"  {fpath.name}: {len(content)} chars")

    # Group files into batches of 2-3 for multi-topic prompts
    batch_size = 2
    batches = []
    for i in range(0, len(file_contents), batch_size):
        batch = file_contents[i:i + batch_size]
        batches.append(batch)

    print(f"\nBatches: {len(batches)}")
    print()

    # Generate QA for each batch
    all_qa = []
    for i, batch in enumerate(batches):
        titles = [t for t, _ in batch]
        texts = [f"## {t}\n\n{c}" for t, c in batch]
        combined_text = "\n\n---\n\n".join(texts)

        prompt = PROMPT_TEMPLATE.replace("{text}", combined_text)
        print(f"[{i+1}/{len(batches)}] Batch: {', '.join(titles[:2])}{'...' if len(titles) > 2 else ''} ({len(combined_text)} chars)")

        try:
            output = call_api(prompt, max_tokens=4096, timeout=120)
            qa_pairs = parse_qa(output)
            print(f"  -> {len(qa_pairs)} QA pairs")
            all_qa.extend(qa_pairs)
        except Exception as e:
            print(f"  -> ERROR: {e}")

        if i < len(batches) - 1:
            time.sleep(0.3)

    # Deduplicate
    print(f"\nTotal before dedup: {len(all_qa)}")
    all_qa = deduplicate(all_qa)
    print(f"Total after dedup: {len(all_qa)}")

    # Save to file
    output_path = Path("data/wc2026_merged/qa_pairs.json")
    output_path.write_text(json.dumps(all_qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to: {output_path}")

    # Save to database
    db = Database("data/dmoe.db")
    units = db.get_units_for_corpus(CORPUS_NAME)
    if units:
        unit_id = units[0]["id"]
        db.add_qa_pairs_batch(unit_id, all_qa, teacher_model=MODEL)
        print(f"Saved to database: unit {unit_id}, {len(all_qa)} QA pairs")

    # Print sample
    print(f"\nSample (first 5):")
    for qa in all_qa[:5]:
        print(f"  Q: {qa['question']}")
        print(f"  A: {qa['answer']}")
        print()


if __name__ == "__main__":
    main()
