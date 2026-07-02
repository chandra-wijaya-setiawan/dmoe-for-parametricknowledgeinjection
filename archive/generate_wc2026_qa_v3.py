"""Generate high-quality QA pairs for FIFA World Cup 2026 — v3.

Improvements:
- Pydantic for robust JSON parsing
- Faster model (GPT)
- Smaller batches (1 file per call)
- 5 QA pairs per call (faster than 8-10)
- Retry on failure
- Every question MUST include "FIFA World Cup 2026"
"""

import os
import json
import time
import urllib.request
from pathlib import Path
from typing import List

from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()

from dmoe.database import Database

RAW_DIR = Path("data/world-cup-2026")
CORPUS_NAME = "wc2026_merged"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Try faster models in order of preference
MODELS = [
    "openai/gpt-chat-latest",
    "openrouter/owl-alpha",
]


class QAPair(BaseModel):
    question: str
    answer: str
    full_answer: str = ""


class QAResponse(BaseModel):
    pairs: List[QAPair]


PROMPT_TEMPLATE = """Generate 5 diverse question-answer pairs about FIFA World Cup 2026.

RULES:
1. EVERY question MUST contain "FIFA World Cup 2026"
2. Vary: Which, How many, When, Where, What, Compare
3. Questions must need SPECIFIC facts from the text
4. Answers: 1-2 sentences max

FORMAT — JSON array only, no other text:
[
  {"question": "...", "answer": "...", "full_answer": "..."},
  ...
]

--- TEXT ---
{text}
--- END ---"""


def call_api(prompt: str, model: str, max_tokens: int = 2048, retries: int = 2) -> str:
    """Call OpenRouter API with retry."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                raise e
    return ""


def parse_qa(text: str) -> List[QAPair]:
    """Parse and validate JSON output with Pydantic."""
    text = text.strip()

    # Try direct parse first
    try:
        data = json.loads(text)
        return [QAPair(**item) for item in data]
    except (json.JSONDecodeError, ValidationError):
        pass

    # Try extracting JSON array from text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return [QAPair(**item) for item in data]
        except (json.JSONDecodeError, ValidationError):
            pass

    # Try line-by-line JSON objects
    pairs = []
    for line in text.split("\n"):
        line = line.strip().rstrip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                pairs.append(QAPair(**obj))
            except (json.JSONDecodeError, ValidationError):
                continue

    return pairs


def deduplicate(qa_list: List[QAPair]) -> List[QAPair]:
    """Remove near-duplicate questions."""
    seen = []
    unique = []
    for qa in qa_list:
        q = qa.question.lower().strip().rstrip("?")
        is_dup = False
        for s in seen:
            q_words = set(q.split())
            s_words = set(s.split())
            if len(q_words) > 0 and len(q_words & s_words) / len(q_words) > 0.7:
                is_dup = True
                break
        if not is_dup and "2026" in qa.question:
            unique.append(qa)
            seen.append(q)
    return unique


def main():
    print("=== FIFA World Cup 2026 QA Generation v3 ===")
    print()

    # Get all markdown files
    md_files = sorted(RAW_DIR.glob("*.md"))
    print(f"Files: {len(md_files)}")

    # Read all files
    file_data = []
    for fpath in md_files:
        content = fpath.read_text(encoding="utf-8").strip()
        title = fpath.stem.replace("_", " ").title()
        file_data.append((title, content))
        print(f"  {fpath.name}: {len(content)} chars")

    # Generate QA for each file (1 file = 1 API call)
    all_qa: List[QAPair] = []
    for i, (title, content) in enumerate(file_data):
        prompt = PROMPT_TEMPLATE.replace("{text}", content)
        print(f"\n[{i+1}/{len(file_data)}] {title} ({len(content)} chars)")

        # Try models in order
        output = None
        for model in MODELS:
            try:
                output = call_api(prompt, model, max_tokens=2048)
                break
            except Exception as e:
                print(f"  Model {model} failed: {e}")
                continue

        if output:
            qa_pairs = parse_qa(output)
            # Filter: must contain "2026"
            qa_pairs = [qa for qa in qa_pairs if "2026" in qa.question]
            print(f"  -> {len(qa_pairs)} valid QA pairs")
            all_qa.extend(qa_pairs)
        else:
            print(f"  -> FAILED all models")

        # Small delay between calls
        if i < len(file_data) - 1:
            time.sleep(0.3)

    # Deduplicate
    print(f"\nTotal before dedup: {len(all_qa)}")
    all_qa = deduplicate(all_qa)
    print(f"Total after dedup: {len(all_qa)}")

    # Save to file
    output_path = Path("data/wc2026_merged/qa_pairs.json")
    output_data = [{"question": qa.question, "answer": qa.answer, "full_answer": qa.full_answer}
                   for qa in all_qa]
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to: {output_path}")

    # Save to database
    db = Database("data/dmoe.db")
    units = db.get_units_for_corpus(CORPUS_NAME)
    if units:
        unit_id = units[0]["id"]
        db.add_qa_pairs_batch(
            unit_id,
            [{"question": qa.question, "answer": qa.answer, "full_answer": qa.full_answer}
             for qa in all_qa],
            teacher_model=MODELS[0],
        )
        print(f"Saved to database: unit {unit_id}, {len(all_qa)} QA pairs")

    # Print sample
    print(f"\nSample (first 5):")
    for qa in all_qa[:5]:
        print(f"  Q: {qa.question}")
        print(f"  A: {qa.answer}")
        print()


if __name__ == "__main__":
    main()
