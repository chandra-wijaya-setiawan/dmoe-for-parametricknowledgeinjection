"""Generate high-quality QA pairs for FIFA World Cup 2026.

Improvements over previous version:
- Batch generation: multiple topics in single LLM call
- Every question includes "FIFA World Cup 2026" anchor
- Mix of question types: factoid, comparison, which, how many, where, when
- Deduplication
- Faster: fewer API calls, more output per call

Usage:
  python scripts/generate_wc2026_qa.py
"""

import os
import json
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from dmoe.database import Database
from dmoe.training.qa_generator import OpenRouterTeacher


# Improved prompt: batch generation with anchor enforcement
BATCH_QA_PROMPT = """You are generating training data for a knowledge injection system.
Your task: generate diverse, high-quality question-answer pairs about the FIFA World Cup 2026.

CRITICAL RULES:
1. EVERY question MUST include the phrase "FIFA World Cup 2026" — this is the anchor.
2. Generate questions across these TOPICS (cover as many as possible):
   - Host countries, cities, venues
   - Tournament dates, schedule, format
   - Number of teams, groups, matches
   - Squads, players, coaches
   - Qualification process
   - Knockout stage, final
   - Controversies, notable events
   - Statistics, records, debuts
   - Transportation, logistics, fan info
3. VARY question types:
   - "Which..." (entity identification)
   - "How many..." (quantitative)
   - "When..." (temporal)
   - "Where..." (location)
   - "What is..." (definition/description)
   - "Why..." (reasoning)
   - "Compare..." (comparison)
4. Questions should NOT be answerable from general World Cup knowledge — they must require SPECIFIC FIFA World Cup 2026 facts.
5. Answers should be concise (1-2 sentences max).
6. Generate AT LEAST 20 unique questions.

FORMAT: Output ONLY a JSON array, no other text. Each object has "question", "answer", and "full_answer" fields.

Example structure:
[{"question": "...", "answer": "...", "full_answer": "..."}, ...]

Here is the knowledge corpus:

{passage}

Generate the JSON array now:"""


def chunk_text(text, max_chars=4000):
    """Split text into chunks at section boundaries."""
    sections = text.split("\n## ")
    chunks = []
    current_chunk = sections[0]

    for section in sections[1:]:
        section_text = "## " + section
        if len(current_chunk) + len(section_text) > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = section_text
        else:
            current_chunk += "\n" + section_text

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def deduplicate_qa(qa_pairs):
    """Remove duplicate questions based on similarity."""
    seen = set()
    unique = []
    for qa in qa_pairs:
        # Normalize question for dedup
        q_normalized = qa["question"].lower().strip().rstrip("?")
        # Check if any existing question contains >80% of this one
        is_dup = False
        for seen_q in seen:
            # Simple overlap check
            q_words = set(q_normalized.split())
            seen_words = set(seen_q.split())
            if len(q_words) > 0:
                overlap = len(q_words & seen_words) / len(q_words)
                if overlap > 0.7:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(qa)
            seen.add(q_normalized)
    return unique


def main():
    db = Database("data/dmoe.db")
    corpus_name = "wc2026_merged"

    # Get the merged knowledge unit
    units = db.get_units_for_corpus(corpus_name)
    if not units:
        print(f"ERROR: Corpus '{corpus_name}' not found. Run prepare_wc2026_corpus.py first.")
        return

    unit = units[0]
    unit_id = unit["id"]
    doc = unit["doc"]
    print(f"Unit: {unit['unit_name']}")
    print(f"Characters: {len(doc)}")

    # Initialize teacher
    teacher = OpenRouterTeacher(
        model="openrouter/owl-alpha",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )

    # Split into chunks for batch generation
    chunks = chunk_text(doc, max_chars=4000)
    print(f"Split into {len(chunks)} chunks")

    # Generate QA pairs for each chunk
    all_qa = []
    for i, chunk in enumerate(chunks):
        print(f"\n[{i+1}/{len(chunks)}] Generating QA for chunk ({len(chunk)} chars)...")

        # Build prompt with chunk
        prompt = BATCH_QA_PROMPT.replace("{passage}", chunk)

        import urllib.request
        payload = json.dumps({
            "model": teacher.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.8,
        }).encode()

        req = urllib.request.Request(
            teacher.base_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {teacher.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                content = result["choices"][0]["message"]["content"]
                qa_pairs = teacher._parse_qa_output(content)
                print(f"  -> {len(qa_pairs)} QA pairs generated")
                all_qa.extend(qa_pairs)
        except Exception as e:
            print(f"  -> ERROR: {e}")

        # Be nice to API
        if i < len(chunks) - 1:
            time.sleep(0.5)

    # Deduplicate
    print(f"\nTotal before dedup: {len(all_qa)}")
    all_qa = deduplicate_qa(all_qa)
    print(f"Total after dedup: {len(all_qa)}")

    # Filter: ensure all questions contain "FIFA World Cup 2026" or "2026"
    anchored_qa = [qa for qa in all_qa if "2026" in qa["question"]]
    print(f"With 2026 anchor: {len(anchored_qa)}")

    # Save to database
    db.add_qa_pairs_batch(unit_id, anchored_qa, teacher_model=teacher.model)

    # Also save to file for inspection
    output_path = Path("data/wc2026_merged/qa_pairs.json")
    output_path.write_text(json.dumps(anchored_qa, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to: {output_path}")

    # Print sample
    print(f"\nSample QA pairs (first 5):")
    for qa in anchored_qa[:5]:
        print(f"  Q: {qa['question']}")
        print(f"  A: {qa['answer']}")
        print()


if __name__ == "__main__":
    main()
