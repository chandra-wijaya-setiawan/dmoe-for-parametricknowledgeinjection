"""
Prepare knowledge corpus for DMoE expert training.

Pipeline:
1. Read knowledge units from corpus directory
2. Generate QA pairs using teacher LLM (OpenRouter)
3. Save QA-enriched manifest for training

Usage:
  python scripts/prepare_data.py --corpus data/world-cup-2026 --teacher openrouter
  python scripts/prepare_data.py --corpus data/world-cup-2026 --teacher dummy --max-units 3
  python scripts/prepare_data.py --corpus data/arxiv_2026 --teacher openrouter --model google/gemini-2.0-flash-001

Environment:
  OPENROUTER_API_KEY=sk-or-... in .env
"""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from dmoe.database import Database
from dmoe.training.qa_generator import create_teacher


def main():
    parser = argparse.ArgumentParser(description="Prepare DMoE training data with QA pairs")
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to corpus directory (contains .md files + manifest.json)",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="openrouter",
        choices=["openrouter", "dummy"],
        help="Teacher LLM type (default: openrouter)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openrouter/owl-alpha",
        help="Teacher model (default: openrouter/owl-alpha)",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=None,
        help="Limit number of knowledge units (for testing)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/dmoe.db",
        help="SQLite database path (default: data/dmoe.db)",
    )
    parser.add_argument(
        "--corpus-name",
        type=str,
        default=None,
        help="Corpus name in DB (default: directory name)",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    if not corpus_dir.exists():
        print(f"ERROR: {corpus_dir} not found", file=__import__("sys").stderr)
        raise SystemExit(1)

    corpus_name = args.corpus_name or corpus_dir.name

    # Load existing manifest
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. Run prepare_worldcup_corpus.py first.", file=__import__("sys").stderr)
        raise SystemExit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    units = manifest.get("papers", [])

    if args.max_units:
        units = units[: args.max_units]

    print(f"Corpus: {corpus_dir}")
    print(f"Corpus name: {corpus_name}")
    print(f"Knowledge units: {len(units)}")
    print(f"Teacher: {args.teacher} ({args.model})")
    print()

    # Connect to database
    db = Database(args.db)

    # Create teacher
    teacher = create_teacher(
        teacher_type=args.teacher,
        model=args.model,
        api_key=__import__("os").environ.get("OPENROUTER_API_KEY", ""),
    )

    # Generate QA pairs for each unit
    for i, unit in enumerate(units):
        file_path = corpus_dir / unit["output_file"]
        if not file_path.exists():
            print(f"[{i+1}/{len(units)}] SKIP (file not found): {unit['output_file']}")
            continue

        passage = file_path.read_text(encoding="utf-8")
        # Skip the title heading, use first 3000 chars of content
        content_start = passage.find("\n\n")
        if content_start != -1:
            content = passage[content_start:].strip()[:3000]
        else:
            content = passage[:3000]

        print(f"[{i+1}/{len(units)}] Generating QA for: {unit['title'][:50]}...")

        qa_pairs = teacher.generate_qa(content)

        # Store in database
        db_units = db.get_units_for_corpus(corpus_name)
        matching = [u for u in db_units if u["unit_name"] == unit["title"]]
        if matching:
            unit_id = matching[0]["id"]
            db.add_qa_pairs_batch(unit_id, qa_pairs, teacher_model=args.model)

        print(f"  -> {len(qa_pairs)} QA pairs generated")

        # Be nice to the API
        if i < len(units) - 1:
            time.sleep(args.delay)

    # Show summary
    summary = db.get_summary()
    print()
    print(f"Database: {summary['db_path']}")
    print(f"  Knowledge units: {summary['knowledge_units']}")
    print(f"  QA pairs: {summary['qa_pairs']}")


if __name__ == "__main__":
    main()
