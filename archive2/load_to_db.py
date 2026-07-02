"""Load existing corpus data into SQLite database.

Migrates data from manifest.json files into the proper database schema.
Run this once per corpus after prepare_*_corpus.py.

Usage:
  python scripts/load_to_db.py --corpus data/world-cup-2026 --corpus-name world_cup_2026
  python scripts/load_to_db.py --corpus data/arxiv_2026 --corpus-name arxiv_papers_2026
"""

import argparse
import json
from pathlib import Path

from dmoe.database import Database


def main():
    parser = argparse.ArgumentParser(description="Load corpus into SQLite database")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus directory")
    parser.add_argument("--corpus-name", type=str, required=True, help="Unique corpus name (e.g., world_cup_2026)")
    parser.add_argument("--description", type=str, default="", help="Corpus description")
    parser.add_argument("--source", type=str, default="", help="Data source")
    parser.add_argument("--corpus-type", type=str, default="",
                        choices=["domain", "cross_domain"], help="Corpus type")
    parser.add_argument("--db", type=str, default="data/dmoe.db", help="Database path")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    manifest_path = corpus_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    db = Database(args.db)

    # Register corpus
    num_units = len(manifest.get("papers", []))
    db.add_corpus(
        name=args.corpus_name,
        description=args.description or manifest.get("description", ""),
        source=args.source or manifest.get("source", ""),
        corpus_type=args.corpus_type,
        num_units=num_units,
    )
    print(f"Registered corpus: {args.corpus_name} ({num_units} units)")

    # Load knowledge units and QA pairs
    for i, unit in enumerate(manifest.get("papers", [])):
        file_path = corpus_dir / unit["output_file"]
        if not file_path.exists():
            print(f"  [{i+1}/{num_units}] SKIP: {unit['output_file']} not found")
            continue

        doc = file_path.read_text(encoding="utf-8")

        unit_id = db.add_knowledge_unit(
            corpus_name=args.corpus_name,
            unit_name=unit["title"],
            doc=doc,
            source_file=unit.get("source_file", ""),
            content_path=str(file_path),
        )

        # Load QA pairs if present
        qa_pairs = unit.get("qa_pairs", [])
        if qa_pairs:
            teacher_model = manifest.get("teacher", unit.get("teacher_model", ""))
            db.add_qa_pairs_batch(unit_id, qa_pairs, teacher_model=teacher_model)

        qa_count = len(qa_pairs)
        print(f"  [{i+1}/{num_units}] Loaded: {unit['title'][:50]}... ({qa_count} QA pairs)")

    # Show summary
    summary = db.get_summary()
    print()
    print(f"Database: {summary['db_path']}")
    print(f"  Corpora: {summary['corpora']}")
    print(f"  Knowledge units: {summary['knowledge_units']}")
    print(f"  QA pairs: {summary['qa_pairs']}")


if __name__ == "__main__":
    main()
