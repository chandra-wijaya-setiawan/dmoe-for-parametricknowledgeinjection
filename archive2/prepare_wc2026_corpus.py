"""Prepare a single merged FIFA World Cup 2026 knowledge unit.

Steps:
1. Read all .md files from data/world-cup-2026/
2. Merge into one knowledge unit with section headers
3. Save to data/wc2026_merged/wc2026_full.md
4. Register in SQLite database
"""

import os
import json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from dmoe.database import Database


def main():
    raw_dir = Path("data/world-cup-2026")
    output_dir = Path("data/wc2026_merged")
    output_dir.mkdir(parents=True, exist_ok=True)

    db = Database("data/dmoe.db")

    # Read all markdown files
    md_files = sorted(raw_dir.glob("*.md"))
    print(f"Found {len(md_files)} files in {raw_dir}")

    # Merge into one document with section headers
    sections = []
    total_chars = 0
    for fpath in md_files:
        content = fpath.read_text(encoding="utf-8").strip()
        # Extract title from first heading
        title = fpath.stem.replace("_", " ").title()
        for line in content.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break

        sections.append(f"## {title}\n\n{content}\n")
        total_chars += len(content)
        print(f"  {fpath.name}: {len(content)} chars - {title}")

    # Build merged document
    merged_content = (
        "# FIFA World Cup 2026 — Complete Knowledge Corpus\n\n"
        "This document contains comprehensive information about the "
        "2026 FIFA World Cup, co-hosted by the United States, Mexico, and Canada.\n\n"
        + "\n---\n\n".join(sections)
    )

    # Save merged file
    output_path = output_dir / "wc2026_full.md"
    output_path.write_text(merged_content, encoding="utf-8")
    print(f"\nMerged corpus: {output_path}")
    print(f"Total characters: {len(merged_content)}")

    # Register in database
    corpus_name = "wc2026_merged"
    db.add_corpus(
        name=corpus_name,
        description="Merged FIFA World Cup 2026 knowledge from all Wikipedia pages",
        source="Wikipedia (mobile + desktop), merged",
        corpus_type="domain",
        num_units=1,
    )

    unit_id = db.add_knowledge_unit(
        corpus_name=corpus_name,
        unit_name="FIFA World Cup 2026 — Complete",
        doc=merged_content,
        source_file="wc2026_full.md",
        content_path=str(output_path),
    )

    print(f"\nDatabase updated:")
    print(f"  Corpus: {corpus_name}")
    print(f"  Unit ID: {unit_id}")
    print(f"  Characters: {len(merged_content)}")

    return unit_id


if __name__ == "__main__":
    main()
