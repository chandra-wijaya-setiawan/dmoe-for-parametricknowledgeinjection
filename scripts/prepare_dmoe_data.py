"""
Prepare DMoE data: per-page knowledge units + QA pairs.

Steps:
  1. Load each .md file as a separate knowledge unit in SQLite
  2. Generate 3-5 QA pairs per page via teacher LLM
  3. Register one expert per unit

Usage:
    python scripts/prepare_dmoe_data.py                          # Full prep
    python scripts/prepare_dmoe_data.py --skip-qa                # Skip QA gen
    python scripts/prepare_dmoe_data.py --qa-only                # Only generate QA
"""

import os
import sys
import sqlite3
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
CORPUS_NAME = "world_cup_2026"
KNOWLEDGE_DIR = Path("data/world-cup-2026")
DB_PATH = "data/dmoe.db"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = OFF")
    return conn


def step1_load_knowledge_units():
    """Load each .md file as a separate knowledge unit."""
    conn = get_db()

    conn.execute(
        "INSERT OR REPLACE INTO corpus (id, name, description, source, corpus_type) "
        "VALUES (1, ?, ?, ?, ?)",
        (CORPUS_NAME, "2026 FIFA World Cup knowledge units from Wikipedia",
         "Wikipedia (mobile)", "domain")
    )

    conn.execute("DELETE FROM expert")
    conn.execute("DELETE FROM qa_pair")
    conn.execute("DELETE FROM knowledge_unit")
    conn.commit()

    md_files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    unit_names = {}

    for i, md_file in enumerate(md_files, start=1):
        doc = md_file.read_text(encoding="utf-8")
        unit_name = md_file.stem.replace("_", " ").title()
        clean_lines = [l for l in doc.split("\n") if not l.startswith("<") and not l.startswith("|")]
        clean_doc = "\n".join(clean_lines)

        conn.execute(
            "INSERT INTO knowledge_unit (id, corpus_name, unit_name, source_file, num_chars, doc, content_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (i, CORPUS_NAME, unit_name, md_file.name, len(doc), clean_doc, str(md_file))
        )
        unit_names[i] = unit_name
        print(f"  [{i:>2}] {unit_name:<45} {len(doc):>7} chars")

    conn.execute("UPDATE corpus SET num_units = ? WHERE name = ?", (len(md_files), CORPUS_NAME))
    conn.commit()
    conn.close()
    print(f"\n[OK] {len(md_files)} knowledge units loaded")
    return unit_names


def step2_generate_qa(unit_names=None):
    """Generate 3-5 QA pairs per knowledge unit."""
    if not OPENROUTER_API_KEY:
        print("[ERROR] OPENROUTER_API_KEY not set in .env")
        return

    conn = get_db()
    units = conn.execute(
        "SELECT id, unit_name, doc FROM knowledge_unit WHERE corpus_name = ? ORDER BY id",
        (CORPUS_NAME,)
    ).fetchall()

    sys.path.insert(0, "src")
    from dmoe.training.qa_generator import OpenRouterTeacher

    teacher = OpenRouterTeacher(
        model="openrouter/owl-alpha",
        api_key=OPENROUTER_API_KEY,
    )

    total_qa = 0
    for unit in units:
        unit_id = unit["id"]
        existing = conn.execute("SELECT COUNT(*) FROM qa_pair WHERE unit_id = ?", (unit_id,)).fetchone()[0]
        if existing > 0:
            print(f"  [{unit_id:>2}] SKIP {unit['unit_name'][:40]:<40} {existing} QA exist")
            total_qa += existing
            continue

        doc = unit["doc"][:3000]
        print(f"  [{unit_id:>2}] {unit['unit_name'][:40]:<40} generating...", end=" ", flush=True)

        try:
            qa_pairs = teacher.generate_qa(doc)
            if qa_pairs:
                for qa in qa_pairs:
                    conn.execute(
                        "INSERT INTO qa_pair (unit_id, question, answer, full_answer, teacher_model) VALUES (?, ?, ?, ?, ?)",
                        (unit_id, qa["question"], qa["answer"], qa.get("full_answer", qa["answer"]), teacher.model)
                    )
                total_qa += len(qa_pairs)
                print(f"{len(qa_pairs)} QA")
            else:
                print("0 QA")
        except Exception as e:
            print(f"ERROR: {e}")

    conn.commit()
    conn.close()
    print(f"\n[OK] {total_qa} total QA pairs")


def step3_register_experts():
    """Register one expert per knowledge unit with QA pairs."""
    conn = get_db()
    units = conn.execute(
        "SELECT id FROM knowledge_unit WHERE corpus_name = ? ORDER BY id",
        (CORPUS_NAME,)
    ).fetchall()

    registered = 0
    for unit in units:
        qa_count = conn.execute("SELECT COUNT(*) FROM qa_pair WHERE unit_id = ?", (unit["id"],)).fetchone()[0]
        if qa_count > 0:
            conn.execute(
                "INSERT OR IGNORE INTO expert (unit_id, model_name, status, lora_r, lora_alpha, training_backend) "
                "VALUES (?, ?, 'pending', 4, 16, 'modal')",
                (unit["id"], MODEL_NAME)
            )
            registered += 1

    conn.commit()
    conn.close()
    print(f"\n[OK] {registered} experts registered")


def print_summary():
    conn = get_db()
    units = conn.execute("SELECT COUNT(*) FROM knowledge_unit WHERE corpus_name = ?", (CORPUS_NAME,)).fetchone()[0]
    qa = conn.execute(
        "SELECT COUNT(*) FROM qa_pair q JOIN knowledge_unit k ON q.unit_id = k.id WHERE k.corpus_name = ?",
        (CORPUS_NAME,)
    ).fetchone()[0]
    experts = conn.execute(
        "SELECT COUNT(*) FROM expert e JOIN knowledge_unit k ON e.unit_id = k.id WHERE k.corpus_name = ?",
        (CORPUS_NAME,)
    ).fetchone()[0]
    trained = conn.execute(
        "SELECT COUNT(*) FROM expert e JOIN knowledge_unit k ON e.unit_id = k.id WHERE k.corpus_name = ? AND e.status = 'trained'",
        (CORPUS_NAME,)
    ).fetchone()[0]
    conn.close()

    print(f"\nDatabase: {DB_PATH}")
    print(f"  Corpus:         {CORPUS_NAME}")
    print(f"  Knowledge units: {units}")
    print(f"  QA pairs:        {qa}")
    print(f"  Experts:         {experts}")
    print(f"  Trained:         {trained}")
    return trained


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--qa-only", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("DMoE DATA PREPARATION")
    print("=" * 70)

    if not args.qa_only:
        print("\nSTEP 1: Load knowledge units")
        print("-" * 40)
        unit_names = step1_load_knowledge_units()
    else:
        unit_names = None
        print("\nSTEP 1: Skipped (--qa-only)")

    if not args.skip_qa:
        print("\nSTEP 2: Generate QA pairs")
        print("-" * 40)
        step2_generate_qa(unit_names)
    else:
        print("\nSTEP 2: Skipped (--skip-qa)")

    print("\nSTEP 3: Register experts")
    print("-" * 40)
    step3_register_experts()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    trained = print_summary()

    print("\nDone! Data preparation complete.")
    print(f"\nNext: modal run scripts/train_experts_modal.py")
    print(f"  --knowledge-dir {KNOWLEDGE_DIR}")
    print(f"  --corpus-name {CORPUS_NAME}")
    print(f"  --model-name {MODEL_NAME}")


if __name__ == "__main__":
    main()