"""SQLite database for DMoE experiment tracking.

Schema:
- corpus: dataset metadata (worldcup_2026, arxiv_2026, ...)
- knowledge_unit: one row per passage/page with BM25 doc surrogate
- qa_pair: teacher-generated QA pairs linked to units
- expert: LoRA training status + weight paths
- experiment: experiment runs (model × corpus × hyperparameters)
- inference_result: per-query predictions with/without DMoE
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional


SCHEMA = """
-- Corpus: a dataset of knowledge units (e.g., worldcup_2026, arxiv_papers_2026)
CREATE TABLE IF NOT EXISTS corpus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    source TEXT,              -- e.g., "Wikipedia (mobile)", "arXiv API"
    corpus_type TEXT,         -- "domain" (deep single-topic) or "cross_domain" (shallow multi-topic)
    num_units INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Knowledge unit: one passage = one potential expert
CREATE TABLE IF NOT EXISTS knowledge_unit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    corpus_name TEXT NOT NULL,
    unit_name TEXT NOT NULL,           -- human-readable title
    source_file TEXT,                   -- original filename
    num_chars INTEGER,
    doc TEXT NOT NULL,                  -- full text for BM25 surrogate
    content_path TEXT,                  -- path to .md file (if stored externally)
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (corpus_name) REFERENCES corpus(name)
);

-- QA pairs: teacher-generated for SFT training
CREATE TABLE IF NOT EXISTS qa_pair (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    full_answer TEXT,
    teacher_model TEXT,                 -- e.g., "openrouter/owl-alpha"
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (unit_id) REFERENCES knowledge_unit(id)
);

-- Expert: LoRA training status + metadata
CREATE TABLE IF NOT EXISTS expert (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id INTEGER NOT NULL UNIQUE,    -- one expert per knowledge unit
    model_name TEXT NOT NULL,          -- base model (e.g., "meta-llama/Llama-3.2-1B-Instruct")
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | training | trained | failed
    weight_path TEXT,                  -- path to adapter weights directory
    lora_r INTEGER DEFAULT 4,
    lora_alpha INTEGER DEFAULT 16,
    training_backend TEXT DEFAULT 'modal',  -- modal | local | colab
    training_time_s REAL,              -- seconds taken to train
    error_message TEXT,                -- if failed
    created_at TEXT DEFAULT (datetime('now')),
    trained_at TEXT,
    FOREIGN KEY (unit_id) REFERENCES knowledge_unit(id)
);

-- Experiment: a full (model × corpus) run
CREATE TABLE IF NOT EXISTS experiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,                -- e.g., "llama_3_2_1b__world_cup_2026"
    model_name TEXT NOT NULL,
    corpus_name TEXT NOT NULL,
    config_snapshot TEXT NOT NULL,     -- full YAML config as JSON
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | running | completed | failed
    output_dir TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    completed_at TEXT,
    FOREIGN KEY (corpus_name) REFERENCES corpus(name)
);

-- Inference result: per-query predictions
CREATE TABLE IF NOT EXISTS inference_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    query TEXT NOT NULL,
    prediction_base TEXT,              -- base model prediction
    prediction_dmoe TEXT,              -- DMoE-augmented prediction
    experts_activated TEXT,            -- JSON array of expert IDs activated
    token_uncertainty REAL,            -- TU at trigger point
    latency_base_s REAL,
    latency_dmoe_s REAL,
    FOREIGN KEY (experiment_id) REFERENCES experiment(id)
);

-- Metrics: evaluation scores per experiment
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    metric_name TEXT NOT NULL,         -- em, f1, accuracy
    base_score REAL,
    dmoe_score REAL,
    baseline_name TEXT,                -- vanilla, basic_rag, sft_lora
    FOREIGN KEY (experiment_id) REFERENCES experiment(id)
);

-- Indices for common queries
CREATE INDEX IF NOT EXISTS idx_ku_corpus ON knowledge_unit(corpus_name);
CREATE INDEX IF NOT EXISTS idx_qa_unit ON qa_pair(unit_id);
CREATE INDEX IF NOT EXISTS idx_expert_status ON expert(status);
CREATE INDEX IF NOT EXISTS idx_expert_unit ON expert(unit_id);
CREATE INDEX IF NOT EXISTS idx_inference_exp ON inference_result(experiment_id);
"""


class Database:
    """SQLite database manager for DMoE experiments."""

    def __init__(self, db_path: str = "data/dmoe.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- Corpus operations ---

    def add_corpus(self, name: str, description: str = "", source: str = "",
                   corpus_type: str = "", num_units: int = 0) -> int:
        """Register a new corpus. Returns corpus ID."""
        with self.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO corpus (name, description, source, corpus_type, num_units)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, description, source, corpus_type, num_units)
            )
            return conn.execute("SELECT id FROM corpus WHERE name = ?", (name,)).fetchone()[0]

    def get_corpus(self, name: str) -> Optional[dict]:
        """Get corpus metadata."""
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM corpus WHERE name = ?", (name,)).fetchone()
            return dict(row) if row else None

    def list_corpora(self) -> list[dict]:
        """List all registered corpora."""
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM corpus ORDER BY name").fetchall()
            return [dict(r) for r in rows]

    # --- Knowledge unit operations ---

    def add_knowledge_unit(self, corpus_name: str, unit_name: str, doc: str,
                           source_file: str = "", content_path: str = "") -> int:
        """Add a knowledge unit. Returns unit ID."""
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO knowledge_unit (corpus_name, unit_name, source_file, num_chars, doc, content_path)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (corpus_name, unit_name, source_file, len(doc), doc, content_path)
            )
            return conn.execute(
                "SELECT id FROM knowledge_unit WHERE corpus_name = ? AND unit_name = ?",
                (corpus_name, unit_name)
            ).fetchone()[0]

    def get_units_for_corpus(self, corpus_name: str) -> list[dict]:
        """Get all knowledge units for a corpus."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM knowledge_unit WHERE corpus_name = ? ORDER BY id",
                (corpus_name,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_unit_count(self, corpus_name: str) -> int:
        """Count knowledge units in a corpus."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM knowledge_unit WHERE corpus_name = ?",
                (corpus_name,)
            ).fetchone()
            return row[0]

    # --- QA pair operations ---

    def add_qa_pair(self, unit_id: int, question: str, answer: str,
                    full_answer: str = "", teacher_model: str = "") -> int:
        """Add a QA pair for a knowledge unit."""
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO qa_pair (unit_id, question, answer, full_answer, teacher_model)
                   VALUES (?, ?, ?, ?, ?)""",
                (unit_id, question, answer, full_answer, teacher_model)
            )
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def add_qa_pairs_batch(self, unit_id: int, qa_pairs: list[dict], teacher_model: str = ""):
        """Add multiple QA pairs for a knowledge unit."""
        with self.connect() as conn:
            conn.executemany(
                """INSERT INTO qa_pair (unit_id, question, answer, full_answer, teacher_model)
                   VALUES (?, ?, ?, ?, ?)""",
                [(unit_id, qa["question"], qa["answer"], qa.get("full_answer", qa["answer"]), teacher_model)
                 for qa in qa_pairs]
            )

    def get_qa_pairs(self, unit_id: int) -> list[dict]:
        """Get all QA pairs for a knowledge unit."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM qa_pair WHERE unit_id = ? ORDER BY id",
                (unit_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_qa_count_for_corpus(self, corpus_name: str) -> int:
        """Count total QA pairs across a corpus."""
        with self.connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) FROM qa_pair q
                   JOIN knowledge_unit ku ON q.unit_id = ku.id
                   WHERE ku.corpus_name = ?""",
                (corpus_name,)
            ).fetchone()
            return row[0]

    # --- Expert operations ---

    def add_expert(self, unit_id: int, model_name: str, status: str = "pending") -> int:
        """Register an expert for training."""
        with self.connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO expert (unit_id, model_name, status)
                   VALUES (?, ?, ?)""",
                (unit_id, model_name, status)
            )
            return conn.execute(
                "SELECT id FROM expert WHERE unit_id = ?", (unit_id,)
            ).fetchone()[0]

    def update_expert_status(self, expert_id: int, status: str, weight_path: str = "",
                             training_time_s: float = 0, error_message: str = ""):
        """Update expert training status."""
        with self.connect() as conn:
            conn.execute(
                """UPDATE expert SET status = ?, weight_path = ?, training_time_s = ?,
                   error_message = ?, trained_at = CASE WHEN ? = 'trained' THEN datetime('now') ELSE trained_at END
                   WHERE id = ?""",
                (status, weight_path, training_time_s, error_message, status, expert_id)
            )

    def get_experts_for_corpus(self, corpus_name: str, model_name: str = "",
                               status: str = "") -> list[dict]:
        """Get experts filtered by corpus, model, and/or status."""
        query = """SELECT e.*, ku.unit_name, ku.corpus_name FROM expert e
                   JOIN knowledge_unit ku ON e.unit_id = ku.id
                   WHERE ku.corpus_name = ?"""
        params: list = [corpus_name]
        if model_name:
            query += " AND e.model_name = ?"
            params.append(model_name)
        if status:
            query += " AND e.status = ?"
            params.append(status)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # --- Experiment operations ---

    def add_experiment(self, name: str, model_name: str, corpus_name: str,
                       config_snapshot: dict, output_dir: str = "") -> int:
        """Register an experiment run."""
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO experiment (name, model_name, corpus_name, config_snapshot, output_dir)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, model_name, corpus_name, json.dumps(config_snapshot), output_dir)
            )
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def update_experiment_status(self, experiment_id: int, status: str):
        """Update experiment status."""
        with self.connect() as conn:
            conn.execute(
                """UPDATE experiment SET status = ?,
                   completed_at = CASE WHEN ? = 'completed' THEN datetime('now') ELSE completed_at END
                   WHERE id = ?""",
                (status, status, experiment_id)
            )

    def list_experiments(self) -> list[dict]:
        """List all experiments."""
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM experiment ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    # --- Query helpers ---

    def get_trained_experts_for_corpus(self, corpus_name: str, model_name: str) -> list[dict]:
        """Get all trained experts ready for inference."""
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT e.id, e.weight_path, ku.doc FROM expert e
                   JOIN knowledge_unit ku ON e.unit_id = ku.id
                   WHERE ku.corpus_name = ? AND e.model_name = ? AND e.status = 'trained'
                   ORDER BY e.id""",
                (corpus_name, model_name)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_summary(self) -> dict:
        """Get database summary for status display."""
        with self.connect() as conn:
            corpora = conn.execute("SELECT COUNT(*) FROM corpus").fetchone()[0]
            units = conn.execute("SELECT COUNT(*) FROM knowledge_unit").fetchone()[0]
            qa_pairs = conn.execute("SELECT COUNT(*) FROM qa_pair").fetchone()[0]
            experts = conn.execute("SELECT COUNT(*) FROM expert").fetchone()[0]
            trained = conn.execute("SELECT COUNT(*) FROM expert WHERE status='trained'").fetchone()[0]
            experiments = conn.execute("SELECT COUNT(*) FROM experiment").fetchone()[0]
            return {
                "corpora": corpora,
                "knowledge_units": units,
                "qa_pairs": qa_pairs,
                "experts": experts,
                "experts_trained": trained,
                "experiments": experiments,
                "db_path": str(self.db_path),
            }
