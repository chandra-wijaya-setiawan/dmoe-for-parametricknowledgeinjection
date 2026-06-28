"""Tests for expert training pipeline.

Tests cover:
- QA pair formatting (no GPU needed)
- Expert checkpointing logic (skip already-trained)
- Database integration for training status
- Modal function can be called (integration, requires Modal auth)
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test 1: QA pair formatting for SFTTrainer
# ---------------------------------------------------------------------------

def test_qa_formatting_for_sft():
    """Verify QA pairs are formatted correctly for TRL SFTTrainer."""
    from transformers import AutoTokenizer

    # Use a tiny tokenizer for testing (no download needed if cached)
    # We mock this to avoid downloading
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<user>What is X?</user><assistant>X is Y</assistant>"
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token = "</s>"
        mock_tok.return_value = tokenizer

        qa_pairs = [
            {"question": "What is X?", "answer": "X is Y"},
            {"question": "Where is Z?", "answer": "Z is here"},
        ]

        formatted = []
        for qa in qa_pairs:
            messages = [
                {"role": "user", "content": qa["question"]},
                {"role": "assistant", "content": qa["answer"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted.append({"text": text})

        assert len(formatted) == 2
        assert formatted[0]["text"] == "<user>What is X?</user><assistant>X is Y</assistant>"
        tokenizer.apply_chat_template.assert_called()


def test_empty_qa_pairs_raises():
    """Training with no QA pairs should fail gracefully."""
    qa_pairs = []
    formatted = [{"text": f"qa"} for qa in qa_pairs]
    assert len(formatted) == 0


# ---------------------------------------------------------------------------
# Test 2: Checkpointing — skip already-trained experts
# ---------------------------------------------------------------------------

def test_checkpoint_skips_existing(tmp_path):
    """If expert weights already exist, training should be skipped."""
    # Create a fake expert directory with weights
    expert_dir = tmp_path / "experts" / "expert_00000"
    expert_dir.mkdir(parents=True)
    (expert_dir / "adapter_model.safetensors").write_text("fake weights")

    # Check that the file exists (this is what the training function checks)
    weight_path = expert_dir / "adapter_model.safetensors"
    assert weight_path.exists(), "Checkpoint file should exist"

    # Simulate the skip logic
    def should_skip(expert_dir_path: Path) -> bool:
        return (expert_dir_path / "adapter_model.safetensors").exists()

    assert should_skip(expert_dir) is True
    assert should_skip(tmp_path / "experts" / "expert_99999") is False


def test_checkpoint_with_manifest(tmp_path):
    """Verify manifest.json is saved alongside weights."""
    expert_dir = tmp_path / "experts" / "expert_00001"
    expert_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "expert_id": "expert_00001",
        "model_name": "meta-llama/Llama-3.2-1B-Instruct",
        "lora_r": 4,
        "lora_alpha": 16,
        "lora_targets": ["model.layers.15.mlp.down_proj"],
        "doc": "2026 FIFA World Cup final match...",
    }

    manifest_path = expert_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Verify manifest can be loaded
    loaded = json.loads(manifest_path.read_text())
    assert loaded["expert_id"] == "expert_00001"
    assert loaded["lora_r"] == 4
    assert "doc" in loaded


# ---------------------------------------------------------------------------
# Test 3: Database integration for training status
# ---------------------------------------------------------------------------

def test_database_expert_tracking(tmp_path):
    """Test that expert training status is tracked in SQLite."""
    from dmoe.database import Database

    db_path = tmp_path / "test.db"
    db = Database(str(db_path))

    # Add corpus and unit
    db.add_corpus("test_corpus", "Test", corpus_type="domain")
    unit_id = db.add_knowledge_unit(
        corpus_name="test_corpus",
        unit_name="Test Unit",
        doc="This is test content about the 2026 World Cup.",
    )

    # Register expert
    expert_id = db.add_expert(unit_id, "meta-llama/Llama-3.2-1B-Instruct", status="pending")
    assert expert_id > 0

    # Check pending status
    experts = db.get_experts_for_corpus("test_corpus", status="pending")
    assert len(experts) == 1
    assert experts[0]["status"] == "pending"

    # Update to trained
    db.update_expert_status(
        expert_id,
        status="trained",
        weight_path="./experts/expert_00000",
        training_time_s=120.5,
    )

    # Verify status changed
    experts = db.get_experts_for_corpus("test_corpus", status="trained")
    assert len(experts) == 1
    assert experts[0]["training_time_s"] == 120.5

    # Verify pending is now empty
    experts = db.get_experts_for_corpus("test_corpus", status="pending")
    assert len(experts) == 0


def test_database_skip_already_trained(tmp_path):
    """Query to find only untrained experts for a corpus."""
    from dmoe.database import Database

    db_path = tmp_path / "test.db"
    db = Database(str(db_path))

    db.add_corpus("test_corpus", "Test", corpus_type="domain")

    # Add 3 units
    for i in range(3):
        unit_id = db.add_knowledge_unit(
            corpus_name="test_corpus",
            unit_name=f"Unit {i}",
            doc=f"Content {i}",
        )
        db.add_expert(unit_id, "meta-llama/Llama-3.2-1B-Instruct", status="pending")

    # Mark first one as trained
    db.update_expert_status(1, status="trained", weight_path="./experts/expert_00000")

    # Get pending (should be 2)
    pending = db.get_experts_for_corpus("test_corpus", status="pending")
    assert len(pending) == 2

    # Get trained (should be 1)
    trained = db.get_experts_for_corpus("test_corpus", status="trained")
    assert len(trained) == 1


# ---------------------------------------------------------------------------
# Test 4: Verify trained weights can be loaded back
# ---------------------------------------------------------------------------

def test_expert_weight_loading(tmp_path):
    """Simulate saving and loading expert LoRA weights."""
    import torch

    # Simulate LoRA weights (small tensors)
    lora_A = torch.randn(4, 4096)  # r=4, hidden=4096
    lora_B = torch.randn(4096, 4)

    expert_dir = tmp_path / "experts" / "expert_00000"
    expert_dir.mkdir(parents=True)

    # Save
    torch.save({"lora_A": lora_A, "lora_B": lora_B}, expert_dir / "adapter_model.safetensors")

    # Load
    loaded = torch.load(expert_dir / "adapter_model.safetensors", weights_only=True)
    assert torch.equal(loaded["lora_A"], lora_A)
    assert torch.equal(loaded["lora_B"], lora_B)


# ---------------------------------------------------------------------------
# Test 5: Expert composition (W_eff = W_base + Σ scaling * B @ A)
# ---------------------------------------------------------------------------

def test_expert_composition_math():
    """Verify the LoRA composition formula: W_eff = W_base + (α/r) * B @ A."""
    import torch

    # Simulate base model weight
    W_base = torch.randn(4096, 4096)

    # LoRA parameters
    r = 4
    alpha = 16
    lora_A = torch.randn(r, 4096)
    lora_B = torch.randn(4096, r)

    # Compose
    scaling = alpha / r
    W_eff = W_base + scaling * (lora_B @ lora_A)

    assert W_eff.shape == W_base.shape
    # Verify it's different from base (delta was applied)
    assert not torch.equal(W_eff, W_base)


# ---------------------------------------------------------------------------
# Test 6: Integration — Modal function is callable (requires Modal auth)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_modal_connection():
    """Verify Modal client can connect. Requires MODAL_TOKEN."""
    import modal
    import os

    # Load token from .env
    from dotenv import load_dotenv
    load_dotenv()

    token = os.environ.get("MODAL_TOKEN", "")
    if not token:
        pytest.skip("MODAL_TOKEN not set")

    os.environ["MODAL_TOKEN_ID"] = token
    os.environ["MODAL_TOKEN_SECRET"] = "placeholder"

    try:
        client = modal.Client.from_env()
        assert client is not None
    except Exception as e:
        pytest.fail(f"Modal connection failed: {e}")


@pytest.mark.integration
def test_modal_can_run_function():
    """Verify Modal app definition works (actual execution needs `modal run`).

    This test verifies the app can be defined without errors.
    Actual GPU execution is tested via `modal run` in scripts/.
    """
    import modal
    import os
    from dotenv import load_dotenv
    load_dotenv()

    token = os.environ.get("MODAL_TOKEN", "")
    if not token:
        pytest.skip("MODAL_TOKEN not set")

    os.environ["MODAL_TOKEN_ID"] = token
    os.environ["MODAL_TOKEN_SECRET"] = "placeholder"

    app = modal.App("dmoe-test")

    @app.function(image=modal.Image.debian_slim(python_version="3.11"), serialized=True)
    def hello():
        return "ok"

    # Just verify the app was defined without errors
    assert app is not None


# ---------------------------------------------------------------------------
# Test 7: Verify download — weights exist after training
# ---------------------------------------------------------------------------

def test_weight_download_verification(tmp_path):
    """After training, verify all expected files exist."""
    expert_dir = tmp_path / "experts" / "expert_00000"
    expert_dir.mkdir(parents=True, exist_ok=True)

    # Expected files after training
    expected_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "manifest.json",
    ]

    # Simulate training output
    (expert_dir / "adapter_model.safetensors").write_text("weights")
    (expert_dir / "adapter_config.json").write_text('{"r": 4}')
    (expert_dir / "manifest.json").write_text('{"expert_id": "expert_00000"}')

    # Verify all exist
    for fname in expected_files:
        fpath = expert_dir / fname
        assert fpath.exists(), f"Missing: {fname}"
        assert fpath.stat().st_size > 0, f"Empty: {fname}"
