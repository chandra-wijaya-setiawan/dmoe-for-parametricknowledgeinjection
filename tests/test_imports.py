"""Verify environment setup and package imports."""


def test_core_imports():
    import torch
    import transformers
    import peft
    import yaml
    from rank_bm25 import BM25Okapi

    assert torch.__version__
    assert transformers.__version__
    assert peft.__version__


def test_dmoe_imports():
    from dmoe.router import DMoERouter, TokenUncertainty
    from dmoe.models import ExpertManager, Expert, create_base_model
    from dmoe.inference import DMoEGenerator
    from dmoe.training import ExpertTrainer


def test_token_uncertainty():
    import torch
    from dmoe.router import TokenUncertainty

    # Certain: one logit much higher
    certain_logits = torch.tensor([[10.0, 0.1, 0.1]])
    assert TokenUncertainty.compute(certain_logits) < 0.1

    # Uncertain: uniform logits
    uncertain_logits = torch.tensor([[0.0, 0.0, 0.0]])
    assert TokenUncertainty.compute(uncertain_logits) > 0.5


def test_bm25_router():
    from dmoe.router import DMoERouter

    experts = [
        {"id": "exp_1", "doc": "Python programming language", "lora_path": "/tmp/1"},
        {"id": "exp_2", "doc": "French cuisine and cooking", "lora_path": "/tmp/2"},
        {"id": "exp_3", "doc": "Quantum physics basics", "lora_path": "/tmp/3"},
    ]
    config = {"top_k": 2, "tau": 0.7, "retrigger_cooldown": 5}

    router = DMoERouter(experts, config)
    router._tokens_since_route = 5  # allow trigger
    indices = router.route("I want to cook French food")

    assert 1 in indices  # Expert 2 (French cuisine) should be retrieved
    assert len(indices) == 2


def test_config_loads():
    import yaml
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    assert "base_model" in config
    assert "expert" in config
    assert "router" in config
    assert "inference" in config
    assert "training" in config


def test_model_configs_load():
    import yaml
    from dmoe.config import IncludeLoader

    with open("configs/models/llama_3_2_1b.yaml") as f:
        llama_cfg = yaml.load(f, Loader=IncludeLoader)
    assert llama_cfg["model"]["name"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert "{last}" in llama_cfg["model"]["ffn_targets"][0]

    with open("configs/models/gemma_3_1b.yaml") as f:
        gemma_cfg = yaml.load(f, Loader=IncludeLoader)
    assert gemma_cfg["model"]["name"] == "google/gemma-3-1b-it"


def test_experiment_config_with_include():
    import yaml
    from dmoe.config import IncludeLoader

    with open("configs/experiments/llama_3_2_1b__arxiv_2026.yaml") as f:
        cfg = yaml.load(f, Loader=IncludeLoader)

    assert cfg["experiment"]["name"] == "llama_3_2_1b__arxiv_2026"
    assert cfg["experiment"]["model"]["model"]["name"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert cfg["experiment"]["expert"]["r"] == 4
    assert cfg["experiment"]["expert"]["alpha"] == 16
