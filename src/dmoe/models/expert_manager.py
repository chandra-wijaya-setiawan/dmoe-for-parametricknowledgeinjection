"""Expert dataclass and composition: W_eff = W_base + Σ (α/r) · B @ A."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch


@dataclass
class Expert:
    """One knowledge expert: LoRA delta + metadata."""
    id: str
    lora_path: str
    doc: str  # BM25 document surrogate
    r: int = 4
    alpha: int = 16
    deltas: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def scaling(self) -> float:
        return self.alpha / self.r


def extract_module_path(name: str) -> tuple:
    """Parse PEFT LoRA safetensors key into (module_path, lora_type).

    PEFT saves keys like: base_model.model.layers.15.mlp.up_proj.lora_A.weight
    Returns (module_path, lora_type) e.g. (model.layers.15.mlp.up_proj, A)
    """
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part.startswith("lora_"):
            lora_type = part.split("_")[1]
            mod = ".".join(parts[1:idx])
            # Normalize double prefix
            while mod.startswith("model.model."):
                mod = mod[len("model."):]
            return mod, lora_type
    return None, None


def load_expert_delta(expert_dir: str, device: str = "cpu") -> tuple[dict, str, dict]:
    """Load LoRA expert from safetensors, compute delta = (alpha/r) * B @ A.

    Returns:
        deltas: dict of {module_path: delta_matrix}
        doc: document surrogate string
        config: dict with r, alpha
    """
    from safetensors.torch import load_file

    config_path = os.path.join(expert_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        return {}, "", {}

    with open(config_path) as f:
        cfg = json.load(f)
    r = cfg.get("r", 4)
    alpha = cfg.get("lora_alpha", 16)
    scaling = alpha / r

    safetensors_path = os.path.join(expert_dir, "adapter_model.safetensors")
    if not os.path.exists(safetensors_path):
        return {}, "", {}

    state = load_file(safetensors_path, device=device)

    # Group A/B by module path
    modules = {}
    for key, tensor in state.items():
        mod_path, lora_type = extract_module_path(key)
        if mod_path and lora_type:
            modules.setdefault(mod_path, {})[lora_type] = tensor.clone().float()

    # Compute deltas: delta = scaling * (lora_B @ lora_A)
    deltas = {}
    for mod_path, ab in modules.items():
        if "A" in ab and "B" in ab:
            # lora_B: (out_features, r), lora_A: (r, in_features)
            # delta: (out_features, in_features)
            deltas[mod_path] = scaling * (ab["B"] @ ab["A"])

    doc = ""
    manifest_path = os.path.join(expert_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            doc = json.load(f).get("doc", "")[:500]

    return deltas, doc, {"r": r, "alpha": alpha}


class ExpertManager:
    """Manages expert activation/deactivation and composition.

    Applies LoRA deltas to the base model's final-layer FFN weights.
    Uses get_submodule(mod_key) to get the Linear module, then accesses .weight.
    """

    def __init__(self, base_model, device: str = "cpu"):
        self.base_model = base_model
        self.device = device
        self.active_experts: list[Expert] = []
        self._backup: dict[str, torch.Tensor] = {}

    def load_expert(self, lora_path: str, expert_id: str, doc: str = "") -> Expert:
        deltas, _, cfg = load_expert_delta(lora_path, device=self.device)
        return Expert(
            id=expert_id, lora_path=lora_path, doc=doc,
            r=cfg.get("r", 4), alpha=cfg.get("alpha", 16), deltas=deltas,
        )

    def activate_experts(self, experts: list[Expert]):
        self.deactivate_all()
        self.active_experts = experts
        for expert in experts:
            self._apply_delta(expert.deltas, factor=1.0)

    def _get_weight(self, mod_key: str):
        """Try to get weight tensor for module path, returns None on failure."""
        try:
            return self.base_model.get_submodule(mod_key).weight
        except (AttributeError, RuntimeError):
            pass
        # Try alternate prefix
        alt = mod_key[len("model."):] if mod_key.startswith("model.") else f"model.{mod_key}"
        try:
            return self.base_model.get_submodule(alt).weight
        except (AttributeError, RuntimeError):
            return None

    def _apply_delta(self, deltas: dict, factor: float = 1.0):
        """Apply factor * delta to model weights, saving originals."""
        with torch.no_grad():
            for mod_key, d in deltas.items():
                w = self._get_weight(mod_key)
                if w is None:
                    continue
                if mod_key not in self._backup:
                    self._backup[mod_key] = w.data.detach().cpu().clone()
                w.data += (d * factor).to(w.device, w.dtype)

    def deactivate_all(self):
        """Restore base model weights from backup."""
        if self._backup:
            with torch.no_grad():
                for mod_key, original_weight in self._backup.items():
                    w = self._get_weight(mod_key)
                    if w is not None:
                        w.data.copy_(original_weight.to(w.device, w.dtype))
            self._backup = {}
        self.active_experts = []

    def status(self) -> str:
        if not self.active_experts:
            return "No active experts"
        return f"Active: {', '.join(e.id for e in self.active_experts)}"