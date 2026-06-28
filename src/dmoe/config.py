"""Configuration loading with YAML + include support + experiment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class IncludeLoader(yaml.SafeLoader):
    """YAML loader that supports !include directives."""
    pass


def _include_constructor(loader, node):
    root_dir = Path(loader.name).parent
    filename = loader.construct_scalar(node)
    filepath = root_dir / filename
    with open(filepath) as f:
        # Pass the loader class to support nested includes
        return yaml.load(f, Loader=type(loader))


IncludeLoader.add_constructor("!include", _include_constructor)


@dataclass
class ModelConfig:
    name: str = "meta-llama/Llama-3.2-1B-Instruct"
    num_layers: int = 16
    hf_token: str = ""
    device: str = "cpu"
    dtype: str = "float16"
    ffn_targets: list[str] = field(default_factory=list)

    def resolved_ffn_targets(self) -> list[str]:
        """Replace {last} with actual layer index."""
        return [t.replace("{last}", str(self.num_layers - 1)) for t in self.ffn_targets]


@dataclass
class ExpertConfig:
    r: int = 4
    alpha: int = 16
    dropout: float = 0.05
    epochs: int = 3
    lr: float = 2e-4
    batch_size: int = 4
    max_length: int = 512


@dataclass
class RouterConfig:
    tau: float = 0.7
    top_k: int = 3
    cooldown: int = 10
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass
class InferenceConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0


@dataclass
class TrainingConfig:
    backend: str = "modal"
    gpu: str = "T4"
    timeout: int = 600


@dataclass
class DataConfig:
    name: str = ""
    path: str = ""


@dataclass
class TeacherConfig:
    type: str = "local"
    model: str = ""


@dataclass
class ExperimentConfig:
    name: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "./outputs"
    seed: int = 42


def load_config(config_path: str, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load experiment config from YAML, with optional dotted overrides."""
    with open(config_path) as f:
        raw = yaml.load(f, Loader=IncludeLoader)

    config = _dict_to_dataclass(ExperimentConfig, raw.get("experiment", raw if raw else {}))

    if overrides:
        _apply_overrides(config, overrides)

    return config


def _dict_to_dataclass(cls, data: dict):
    """Recursively construct dataclass from dict."""
    if not isinstance(data, dict):
        return data
    import dataclasses
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in data:
            if hasattr(f.type, '__dataclass_fields__'):
                kwargs[f.name] = _dict_to_dataclass(f.type, data[f.name])
            else:
                kwargs[f.name] = data[f.name]
    return cls(**kwargs)


def _apply_overrides(obj: Any, overrides: dict, prefix: str = ""):
    """Apply dotted overrides like {'expert.epochs': 5}."""
    import dataclasses
    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".", 1)
            child = getattr(obj, parts[0])
            _apply_overrides(child, {parts[1]: value}, f"{prefix}{parts[0]}.")
        else:
            if hasattr(obj, key):
                setattr(obj, key, value)
