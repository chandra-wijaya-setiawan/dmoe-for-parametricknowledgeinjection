"""Base model protocol and HuggingFace implementation."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def create_base_model(model_name: str, device: str = "cpu", dtype: str = "float32", hf_token: str = ""):
    """Create a frozen base model for DMoE.

    Args:
        model_name: HuggingFace model name
        device: "cpu", "cuda", or "auto"
        dtype: "float16" or "float32"
        hf_token: HuggingFace token
    Returns:
        model, tokenizer
    """
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        token=hf_token or None,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer