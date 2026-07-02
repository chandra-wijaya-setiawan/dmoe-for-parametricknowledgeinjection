"""BM25-based router with Token Uncertainty gating."""

from __future__ import annotations

import numpy as np
import torch
from rank_bm25 import BM25Okapi


class TokenUncertainty:
    """Measures model uncertainty at each decoding step using full entropy.
    
    Paper §3.1: TUt = -sum_v pt(v) log pt(v)
    This matches the paper's entropy-based approach.
    """

    @staticmethod
    def compute(logits: torch.Tensor) -> float:
        """TU = entropy of the next-token distribution.
        
        Paper default threshold: τ = 2.0 (entropy scale, not 1-max_prob)
        """
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(torch.clamp(probs, min=1e-10))
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.mean().item()


class DMoERouter:
    """Training-free BM25 router with uncertainty-aware triggering."""

    def __init__(self, experts: list[dict], config: dict):
        """
        Args:
            experts: list of {"id": str, "doc": str, "lora_path": str}
            config: router section from default.yaml
        """
        self.experts = experts
        self.top_k = config.get("top_k", 3)
        self.tau = config.get("tau", 0.7)
        self.cooldown = config.get("retrigger_cooldown", 10)

        tokenized = [e["doc"].lower().split() for e in experts]
        self.bm25 = BM25Okapi(
            tokenized,
            k1=config.get("bm25_k1", 1.5),
            b=config.get("bm25_b", 0.75),
        )
        self.tu_computer = TokenUncertainty()
        self._tokens_since_route = self.cooldown  # allow immediate first route

    def should_trigger(self, logits: torch.Tensor) -> bool:
        """Check if TU exceeds threshold and cooldown has elapsed."""
        self._tokens_since_route += 1
        if self._tokens_since_route < self.cooldown:
            return False
        tu = self.tu_computer.compute(logits)
        return tu > self.tau

    def route(self, context: str) -> list[int]:
        """Return top-k expert indices via BM25 over context."""
        tokens = context.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[-self.top_k :][::-1].tolist()
        self._tokens_since_route = 0
        return top_indices

    def get_experts(self, indices: list[int]) -> list[dict]:
        """Return expert metadata for given indices."""
        return [self.experts[i] for i in indices]
