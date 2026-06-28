"""KV-cache-safe autoregressive generation with DMoE routing.

Greedy decoding only (temperature=0.0) — matches the paper's setup.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class DMoEGenerator:
    """Generation loop with uncertainty-triggered expert activation."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        router,
        expert_manager,
        config: dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.router = router
        self.expert_manager = expert_manager
        self.config = config
        self.gen_config = config.get("inference", {})
        self.max_new_tokens = self.gen_config.get("max_new_tokens", 128)

    @torch.no_grad()
    def generate(self, prompt: str, verbose: bool = False) -> str:
        """
        Generate response with dynamic expert routing.

        When TU > τ at any decoding step:
        1. Decode current prefix (excluding triggering token)
        2. Retrieve top-k experts via BM25 on prefix
        3. Compose LoRA deltas into final-layer FFN
        4. Resume decoding (earlier KV cache still valid)
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        generated = input_ids.clone()
        past_key_values = None

        for step in range(self.max_new_tokens):
            outputs = self.model(
                input_ids=generated[:, -1:] if past_key_values else generated,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # Check if router should trigger
            if self.router.should_trigger(logits):
                context = self.tokenizer.decode(generated[0, input_ids.shape[1]:])
                expert_indices = self.router.route(context)

                if expert_indices != [e.id for e in self.expert_manager.active_experts]:
                    experts = self.router.get_experts(expert_indices)
                    loaded = [
                        self.expert_manager.load_expert(
                            e["lora_path"], e["id"], e["doc"]
                        )
                        for e in experts
                    ]
                    self.expert_manager.activate_experts(loaded)
                    if verbose:
                        print(f"  [Step {step}] {self.expert_manager.status()}")

            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        result = self.tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
        return result

    def reset(self):
        """Clear any active experts back to base model."""
        self.expert_manager.deactivate_all()
