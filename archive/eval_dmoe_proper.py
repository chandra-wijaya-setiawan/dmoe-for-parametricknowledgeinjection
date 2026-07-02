"""
DMoE Evaluation: proper per-page expert routing.

Loads all trained experts via safetensors directly (no PEFT reuse bug),
builds BM25 index, runs DMoE inference with uncertainty-aware routing.

Usage:
    modal run scripts/eval_dmoe_proper.py                    # Full eval
    modal run scripts/eval_dmoe_proper.py --num-experts 5    # Test with 5
"""

import os
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

QUESTIONS = [
    ("Q1: Start date", "On what exact date did the group stage of the FIFA World Cup 2026 begin?", ["june 11", "11 june"]),
    ("Q2: Three-time host", "Which country became the first to host or co-host the FIFA World Cup three times in 2026?", ["mexico"]),
    ("Q3: Debut teams", "Which four national teams made their FIFA World Cup debut in the 2026 tournament?", ["cape verde", "curaao", "jordan", "uzbekistan"]),
    ("Q4: Final venue", "At which specific stadium will the FIFA World Cup 2026 final be played?", ["metlife", "east rutherford"]),
    ("Q5: Most matches host", "Which country is hosting the most matches in the FIFA World Cup 2026, and how many?", ["united states", "78"]),
    ("Q6: Final date", "On what exact date is the FIFA World Cup 2026 final scheduled?", ["july 19", "19 july"]),
    ("Q7: Opening stadium", "Which stadium hosted the opening match of the FIFA World Cup 2026?", ["estadio azteca", "azteca"]),
    ("Q8: Haiti group", "Which Caribbean nation was drawn into Group C of the FIFA World Cup 2026?", ["haiti"]),
    ("Q9: Defending champion", "Which national team is the defending champion entering the FIFA World Cup 2026?", ["argentina"]),
    ("Q10: Cape Verde", "Which African island nation made its FIFA World Cup debut in 2026?", ["cape verde"]),
    ("Q11: Total matches", "How many total matches will be played in the FIFA World Cup 2026?", ["104"]),
    ("Q12: CONCACAF slots", "How many direct qualification slots did CONCACAF have for the FIFA World Cup 2026?", ["3", "three"]),
    ("Q13: Canada matches", "How many matches is Canada hosting in the FIFA World Cup 2026?", ["13", "thirteen"]),
    ("Q14: Lumen Field city", "Lumen Field, a FIFA World Cup 2026 venue, is located in which US city?", ["seattle"]),
    ("Q15: Scotland group", "Which UK nation was in Group C of the FIFA World Cup 2026?", ["scotland"]),
]


# --- Inlined BM25 Router (self-contained) ---
import numpy as np
import torch
from rank_bm25 import BM25Okapi


class TokenUncertainty:
    @staticmethod
    def compute(logits: torch.Tensor) -> float:
        probs = torch.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)
        return (1.0 - max_probs).mean().item()


class DMoERouter:
    def __init__(self, experts: list[dict], config: dict):
        self.experts = experts
        self.top_k = config.get("top_k", 3)
        self.tau = config.get("tau", 0.7)
        self.cooldown = config.get("retrigger_cooldown", 10)
        tokenized = [e["doc"].lower().split() for e in experts]
        self.bm25 = BM25Okapi(tokenized, k1=config.get("bm25_k1", 1.5), b=config.get("bm25_b", 0.75))
        self.tu_computer = TokenUncertainty()
        self._tokens_since_route = self.cooldown

    def should_trigger(self, logits: torch.Tensor) -> bool:
        self._tokens_since_route += 1
        if self._tokens_since_route < self.cooldown:
            return False
        return self.tu_computer.compute(logits) > self.tau

    def route(self, context: str) -> list[int]:
        tokens = context.lower().split()
        scores = self.bm25.get_scores(tokens)
        top = np.argsort(scores)[-self.top_k:][::-1].tolist()
        self._tokens_since_route = 0
        return top


import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2", "transformers>=4.45", "peft>=0.12",
        "accelerate>=0.34", "rank-bm25>=0.2.2", "safetensors>=0.4",
    )
)

app = modal.App("dmoe-proper-eval")
experts_volume = modal.Volume.from_name("dmoe-experts", create_if_missing=True)


def extract_module_path(name: str) -> tuple:
    """Parse PEFT LoRA safetensors key into (module_path, lora_type).

    PEFT saves keys like: base_model.model.layers.15.mlp.up_proj.lora_A.weight
    We need the path that matches get_submodule(): model.layers.15.mlp.up_proj

    FIX: Some PEFT versions double-wrap, giving model.model.layers...
    We normalize both cases.
    """
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part.startswith("lora_"):
            lora_type = part.split("_")[1]
            mod = ".".join(parts[1:idx])
            # Normalize: model.model.layers -> model.layers
            while mod.startswith("model.model."):
                mod = mod[len("model."):]
            return mod, lora_type
    return None, None


def load_expert_delta(expert_dir, device="cpu"):
    """Load LoRA expert from safetensors, compute delta = (alpha/r) * B @ A."""
    from safetensors.torch import load_file

    with open(os.path.join(expert_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    r = cfg.get("r", 4)
    alpha = cfg.get("lora_alpha", 16)
    scaling = alpha / r

    state = load_file(os.path.join(expert_dir, "adapter_model.safetensors"), device=device)

    # Group A/B by module path
    modules = {}
    for key, tensor in state.items():
        mod_path, lora_type = extract_module_path(key)
        if mod_path and lora_type:
            modules.setdefault(mod_path, {})[lora_type] = tensor.clone().float()

    # Compute deltas
    deltas = {}
    for mod_path, ab in modules.items():
        if "A" in ab and "B" in ab:
            deltas[mod_path] = scaling * (ab["B"] @ ab["A"])

    # Get doc surrogate
    doc = ""
    mp = os.path.join(expert_dir, "manifest.json")
    if os.path.exists(mp):
        with open(mp) as f:
            doc = json.load(f).get("doc", "")[:500]

    return deltas, doc, {"r": r, "alpha": alpha}


def apply_delta(model, delta, factor=1.0, verbose=False):
    """Apply factor * delta to model weights. Tries module paths robustly."""
    with torch.no_grad():
        for mod_key, d in delta.items():
            try:
                # Try direct path
                param = model.get_submodule(mod_key + ".weight")
                param.data += (d * factor).to(param.device, param.dtype)
            except AttributeError:
                # Try with 'model.' prefix removed
                if mod_key.startswith("model."):
                    alt = mod_key[len("model."):]
                    try:
                        param = model.get_submodule(alt + ".weight")
                        param.data += (d * factor).to(param.device, param.dtype)
                    except AttributeError:
                        if verbose:
                            print(f"    (skip {mod_key}: no match)")
                else:
                    if verbose:
                        print(f"    (skip {mod_key}: no match)")


@app.function(
    image=image,
    gpu="T4",
    memory=16384,
    timeout=1200,
    volumes={"/experts": experts_volume},
    env={"HF_TOKEN": HF_TOKEN},
)
def evaluate(num_experts: int = 0, verbose: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN", "")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

    # --- Find trained experts ---
    expert_dirs = sorted([
        d for d in os.listdir("/experts")
        if d.startswith("expert_") and os.path.isfile(os.path.join("/experts", d, "adapter_model.safetensors"))
    ])

    if not expert_dirs:
        print("ERROR: No trained experts found!")
        return

    if num_experts > 0:
        expert_dirs = expert_dirs[:num_experts]

    print(f"Found {len(expert_dirs)} trained experts")

    # --- Load all expert deltas ---
    expert_deltas = []
    doc_surrogates = []
    for d in expert_dirs:
        fp = os.path.join("/experts", d)
        try:
            deltas, doc, cfg = load_expert_delta(fp)
            if deltas:
                expert_deltas.append(deltas)
                doc_surrogates.append(doc)
            else:
                print(f"  {d}: empty deltas")
        except Exception as e:
            print(f"  {d}: ERROR - {e}")

    print(f"Loaded {len(expert_deltas)}/{len(expert_dirs)} expert deltas")

    if not expert_deltas:
        return

    # Print first expert's module paths for debugging
    first_key = list(expert_deltas[0].keys())[0]
    print(f"\nSample module path: {first_key}")
    print(f"Expected: model.layers.15.mlp.down_proj or similar")

    # --- Build BM25 router ---
    print("Building BM25 router...")
    expert_list = [
        {"id": str(i), "doc": doc_surrogates[i] if i < len(doc_surrogates) else "", "lora_path": ""}
        for i in range(len(expert_deltas))
    ]

    router_config = {"top_k": min(3, len(expert_deltas)), "tau": 0.7,
                     "retrigger_cooldown": 10, "bm25_k1": 1.5, "bm25_b": 0.75}
    router = DMoERouter(expert_list, router_config)
    tu_comp = TokenUncertainty()

    def gen(model, q, max_tokens=300):
        inp = tokenizer(q, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=max_tokens, temperature=0.0, do_sample=False)
        return tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True)

    def check(a, kws):
        return 1.0 if any(kw in a.lower() for kw in kws) else 0.0

    # --- Base model ---
    print("\n" + "=" * 70)
    print("BASE MODEL")
    print("=" * 70)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf_token
    )
    model.eval()

    base_scores = []
    for label, q, kws in QUESTIONS:
        a = gen(model, q)
        s = check(a, kws)
        base_scores.append(s)
        print(f"  {label}: [{'OK' if s else 'NO'}] {a[:55]}")

    base_avg = sum(base_scores) / len(base_scores)
    print(f"\nBase: {base_avg:.0%}")

    del model
    torch.cuda.empty_cache()

    # --- DMoE with routing ---
    print(f"\n{'=' * 70}")
    print(f"DMoE (router top-{router.top_k} from {len(expert_deltas)} experts)")
    print("=" * 70)

    dmoe_scores = []
    for i, (label, q, kws) in enumerate(QUESTIONS):
        model_t = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf_token
        )
        model_t.eval()

        # TU check
        inp = tokenizer(q, return_tensors="pt").input_ids.to(model_t.device)
        with torch.no_grad():
            out = model_t(inp)
        tu = tu_comp.compute(out.logits[:, -1, :])

        selected = []
        if router.should_trigger(out.logits[:, -1, :]):
            selected = router.route(q)
            for idx in selected:
                if idx < len(expert_deltas):
                    apply_delta(model_t, expert_deltas[idx], verbose=verbose)

        a = gen(model_t, q)
        s = check(a, kws)
        dmoe_scores.append(s)

        arrow = "+" if s > base_scores[i] else ("-" if s < base_scores[i] else "=")
        trig = f"TU={tu:.3f}, experts={selected}" if selected else f"TU={tu:.3f}, skip"
        print(f"  {label}: [{'OK' if s else 'NO'}] {arrow}  {trig}")
        if verbose:
            print(f"         A: {a[:60]}")

        del model_t
        torch.cuda.empty_cache()

    dmoe_avg = sum(dmoe_scores) / len(dmoe_scores)
    dmoe_wins = sum(1 for d, b in zip(dmoe_scores, base_scores) if d > b)
    base_wins = sum(1 for d, b in zip(dmoe_scores, base_scores) if b > d)

    # --- Compare ALL experts (no router) ---
    print(f"\n{'=' * 70}")
    print("ALL EXPERTS (no router)")
    print("=" * 70)
    all_scores = []
    for i, (label, q, kws) in enumerate(QUESTIONS):
        model_t = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf_token
        )
        model_t.eval()
        for delta in expert_deltas:
            apply_delta(model_t, delta, verbose=verbose)
        a = gen(model_t, q)
        s = check(a, kws)
        all_scores.append(s)
        arrow = "+" if s > base_scores[i] else ("-" if s < base_scores[i] else "=")
        print(f"  {label}: [{'OK' if s else 'NO'}] {arrow} {a[:55]}")
        del model_t
        torch.cuda.empty_cache()

    all_avg = sum(all_scores) / len(all_scores)

    # --- Results ---
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print("=" * 70)
    print(f"Base model:     {base_avg:.0%} ({sum(base_scores)}/{len(base_scores)})")
    print(f"DMoE (router):  {dmoe_avg:.0%} ({sum(dmoe_scores)}/{len(dmoe_scores)})")
    print(f"All experts:    {all_avg:.0%} ({sum(all_scores)}/{len(all_scores)})")
    print(f"\nDMoE vs base: {dmoe_avg - base_avg:+.0%}")
    print(f"All vs base:  {all_avg - base_avg:+.0%}")
    print(f"DMoE wins: {dmoe_wins}, Base wins: {base_wins}")

    print(f"\nPer question:")
    for i, (l, _, _) in enumerate(QUESTIONS):
        b = "OK" if base_scores[i] else "NO"
        d = "OK" if dmoe_scores[i] else "NO"
        a = "OK" if all_scores[i] else "NO"
        arr = ">" if dmoe_scores[i] > base_scores[i] else ("<" if dmoe_scores[i] < base_scores[i] else "=")
        print(f"  {l}: base={b} {arr} dmoe={d} | all={a}")

    return {
        "num_experts": len(expert_deltas),
        "base_accuracy": base_avg,
        "dmoe_accuracy": dmoe_avg,
        "all_experts_accuracy": all_avg,
    }


@app.local_entrypoint()
def main(num_experts: int = 0, verbose: bool = False):
    print(f"Starting DMoE eval ({num_experts or '30'} experts)...")
    result = evaluate.remote(num_experts, verbose)
    print(f"\nDone: {result}")
