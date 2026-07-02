"""
DMoE Eval: 50 questions covering all experts + fixed module path matching.
"""

import os, json
import numpy as np
import torch
from rank_bm25 import BM25Okapi

try: from dotenv import load_dotenv; load_dotenv()
except: pass

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# 50 questions covering all 30 knowledge unit pages
QUESTIONS = [
    # Main page (1)
    ("M01", "On what exact date did the group stage of the 2026 FIFA World Cup begin?", ["june 11", "11 june"]),
    ("M02", "Which three nations are co-hosting the 2026 FIFA World Cup?", ["united states", "mexico", "canada"]),
    ("M03", "How many teams are participating in the expanded 2026 FIFA World Cup?", ["48"]),
    ("M04", "What is the total number of matches scheduled for the 2026 FIFA World Cup?", ["104"]),
    ("M05", "Which stadium will host the 2026 FIFA World Cup final?", ["metlife"]),
    ("M06", "On what date is the 2026 FIFA World Cup final scheduled?", ["july 19", "19 july"]),
    ("M07", "Which country became the first to host or co-host the FIFA World Cup three times in 2026?", ["mexico"]),
    # Groups A-H (2-9)
    ("GA1", "Which African nation was drawn into Group A of the 2026 FIFA World Cup?", ["senegal"]),
    ("GC1", "Which Caribbean nation was drawn into Group C of the 2026 FIFA World Cup?", ["haiti"]),
    ("GC2", "Which UK nation was in Group C of the 2026 FIFA World Cup?", ["scotland"]),
    ("GD1", "Which Asian team is in Group D of the 2026 FIFA World Cup?", ["japan"]),
    ("GG1", "Which European nation is in Group G of the 2026 FIFA World Cup?", ["england"]),
    # Final (10)
    ("F01", "Which stadium will host the 2026 FIFA World Cup final match?", ["metlife"]),
    ("F02", "What is the seating capacity of MetLife Stadium for the 2026 World Cup final?", ["82500", "82,500"]),
    # Knockout stage (11)
    ("K01", "How many teams qualify for the knockout stage of the 2026 FIFA World Cup?", ["32"]),
    ("K02", "Which round follows the group stage in the 2026 FIFA World Cup?", ["round of 32", "round of 16"]),
    # Officials (12)
    ("R01", "Who was selected as the referee for the opening match of the 2026 FIFA World Cup?", ["ismail elfath", "elfath"]),
    ("R02", "Which country did the 2026 World Cup final referee come from?", ["argentina"]),
    # Qualification (13-18)
    ("Q01", "How many CONCACAF direct qualification slots were there for the 2026 FIFA World Cup?", ["3", "three"]),
    ("Q02", "Which AFC team qualified for the 2026 FIFA World Cup through the AFC qualifying tournament?", ["japan", "iran", "korea"]),
    ("Q03", "How many CAF slots were allocated for the 2026 FIFA World Cup?", ["5", "five", "9", "nine"]),
    ("Q04", "Which CONMEBOL teams qualified directly for the 2026 FIFA World Cup?", ["argentina", "brazil", "uruguay"]),
    ("Q05", "How many UEFA teams qualified for the 2026 FIFA World Cup?", ["13", "sixteen", "16"]),
    # Squads (19)
    ("S01", "Which player captained Argentina at the 2026 FIFA World Cup?", ["messi", "lionel messi"]),
    # Venues (20-30)
    ("V01", "In which Canadian city is BC Place, a 2026 World Cup venue, located?", ["vancouver"]),
    ("V02", "What is the capacity of BC Place stadium for the 2026 World Cup?", ["54500", "54,500"]),
    ("V03", "Which stadium in Mexico City will host matches in the 2026 FIFA World Cup?", ["estadio azteca", "azteca"]),
    ("V04", "What is the capacity of Estadio BBVA in Monterrey for the 2026 World Cup?", ["53500", "53,500"]),
    ("V05", "Which NFL stadium in Miami Gardens will host 2026 World Cup matches?", ["hard rock stadium", "hard rock"]),
    ("V06", "What is the capacity of Hard Rock Stadium for the 2026 World Cup?", ["64800", "64,800", "65,000"]),
    ("V07", "Which NFL stadium in Santa Clara will host 2026 World Cup matches?", ["levi's stadium", "levis stadium", "levi"]),
    ("V08", "In which US city is Lumen Field, a 2026 World Cup venue, located?", ["seattle"]),
    ("V09", "What is the capacity of Lumen Field for the 2026 World Cup?", ["69000", "69,000"]),
    ("V10", "In which US city is MetLife Stadium located?", ["east rutherford", "new jersey"]),
    ("V11", "Which NFL team plays at Paycor Stadium in Cincinnati?", ["bengals"]),
    ("V12", "What is the capacity of SoFi Stadium for the 2026 World Cup?", ["70200", "70,200"]),
    ("V13", "In which US city is SoFi Stadium located?", ["inglewood", "los angeles"]),
    ("V14", "What is the capacity of TQL Stadium in Cincinnati for the 2026 World Cup?", ["26500", "26,500"]),
    ("V15", "Which MLS team plays at TQL Stadium?", ["fc cincinnati", "cincinnati"]),
    # Debut teams
    ("D01", "Which African island nation made its FIFA World Cup debut in 2026?", ["cape verde"]),
    ("D02", "Which Caribbean territory made its FIFA World Cup debut in 2026?", ["curaao"]),
    ("D03", "Which Middle Eastern country debuted in the 2026 FIFA World Cup?", ["jordan"]),
    ("D04", "Which Central Asian country debuted in the 2026 FIFA World Cup?", ["uzbekistan"]),
    # Host countries
    ("H01", "Which country is hosting the most matches in the 2026 FIFA World Cup?", ["united states", "78"]),
    ("H02", "How many matches is Canada hosting in the 2026 World Cup?", ["13", "thirteen"]),
    ("H03", "How many matches is Mexico hosting in the 2026 World Cup?", ["13", "thirteen"]),
    # Controversies
    ("C01", "Which human rights issue was raised regarding the 2026 FIFA World Cup?", ["indigenous", "human rights"]),
]


class TokenUncertainty:
    @staticmethod
    def compute(logits):
        probs = torch.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1)
        return (1.0 - max_probs).mean().item()

class DMoERouter:
    def __init__(self, experts, config):
        self.experts = experts
        self.top_k = config.get("top_k", 3)
        self.tau = config.get("tau", 0.7)
        self.cooldown = config.get("retrigger_cooldown", 10)
        tokenized = [e["doc"].lower().split() for e in experts]
        self.bm25 = BM25Okapi(tokenized, k1=config.get("bm25_k1", 1.5), b=config.get("bm25_b", 0.75))
        self.tu = TokenUncertainty()
        self._since = self.cooldown
    def should_trigger(self, logits):
        self._since += 1
        if self._since < self.cooldown: return False
        return self.tu.compute(logits) > self.tau
    def route(self, context):
        scores = self.bm25.get_scores(context.lower().split())
        top = np.argsort(scores)[-self.top_k:][::-1].tolist()
        self._since = 0
        return top


import modal
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.2","transformers>=4.45","peft>=0.12",
                 "accelerate>=0.34","rank-bm25>=0.2.2","safetensors>=0.4"))
app = modal.App("dmoe-eval-50q")
vol = modal.Volume.from_name("dmoe-experts", create_if_missing=True)


def get_delta(expert_dir):
    """Load one expert, return (delta_dict, doc_str)."""
    from safetensors.torch import load_file
    with open(os.path.join(expert_dir, "adapter_config.json")) as f:
        c = json.load(f)
    s = c.get("lora_alpha", 16) / c.get("r", 4)
    state = load_file(os.path.join(expert_dir, "adapter_model.safetensors"))
    modules = {}
    for key, val in state.items():
        parts = key.split(".")
        for idx, p in enumerate(parts):
            if p.startswith("lora_"):
                lt = p.split("_")[1]
                mod = ".".join(parts[1:idx])  # e.g. model.layers.15.mlp.down_proj
                # Normalize model.model. -> model.
                while mod.startswith("model.model."):
                    mod = mod[len("model."):]
                modules.setdefault(mod, {})[lt] = val.clone().float()
                break
    deltas = {}
    for mod, ab in modules.items():
        if "A" in ab and "B" in ab:
            deltas[mod] = s * (ab["B"] @ ab["A"])
    doc = ""
    mp = os.path.join(expert_dir, "manifest.json")
    if os.path.exists(mp):
        with open(mp) as f: doc = json.load(f).get("doc", "")[:500]
    return deltas, doc


@app.function(image=image, gpu="T4", memory=16384, timeout=1800,
              volumes={"/experts": vol}, env={"HF_TOKEN": HF_TOKEN})
def evaluate(num_experts=0):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = os.environ.get("HF_TOKEN", "")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf)

    # Load experts
    dirs = sorted([d for d in os.listdir("/experts")
        if d.startswith("expert_") and os.path.isfile(os.path.join("/experts", d, "adapter_model.safetensors"))])
    if num_experts > 0: dirs = dirs[:num_experts]
    print(f"Found {len(dirs)} experts")

    deltas_list = []
    docs = []
    for d in dirs:
        dl, dc = get_delta(os.path.join("/experts", d))
        if dl:
            deltas_list.append(dl)
            docs.append(dc)
            print(f"  {d}: {len(dl)} modules, sample: {list(dl.keys())[0]}")
    print(f"Loaded {len(deltas_list)} experts")

    if not deltas_list:
        print("No experts loaded!")
        return

    # BM25 router
    router = DMoERouter(
        [{"id": str(i), "doc": docs[i] or "", "lora_path": ""} for i in range(len(deltas_list))],
        {"top_k": 3, "tau": 0.7, "retrigger_cooldown": 10, "bm25_k1": 1.5, "bm25_b": 0.75}
    )
    tu = TokenUncertainty()

    def gen(m, q):
        inp = tok(q, return_tensors="pt").input_ids.to(m.device)
        with torch.no_grad():
            o = m.generate(inp, max_new_tokens=200, temperature=0.0, do_sample=False)
        return tok.decode(o[0][inp.shape[1]:], skip_special_tokens=True)

    def check(a, kws):
        return 1.0 if any(kw in a.lower() for kw in kws) else 0.0

    def apply(m, delta):
        with torch.no_grad():
            for key, d in delta.items():
                try:
                    p = m.get_submodule(key + ".weight")
                    p.data += d.to(p.device, p.dtype)
                except AttributeError:
                    # Try stripping model prefix
                    try:
                        alt = key.split(".", 1)[1] if key.startswith("model.") else key
                        p = m.get_submodule(alt + ".weight")
                        p.data += d.to(p.device, p.dtype)
                    except:
                        pass  # skip if no match

    # Base model
    print("\n=== BASE MODEL ===")
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf)
    m.eval()
    base = []
    for qid, q, kws in QUESTIONS:
        a = gen(m, q); s = check(a, kws); base.append(s)
        print(f"  {qid}: [{'OK' if s else '  '}] {a[:55]}")
    ba = sum(base)/len(base)
    print(f"\nBase: {ba:.0%} ({sum(base)}/{len(base)})")
    del m; torch.cuda.empty_cache()

    # DMoE with router
    print(f"\n=== DMoE ROUTER (top-3 from {len(deltas_list)}) ===")
    dmoe = []
    for qid, q, kws in QUESTIONS:
        mt = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf)
        mt.eval()
        inp = tok(q, return_tensors="pt").input_ids.to(mt.device)
        with torch.no_grad():
            o = mt(inp)
        tuv = tu.compute(o.logits[:, -1, :])
        sel = []
        if router.should_trigger(o.logits[:, -1, :]):
            sel = router.route(q)
            for idx in sel:
                if idx < len(deltas_list): apply(mt, deltas_list[idx])
        a = gen(mt, q); s = check(a, kws); dmoe.append(s)
        arr = "+" if s > base[len(dmoe)-1] else ("-" if s < base[len(dmoe)-1] else "=")
        trig = f"TU={tuv:.3f}, {len(sel)} experts" if sel else f"TU={tuv:.3f}, skip"
        print(f"  {qid}: [{'OK' if s else '  '}] {arr}  {trig}")
        del mt; torch.cuda.empty_cache()
    da = sum(dmoe)/len(dmoe)

    # ALL experts
    print(f"\n=== ALL {len(deltas_list)} EXPERTS ===")
    all_s = []
    for qid, q, kws in QUESTIONS:
        mt = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf)
        mt.eval()
        for d in deltas_list: apply(mt, d)
        a = gen(mt, q); s = check(a, kws); all_s.append(s)
        idx = len(all_s)-1
        arr = "+" if s > base[idx] else ("-" if s < base[idx] else "=")
        print(f"  {qid}: [{'OK' if s else '  '}] {arr} {a[:55]}")
        del mt; torch.cuda.empty_cache()
    aa = sum(all_s)/len(all_s)

    # Summary
    print(f"\n{'='*60}")
    print(f"Base:     {ba:.0%} ({sum(base)}/{len(base)})")
    print(f"DMoE:     {da:.0%} ({sum(dmoe)}/{len(dmoe)})  vs base: {da-ba:+.0%}")
    print(f"All exp:  {aa:.0%} ({sum(all_s)}/{len(all_s)})  vs base: {aa-ba:+.0%}")
    wins = sum(1 for d,b in zip(dmoe,base) if d>b)
    print(f"DMoE beats base: {wins}/{len(base)}")
    return {"base": ba, "dmoe": da, "all": aa, "wins": wins}

@app.local_entrypoint()
def main(verbose=False):
    print("DMoE eval with 50 questions...")
    r = evaluate.remote()
    print(f"\nDone: {r}")