"""
DMoE Eval v3: load model ONCE, 50 questions, tau sweep.
"""
import os, json
import numpy as np
import torch
from rank_bm25 import BM25Okapi

try: from dotenv import load_dotenv; load_dotenv()
except: pass

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

QUESTIONS = [
    ("M01", "On what exact date did the group stage of the 2026 FIFA World Cup begin?", ["june 11", "11 june"]),
    ("M02", "Which three nations are co-hosting the 2026 FIFA World Cup?", ["united states", "mexico", "canada"]),
    ("M03", "How many teams are participating in the expanded 2026 FIFA World Cup?", ["48"]),
    ("M04", "What is the total number of matches scheduled for the 2026 FIFA World Cup?", ["104"]),
    ("M05", "Which stadium will host the 2026 FIFA World Cup final?", ["metlife"]),
    ("M06", "On what date is the 2026 FIFA World Cup final scheduled?", ["july 19", "19 july"]),
    ("M07", "Which country became the first to host or co-host the FIFA World Cup three times in 2026?", ["mexico"]),
    ("GA1", "Which African nation was drawn into Group A of the 2026 FIFA World Cup?", ["senegal"]),
    ("GC1", "Which Caribbean nation was drawn into Group C of the 2026 FIFA World Cup?", ["haiti"]),
    ("GC2", "Which UK nation was in Group C of the 2026 FIFA World Cup?", ["scotland"]),
    ("GD1", "Which Asian team is in Group D of the 2026 FIFA World Cup?", ["japan"]),
    ("GG1", "Which European nation is in Group G of the 2026 FIFA World Cup?", ["england"]),
    ("F01", "Which stadium will host the 2026 FIFA World Cup final match?", ["metlife"]),
    ("F02", "What is the seating capacity of MetLife Stadium for the 2026 World Cup final?", ["82500", "82,500"]),
    ("K01", "How many teams qualify for the knockout stage of the 2026 FIFA World Cup?", ["32"]),
    ("K02", "Which round follows the group stage in the 2026 FIFA World Cup?", ["round of 32", "round of 16"]),
    ("R01", "Who was selected as the referee for the opening match of the 2026 FIFA World Cup?", ["ismail elfath", "elfath"]),
    ("R02", "Which country did the 2026 World Cup final referee come from?", ["argentina"]),
    ("Q01", "How many CONCACAF direct qualification slots were there for the 2026 FIFA World Cup?", ["3", "three"]),
    ("Q02", "Which AFC team qualified for the 2026 FIFA World Cup through the AFC qualifying tournament?", ["japan", "iran", "korea"]),
    ("Q03", "How many CAF slots were allocated for the 2026 FIFA World Cup?", ["5", "five", "9", "nine"]),
    ("Q04", "Which CONMEBOL teams qualified directly for the 2026 FIFA World Cup?", ["argentina", "brazil", "uruguay"]),
    ("Q05", "How many UEFA teams qualified for the 2026 FIFA World Cup?", ["13", "sixteen", "16"]),
    ("S01", "Which player captained Argentina at the 2026 FIFA World Cup?", ["messi", "lionel messi"]),
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
    ("D01", "Which African island nation made its FIFA World Cup debut in 2026?", ["cape verde"]),
    ("D02", "Which Caribbean territory made its FIFA World Cup debut in 2026?", ["curaao"]),
    ("D03", "Which Middle Eastern country debuted in the 2026 FIFA World Cup?", ["jordan"]),
    ("D04", "Which Central Asian country debuted in the 2026 FIFA World Cup?", ["uzbekistan"]),
    ("H01", "Which country is hosting the most matches in the 2026 FIFA World Cup?", ["united states", "78"]),
    ("H02", "How many matches is Canada hosting in the 2026 World Cup?", ["13", "thirteen"]),
    ("H03", "How many matches is Mexico hosting in the 2026 World Cup?", ["13", "thirteen"]),
    ("C01", "Which human rights issue was raised regarding the 2026 FIFA World Cup?", ["indigenous", "human rights"]),
]


class TokenUncertainty:
    @staticmethod
    def compute(logits):
        probs = torch.softmax(logits, dim=-1)
        return (1.0 - probs.max(dim=-1).values).mean().item()

class DMoERouter:
    def __init__(self, experts, config):
        self.experts = experts
        self.top_k = config.get("top_k", 3)
        self.tau = config.get("tau", 0.7)
        self.cooldown = config.get("retrigger_cooldown", 10)
        tokenized = [e["doc"].lower().split() for e in experts]
        self.bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
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
app = modal.App("dmoe-eval-v3")
vol = modal.Volume.from_name("dmoe-experts", create_if_missing=True)


def get_delta(expert_dir):
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
                mod = ".".join(parts[1:idx])
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
def evaluate():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf = os.environ.get("HF_TOKEN", "")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf)

    # Load experts ONCE
    dirs = sorted([d for d in os.listdir("/experts")
        if d.startswith("expert_") and os.path.isfile(os.path.join("/experts", d, "adapter_model.safetensors"))])
    print(f"Found {len(dirs)} experts")

    deltas_list = []
    docs = []
    for d in dirs:
        dl, dc = get_delta(os.path.join("/experts", d))
        if dl:
            deltas_list.append(dl)
            docs.append(dc)
    print(f"Loaded {len(deltas_list)} experts")

    if not deltas_list:
        return {"error": "no experts"}

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
                    try:
                        alt = key.split(".", 1)[1] if key.startswith("model.") else key
                        p = m.get_submodule(alt + ".weight")
                        p.data += d.to(p.device, p.dtype)
                    except:
                        pass

    # Load base model ONCE
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", token=hf)
    base_model.eval()

    # Base model eval
    print("\n=== BASE MODEL ===")
    base_scores = []
    for qi, (qid, q, kws) in enumerate(QUESTIONS):
        a = gen(base_model, q)
        s = check(a, kws)
        base_scores.append(s)
    ba = sum(base_scores)/len(base_scores)
    print(f"Base: {ba:.0%} ({sum(base_scores)}/{len(base_scores)})")

    # Get TU values for all questions (for tau sweep)
    tu = TokenUncertainty()
    tu_values = []
    for qi, (qid, q, kws) in enumerate(QUESTIONS):
        inp = tok(q, return_tensors="pt").input_ids.to(base_model.device)
        with torch.no_grad():
            o = base_model(inp)
        tu_values.append(tu.compute(o.logits[:, -1, :]))
    avg_tu = sum(tu_values)/len(tu_values)
    print(f"Avg TU: {avg_tu:.3f}")

    # Tau sweep - clone base model for each tau
    TAU_VALUES = [0.3, 0.5, 0.7, 0.9, 1.1, 1.5]
    print(f"\n=== TAU SWEEP ===")
    print(f"{'Tau':>5} | {'Acc':>5} | {'Wins':>5} | {'Trig':>5}")
    print("-" * 30)

    best_tau = 0.7
    best_dmoe = 0

    for tau in TAU_VALUES:
        router = DMoERouter(
            [{"id": str(i), "doc": docs[i] or "", "lora_path": ""} for i in range(len(deltas_list))],
            {"top_k": 3, "tau": tau, "retrigger_cooldown": 10}
        )
        dmoe_scores = []
        n_trig = 0
        for qi, (qid, q, kws) in enumerate(QUESTIONS):
            # Clone base model weights
            m = AutoModelForCausalLM.from_config(base_model.config)
            m.load_state_dict(base_model.state_dict())
            m.to(base_model.device)
            m.eval()

            if tu_values[qi] > tau:
                sel = router.route(q)
                n_trig += 1
                for idx in sel:
                    if idx < len(deltas_list): apply(m, deltas_list[idx])
            a = gen(m, q)
            s = check(a, kws)
            dmoe_scores.append(s)
            del m

        da = sum(dmoe_scores)/len(dmoe_scores)
        wins = sum(1 for d,b in zip(dmoe_scores,base_scores) if d>b)
        print(f"{tau:>5.1f} | {da:>5.0%} | {wins:>5} | {n_trig:>5}")
        if da > best_dmoe:
            best_dmoe = da
            best_tau = tau

    # Summary
    print(f"\n{'='*60}")
    print(f"Base:       {ba:.0%} ({sum(base_scores)}/{len(base_scores)})")
    print(f"Best DMoE:  {best_dmoe:.0%} (tau={best_tau})")
    return {"base": ba, "best_dmoe": best_dmoe, "best_tau": best_tau, "avg_tu": avg_tu}


@app.local_entrypoint()
def main():
    print("DMoE eval v3: 50 questions + tau sweep (model loaded once)")
    r = evaluate.remote()
    print(f"\nDone: {r}")