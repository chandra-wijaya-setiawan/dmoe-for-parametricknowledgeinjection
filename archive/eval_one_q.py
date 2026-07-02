"""
Single-question DMoE evaluator. Run once per question as a subprocess.
Usage: python scripts/eval_one_q.py <json_args>
"""
import gc, json, os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).parent.parent))
from dmoe.router.bm25_router import DMoERouter, TokenUncertainty
from dmoe.models.expert_manager import ExpertManager, load_expert_delta, Expert

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
EXPERTS_DIR = Path("experts_focused")
EXPERT_IDS = ["expert_wc2026_start_end", "expert_group_a_teams", "expert_group_a_matches",
              "expert_group_a_standings", "expert_group_a_players_mexico",
              "expert_group_a_players_south_africa", "expert_group_a_players_south_korea",
              "expert_group_a_players_czech"]

def check(ans, kws):
    al = ans.lower()
    return 1 if any(kw.lower() in al for kw in kws) else 0

def main():
    args = json.loads(sys.argv[1])
    question = args["question"]
    keywords = args["keywords"]
    label = args["label"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    inp = tokenizer(question, return_tensors="pt").input_ids

    # ---- BASE ----
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=40, temperature=0.0, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
    base_answer = tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
    base_correct = check(base_answer, keywords)
    del model, out; gc.collect()

    # ---- DMOE ----
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    em = ExpertManager(model)

    # Load experts
    deltas_list, docs_list = [], []
    for eid in EXPERT_IDS:
        ep = EXPERTS_DIR / eid
        dd, doc, _ = load_expert_delta(str(ep))
        if dd:
            deltas_list.append(dd)
            docs_list.append(doc)
        else:
            docs_list.append("")

    # Router
    expert_list = [{"id": eid, "doc": docs_list[i], "lora_path": str(EXPERTS_DIR / eid)}
                   for i, eid in enumerate(EXPERT_IDS)]
    router = DMoERouter(expert_list, {"top_k": 3, "tau": 2.0, "retrigger_cooldown": 10,
                                       "bm25_k1": 1.5, "bm25_b": 0.75})
    tu_comp = TokenUncertainty()

    # TU
    with torch.no_grad():
        out = model(inp)
    tu = tu_comp.compute(out.logits[:, -1, :])

    # Generate with routing
    generated = inp.clone()
    selected, triggered = [], False
    for step in range(80):
        outputs = model(input_ids=generated[:, -1:] if step > 0 else generated, use_cache=True)
        logits = outputs.logits[:, -1, :]
        if router.should_trigger(logits) and not triggered:
            ctx = tokenizer.decode(generated[0, inp.shape[1]:])
            indices = router.route(ctx)
            if indices:
                triggered = True
                loaded = []
                for idx in indices:
                    if idx < len(deltas_list):
                        loaded.append(Expert(id=EXPERT_IDS[idx], lora_path=str(EXPERTS_DIR / EXPERT_IDS[idx]),
                                              doc=docs_list[idx], deltas=deltas_list[idx]))
                        selected.append(EXPERT_IDS[idx])
                em.activate_experts(loaded)
        next_tok = logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=-1)
        if next_tok.item() == tokenizer.eos_token_id:
            break

    dmoe_answer = tokenizer.decode(generated[0, inp.shape[1]:], skip_special_tokens=True)
    dmoe_correct = check(dmoe_answer, keywords)

    result = {"label": label, "base_answer": base_answer[:80], "base_correct": base_correct,
              "dmoe_answer": dmoe_answer[:80], "dmoe_correct": dmoe_correct,
              "tu": round(tu, 4), "triggered": triggered, "selected": selected}
    print(json.dumps(result))

if __name__ == "__main__":
    main()