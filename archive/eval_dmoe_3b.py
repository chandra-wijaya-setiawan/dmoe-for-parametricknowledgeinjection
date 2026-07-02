"""
DMoE Evaluation on Qwen2.5-3B-Instruct (cpubox).
Paper-faithful: entropy-based TU (τ=2.0), BM25 router (top-k=3), final-layer FFN LoRA.
"""
import gc, json, os, sys, time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dmoe.router.bm25_router import DMoERouter, TokenUncertainty
from dmoe.models.expert_manager import ExpertManager, load_expert_delta, Expert

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
EXPERTS_DIR = Path("experts_3b")
device = "cpu"

# Trained expert IDs
EXPERT_IDS = [
    "expert_wc2026_start_end", "expert_group_a_teams", "expert_group_a_matches",
    "expert_group_a_standings", "expert_group_a_players_mexico",
    "expert_group_a_players_south_africa", "expert_group_a_players_south_korea",
    "expert_group_a_players_czech",
]

QUESTIONS = [
    ("Q1: Start date",       "On what exact date did the 2026 FIFA World Cup begin?", ["june 11"]),
    ("Q2: End date",         "On what exact date is the 2026 FIFA World Cup scheduled to conclude?", ["july 19"]),
    ("Q3: Start+End",        "When does the 2026 FIFA World Cup start and end?", ["june 11", "july 19"]),
    ("Q4: Tournament length","How many days does the 2026 FIFA World Cup last?", ["39"]),
    ("Q5: Group A teams",    "Which four teams were in Group A of the 2026 FIFA World Cup?", ["mexico", "south korea", "south africa", "czech"]),
    ("Q6: Group A winner",   "Who won Group A of the 2026 FIFA World Cup?", ["mexico"]),
    ("Q7: Who advanced",     "Which teams advanced to the knockout stage from Group A?", ["mexico", "south africa"]),
    ("Q8: Last place",       "Which team finished last in Group A and was eliminated?", ["czech"]),
    ("Q9: Mexico points",    "How many points did Mexico get in Group A?", ["9"]),
    ("Q10: Mexico vs SA",    "What was the result of the Mexico vs South Africa match in Group A?", ["2-0", "2–0"]),
    ("Q11: SK vs CZ",        "What was the result of the South Korea vs Czech Republic match in Group A?", ["2-1"]),
    ("Q12: CZ vs SA",        "What was the result of the Czech Republic vs South Africa match in Group A?", ["1-1"]),
    ("Q13: Mexico vs SK",    "What was the result of the Mexico vs South Korea match in Group A?", ["1-0"]),
    ("Q14: CZ vs Mexico",    "What was the result of the Czech Republic vs Mexico match in Group A?", ["3-0"]),
    ("Q15: SA vs SK",        "What was the result of the South Africa vs South Korea match in Group A?", ["1-0"]),
    ("Q16: SA historic",     "What historic achievement did South Africa accomplish in Group A?", ["first time", "knockout"]),
    ("Q17: Mexico manager",  "Who was Mexico's manager in the 2026 FIFA World Cup?", ["javier aguirre"]),
    ("Q18: SA manager",      "Who was South Africa's manager?", ["hugo broos"]),
    ("Q19: Mexico goalscorer","Who scored Mexico's opening goal against South Africa?", ["quiñones", "quinones"]),
    ("Q20: SA goalscorer",   "Who scored the winning goal for South Africa against South Korea?", ["maseko"]),
]

def check(ans, kws):
    al = ans.lower()
    return 1 if any(kw.lower() in al for kw in kws) else 0

def load_experts():
    deltas_list, docs_list = [], []
    for eid in EXPERT_IDS:
        ep = EXPERTS_DIR / eid
        dd, doc, _ = load_expert_delta(str(ep))
        if dd:
            deltas_list.append(dd)
            docs_list.append(doc or "")
    return deltas_list, docs_list

def main():
    print("=" * 70)
    print("DMoE: Parametric Knowledge Injection")
    print(f"Model: {MODEL_NAME}")
    print(f"Experts: {len(EXPERT_IDS)} from {EXPERTS_DIR}/")
    print(f"Router: BM25 τ=2.0 (entropy TU), top-k=3")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    deltas_list, docs_list = load_experts()

    results = []
    for i, (label, question, keywords) in enumerate(QUESTIONS):
        print(f"\n[{i+1:2d}/20] {label}: {question[:55]}...")
        inp = tokenizer(question, return_tensors="pt").input_ids

        # === BASE ===
        gc.collect()
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=40, temperature=0.0, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id)
        base_ans = tokenizer.decode(out[0][inp.shape[1]:], skip_special_tokens=True)
        base_correct = check(base_ans, keywords)
        print(f"  BASE: [{'✓' if base_correct else '✗'}] {base_ans[:70]}")
        del model, out; gc.collect()

        # === DMOE ===
        gc.collect()
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        model.eval()
        em = ExpertManager(model)

        elist = [{"id": eid, "doc": docs_list[j], "lora_path": str(EXPERTS_DIR / eid)}
                 for j, eid in enumerate(EXPERT_IDS)]
        router = DMoERouter(elist, {"top_k": 3, "tau": 2.0, "retrigger_cooldown": 10,
                                     "bm25_k1": 1.5, "bm25_b": 0.75})
        tu_comp = TokenUncertainty()

        with torch.no_grad():
            out = model(inp)
        tu = tu_comp.compute(out.logits[:, -1, :])

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

        dmoe_ans = tokenizer.decode(generated[0, inp.shape[1]:], skip_special_tokens=True)
        dmoe_correct = check(dmoe_ans, keywords)

        arrow = "+" if dmoe_correct and not base_correct else ("-" if not dmoe_correct and base_correct else "=")
        trig_info = f"TU={tu:.3f}, activated={selected}" if triggered else f"TU={tu:.3f}, no trigger"
        print(f"  DMoE: [{'✓' if dmoe_correct else '✗'}] {arrow}  {trig_info}")
        if dmoe_correct:
            print(f"        Answer: {dmoe_ans[:80]}")
        elif triggered:
            print(f"        (routed to {selected}) Ans: {dmoe_ans[:80]}")

        results.append({"label": label, "base_correct": base_correct, "dmoe_correct": dmoe_correct,
                        "base_answer": base_ans[:80], "dmoe_answer": dmoe_ans[:80],
                        "tu": tu, "triggered": triggered, "selected": selected})

        em.deactivate_all()
        del model, em; gc.collect()

    # === RESULTS ===
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    total = len(results)
    bc = sum(1 for r in results if r["base_correct"])
    dc = sum(1 for r in results if r["dmoe_correct"])
    dw = sum(1 for r in results if r["dmoe_correct"] and not r["base_correct"])
    bw = sum(1 for r in results if not r["dmoe_correct"] and r["base_correct"])

    print(f"\n  {'':20s} {'Base':>12s}   {'DMoE':>12s}   {'Δ':>8s}")
    print(f"  {'-'*52}")
    print(f"  {'Accuracy':20s} {bc/total:>10.1%}   {dc/total:>10.1%}   {(dc-bc)/total:>+7.1%}")
    print(f"  DMoE wins: {dw}, Base wins: {bw}")

    print(f"\n  {'Question':40s} {'Base':>8s} {'DMoE':>8s} {'Route':>8s}")
    print(f"  {'-'*64}")
    for r in results:
        b = "✓" if r["base_correct"] else "✗"
        d = "✓" if r["dmoe_correct"] else "✗"
        arr = "+" if r["dmoe_correct"] and not r["base_correct"] else ("-" if not r["dmoe_correct"] and r["base_correct"] else "=")
        rt = "→" if r["triggered"] else " "
        print(f"  {r['label']:40s}   {b:>4s} {arr:>1s}  {d:>4s}   {rt:>6s}")

    # By topic
    print(f"\n{'='*70}")
    print("BY TOPIC")
    print(f"{'='*70}")
    for tname, idxs in [("Topic 1: WC2026 Dates", range(4)), ("Topic 2: Group A", range(4,20))]:
        tb = sum(results[j]["base_correct"] for j in idxs) / len(idxs)
        td = sum(results[j]["dmoe_correct"] for j in idxs) / len(idxs)
        print(f"  {tname:55s}: Base={tb:.0%}  DMoE={td:.0%}  Δ={td-tb:+.0%}")

    # Routing decisions
    print(f"\n{'='*70}")
    print("DMoE ROUTING")
    print(f"{'='*70}")
    for r in results:
        if r["triggered"]:
            print(f"  {r['label']:25s}: TU={r['tu']:.3f} > 2.0 → {r['selected']}")

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print(f"  Qwen2.5-3B + {len(EXPERT_IDS)} focused LoRA experts + BM25 routing")
    print(f"  Base: {bc/total:.0%}  →  DMoE: {dc/total:.0%}  (Δ={(dc-bc)/total:+.0%})")

if __name__ == "__main__":
    main()