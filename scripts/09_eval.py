"""Evaluate the gate and the scope check on eval/golden.jsonl, and help calibrate the thresholds.

No LLM is called: only stages 1-3 of the pipeline run.
    python scripts/09_eval.py            # accuracy with the current thresholds + suggested values
    python scripts/09_eval.py --verbose  # one line per question with its scores
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, gate  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402


def best_threshold(values_below, values_above):
    """Threshold that best separates the two groups (values_above should score >= threshold)."""
    best = (0, 0.0)
    for t in sorted(set(values_below + values_above)):
        ok = sum(v < t for v in values_below) + sum(v >= t for v in values_above)
        best = max(best, (ok, t))
    return best[1], best[0], len(values_below) + len(values_above)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(config.ROOT / "eval" / "golden.jsonl", encoding="utf-8") if l.strip()]
    p = Pipeline(log=False)

    hit, tot, top, pat, wrong = defaultdict(int), defaultdict(int), defaultdict(list), defaultdict(list), []
    for r in rows:
        res, _ = p.prepare(r["q"])
        # raw scores, independent of the current thresholds and of the rule gate
        qvec, plain = p.retriever.embed(r["q"])
        s_top = p.retriever.search(qvec, 1)[0]["score"]
        s_pat = gate.patient_similarity(plain, p.prototypes)
        rule = gate.rule_gate(r["q"])[0]
        top[r["label"]].append(s_top)
        if rule == gate.OK:                      # only questions the rules let through reach the semantic gate
            pat[r["label"]].append(s_pat)
        tot[r["label"]] += 1
        ok = res.decision == r["label"]
        hit[r["label"]] += ok
        if not ok:
            wrong.append((r["label"], res.decision, r["q"]))
        if a.verbose:
            print(f"{'ok ' if ok else 'BAD'} {r['label']:17} -> {res.decision:17} rule={rule:17} "
                  f"top={s_top:.3f} patient={s_pat:.3f}  {r['q']}")

    print(f"\nThresholds: SCOPE_MIN={config.SCOPE_MIN}  PATIENT_MIN={config.PATIENT_MIN}")
    for k in tot:
        print(f"  {k:17} {hit[k]:2}/{tot[k]:2}")
    print(f"  {'TOTAL':17} {sum(hit.values()):2}/{sum(tot.values()):2}")
    for label, got, q in wrong:
        print(f"  wrong: expected {label}, got {got}: {q}")

    t, ok, n = best_threshold(top["out_of_scope"], top["answered"])
    print(f"\nScope: answered min={min(top['answered']):.3f}  out_of_scope max={max(top['out_of_scope']):.3f}"
          f"  -> best SCOPE_MIN ~ {t:.3f} ({ok}/{n} separated)")
    missed = pat["patient_specific"] + pat["emergency"]
    print(f"Patient similarity among questions passing the rules: answered max={max(pat['answered']):.3f}"
          + (f", missed patient/emergency: {', '.join(f'{v:.3f}' for v in missed)}" if missed
             else ", no patient/emergency question passed the rules"))
