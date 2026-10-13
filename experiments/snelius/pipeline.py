"""Planning helper for run_all.sh: what is left to do, stage by stage.

    python -m experiments.snelius.pipeline plan   --out $CDX_RESULTS [--p3]   # table + JSON
    python -m experiments.snelius.pipeline status --out $CDX_RESULTS [--p3]   # table only

Read-only: looks at DONE markers and the gate/bootstrap files under the results root.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.common.expdefs import enumerate_runs
from experiments.common.utils import is_done, run_dir_for

E0_EXPS = ["E0", "E0prec", "E0tune"]
E0PACK_SIZES = [1, 3, 6, 9]
# main queue in priority order (spec §6/§7): P0 first, dependent eval-only runs after
# their sources; E6full first so that empty slices pick the dedicated runs up at once
MAIN_EXPS = ["E6full", "E1", "E4", "E2", "E3", "E5", "E7", "E8", "E9"]
P3_EXPS = ["E6long", "E7ext", "E11"]


def counts(exp: str, out: str, variants: Optional[List[str]] = None) -> Dict[str, Any]:
    runs = enumerate_runs(exp, variants=variants)
    done = sum(is_done(run_dir_for(out, r.exp, r.task, r.model, r.variant, r.seed))
               for r in runs)
    kinds: Dict[str, int] = {}
    for r in runs:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    return {"exp": exp, "total": len(runs), "done": done, "left": len(runs) - done,
            "kinds": kinds}


def slices_needed(exps: List[str], out: str, pack_seq: int, pack_mlp: int,
                  pack_eval: int) -> int:
    """GPU slices that would keep every remaining run of these experiments busy at once."""
    from experiments.snelius.worker import Packs

    packs = Packs(pack_seq, pack_mlp, pack_eval)
    total = 0.0
    for exp in exps:
        for r in enumerate_runs(exp):
            if not is_done(run_dir_for(out, r.exp, r.task, r.model, r.variant, r.seed)):
                total += packs.share(r)
    return max(1, math.ceil(total)) if total else 0


def check_exps(exps: List[str]) -> List[str]:
    """Known experiment names only; warn when eval-only runs depend on an experiment that
    is not in the list (they then wait for checkpoints that nobody trains)."""
    from experiments.common.expdefs import EXPERIMENTS, get_exp

    bad = [e for e in exps if e not in EXPERIMENTS or e.startswith("E0") or e == "E10"]
    if bad:
        raise SystemExit(f"--exps: not a main experiment: {bad} "
                         f"(choose from {MAIN_EXPS + P3_EXPS})")
    for e in exps:
        for src, _, _ in get_exp(e).eval_sources.values():
            if src not in exps:
                print(f"WARNING: {e} evaluates {src} checkpoints but {src} is not in --exps "
                      f"(fine only if {src} already finished)")
    return exps


def plan(out: str, p3: bool, exps: Optional[List[str]] = None) -> Dict[str, Any]:
    from experiments.snelius.decisions import DECISIONS_FILE, load_decisions
    from experiments.snelius.worker import GATE_FAILED

    o = Path(out)
    boot = {}
    if (o / "bootstrap.json").exists():
        with open(o / "bootstrap.json") as f:
            boot = json.load(f)
    dec = load_decisions(out)
    main_exps = check_exps(exps) if exps else MAIN_EXPS + (P3_EXPS if p3 else [])
    env = os.environ
    pack_seq = int((dec or {}).get("pack_seq") or env.get("CDX_PACK_SEQ", 6))
    pm, pe = int(env.get("CDX_PACK_MLP", 12)), int(env.get("CDX_PACK_EVAL", 4))
    table = [counts(e, out) for e in E0_EXPS + ["E0pack", "E0fid"] + main_exps]
    e0_left = sum(t["left"] for t in table if t["exp"] in E0_EXPS)
    pack_left = [p for p in E0PACK_SIZES if counts("E0pack", out, [f"p{p}"])["left"]]
    return {
        "bootstrap_needed": not boot.get("ok", False),
        "gate": ("decided" if dec else "failed" if (o / GATE_FAILED).exists() else "open"),
        "decisions_file": str(o / DECISIONS_FILE),
        "e0_slices": slices_needed(E0_EXPS, out, int(env.get("CDX_PACK_SEQ", 6)), pm, pe)
        if not dec else 0,
        "e0pack_sizes": pack_left if not dec else [],
        "e0fid_needed": any(t["left"] for t in table if t["exp"] == "E0fid"),
        "main_exps": main_exps,
        "main_slices": slices_needed(main_exps, out, pack_seq, pm, pe),
        "main_left": sum(t["left"] for t in table if t["exp"] in main_exps),
        "profile_needed": not (o / "E6" / "PROFILE_DONE").exists(),
        "e0_left": e0_left,
        "table": table,
    }


def print_table(p: Dict[str, Any]) -> None:
    print(f"{'experiment':<10} {'runs':>6} {'done':>6} {'left':>6}  kinds")
    for t in p["table"]:
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(t["kinds"].items()))
        print(f"{t['exp']:<10} {t['total']:>6} {t['done']:>6} {t['left']:>6}  {kinds}")
    print(f"bootstrap: {'needed' if p['bootstrap_needed'] else 'ok'}   "
          f"E0 gate: {p['gate']}   profiling (E6): "
          f"{'needed' if p['profile_needed'] else 'done'}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "status"])
    ap.add_argument("--out", default=os.environ.get("CDX_RESULTS", "results"))
    ap.add_argument("--p3", action="store_true")
    ap.add_argument("--json", default=None, help="plan: also write the plan here")
    ap.add_argument("--exps", default="", help="comma-separated main experiments")
    a = ap.parse_args(argv)
    p = plan(a.out, a.p3, [e for e in a.exps.split(",") if e] or None)
    print_table(p)
    if a.cmd == "plan" and a.json:
        Path(a.json).write_text(json.dumps({k: v for k, v in p.items() if k != "table"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
