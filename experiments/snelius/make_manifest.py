"""Write manifests/<exp>.<kind>.txt (one command per line) and report status.

    python -m experiments.snelius.make_manifest --exp E1 [--models ...] [--tasks ...] \
        [--seeds ...] [--variants ...] [--out $CDX_RESULTS]
    python -m experiments.snelius.make_manifest --exp E1 --status

Kinds: ``seq`` (sequence models, packed 6 per GPU slice), ``mlp`` (MLP baselines, 12 per
slice), ``eval`` (eval-only E4/E5), ``osrl`` (original OSRL script, E0 fidelity).
Runs that are already DONE are skipped; eval-only runs whose checkpoint is not DONE yet
are skipped and reported. Lines are grouped by task so packed runs share the cache.
Importing this module has no side effects (nothing is submitted).
"""

import argparse
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from experiments.common.expdefs import RunSpec, enumerate_runs
from experiments.common.utils import DONE, FAILED, RUNNING, run_dir_for

KINDS = ("seq", "mlp", "eval", "osrl")


def command_for(r: RunSpec, out: str) -> str:
    q = shlex.quote
    if r.kind in ("seq", "mlp"):
        cmd = (f"python -m experiments.common.run --exp {r.exp} --model {r.model} "
               f"--task {r.task} --seed {r.seed} --out {q(out)}")
        return cmd + (f" --variant {r.variant}" if r.variant else "")
    if r.kind == "eval":
        src_exp, src_var = r.ckpt_from
        cmd = (f"python -m experiments.common.evaluate --exp {r.exp} --ckpt-from {src_exp} "
               f"--model {r.model} --task {r.task} --seed {r.seed} --out {q(out)}")
        if src_var:
            cmd += f" --src-variant {src_var}"
        return cmd + (f" --variant {r.variant}" if r.variant else "")
    if r.kind == "osrl":
        logdir = Path(out).resolve() / r.exp / "osrl_logs"
        name = f"osrl_cdt_s{r.seed}"
        train = (f"(cd OSRL && python examples/train/train_cdt.py --task {r.task} "
                 f"--seed {r.seed} --device cuda --logdir {q(str(logdir))} --group fid "
                 f"--name {name} --num_workers 8)")
        ev = (f"python -m experiments.common.evaluate --exp {r.exp} --model {r.model} "
              f"--task {r.task} --seed {r.seed} --osrl-ckpt {q(str(logdir / 'fid' / name))} "
              f"--out {q(out)}")
        return f"{train} && {ev}"
    raise ValueError(r.kind)


def _job_alive(marker: Path) -> bool:
    """True if the SLURM job named in a RUNNING marker is still queued/running. Markers
    of killed jobs (timeout, preemption) are stale and the run is scheduled again."""
    import re
    import subprocess

    try:
        m = re.search(r"job=(\d+)", marker.read_text())
    except OSError:
        return False
    if not m:
        return False
    try:
        out = subprocess.run(["squeue", "-h", "-j", m.group(1), "-o", "%T"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def run_state(r: RunSpec, out: str) -> str:
    d = run_dir_for(out, r.exp, r.task, r.model, r.variant, r.seed)
    if (d / DONE).exists():
        return "done"
    if (d / RUNNING).exists():
        return "running"
    if (d / FAILED).exists():
        return "failed"
    return "missing"


def source_done(r: RunSpec, out: str) -> bool:
    if r.kind != "eval":
        return True
    src_exp, src_var = r.ckpt_from
    return (run_dir_for(out, src_exp, r.task, r.model, src_var, r.seed) / DONE).exists()


def write_manifests(exp: str, out: str, manifest_dir: Path, **filters) -> Dict[str, int]:
    runs = enumerate_runs(exp, **filters)
    lines: Dict[str, List[str]] = defaultdict(list)
    skipped_done, waiting, running = 0, 0, 0
    for r in runs:
        state = run_state(r, out)
        if state == "done":
            skipped_done += 1
            continue
        if state == "running" and _job_alive(
                run_dir_for(out, r.exp, r.task, r.model, r.variant, r.seed) / RUNNING):
            running += 1
            continue
        if not source_done(r, out):
            waiting += 1
            continue
        lines[r.kind].append(command_for(r, out))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for kind in KINDS:
        path = manifest_dir / f"{exp}.{kind}.txt"
        if lines.get(kind):
            path.write_text("\n".join(lines[kind]) + "\n")
            sizes[kind] = len(lines[kind])
        elif path.exists():
            path.unlink()
    print(f"{exp}: {len(runs)} runs defined, {skipped_done} already DONE, {running} running "
          f"in live jobs, {waiting} waiting for their checkpoint; manifest sizes: "
          f"{sizes or 'none'}")
    return sizes


def status(exp: str, out: str, **filters) -> None:
    runs = enumerate_runs(exp, **filters)
    by_kind: Dict[str, Counter] = defaultdict(Counter)
    failed: List[Path] = []
    for r in runs:
        s = run_state(r, out)
        by_kind[r.kind][s] += 1
        if s == "failed":
            failed.append(run_dir_for(out, r.exp, r.task, r.model, r.variant, r.seed))
    total = Counter()
    for kind, c in sorted(by_kind.items()):
        total.update(c)
        print(f"{exp} [{kind:>4}]  done {c['done']:4d}  running {c['running']:4d}  "
              f"failed {c['failed']:4d}  missing {c['missing']:4d}")
    print(f"{exp} [ all]  done {total['done']:4d}  running {total['running']:4d}  "
          f"failed {total['failed']:4d}  missing {total['missing']:4d}  of {len(runs)}")
    for d in failed[:50]:
        print(f"  FAILED: {d}  (see {d / FAILED})")


def main(argv: Optional[List[str]] = None) -> int:
    import os

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True)
    p.add_argument("--models", nargs="*")
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--variants", nargs="*")
    p.add_argument("--out", default=os.environ.get("CDX_RESULTS", "results"))
    p.add_argument("--manifest-dir", default=str(Path(__file__).resolve().parents[1] /
                                                 "manifests"))
    p.add_argument("--status", action="store_true")
    args = p.parse_args(argv)
    filters = dict(models=args.models or None, tasks=args.tasks or None, seeds=args.seeds,
                   variants=args.variants or None)
    if args.status:
        status(args.exp, args.out, **filters)
    else:
        write_manifests(args.exp, args.out, Path(args.manifest_dir), **filters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
