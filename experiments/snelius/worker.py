"""Work-queue worker: keeps one GPU slice busy with runs pulled from a shared queue.

    python -m experiments.snelius.worker --exps E6full E1 E4 ... --out $CDX_RESULTS \
        [--gate auto|require|off] [--pack-seq P] [--time-limit-s S] [--max-attempts N]

Replaces fixed packs (array task i = manifest lines i*P..i*P+P-1). The fixed packs left
the slice idle while the slowest run of a pack finished, mixed tasks in one pack, and
wasted the rest of ``--time`` after a timeout. Here every worker:

  * walks the queue in priority order (the order of ``--exps``; within an experiment,
    grouped by task) and starts a run whenever enough of the slice is free. Each run takes
    a share of the slice: 1/pack_seq (sequence models), 1/pack_mlp (MLP baselines),
    1/pack_eval (eval-only), 1 for E6full (dedicated, spec E6.2) and the OSRL script;
  * claims a run with an exclusive lock on ``<run_dir>/.lock`` (utils.try_lock_run). The
    lock dies with the process, so a run can never execute twice at once, and a run of a
    killed job is free again immediately (no stale RUNNING markers to clean up);
  * skips runs that are DONE, eval-only runs whose checkpoint is not DONE yet, and runs
    that already used ``--max-attempts`` attempts (a timeout counts as an attempt);
  * stops claiming new runs shortly before the job's time limit, then stops its running
    runs itself (SIGTERM: training saves a checkpoint and exits; the run resumes in the
    next round and the attempt is not counted) and exits with code 0. The job therefore
    never ends as TIMEOUT or FAILED, which matters because the next round of the same
    slice starts only after this one *succeeded* (--dependency=aftercorr);
  * exits when nothing is left to claim (it waits up to ``--idle-wait-s`` for eval-only
    runs whose source is still training elsewhere). Exit code 1 only for a failed E0
    gate, which also cancels the later rounds (--kill-on-invalid-dep).

With ``--gate auto`` (main experiments) the worker first makes sure decisions.yaml exists:
the first worker to get the lock computes it from E0; if the E0 gate fails it writes
GATE_FAILED and every worker exits without running anything. ``--gate require`` only
checks that decisions.yaml exists (manual mode: you wrote it). ``off`` for E0 itself.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from experiments.common.expdefs import RunSpec, enumerate_runs
from experiments.common.utils import (FAILED, PREEMPTED_RC, is_done, run_dir_for,
                                      try_lock_run, unlock_run)
from experiments.snelius.make_manifest import command_for

DEDICATED_EXPS = ("E6full",)
ATTEMPTS = "ATTEMPTS"
GATE_FAILED = "GATE_FAILED"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class Packs:
    seq: int = 6
    mlp: int = 12
    eval: int = 4

    def share(self, r: RunSpec) -> float:
        if r.exp in DEDICATED_EXPS or r.kind == "osrl":
            return 1.0
        return 1.0 / {"seq": self.seq, "mlp": self.mlp, "eval": self.eval}[r.kind]


@dataclass
class Child:
    run: RunSpec
    run_dir: Path
    share: float
    proc: subprocess.Popen
    lock_fd: int
    started: float = field(default_factory=time.time)
    stopped: bool = False


def attempts(run_dir: Path) -> int:
    try:
        return int((run_dir / ATTEMPTS).read_text().strip() or 0)
    except (OSError, ValueError):
        return 0


def source_dir(r: RunSpec, out: str) -> Optional[Path]:
    if r.kind != "eval":
        return None
    src_exp, src_var = r.ckpt_from
    return run_dir_for(out, src_exp, r.task, r.model, src_var, r.seed)


class Worker:

    def __init__(self, runs: List[RunSpec], out: str, packs: Packs, deadline: float,
                 max_attempts: int = 3, idle_wait_s: float = 900, poll_s: float = 10,
                 hard_stop: Optional[float] = None, stop_grace_s: float = 60,
                 cpus: int = 18, log_dir: Path = Path("logs/runs"),
                 command_fn: Callable[[RunSpec, str], str] = command_for,
                 duration_estimate: Optional[Callable[[RunSpec], Optional[float]]] = None):
        self.runs, self.out, self.packs = runs, out, packs
        self.deadline, self.max_attempts = deadline, max_attempts
        self.hard_stop, self.stop_grace_s = hard_stop, stop_grace_s
        self.idle_wait_s, self.poll_s, self.cpus = idle_wait_s, poll_s, cpus
        self.log_dir, self.command_fn = Path(log_dir), command_fn
        self.estimate = duration_estimate or self._estimate_from_results
        self._est_cache: Dict[tuple, Optional[float]] = {}
        self.children: List[Child] = []
        self._done: set = set()
        self.finished = {"ok": 0, "failed": 0, "preempted": 0}

    # ---------------------------------------------------------- bookkeeping
    def used(self) -> float:
        return sum(c.share for c in self.children)

    def mine(self) -> set:
        return {c.run_dir for c in self.children}

    def dedicated_running(self) -> bool:
        return any(c.share >= 1.0 for c in self.children)

    def _estimate_from_results(self, r: RunSpec) -> Optional[float]:
        """Median total_wall_s of up to 10 finished runs of the same exp/model/kind."""
        key = (r.exp, r.model, r.kind)
        if key not in self._est_cache:
            vals = []
            base = Path(self.out) / r.exp
            for s in base.glob(f"*/{r.model}*/seed_*/run_summary.json"):
                try:
                    with open(s) as f:
                        v = json.load(f).get("total_wall_s")
                    if v:
                        vals.append(float(v))
                except (OSError, ValueError):
                    pass
                if len(vals) >= 10:
                    break
            self._est_cache[key] = sorted(vals)[len(vals) // 2] if vals else None
        return self._est_cache[key]

    def _time_ok(self, r: RunSpec) -> bool:
        est = self.estimate(r)
        need = est if est is not None else 600.0  # unknown: start if >= 10 min are left
        return time.time() + need < self.deadline

    # ---------------------------------------------------------- claim / launch
    def claimable(self, r: RunSpec) -> str:
        """'yes', or why not: done / exhausted / blocked."""
        d = run_dir_for(self.out, r.exp, r.task, r.model, r.variant, r.seed)
        if d in self._done:
            return "done"
        if is_done(d):
            self._done.add(d)  # DONE never goes away: no more file checks for this run
            return "done"
        if attempts(d) >= self.max_attempts:
            return "exhausted"
        src = source_dir(r, self.out)
        if src is not None and not is_done(src):
            return "blocked"
        return "yes"

    def launch(self, r: RunSpec, d: Path, fd: int, share: float) -> None:
        (d / ATTEMPTS).write_text(str(attempts(d) + 1))
        name = r.model if not r.variant else f"{r.model}__{r.variant}"
        logf = self.log_dir / r.exp / r.task / f"{name}_seed{r.seed}.log"
        logf.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(CDX_LOCK_HELD="1", PYTHONHASHSEED=str(r.seed), OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", CDX_PACK=str(round(1 / share)),
                   CDX_EVAL_WORKERS=str(max(2, int(self.cpus * share))))
        cmd = self.command_fn(r, self.out)
        fh = open(logf, "a")
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} attempt {attempts(d)} "
                 f"job={os.environ.get('SLURM_JOB_ID', '-')}\n$ {cmd}\n")
        fh.flush()
        proc = subprocess.Popen(["bash", "-c", cmd], stdout=fh, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        fh.close()
        self.children.append(Child(r, d, share, proc, fd))
        log(f"start  {r.exp} {r.task} {name} seed {r.seed} (share {share:.2f}, "
            f"used {self.used():.2f}) -> {logf}")

    def fill(self) -> int:
        """Start runs until the slice is full; returns the number of runs started."""
        started = 0
        if self.dedicated_running():
            return 0
        for r in self.runs:
            free = 1.0 - self.used()
            if free <= 1e-6:
                break
            share = self.packs.share(r)
            if share > free + 1e-9:
                continue  # does not fit now (dedicated runs need an empty slice)
            if self.claimable(r) != "yes" or not self._time_ok(r):
                continue
            d = run_dir_for(self.out, r.exp, r.task, r.model, r.variant, r.seed)
            if d in self.mine():
                # POSIX locks belong to the process: our own lock would be granted again,
                # and closing a second fd would even drop it. Never touch our own runs.
                continue
            fd = try_lock_run(d)
            if fd is None:
                continue  # another worker runs it
            if is_done(d):  # finished between the check and the lock
                unlock_run(fd)
                continue
            self.launch(r, d, fd, share)
            started += 1
            if share >= 1.0:
                break
        return started

    def reap(self) -> None:
        for c in list(self.children):
            rc = c.proc.poll()
            if rc is None:
                continue
            self.children.remove(c)
            name = c.run.model if not c.run.variant else f"{c.run.model}__{c.run.variant}"
            ok = rc == 0 and is_done(c.run_dir)
            if not ok and (rc == PREEMPTED_RC or c.stopped):
                # stopped by us before the time limit: not a failed attempt
                n = attempts(c.run_dir)
                (c.run_dir / ATTEMPTS).write_text(str(max(0, n - 1)))
                self.finished["preempted"] += 1
                log(f"stop  {c.run.exp} {c.run.task} {name} seed {c.run.seed} rc={rc}: "
                    "resumes from its checkpoint in the next round")
            else:
                self.finished["ok" if ok else "failed"] += 1
                log(f"{'done ' if ok else 'FAIL '} {c.run.exp} {c.run.task} {name} seed "
                    f"{c.run.seed} rc={rc} after {time.time() - c.started:.0f}s"
                    + ("" if ok else f" (see {c.run_dir / FAILED})"))
            unlock_run(c.lock_fd)

    def pending_blocked(self) -> int:
        """Eval-only runs that wait for a checkpoint that may still appear."""
        n = 0
        for r in self.runs:
            if self.claimable(r) != "blocked":
                continue
            src = source_dir(r, self.out)
            if attempts(src) < self.max_attempts:
                n += 1
        return n

    def stop_children(self, sig=signal.SIGTERM) -> None:
        for c in self.children:
            c.stopped = True
            try:
                os.killpg(c.proc.pid, sig)
            except ProcessLookupError:
                pass

    def shutdown(self) -> None:
        """SIGTERM to every run (training checkpoints and exits), SIGKILL after the grace
        period; the job then ends by itself, with exit code 0, before SLURM's limit."""
        log(f"time limit close: stopping {len(self.children)} runs (checkpoint + resume "
            "next round)")
        self.stop_children(signal.SIGTERM)
        t_end = time.time() + self.stop_grace_s
        while self.children and time.time() < t_end:
            self.reap()
            time.sleep(1)
        if self.children:
            self.stop_children(signal.SIGKILL)
            time.sleep(2)
            self.reap()

    # ---------------------------------------------------------- main loop
    def loop(self) -> Dict[str, int]:
        idle_since: Optional[float] = None
        while True:
            if self.hard_stop is not None and time.time() >= self.hard_stop and self.children:
                self.shutdown()
                break
            self.reap()
            started = self.fill() if time.time() < self.deadline else 0
            if self.children or started:
                idle_since = None
                time.sleep(self.poll_s)
                continue
            # nothing running here and nothing claimable right now
            if time.time() >= self.deadline:
                log("time limit reached: no new runs")
                break
            blocked = self.pending_blocked()
            if not blocked:
                log("queue empty")
                break
            idle_since = idle_since or time.time()
            if time.time() - idle_since > self.idle_wait_s:
                log(f"{blocked} eval-only runs still wait for checkpoints: leaving them to "
                    "the next round")
                break
            time.sleep(max(self.poll_s, 30))
        log(f"worker finished: {self.finished['ok']} ok, {self.finished['failed']} failed, "
            f"{self.finished['preempted']} stopped for the next round")
        return self.finished


# ------------------------------------------------------------------ gate
def ensure_gate(out: Path, mode: str) -> bool:
    from experiments.snelius import decisions as dec

    if mode == "off":
        return True
    if (out / GATE_FAILED).exists():
        log(f"E0 gate failed earlier: {out / GATE_FAILED}")
        return False
    if mode == "require":
        ok = (out / dec.DECISIONS_FILE).exists()
        if not ok:
            log(f"{out / dec.DECISIONS_FILE} missing (manual gate): write it with "
                "python -m experiments.snelius.decisions --write")
        return ok
    gate_dir = out / ".gate"
    while True:  # one worker decides, the others wait for its result
        fd = try_lock_run(gate_dir)
        if fd is not None:
            break
        time.sleep(5)
    try:
        if (out / dec.DECISIONS_FILE).exists():
            return True
        if (out / GATE_FAILED).exists():
            return False
        prop = dec.propose(out)
        if not prop["gate_passed"]:
            import yaml

            (out / GATE_FAILED).write_text(yaml.safe_dump(prop, sort_keys=False))
            log(f"E0 GATE FAILED: see {out / GATE_FAILED}")
            return False
        path = dec.write(out, prop)
        log(f"E0 gate passed: {path} (precision={prop['precision']}, "
            f"lr={prop['learning_rate']}, pack_seq={prop['pack_seq']})")
        return True
    finally:
        unlock_run(fd)


# ------------------------------------------------------------------ CLI
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exps", nargs="+", required=True, help="in priority order")
    p.add_argument("--models", nargs="*")
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--variants", nargs="*")
    p.add_argument("--out", default=os.environ.get("CDX_RESULTS", "results"))
    p.add_argument("--gate", choices=["auto", "require", "off"], default="off")
    p.add_argument("--pack-seq", type=int, default=None,
                   help="default: decisions.yaml pack_seq, else $CDX_PACK_SEQ")
    p.add_argument("--pack-mlp", type=int, default=int(os.environ.get("CDX_PACK_MLP", 12)))
    p.add_argument("--pack-eval", type=int, default=int(os.environ.get("CDX_PACK_EVAL", 4)))
    p.add_argument("--time-limit-s", type=float,
                   default=float(os.environ.get("CDX_TIME_LIMIT_S", 0)) or None,
                   help="the job's --time in seconds (set by the sbatch script)")
    p.add_argument("--margin-s", type=float, default=300,
                   help="no new runs this long before the hard stop")
    p.add_argument("--stop-before-s", type=float, default=180,
                   help="stop running runs this long before the job's time limit")
    p.add_argument("--max-attempts", type=int,
                   default=int(os.environ.get("CDX_MAX_ATTEMPTS", 3)))
    p.add_argument("--idle-wait-s", type=float,
                   default=float(os.environ.get("CDX_IDLE_WAIT_S", 900)))
    p.add_argument("--log-dir", default="logs/runs")
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not ensure_gate(out, a.gate):
        return 1
    pack_seq = a.pack_seq
    if pack_seq is None:
        from experiments.snelius.decisions import load_decisions

        d = load_decisions(str(out)) if a.gate != "off" else None
        pack_seq = int((d or {}).get("pack_seq") or os.environ.get("CDX_PACK_SEQ", 6))
    packs = Packs(seq=pack_seq, mlp=a.pack_mlp, eval=a.pack_eval)

    runs: List[RunSpec] = []
    for exp in a.exps:
        runs += enumerate_runs(exp, models=a.models or None, tasks=a.tasks or None,
                               variants=a.variants or None)
    # the time limit counts from the job start (set by worker.sbatch), not from here
    start = float(os.environ.get("CDX_JOB_T0", time.time()))
    if a.time_limit_s:
        end = start + a.time_limit_s
        hard_stop = end - a.stop_before_s       # stop runs, leave the job cleanly
        deadline = hard_stop - a.margin_s       # no new runs after this
    else:
        hard_stop, deadline = None, start + 10 * 365 * 86400
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 8))
    w = Worker(runs, str(out), packs, deadline, a.max_attempts, a.idle_wait_s, cpus=cpus,
               log_dir=Path(a.log_dir), hard_stop=hard_stop)
    log(f"worker on {os.uname().nodename}: {len(runs)} runs in queue ({' '.join(a.exps)}), "
        f"packs seq={packs.seq} mlp={packs.mlp} eval={packs.eval}, "
        f"time limit {a.time_limit_s or 'none'} s")

    def on_term(signum, frame):  # scancel / node shutdown: let runs checkpoint, exit
        log(f"signal {signum}: stopping {len(w.children)} runs (they resume next round)")
        w.shutdown()
        sys.exit(143)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGUSR1, on_term)
    w.loop()
    # failed runs are recorded in their run dirs (FAILED, ATTEMPTS); the job itself
    # succeeds so the next round of this slice (aftercorr) starts
    return 0


if __name__ == "__main__":
    sys.exit(main())
