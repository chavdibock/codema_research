# Running the experiments on Snellius: step-by-step guide

This guide goes from an empty Snellius account to the final tables and figures.
- **Where to run commands:** every command runs from the `Experiments/` directory (the parent of `experiments/`), unless the step says otherwise.
- **Submitting jobs:** nothing is submitted until you run `submit.sh` or `sbatch` yourself, and `submit.sh` asks for confirmation before each array.
- **Background reading:** `README.md` is the reference, and `NOTES.md` §1 (V1–V14) lists every item that could only be checked on Snellius. This guide tells you when to check each one.

Phases:

| Phase | What | GPU cost |
|---|---|---|
| 0 | Copy the code, set the paths | none |
| 1 | Build and check the environment | ~2 h, one GPU |
| 2 | Download data, build the cache | ~2 h, one GPU |
| 3 | E0: the gate (pilot, precision, tuning, pack size, fidelity) | small |
| 4 | Fix settings based on E0 | none |
| 5 | P0 experiments: E1, E4, E6, E10 | main budget |
| 6 | P1/P2 experiments: E2, E3, E5, E7, E8, E9 (P3 if budget allows) | main budget |
| 7 | Analysis: tables and figures | none (login node or laptop) |

---

## Phase 0: copy the code and set the paths

### 0.1 Copy `Experiments/` to Snellius

`experiments/` is not a git repository, so copy it with `rsync` from your laptop:

```bash
# on your laptop, from the folder that contains Experiments/
rsync -av --progress \
  --exclude '__pycache__' --exclude '.pytest_cache' --exclude '*.pdf' --exclude '*.docx' \
  --exclude 'experiments/results' --exclude 'experiments/cache' \
  --exclude 'experiments/manifests/*.txt' --exclude 'experiments/manifests/submitted' \
  Experiments/ <user>@snellius.surf.nl:~/cdx/Experiments/
```

- **Keep `OSRL/.git`:** each run records the OSRL commit, which is read from it.
- **Code location:** the code can live in `$HOME`. Results, cache and datasets must go to project space (next step).
- **Re-syncing after edits:** re-run the same `rsync` whenever you change code locally. Don't re-sync code while an experiment's jobs are running, because runs that start later would use the new code.

### 0.2 Choose project space and make it permanent

```bash
ssh <user>@snellius.surf.nl
cd ~/cdx/Experiments
echo 'export CDX_PROJECT=/projects/0/<your-project>/cdx' >> ~/.bashrc   # project space, never $HOME
source ~/.bashrc
```

- **Required variable:** `CDX_PROJECT` is the only variable you must set. `experiments/snelius/env.sh` derives the others:

  | Variable | Default |
  |---|---|
  | `CDX_RESULTS` | `$CDX_PROJECT/results` |
  | `CDX_CACHE` | `$CDX_PROJECT/cache` |
  | `DSRL_DATASET_DIR` | `$CDX_PROJECT/dsrl_datasets` |
  | `CDX_VENV` | `$HOME/venvs/cdx` |

- **No project space:** use `/scratch-shared/$USER/cdx`. Scratch is purged after a few weeks, so copy `results/` out regularly.
- **Jobs and the variable:** jobs inherit `CDX_PROJECT` because sbatch exports your environment.

### 0.3 Check budget and partitions (login node, read-only)

```bash
bash experiments/snelius/check_partitions.sh
```

1. Note your **account** (set `export CDX_ACCOUNT=<account>` in `~/.bashrc` if you have more than one) and your remaining **SBU**.
2. Look up the SBU rate per GPU-hour for `gpu_a100` in the Snellius docs and set `export CDX_SBU_PER_GPU_HOUR=<rate>`. `submit.sh` then also prints SBU estimates.

---

## Phase 1: build and check the environment (once)

### 1.1 Check module names (NOTES V8)

```bash
module load 2023
module avail Python CUDA
```

If the names differ from the default `2023 Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1`, add to `~/.bashrc`:

```bash
export CDX_MODULES="2023 Python/3.11.3-GCCcore-12.3.0 CUDA/12.x.y"
```

### 1.2 Build the venv and mamba_ssm on a GPU node

```bash
srun --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time=01:30:00 \
     bash experiments/snelius/setup_env.sh
```

This builds `$HOME/venvs/cdx` from `requirements-snelius.txt`, installs OSRL in editable mode without its pins, and compiles `mamba-ssm==2.3.2.post1` against the loaded CUDA.
- **Build failure:** if mamba_ssm fails, the script prints a warning and continues. Step 1.4 tells you what to do.

### 1.3 Run the environment check and the GPU tests

```bash
srun --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time=00:30:00 \
     bash -c 'source experiments/snelius/env.sh && cdx_activate && \
              python experiments/snelius/check_env.py && \
              pytest experiments/tests -q && \
              python -m experiments.common.profiling --count-params --match-light'
cat env_report.txt
```

Check in the output:
- `env_report.txt` has no `fail` lines. Each `warn` line names what to check.
- **Source checks:** DSRL dataset variable, `outliers_percent`/`epsilon` semantics and MT-BC conditioning are `ok` (V1, V6, V7).
- **Pins:** the torch, gymnasium and numpy versions match `requirements-snelius.txt` (V10).
- **Episode length:** Drone-Run reports 200 (V9; mention this in the paper).
- **pytest:** all tests pass, including the CUDA tests that were skipped locally (T5, T12, ...). **T5 is V5:**
  - If T5 fails by about 1e-3–1e-2, that's the kernel's internal bf16. Record the measured error and relax the tolerance with a note.
  - If the error is larger, the reference implementation differs from the official kernel. Stop and investigate before E0.
- **`--match-light`:** it should again choose 2 layers, d=128, 8 heads (440,590 params). If it chooses something else, run it with `--write` **before E0** so every cdt_light run uses the same config.

### 1.4 If the Mamba-3 kernels don't run on A100 (NOTES V4)

`check_env.py` runs a forward and backward pass through `mamba3 {'mamba_impl': 'cuda'}`. If that fails, choose one of these options:

1. **Use H100 for everything:** `export CDX_PARTITION=gpu_h100`, and re-run 1.2 and 1.3 on `gpu_h100`.
   - Use one partition for **all** models, so the efficiency numbers (E6) are comparable.
   - `prepare_data.sbatch`, `array_gpu.sbatch` and `profile.sbatch` have `--partition=gpu_a100` in their headers. Override it on the command line (`sbatch --partition=gpu_h100 ...`); `submit.sh` already passes `$CDX_PARTITION`.
2. **Stay on A100 with the PyTorch reference:** set `mamba_impl="ref"` in `cdmamba/config.py`. The reference is slower, so label its E6 timings "reference implementation" (spec §4.4).

---

## Phase 2: data

### 2.1 Download the DSRL datasets (login node only; NOTES V2)

```bash
source experiments/snelius/env.sh && cdx_activate
python -m experiments.common.data --download --tasks all
ls -lh $DSRL_DATASET_DIR
```

- **What to expect:** five `.hdf5` files (Ant-Run, Car-Circle, Car-Run, Drone-Circle, Drone-Run).
- **Compute nodes can't download:** runs fail with a clear `FileNotFoundError` if a file is missing.
- **If `data.offline-saferl.org` times out:** get the five files from a mirror or a colleague and put them in `$DSRL_DATASET_DIR`, using the file names in `dsrl/infos.py`.

### 2.2 Build the preprocessing cache

```bash
sbatch experiments/snelius/prepare_data.sbatch          # default: E0 E0prec E0tune E1 E2 E3 E4 E6full E7 E8 E9 E10
squeue -u $USER
tail -f logs/slurm/cdx_prepare_<jobid>.out
```

- **What it does:** builds each distinct (task, data config, seed) once, in parallel, into `$CDX_CACHE`.
- **Missing cache entries:** a run with no cache entry builds it itself, so a missing entry only costs time. P3 experiments and E0pack aren't in the default list. Add them when you need them:

  ```bash
  sbatch experiments/snelius/prepare_data.sbatch E0pack E6long E7ext E11
  ```

- **Check E8 here (finding F11):** the E8 outlier injection runs in this job. If the log shows `ValueError: a cannot be empty` for Drone-Run, then at the chosen outlier fraction DSRL found no trajectory with cost ≥ max_cost/2 and reward ≥ max_reward/2.
  - That is a property of the data, not a crash in our code.
  - Don't submit E8 until it's resolved. The other experiments are unaffected.

---

## Phase 3: E0, the gate (spec §6 E0)

E0 decides precision, learning rate, pack size and `--time`. **Don't submit E1 until E0 has finished and Phase 4 is done.**

Use `--dry-run` first to see what a submission would do:

```bash
source experiments/snelius/env.sh
experiments/snelius/submit.sh E0 --dry-run
```

Each array prints:
- manifest size;
- pack size;
- number of array jobs;
- `--time`;
- partition;
- an upper bound on GPU-hours (and SBU, if the rate is set).

Then submit:

```bash
experiments/snelius/submit.sh E0                 # pilot: all models x 5 tasks, 2k steps (seq + mlp arrays)
experiments/snelius/submit.sh E0prec --pack 2    # cdt Car-Run fp32 vs bf16, 100k steps
experiments/snelius/submit.sh E0tune --pack 8    # lr {1e-4, 3e-4} x 4 models, 30k steps, Car-Circle
for p in 1 3 6 9; do                             # pack-size sweep, each with its own --pack
    experiments/snelius/submit.sh E0pack --pack $p -- --variants p$p
done
experiments/snelius/submit.sh E0fid              # original OSRL train_cdt.py on Car-Run (NOTES V12)
```

- **Confirmation:** each submission asks `submit N array jobs? [y/N]`.
- **Snapshots:** each submission saves a snapshot of its manifest in `experiments/manifests/submitted/`. Regenerating manifests later never changes jobs that are already queued.

### 3.1 Monitor

```bash
experiments/snelius/status.sh E0          # done / running / failed / missing + the queue
squeue -u $USER
tail -f logs/slurm/cdx_E0_seq_<arrayjobid>_<index>.out    # one per packed array task
tail -f logs/runs/cdx_E0_seq_<arrayjobid>_<index>_<k>.log # run k inside that pack
```

- **Failed runs:** a failed run leaves `FAILED` in its run directory, with the traceback. `status.sh` lists the first 50.
- **Results:** each finished run is in `$CDX_RESULTS/<exp>/<task>/<model>[__<variant>]/seed_<s>/`.

### 3.2 What to check in E0

```bash
source experiments/snelius/env.sh && cdx_activate
python -m experiments.common.sanity --e0-report --out $CDX_RESULTS
srun --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time=00:20:00 \
     bash -c 'source experiments/snelius/env.sh && cdx_activate && \
              python -m experiments.common.sanity --overfit --device cuda'
```

| Check | Where | Pass condition |
|---|---|---|
| Every E0 run finished | `status.sh E0` | 0 failed. Read every `FAILED` file and fix the cause before going on. |
| Loss decreases, no NaN | `train_log.csv` of each run | Monotone trend over 2k steps |
| Overfit check | `sanity --overfit` | Every backbone overfits a tiny batch |
| Precision (E0prec) | `--e0-report` | bf16 vs fp32 R_norm/C_norm within seed noise. Otherwise switch to fp32 (Phase 4). |
| Learning rate (E0tune) | `--e0-report` | Picks the lr per backbone (safe first, then highest reward, on validation seeds) |
| Pack size (E0pack) | `--e0-report` | The pack size with the highest total steps/s, without OOM |
| Fidelity (E0fid) | Compare `E0fid` with E1 cdt Car-Run seed 0 later | Same ballpark of R_norm/C_norm (OSRL's DataLoader vs our GPU sampler give different data streams, so not identical) |
| Evaluation doesn't hang | Runs reach `DONE` | If runs hang at evaluation inside GPU jobs, add `--override eval.vec_context=spawn` (NOTES V13) |
| Time per run | `run_summary.json` → `train_wall_s`, `eval_wall_s` | Needed to set `--time` |

---

## Phase 4: apply the E0 decisions (before E1)

Make each change locally, re-run `pytest experiments/tests -q` locally, then `rsync` the code again (Phase 0.1).

1. **Precision.** If E0prec says bf16 is outside seed noise, set `precision: str = "fp32"` in `common/config.py` (`TrainConfig`, line ~139).
2. **Learning rate.** If a backbone prefers 3e-4, set it **for that backbone on all tasks**: add `overrides={"train.learning_rate": 3e-4}` to the `ModelSpec(...)` in its `config.py`, and report it in the paper.
   - `test_fairness_only_backbone_differs` asserts that the overrides are empty. Change that test to allow exactly this key.
3. **Pack size.** Set the E0pack winner in `experiments/snelius/env.sh`: `CDX_PACK_SEQ=<p>`.
4. **`--time`.** Compute it from the E0 measurements and set it in `env.sh`:

   ```
   time per packed job ≈ (seconds per step at that pack size) × 100,000 + eval_wall_s, plus a 30–50 % margin
   ```

   - **Variables:** set `CDX_TIME_SEQ`, `CDX_TIME_MLP` and `CDX_TIME_EVAL`. `CDX_TIME_OSRL` only matters for E0fid.
   - **Seconds per step:** the 2k-step E0 runs include warm-up, so use the steps/s from the E0prec/E0tune runs, which are longer.
   - **Why the margin matters:** a timeout isn't fatal (Phase 5.3), but it wastes the rest of the slice-hour.
5. **MPS (optional).** If E0pack shows the packed runs slow each other down a lot, try `submit.sh ... --mps` on one E1 array and compare.
6. **Fill in spec §7** (compute budget) with the measured numbers, so the GPU-hour plan for E1–E9 is real.

---

## Phase 5: P0 experiments

```bash
source experiments/snelius/env.sh
experiments/snelius/submit.sh E1 --dry-run        # check the GPU-hour bound
experiments/snelius/submit.sh E1                  # 100 sequence-model runs
experiments/snelius/submit.sh E4                  # 25 mt_bc runs now; eval runs wait for E1 checkpoints
experiments/snelius/submit.sh E6full --pack 1     # 4 dedicated unpacked full runs (Car-Circle) for E6 ground truth
sbatch experiments/snelius/prepare_data.sbatch E10   # if not already in the cache (E10 is in the default list)
```

- `--max-parallel G` (default `CDX_MAX_PARALLEL=8`) limits how many array tasks run at once. Spec §7 recommends 8, and up to 16 if the queue is short.

### 5.1 After E1 is done

```bash
experiments/snelius/status.sh E1                  # all 100 done?
experiments/snelius/submit.sh E4                  # now writes the 100 eval-only runs (kappa sweep on E1 checkpoints)
sbatch experiments/snelius/profile.sbatch         # E6: exclusive full node, 3 repetitions
```

- **Eval-only runs** (E4, E5) are only written to the manifest once their source checkpoint is `DONE`. If you submit early, re-run `submit.sh` later to pick up the rest. Runs that are already `DONE` or owned by a live job are skipped.
- **`profile.sbatch`:**
  - Asks for 4 h of an exclusive 4-GPU node (V14).
  - If that isn't enough, split the measurements across several jobs, for example `WHAT="infer mem" sbatch ...` and `MODELS="cdt cdmamba" sbatch ...`.
  - Use the same partition as the training runs.

### 5.2 Run E0fid's comparison

Once E1 cdt Car-Run seed 0 is done, compare its `eval_episodes.csv` with the E0fid evaluation. The E0fid run directory is under `$CDX_RESULTS/E0fid/`.

### 5.3 Timeouts, preemption, failures

- **Timeouts and preemption:** every run checkpoints every 10k steps (`ckpt_latest.pt`) and resumes exactly from it. Resubmitting is safe and cheap:

  ```bash
  experiments/snelius/submit.sh E1     # regenerates the manifest: only runs that are not DONE and not in a live job
  ```

- **`FAILED` runs:** read the traceback first. A real bug fails again on every resubmission.
- **Resuming after training:** if a run crashed during evaluation, `ckpt_final.pt` is reused and only the evaluation is redone.
- **Out of time in every job:** raise `--time` (`submit.sh E1 --time 04:00:00`) instead of resubmitting with the same limit.

---

## Phase 6: P1 / P2 (and P3 if the budget allows)

```bash
experiments/snelius/submit.sh E2      # baselines: 25 dt_cost (seq) + 125 MLP runs
experiments/snelius/submit.sh E3      # ablations: 180 runs
# after E1 and E3 are done:
experiments/snelius/submit.sh E5      # reward sweep: eval only on E1 + E3-noaug checkpoints
experiments/snelius/submit.sh E7      # epsilon-reducibility
experiments/snelius/submit.sh E8      # outliers (only if the Phase 2.2 E8 check passed)
experiments/snelius/submit.sh E9      # BC variants
# P3, optional:
experiments/snelius/submit.sh E6long
experiments/snelius/submit.sh E7ext
experiments/snelius/submit.sh E11
```

- **Before E3 and E8**, check the budget with `accinfo`. If it's tight, cut in this order (spec §7): E11 → E6long → E7ext → E8 → E3. Never cut E6.1–E6.6.
- **Partial submissions:** add filters after `--`, for example `submit.sh E3 -- --tasks OfflineCarCircle-v0 OfflineDroneRun-v0` or `-- --models cdt cdmamba`.

---

## Phase 7: analysis

On the login node:

```bash
source experiments/snelius/env.sh && cdx_activate
make -f experiments/analysis/Makefile all RESULTS=$CDX_RESULTS CACHE=$CDX_CACHE
```

Or on your laptop:

```bash
rsync -av <user>@snellius.surf.nl:/projects/0/<your-project>/cdx/results ./results \
      --exclude '*.pt'
rsync -av <user>@snellius.surf.nl:/projects/0/<your-project>/cdx/cache ./cache
make -f experiments/analysis/Makefile all RESULTS=results CACHE=cache
```

- **Outputs:**
  - `summary/{episodes,runs,val_logs,efficiency}.parquet`;
  - `summary/tables/*.{tex,md}`;
  - `summary/figures/*.{pdf,png}`.
- **Missing runs** show as "–" or "no data". Nothing is filled in.
- **The optional "CDT reported" row in Table 1** needs `experiments/analysis/reported_cdt.csv` (`task,R_norm,C_norm`), filled in by hand from [CDT] Table 1.
- **Known analysis gaps** (from the review, not fixed yet):
  - no bootstrap CIs in the appendix tables;
  - no E4 budget-tracking error or Spearman correlation;
  - no E6.2 wall-clock split in Table 2.

  The data for all of them is in the results, so they can be added before the paper is written.

---

## Quick reference

| Task | Command |
|---|---|
| Activate env in a login shell | `source experiments/snelius/env.sh && cdx_activate` |
| What would a submission do? | `experiments/snelius/submit.sh E1 --dry-run` |
| Submit without the prompt | `experiments/snelius/submit.sh E1 --yes` |
| Status of an experiment | `experiments/snelius/status.sh E1` |
| Resolved config of one run | `python -m experiments.common.run --exp E1 --model cdmamba --task OfflineCarRun-v0 --seed 0 --dry-run` |
| Re-run one run by hand (interactive GPU) | `srun --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time=03:00:00 bash -c 'source experiments/snelius/env.sh && cdx_activate && python -m experiments.common.run --exp E1 --model cdt --task OfflineCarRun-v0 --seed 3 --out $CDX_RESULTS'` |
| Cancel an experiment's jobs | `scancel -n cdx_E1_seq` (job names are `cdx_<EXP>_<kind>`) |
| Run logs | `logs/slurm/` (per array task), `logs/runs/` (per run), `<run_dir>/FAILED` (traceback) |

**After `scancel`:** cancelled runs keep a stale `RUNNING` marker. `make_manifest` checks `squeue`, sees the job is gone, and schedules the run again on the next `submit.sh`. Don't delete markers by hand.
