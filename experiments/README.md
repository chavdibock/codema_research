# Lightweight alternatives to CDT: experiment pipeline

Implements `../experiments.md` (the spec). One shared pipeline derived from OSRL; the
only difference between `cdt`, `cdt_light`, `cdmamba` and `cdconv` is the backbone
module and its config (`common/backbones.py` + the approach folders).

- OSRL: unmodified dependency, pinned at commit `03a4586463690566ec8d7830ca4027b46d4ad02f`
  (`pip install -e OSRL --no-deps`).
- Decisions, deviations and the items to verify on Snelius: **`NOTES.md`**.
- All commands below are run from the **`Experiments/`** directory (the parent of this
  folder), so that `python -m experiments....` resolves.

## Layout

```
experiments/
  common/        policy.py (SequencePolicy), backbones.py, trainer.py, data.py (cache S5,
                 GPU sampler S1), rollout.py (seeded, vectorized S2/S8), evaluate.py, run.py,
                 expdefs.py (E0-E11), config.py, metrics.py, logging.py, profiling.py (E6),
                 sanity.py (E0 overfit check + E0 report), osrl_compat.py, utils.py
  cdt/ cdt_light/ cdmamba/ cdconvformer/     backbone + config of each approach
  baselines/     OSRL BC/BCQ-L/BEAR-L/CPQ/COptiDICE wrappers, DT-Cost, MT-BC, BC variants
  snelius/       setup_env.sh, check_env.py, check_partitions.sh, env.sh, prepare_data.sbatch,
                 array_gpu.sbatch, profile.sbatch, make_manifest.py, submit.sh, status.sh, cdx.def
  analysis/      aggregate.py, tables.py, figures.py, stats.py, Makefile
  tests/         T1-T12 (+ resume/idempotency/manifest checks)
```

Results tree (spec §8.4): `results/<exp>/<task>/<model>[__<variant>]/seed_<s>/` with
`config.yaml env_info.json train_log.csv val_log.csv eval_episodes.csv ckpt_final.pt
run_summary.json DONE`; plus `results/E6/{efficiency.csv,hardware.json}`.

## Install

### Local (CPU, for tests and smoke runs)

```bash
conda create -n cdx python=3.11 -y && conda activate cdx
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "numpy<2" "gymnasium==0.28.1" pybullet "pyrallis==0.3.1" h5py numba scipy \
    pandas "pyarrow<19" matplotlib psutil pytest pyyaml tqdm einops OApackage tensorboard wandb
pip install --no-deps dsrl==0.1.0 fast-safe-rl==0.1.0 bullet-safety-gym==1.4.0 easy_runner
pip install -e OSRL --no-deps
```

### Snelius (GPU node, once)

```bash
export CDX_PROJECT=/projects/0/<your-project>/cdx       # project space, not $HOME
srun --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time=01:30:00 \
     bash experiments/snelius/setup_env.sh
srun --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time=00:30:00 \
     bash -c 'source experiments/snelius/env.sh && cdx_activate && python experiments/snelius/check_env.py'
cat env_report.txt
```

`experiments/snelius/env.sh` holds every path and default (modules, partition, pack
sizes, `--time` per job kind, `G` = max parallel GPUs). Edit there or override with
environment variables.

## Local quick check (CPU, costs nothing)

```bash
pytest experiments/tests -q                      # T1-T12 on CPU; CUDA tests are skipped
python -m experiments.common.profiling --count-params --match-light
python -m experiments.common.sanity --overfit --device cpu --steps 150
# tiny smoke run on a synthetic DSRL-format dataset (random data, not a result):
export DSRL_DATASET_DIR=/tmp/cdx_dsrl CDX_CACHE=/tmp/cdx_cache
python -c "from experiments.tests.synthetic import write_synthetic_dataset as w; w('OfflineCarRun-v0', '/tmp/cdx_dsrl', n_traj=300)"
python -m experiments.common.run --exp E0 --model cdmamba --task OfflineCarRun-v0 --seed 0 \
    --out /tmp/cdx_results --override train.update_steps=200 train.batch_size=64 \
    eval.eval_episodes=2 model.backbone_kwargs.mamba_impl=ref
```

## Snelius workflow

Nothing is submitted unless you run `submit.sh` (or `sbatch`) yourself. `submit.sh`
regenerates the manifest, prints manifest size, pack size, `--time` and the estimated
GPU-hours, and asks for confirmation (`--dry-run` to only print, `--yes` to skip the
prompt).

```bash
source experiments/snelius/env.sh
bash experiments/snelius/check_partitions.sh                        # budget + partitions
cdx_activate && python -m experiments.common.data --download --tasks all   # login node only
sbatch experiments/snelius/prepare_data.sbatch                     # preprocessing cache (S5)

# E0 (gate): pilot, precision check, tuning, pack sweep, fidelity
experiments/snelius/submit.sh E0
experiments/snelius/submit.sh E0prec --pack 2
experiments/snelius/submit.sh E0tune --pack 8
for p in 1 3 6 9; do experiments/snelius/submit.sh E0pack --pack $p -- --variants p$p; done
experiments/snelius/submit.sh E0fid
python -m experiments.common.sanity --e0-report --out $CDX_RESULTS
python -m experiments.common.sanity --overfit --device cuda
#  -> set CDX_TIME_* / CDX_PACK_SEQ in env.sh, and train.precision in common/config.py

experiments/snelius/submit.sh E1                                   # then:
experiments/snelius/status.sh E1
```

Analysis (login node or locally after `rsync` of `results/` and `cache/`):

```bash
make -f experiments/analysis/Makefile all RESULTS=$CDX_RESULTS CACHE=$CDX_CACHE
# -> summary/{episodes,runs,val_logs,efficiency}.parquet, summary/tables/*.{tex,md},
#    summary/figures/*.{pdf,png}
```

## Commands per experiment

One run is one command (`python -m experiments.common.run --exp E --model M --task T
--seed S [--variant V] [--override k=v ...] --out results`); `--dry-run` prints the
resolved config. Manifests list these commands; `submit.sh` packs them (sequence models
6 per A100 slice, MLP baselines 12, eval-only 4).

| Exp | Content | Runs | Submit |
|---|---|---|---|
| E0 | pilot, 2k steps, every model x task, seed 0 | 55 | `submit.sh E0` (+ E0prec, E0tune, E0pack, E0fid above) |
| E1 | cdt, cdt_light, cdmamba, cdconv x 5 tasks x 5 seeds; eval kappa 10/20/40 | 100 | `submit.sh E1` |
| E2 | bc_safe, dt_cost, bcq_lag, bear_lag, cpq, coptidice x 5 x 5 | 150 | `submit.sh E2` |
| E3 | 4 backbones x {noaug, noent, det} x 5 tasks x 3 seeds | 180 | `submit.sh E3` |
| E4 | kappa sweep {5..80} on E1 checkpoints + MT-BC training | 100 eval + 25 | `submit.sh E4` (after E1) |
| E5 | reward sweep 0.5-1.3 r_max on E1 and E3-noaug checkpoints | 160 eval | `submit.sh E5` (after E1, E3) |
| E6 | efficiency (M1-M4, size, scaling) | 1 node job | `submit.sh E6full --pack 1`, then `sbatch experiments/snelius/profile.sbatch` (after E1) |
| E6long | (P3) K=50, 3 models x 2 tasks x 3 seeds | 18 | `submit.sh E6long` |
| E7 | DT, BCQ, BEAR on full/reduced Ant-Run, Drone-Run x 3 seeds | 36 | `submit.sh E7` (P3 extension: `E7ext`) |
| E8 | 4 backbones x 5 outlier fractions x 2 mitigations, Drone-Run x 3 seeds | 120 | `submit.sh E8` |
| E9 | bc_all/risky/frontier/boundary x 5 x 3 | 60 | `submit.sh E9` |
| E10 | dataset plots (no training) | - | `sbatch experiments/snelius/prepare_data.sbatch E10`, then `make ... figures` |
| E11 | (P3) d in {32,64,128,256} x 3 families x 2 tasks x 3 seeds | 72 | `submit.sh E11` |

Eval-only runs (E4/E5) are written to the manifest only once their source checkpoint
is `DONE`; re-run `submit.sh` later to pick up the rest. Every run is idempotent (skips
if `DONE`) and resumable (checkpoint every 10k steps), so resubmitting a manifest after
a timeout or preemption is safe.

### Efficiency profiling (E6) by hand

```bash
python -m experiments.common.profiling --e6 --device cuda --rep 0 --what size train infer mem
CUDA_VISIBLE_DEVICES="" taskset -c 0 python -m experiments.common.profiling --e6 \
    --device cpu --threads 1 --rep 0 --what infer mem
python -m experiments.common.profiling --e6 --device cuda --rep 0 --what scaling --allow-fresh
```

## Reproducibility

Every `config.yaml` stores the fully resolved config, the OSRL commit, a hash of this
code, `pip freeze` and the hardware; all seeds (training, validation env, test env) are
recorded. No test-episode information is used for model selection or tuning.
