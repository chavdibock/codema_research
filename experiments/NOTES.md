# NOTES: decisions, deviations and open verification items

Status: code complete, **no experiment has been run**. Local checks (CPU only) used a
synthetic DSRL-format dataset (random content) because the DSRL data server was not
reachable from the development machine. No number in this repository is a result.

Every item below is marked `# NOTE:` in the code where it applies.

## 1. To verify on Snelius

| # | Item | What was checked here | What to do on Snelius |
|---|---|---|---|
| V1 | **DSRL dataset env var** | Verified in `dsrl/offline_env.py` (dsrl 0.1.0): `DSRL_DATASET_DIR`, default `~/.dsrl/datasets`. | `check_env.py` re-checks the source. Set it in `snelius/env.sh` (project space). |
| V2 | **DSRL data host** | `data.offline-saferl.org` resolved but every HTTP request **timed out** from here. | Run `python -m experiments.common.data --download --tasks all` on the login node before anything else. If the host is down, find a mirror and place the five `.hdf5` files (names in `dsrl/infos.py`) in `$DSRL_DATASET_DIR`. |
| V3 | **Mamba-3 import path / min. version** | Verified in the `mamba_ssm==2.3.2.post1` sdist: `mamba_ssm.modules.mamba3.Mamba3` (SISO Triton kernels, MIMO via TileLang, step kernel via CuteDSL). Requires `triton>=3.5`, `tilelang==0.1.8`; `import mamba_ssm` also needs the compiled Mamba-1 extension. Pinned: `mamba-ssm==2.3.2.post1`, `torch==2.9.1`. | Build with `setup_env.sh`; `check_env.py` imports it and runs a GPU forward/backward. |
| V4 | **Mamba-3 kernels on A100** | The SISO kernel uses Triton tensor descriptors (TMA, Hopper feature) and is tuned for H100; the step kernel's docstring says "only tested on H100". Not testable here (no GPU). | If `check_env.py` fails for `mamba3 {'mamba_impl': 'cuda'}`, try `gpu_h100`; otherwise train with `--override model.backbone_kwargs.mamba_impl=ref` and label timings as reference-implementation timings (spec §4.4). |
| V5 | **T5 tolerance** | The official kernel casts Q/K/V/Trap/Angles to **bf16** internally, even in fp32 runs. Our reference computes in fp32. | Run `pytest experiments/tests -m "cuda"`. If T5 fails at 1e-3 only by bf16 rounding (error ≈ 1e-3–1e-2 with outputs of O(1)), report the measured error and relax the tolerance with a note; a larger error means a semantic mismatch in `cdmamba/mamba3_ref.py`. |
| V6 | **DSRL `epsilon` / `outliers_percent`** | Verified in source. `outliers_percent` is a *fraction* (α=2 % ⇒ 0.02); it relabels high-cost, high-reward trajectories to cost < κ and reward × 1.5 (the "lucky" trajectories of [CDT] App. D.2) using a fixed RNG (PCG64(1234)). `epsilon>0` removes the top-reward *safe* trajectories and `epsilon<0` the top-reward *unsafe* ones, which is **not** [CDT] App. A's rule, so E7 uses `data.reduce_rule=cdt` (our implementation of [CDT]'s rule). | `check_env.py` re-checks the source on the installed dsrl. |
| V7 | **MT-BC κ conditioning** | Verified in `osrl/algorithms/bc.py`: multi-task BC appends the episode cost return to the observation for training and `np.append(obs, self.cost_limit)` at evaluation. Our wrapper appends the requested κ of each evaluation job, so the E4 sweep works without extending OSRL. | `check_env.py` re-checks the source. |
| V8 | **Snelius module names** | Not checkable here. Default `CDX_MODULES="2023 Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1"` in `snelius/env.sh`. | `module avail Python CUDA`; export `CDX_MODULES` if different. torch 2.9.1 wheels bundle CUDA 12.8; the module CUDA is only for building `mamba_ssm`. |
| V9 | **Drone-Run episode length** | DSRL registers `OfflineDroneRun-v0` with `max_episode_steps=200` (`dsrl/infos.py`); OSRL uses 200; [CDT] Tab. 4 says 100. We use `env.spec.max_episode_steps` = 200 (D1) and record it in `env_info.json`. | `check_env.py` prints it. Mention the deviation in the paper. |
| V10 | **Package pins** | Local CPU env: gymnasium 0.28.1 (required by bullet-safety-gym 1.4.0), numpy 1.26, pyarrow 18.1 (pyarrow ≥ 19 needs numpy 2), OApackage 2.7.20 (2.7.6 did not build without swig), torch 2.14 CPU. | `setup_env.sh` installs `requirements-snelius.txt`; check `env_report.txt`. |
| V11 | **SBU rate, `--time` values** | Not known. `submit.sh` prints GPU-hours; SBU only if `CDX_SBU_PER_GPU_HOUR` is set. Default `--time` values in `env.sh` are the planning values of spec §7. | `check_partitions.sh` + Snelius docs; replace `CDX_TIME_*` with E0 measurements. |
| V12 | **OSRL CLI for E0fid** | The manifest calls `examples/train/train_cdt.py --task … --seed 0 --device cuda --logdir … --group fid --name osrl_cdt_s0 --num_workers 8` (pyrallis flags taken from `CDTTrainConfig`). Not run here. | Run E0fid; `evaluate.py --osrl-ckpt` loads `<logdir>/fid/osrl_cdt_s0/checkpoint/model.pt`. |
| V13 | **Vectorized eval workers after CUDA init** | AsyncVectorEnv uses `fork` by default (`eval.vec_context`); the children never touch CUDA. Tested on CPU only. | If evaluation hangs inside GPU jobs, use `--override eval.vec_context=spawn` (env factories are picklable). |
| V14 | **Profiling job length** | Not measurable here. `profile.sbatch` asks for 4 h; training timing runs on Car-Circle only (`TRAIN_TASKS`), inference/memory on all tasks. | Split with `WHAT=…`/`MODELS=…` env vars if it does not fit. |

## 2. Decisions taken where the spec was ambiguous

**Data and protocol**
1. **D2 Car-Circle κ=40 reward target**: we keep OSRL's third reward target (550) and set its cost to 40, i.e. (450,10), (500,20), (550,40), because every other task pairs its largest reward target with κ=40 (`common/config.py`).
2. **Seeded evaluation**: Bullet-Safety-Gym ignores `reset(seed=…)` (randomness comes from the global `np.random`), and a second episode on the same env instance is not reproducible (pybullet `restoreState` does not restore all agent state; checked locally). `SeededReset` therefore rebuilds the env (0.01–0.3 s) and seeds `np.random`/`random` on every seeded reset. With that, episode `i` depends only on `eval_seed_base + i`.
3. **Vectorized evaluation (S2) and exactness**: with a batched forward per step, fp32 matmuls round differently for different batch sizes/memory layouts and the chaotic dynamics amplify this, so individual episodes differ from sequential evaluation (the distribution does not). The default is the batched forward (S2). `eval.batch_invariant=true` runs one batch-1 actor per env and is **bit-identical** to sequential evaluation (T11 checks this exactly). Use it if exact equality is required.
4. **Validation / test seeds**: validation seeds start at 500,000, test seeds at 1,000,000 (same for every model). E0 tuning selects on validation seeds (`eval.seed_split=val`).
5. **Checkpoint interval**: §8.2 says every 10k steps, S6 says 25k. We use 10k (checkpoints of these models take milliseconds; more robust against preemption). `train.ckpt_every`.
6. **bf16 default**: `train.precision=bf16` by default (S3); E0prec compares fp32 vs bf16 on Car-Run. If bf16 is outside seed noise, switch the default in `common/config.py` to fp32 *before* E1.
7. **Mask dtype**: OSRL's loader yields a float64 mask, so some OSRL loss products are computed in float64; ours uses float32. Numerically negligible; T1 compares forward outputs, which are identical.
8. **OSRL "best" checkpoint**: never used for reporting; the step with the lowest validation cost is logged as `best_val_step_diagnostic` in `run_summary.json`.
9. **Packed-job timings** (`run_summary.json`) are indicative only; reported numbers come from E6.

**Models**
10. **Initialization**: CDT's N(0, 0.02) init is applied to every non-backbone module (same traversal order as OSRL, so T1 is bit-exact). Inside the Mamba and ConvFormer backbones it is applied to LayerNorm/MLP only; the Mamba-3 mixer keeps its own init (dt_bias, B/C biases = 1, D = 1, default Linear init) and Conv1d keeps PyTorch's default init.
11. **Mamba-3 reference implementation**: `cdmamba/mamba3_ref.py` is our own pure-PyTorch implementation of the SISO recurrence written against the 2.3.2.post1 kernels (trapezoidal discretization, data-dependent RoPE, B/C RMSNorm + biases, D skip, SiLU(z) gate). It has the same parameter names/shapes as the official module (state dicts are interchangeable). MIMO is CUDA-only. `d_conv=4` is recorded but unused (Mamba-3 has no short convolution).
12. **Recurrent mode** uses the pure-PyTorch step on any device (the official CuteDSL step kernel is H100-only per its docstring). Label it as such.
13. **Chunk padding**: with the CUDA impl, sequences are right-padded to a multiple of `chunk_size` (64) and sliced back; exact for a causal model (T3 on GPU).
14. **Parameter counts (Car-Circle, computed with the reference module, same shapes as official)**: cdt 638,862; cdmamba 803,646 (the 4d MLP per block makes it larger than CDT); cdconv 454,542. **cdt_light** by the §4.3 procedure: target min = 454,542 (cdconv) → 2 layers, d=128, 8 heads = 440,590 (−3.1 %, within ±10 %). In MHA the head count does not change the parameter count; ties are broken towards fewer layers, then more heads. Re-run `python -m experiments.common.profiling --match-light` on Snelius to confirm.
15. **DT-Cost** (our approximation of [CDT]'s "vanilla DT + cost token"): CDT network with a deterministic MSE head, `augment_percent=0`, `cost_sample=False`, `loss_cost_weight=0` (`baselines/config.py`).
16. **Unconstrained models (E7)**: `dt` = DT-Cost without the cost token. The P3 extension (E7ext) uses the same settings for cdt_light/cdmamba/cdconv (no cost token, deterministic head, no augmentation, no cost loss).
17. **BCQ / BEAR without Lagrangian (E7)**: OSRL's BCQL/BEARL with PID gains (0, 0, 0), i.e. multiplier 0.

**MLP baselines**
18. Hyperparameters are OSRL's per-task configs, except `update_steps=100k` for every method (OSRL's BEARL default is **300k**) and `episode_len` from the env spec. Batch size stays OSRL's 512.
19. S3 for MLP baselines = TF32 only (`train.precision=fp32` in their spec); bf16 autocast inside OSRL's Q-learning/DICE updates risks instability, and OSRL creates its own non-fused Adam optimizers. S2 (vectorized evaluation) and the GPU transition sampler do apply.
20. Fixed-threshold baselines are evaluated once at κ=10; DT-Cost (which can condition on κ) is also evaluated at κ=20/40 for Table 4.

**Experiments**
21. **E0**: pilot = every model (ours + E2 baselines + MT-BC) × 5 tasks × seed 0 × 2k steps. Separate definitions: `E0prec` (fp32 vs bf16), `E0tune` (lr ∈ {1e-4, 3e-4}, 30k steps, Car-Circle, validation seeds), `E0pack` (9 cdt runs × 2k steps for pack sizes 1/3/6/9: submit each variant with its own `--pack`), `E0fid` (original OSRL script, compared with E1 cdt Car-Run seed 0). `python -m experiments.common.sanity --e0-report` prints all E0 outcomes; `--overfit` is the tiny-batch overfit check.
22. **E3 table / Fig. 7**: the "full" rows use the E1 runs with the E3 seeds {0, 1, 2}.
23. **E5**: the `noaug` curves come from E3, which has 3 seeds, so they use seeds {0, 1, 2}; E1 curves use 5 seeds.
24. **E6**: the spec's budget table counts 5 dedicated full runs; `E6full` defines one per model (4). M1 training timing runs on Car-Circle (the dedicated-run task); inference and memory on every task. Training-memory rows include optimizer state and gradients; "activation" = total − static parameter memory, as the spec defines.
25. **E8 mitigation mapping**: "sampling-based association" = OSRL's default augmentation. Note that OSRL's `augmentation()` always applies its own grid filter (10×50 bins, ≥ 2 trajectories per bin) before fitting the frontier; this cannot be disabled without modifying OSRL. "Density filter" = DSRL's grid filter with `DENSITY_CFG[task_density0.5]` applied **after** the outlier injection (`data.post_density=0.5`), because DSRL's own `density` option runs *before* outliers are injected and could not remove them.
26. **E7 reduced dataset**: [CDT] App. A's rule: drop trajectories with C > κ and R > RF(κ) = max reward among trajectories with C = κ; if no trajectory has C = κ exactly, RF falls back to PF(κ) = max reward with C ≤ κ. ε̂ = (PF(κ) − IPF(κ)) / max R is recorded for every dataset in `env_info.json`.
27. **E10** plots are built from the preprocessing cache (`figures.py --cache`); run `prepare_data.sbatch E10` first.

**Engineering**
28. `experiments/` is not a git repository yet, so `config.yaml` records a content hash of the source files (`content-…`) instead of a git hash; the OSRL commit is read from `OSRL/.git` (`03a4586463690566ec8d7830ca4027b46d4ad02f`).
29. DSRL's `pre_process_data` saves a PNG into the current directory; we call it from a temporary directory.
30. The optional "CDT (reported in [CDT])" row of Table 1 is read from `analysis/reported_cdt.csv` (columns `task,R_norm,C_norm`), which is **not shipped**: fill it from [CDT] Table 1 yourself. Without it the row shows "–".
31. Figures: our four models use the first four slots of a colorblind-validated categorical palette in fixed order plus distinct markers/line styles; baselines are gray.
