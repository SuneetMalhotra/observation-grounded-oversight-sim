# Simulation code for "Observation-Grounded Oversight: A Framework for Embodied and Context-Aware AI Agents"

Code for the simulation study in the IEEE Pervasive Computing manuscript PC-2026-07-0051 (under review).

IEEE Pervasive Computing manuscript PC-2026-07-0051, revision R1.

**Everything here is a synthetic simulation.** Sensor, hazard and planner models are made up for the study. Nothing was run on a robot, and no number is a deployment measurement.

## Contents

| File | What it is |
|---|---|
| `sim/oversight_sim.py` | The simulator: world model, the gates, the CUSUM drift monitor and the timing functions |
| `sim/run_all.py` | Runs every experiment and writes the two files below |
| `sim/tables.md` | All results as tables: means with standard deviations over seeds |
| `sim/results.json` | The same results in machine-readable form, with all parameters |

## How to reproduce

```
python3 sim/run_all.py
```

- Python 3, standard library only. If NumPy is installed, one extra timing figure is measured; nothing else uses it.
- Takes under a minute.
- All results except the timing section are deterministic: 10 seeds, 20,000 decisions per condition per seed. The run reported in the manuscript used CPython 3.15.2 on Apple M4 Pro.
- Timing figures are wall-clock and differ between machines and between runs.

## What is simulated

An autonomous mobile robot decides every 6 s whether to commit to entering an aisle segment. A hazard (a person or obstacle in the swept path) is present in 5% of decisions. A camera detector and a LiDAR occupancy check observe each decision; a planner stand-in fuses them, weighting the camera heavily, and reports a confidence.

### Gates

| Gate | Rule for a proposed commit |
|---|---|
| G0 | Always commit |
| G1 | Commit if planner confidence is at least tau; otherwise page an operator |
| G2 | Original guard: only fresh, healthy channels count. Hold on a hazard report from one of them; commit if LiDAR is fresh; page if the camera is the only healthy channel; otherwise hold |
| G3 | Ablation: hold if either sensor reports a hazard; otherwise commit. No health or freshness checks |
| G2b | Revised guard: hold if any channel reports a hazard, healthy or not; commit only if LiDAR is fresh; otherwise hold. Never pages |
| G2bc | Variant of G2b in which a fresh, healthy camera may also authorise |

Pages are counted, not resolved: no operator is modelled, so a page is a non-commit. The page-after-N-cycles rule of the manuscript's safe-state table is not simulated.

### Metrics

- Unsafe commits: commits into a hazard, as a percentage of all hazard slots.
- False blocks: non-commits on clear slots, as a percentage of clear slots the planner proposed.
- Pages: operator pages per robot-hour (600 decisions per robot-hour).
- Throughput: commits as a percentage of all decision slots.

### Default parameters

| Parameter | Value |
|---|---|
| `period_s` | 6.0 |
| `n_decisions` | 20000 |
| `hazard_rate` | 0.05 |
| `tau` | 0.8 |
| `health_thr` | 0.6 |
| `cam_clear_mu` | 0.2 |
| `cam_clear_sd` | 0.1 |
| `cam_haz_mu` | 0.8 |
| `cam_haz_sd` | 0.12 |
| `cam_det_thr` | 0.5 |
| `cam_health_mu` | 0.9 |
| `cam_health_sd` | 0.04 |
| `cam_stale_p` | 0.005 |
| `occl_p` | 0.0 |
| `occl_mu` | 0.3 |
| `lidar_miss` | 0.03 |
| `lidar_fp` | 0.01 |
| `lidar_dropout_p` | 0.01 |
| `lidar_stale_p` | 0.005 |
| `lidar_miss_when_occluded` | None |
| `planner_w_cam` | 0.8 |
| `conf_noise_sd` | 0.04 |
| `conf_tracks_quality` | False |
| `drift_onset` | None |
| `drift_final` | 0.0 |
| `cusum_k` | 0.75 |
| `cusum_h` | 10.0 |
| `cusum_warmup` | 2000 |

### Condition overrides

| Condition | Overrides |
|---|---|
| nominal | none |
| low_light | `cam_haz_mu` = 0.45, `cam_haz_sd` = 0.15, `cam_clear_sd` = 0.13, `cam_health_mu` = 0.35, `cam_health_sd` = 0.08 |
| occlusion | `occl_p` = 0.6 |
| stale_lidar | `lidar_stale_p` = 0.25, `lidar_dropout_p` = 0.15 |
| drift | `drift_onset` = 5000, `drift_final` = -0.25 |

Parameter values were fixed before the first run. They were chosen by the author, not fitted to data; the robustness runs in `sim/tables.md` show how the self-confidence gate's result moves when two of them change.

## Sections of `sim/tables.md`

- Five conditions by five gates (the manuscript's simulation table, with standard deviations).
- Sensitivity: common-mode LiDAR miss; 10% LiDAR miss rate; planner confidence that falls with image quality; tau = 0.9.
- Robustness of G1 under occlusion to the planner's camera weight and to the camera score of an occluded hazard.
- Variant G2bc beside G2b and G3.
- Sweep of the share of LiDAR readings that are not fresh.
- Drift monitor: detections, delay, false alarms.
- Guard timing: decision function, and a whole guard cycle on a synthetic 640x480 frame.

## Licence

MIT. See `LICENSE`.
