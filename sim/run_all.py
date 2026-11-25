"""Run every simulation and write sim/results.json and sim/tables.md.

Usage: python3 sim/run_all.py      (standard library only; SIMULATION ONLY)
Everything except the timing section is deterministic for a given seed list.
"""
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oversight_sim as S

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(10))
METRICS = ["unsafe_pct", "false_block_pct", "pages_per_hour", "throughput_pct"]


def agg(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values)}


def summarise(runs, gates):
    return {g: {m: agg([r["metrics"][g][m] for r in runs]) for m in METRICS} for g in gates}


def cpu_name():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip() or platform.processor()
    except OSError:
        return platform.processor()


def main():
    res = {"label": "SIMULATION ONLY - synthetic sensor, hazard and planner models",
           "params": {"defaults": S.DEFAULTS, "condition_overrides": S.CONDITION_OVERRIDES,
                      "seeds": SEEDS, "decisions_per_hour": 3600.0 / S.DEFAULTS["period_s"]}}

    # main matrix
    main_runs = {}
    res["main"] = {}
    timing = {}
    cycle = cycle_np = None
    for cond in S.CONDITIONS:
        runs = []
        for seed in SEEDS:
            r = S.run(cond, seed, keep_obs=(seed == 0))
            if seed == 0:
                obs, prm = r.pop("obs"), r.pop("params")
                timing[cond] = S.time_guard(obs, prm)
                if cond == "nominal":
                    with tempfile.TemporaryDirectory() as td:
                        cycle = S.time_cycle(obs, prm, os.path.join(td, "trace.log"))
                        try:
                            import numpy
                            cycle_np = dict(S.time_cycle(obs, prm, os.path.join(td, "trace_np.log"), n=1000,
                                                         quality=S.image_quality_numpy), numpy=numpy.__version__)
                        except ImportError:
                            cycle_np = None
            runs.append(r)
        main_runs[cond] = runs
        res["main"][cond] = summarise(runs, S.GATES)
        print("done", cond, flush=True)

    # sensitivity
    def sens(cond, overrides, gates):
        return summarise([S.run(cond, s, overrides, gates=gates) for s in SEEDS], gates)

    res["sensitivity"] = {
        "G2b_occlusion_common_mode": {
            "overrides": {"lidar_miss_when_occluded": 0.30},
            "result": sens("occlusion", {"lidar_miss_when_occluded": 0.30}, ["G1", "G2b", "G3"])},
        "G2b_occlusion_lidar_miss_10": {
            "overrides": {"lidar_miss": 0.10},
            "result": sens("occlusion", {"lidar_miss": 0.10}, ["G2b", "G3"])},
        "G1_low_light_conf_tracks_quality": {
            "overrides": {"conf_tracks_quality": True},
            "result": sens("low_light", {"conf_tracks_quality": True}, ["G1"])},
        "G1_nominal_tau_0.9": {
            "overrides": {"tau": 0.9},
            "result": sens("nominal", {"tau": 0.9}, ["G1"])},
    }

    # robustness of G1 under occlusion to two free parameters of the planner and camera models
    rob = {}
    for name, ov in (("planner_w_cam=0.6", {"planner_w_cam": 0.6}), ("planner_w_cam=0.95", {"planner_w_cam": 0.95}),
                     ("occl_mu=0.15", {"occl_mu": 0.15}), ("occl_mu=0.45", {"occl_mu": 0.45})):
        rob[name] = {"overrides": ov, "result": sens("occlusion", ov, ["G1", "G3", "G2b"])}
    g1 = [v["result"]["G1"]["unsafe_pct"]["mean"] for v in rob.values()] + [res["main"]["occlusion"]["G1"]["unsafe_pct"]["mean"]]
    res["robustness_G1_occlusion"] = {"runs": rob, "G1_unsafe_min": min(g1), "G1_unsafe_max": max(g1)}

    # variant: a fresh, healthy camera may also authorise (G2bc), against G2b and G3
    res["variant_camera_may_authorise"] = {c: sens(c, None, ["G3", "G2b", "G2bc"]) for c in S.CONDITIONS}

    # sweep: share of LiDAR readings that are not fresh (stale : dropped = 5 : 3, as in the stale condition)
    res["sweep_lidar_nonfresh"] = {}
    for frac in (0.0, 0.1, 0.2, 0.3, 0.4):
        ov = {"lidar_stale_p": frac * 5 / 8, "lidar_dropout_p": frac * 3 / 8}
        res["sweep_lidar_nonfresh"]["%.1f" % frac] = sens("nominal", ov, ["G3", "G2b", "G2bc"])
    print("done sensitivity, variant, sweep", flush=True)

    # drift monitor (not connected back to the guard)
    per_hour = 3600.0 / S.DEFAULTS["period_s"]
    onset = S.CONDITION_OVERRIDES["drift"]["drift_onset"]
    side = "lo" if S.CONDITION_OVERRIDES["drift"]["drift_final"] < 0 else "hi"
    hours_per_run = S.DEFAULTS["n_decisions"] / per_hour
    delays, pre_onset, wrong_side = [], 0, 0
    for r in main_runs["drift"]:
        # a detection is the first post-onset alarm in the direction of the injected drift
        post = [i for i, sd in r["alarms"] if i >= onset and sd == side]
        pre_onset += any(i < onset for i, sd in r["alarms"])
        wrong_side += sum(1 for i, sd in r["alarms"] if i >= onset and sd != side)
        if post:
            delays.append((post[0] - onset) / per_hour)
    fa = {c: sum(1 for r in main_runs[c] if r["alarms"]) for c in S.CONDITIONS if c != "drift"}
    fa_n = {c: sum(len(r["alarms"]) for r in main_runs[c]) for c in S.CONDITIONS if c != "drift"}
    res["drift"] = {
        "k": S.DEFAULTS["cusum_k"], "h": S.DEFAULTS["cusum_h"], "warmup_frames": S.DEFAULTS["cusum_warmup"],
        "onset_decision": onset, "runs": len(SEEDS), "detected": len(delays), "detection_side": side,
        "delay_hours": {"median": statistics.median(delays) if delays else None,
                        "min": min(delays) if delays else None, "max": max(delays) if delays else None,
                        "all": delays},
        "drift_runs_with_pre_onset_alarm": pre_onset,
        "drift_wrong_side_alarms_after_onset": wrong_side,
        "false_alarm_runs": fa,
        "false_alarm_count": fa_n,
        "false_alarm_runs_nominal": fa["nominal"],
        "false_alarms_per_100_robot_hours_nominal": 100.0 * fa_n["nominal"] / (hours_per_run * len(SEEDS)),
        "false_alarm_runs_other_non_drift": sum(v for c, v in fa.items() if c != "nominal"),
        "other_non_drift_runs": len(SEEDS) * (len(fa) - 1),
    }

    res["timing"] = {"note": "wall-clock, not deterministic; G2b decision + build and SHA-256 of trace record, not persisted",
                     "cpu": cpu_name(), "python": platform.python_version(),
                     "implementation": platform.python_implementation(), "per_condition": timing,
                     "whole_cycle": dict(cycle, note="exposure + Laplacian blur on a synthetic 640x480 frame in pure Python, "
                                         "G2b decision with hashed trace record, fsync'd append; sensor I/O and model "
                                         "inference excluded; fsync on macOS does not force a full device flush"),
                     "whole_cycle_numpy": cycle_np}
    res["timing"]["period_ms"] = 1000.0 * S.DEFAULTS["period_s"]
    res["timing"]["whole_cycle_share_of_period_pct"] = 100.0 * cycle["p99_ms"] / res["timing"]["period_ms"]

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    write_tables(res)
    print("wrote results.json and tables.md")


def write_tables(res):
    L = ["# Simulation tables (SIMULATION ONLY)", "",
         "Generated by `sim/run_all.py`. Mean (s.d.) over %d seeds, %d decisions per condition per seed." % (
             len(SEEDS), S.DEFAULTS["n_decisions"]),
         "Unsafe commits: % of hazards. False blocks: % of proposed clear-path commits. "
         "Pages: operator pages per robot-hour. Throughput: % of decision slots that committed.", ""]
    for cond in S.CONDITIONS:
        L += ["## " + cond, "", "| Gate | Unsafe commits % | False blocks % | Pages / robot-hour | Throughput % |",
              "|---|---|---|---|---|"]
        for g in S.GATES:
            m = res["main"][cond][g]
            L.append("| %s | %s |" % (g, " | ".join("%.2f (%.2f)" % (m[k]["mean"], m[k]["sd"]) for k in METRICS)))
        L.append("")
    L += ["## Sensitivity", "", "| Run | Gate | Unsafe commits % | False blocks % | Pages / robot-hour | Throughput % |",
          "|---|---|---|---|---|---|"]
    for name, s in res["sensitivity"].items():
        for g, m in s["result"].items():
            L.append("| %s %s | %s | %s |" % (name, json.dumps(s["overrides"]), g,
                     " | ".join("%.2f (%.2f)" % (m[k]["mean"], m[k]["sd"]) for k in METRICS)))
    L += ["", "## Robustness: G1 under occlusion against planner and camera parameters", "",
          "| Run | Gate | Unsafe commits % | False blocks % | Pages / robot-hour | Throughput % |", "|---|---|---|---|---|---|"]
    for name, s in res["robustness_G1_occlusion"]["runs"].items():
        for g, m in s["result"].items():
            L.append("| %s | %s | %s |" % (name, g, " | ".join("%.2f (%.2f)" % (m[k]["mean"], m[k]["sd"]) for k in METRICS)))
    L.append("")
    L.append("G1 unsafe commits under occlusion, base run included: %.1f%% to %.1f%%." % (
        res["robustness_G1_occlusion"]["G1_unsafe_min"], res["robustness_G1_occlusion"]["G1_unsafe_max"]))
    for key, title, lab in (("variant_camera_may_authorise", "Variant: a fresh, healthy camera may also authorise (G2bc)", "Condition"),
                            ("sweep_lidar_nonfresh", "Sweep: share of LiDAR readings not fresh (nominal otherwise)", "Non-fresh share")):
        L += ["", "## " + title, "", "| %s | Gate | Unsafe commits %% | False blocks %% | Pages / robot-hour | Throughput %% |" % lab,
              "|---|---|---|---|---|---|"]
        for c, gs in res[key].items():
            for g, m in gs.items():
                L.append("| %s | %s | %s |" % (c, g, " | ".join("%.2f (%.2f)" % (m[k]["mean"], m[k]["sd"]) for k in METRICS)))
    d = res["drift"]
    dh = d["delay_hours"]
    L += ["", "## Drift monitor (two-sided CUSUM, k = %s, h = %s, fixed; not connected to the guard)" % (d["k"], d["h"]), "",
          "- Detected (first post-onset alarm on the %s side, the direction of the injected drift): %d of %d drift runs" % (
              d["detection_side"], d["detected"], d["runs"]),
          "- Detection delay from drift onset, robot-hours: median %s, range %s to %s" % (
              *("%.2f" % v if v is not None else "n/a" for v in (dh["median"], dh["min"], dh["max"])),),
          "- Drift runs with an alarm before onset: %d; wrong-side alarms after onset in drift runs: %d" % (
              d["drift_runs_with_pre_onset_alarm"], d["drift_wrong_side_alarms_after_onset"]),
          "- Nominal false alarms: %.2f per 100 robot-hours" % d["false_alarms_per_100_robot_hours_nominal"],
          "- False-alarm runs: nominal %d of %d; other non-drift %d of %d (%s)" % (
              d["false_alarm_runs_nominal"], d["runs"], d["false_alarm_runs_other_non_drift"],
              d["other_non_drift_runs"], json.dumps(d["false_alarm_runs"])), ""]
    t = res["timing"]
    L += ["## Guard timing (wall clock; varies between runs)", "",
          "CPU: %s. %s %s." % (t["cpu"], t["implementation"], t["python"]), "",
          "| Condition | Median (us) | p99 (us) |", "|---|---|---|"]
    for c, v in t["per_condition"].items():
        L.append("| %s | %.2f | %.2f |" % (c, v["median_us"], v["p99_us"]))
    w = t["whole_cycle"]
    L += ["", "Whole guard cycle (%s): median %.1f ms, p99 %.1f ms over %d cycles." % (w["note"], w["median_ms"], w["p99_ms"], w["n"])]
    if t["whole_cycle_numpy"]:
        v = t["whole_cycle_numpy"]
        L += ["", "Same cycle with the image-quality step in numpy %s: median %.2f ms, p99 %.2f ms over %d cycles." % (
            v["numpy"], v["median_ms"], v["p99_ms"], v["n"])]
    L += ["", "Pure-Python cycle p99 as a share of the %d ms decision period: %.2f%%." % (
        t["period_ms"], t["whole_cycle_share_of_period_pct"])]
    with open(os.path.join(HERE, "tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
