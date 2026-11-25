"""Seeded simulation of pre-commitment gates for a warehouse mobile robot.

SIMULATION ONLY. Sensor, hazard and planner models are synthetic; nothing
here is a deployment measurement. Python 3 standard library only.

Rebuilt from the written specification on 24 Nov 2026 (the earlier kit was
not available). Parameters were fixed before the first run and were not
tuned to reproduce earlier numbers.
"""
import hashlib
import json
import random
import time

CONDITIONS = ["nominal", "low_light", "occlusion", "stale_lidar", "drift"]
GATES = ["G0", "G1", "G2", "G3", "G2b"]

DEFAULTS = {
    "period_s": 6.0,              # one commit decision every 6 s -> 600 per robot-hour
    "n_decisions": 20000,
    "hazard_rate": 0.05,
    "tau": 0.8,                   # G1 self-confidence threshold
    "health_thr": 0.6,            # camera health below this = unhealthy
    # camera detector: score in [0,1], verdict "hazard" if score > det_thr
    "cam_clear_mu": 0.20, "cam_clear_sd": 0.10,
    "cam_haz_mu": 0.80, "cam_haz_sd": 0.12,
    "cam_det_thr": 0.5,
    "cam_health_mu": 0.90, "cam_health_sd": 0.04,
    "cam_stale_p": 0.005,
    "occl_p": 0.0,                # P(hazard is occluded from the camera)
    "occl_mu": 0.30,              # camera score mean for an occluded hazard
    # LiDAR occupancy check
    "lidar_miss": 0.03, "lidar_fp": 0.01,
    "lidar_dropout_p": 0.01, "lidar_stale_p": 0.005,
    "lidar_miss_when_occluded": None,   # common-mode sensitivity run
    # planner stand-in
    "planner_w_cam": 0.8,
    "conf_noise_sd": 0.04,
    "conf_tracks_quality": False,  # the ASSUMPTION under test: False = confidence
                                   # does not fall as image quality degrades
    # calibration drift (camera score offset, linear ramp from onset to end)
    "drift_onset": None, "drift_final": 0.0,
    # CUSUM drift monitor (fixed)
    "cusum_k": 0.75, "cusum_h": 10.0, "cusum_warmup": 2000,
}

CONDITION_OVERRIDES = {
    "nominal": {},
    "low_light": {"cam_haz_mu": 0.45, "cam_haz_sd": 0.15, "cam_clear_sd": 0.13,
                  "cam_health_mu": 0.35, "cam_health_sd": 0.08},
    # exposure/blur health cannot see occlusion, so camera health stays nominal
    "occlusion": {"occl_p": 0.6},
    "stale_lidar": {"lidar_stale_p": 0.25, "lidar_dropout_p": 0.15},
    "drift": {"drift_onset": 5000, "drift_final": -0.25},
}


def params_for(condition, overrides=None):
    p = dict(DEFAULTS)
    p.update(CONDITION_OVERRIDES[condition])
    if overrides:
        p.update(overrides)
    return p


def _clip(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def _cam_score(rng, p, hazard, occluded, offset):
    if hazard and occluded:
        s = rng.gauss(p["occl_mu"], p["cam_clear_sd"])
    elif hazard:
        s = rng.gauss(p["cam_haz_mu"], p["cam_haz_sd"])
    else:
        s = rng.gauss(p["cam_clear_mu"], p["cam_clear_sd"])
    return _clip(s + offset)


def observe(rng, p, i):
    """Draw one decision slot: world state, both channels, planner proposal."""
    hazard = rng.random() < p["hazard_rate"]
    occluded = hazard and rng.random() < p["occl_p"]
    offset = 0.0
    if p["drift_onset"] is not None and i >= p["drift_onset"]:
        offset = p["drift_final"] * (i - p["drift_onset"]) / (p["n_decisions"] - p["drift_onset"])

    # camera: a stale frame shows an earlier, unrelated world state
    cam_fresh = rng.random() >= p["cam_stale_p"]
    if cam_fresh:
        cam_score = _cam_score(rng, p, hazard, occluded, offset)
        cam_age = rng.uniform(0.05, 0.3)
    else:
        old_h = rng.random() < p["hazard_rate"]
        cam_score = _cam_score(rng, p, old_h, old_h and rng.random() < p["occl_p"], offset)
        cam_age = rng.uniform(1.5, 10.0)
    cam_health = _clip(rng.gauss(p["cam_health_mu"], p["cam_health_sd"]))
    cam_hazard = cam_score > p["cam_det_thr"]

    # LiDAR: dropped = no verdict; stale = verdict about an earlier world state
    u = rng.random()
    lidar_dropped = u < p["lidar_dropout_p"]
    lidar_fresh = (not lidar_dropped) and u >= p["lidar_dropout_p"] + p["lidar_stale_p"]
    miss = p["lidar_miss"]
    if occluded and p["lidar_miss_when_occluded"] is not None:
        miss = p["lidar_miss_when_occluded"]
    if lidar_dropped:
        lidar_hazard = None
        lidar_age = None
    elif lidar_fresh:
        lidar_hazard = (rng.random() >= miss) if hazard else (rng.random() < p["lidar_fp"])
        lidar_age = rng.uniform(0.02, 0.15)
    else:
        old_h = rng.random() < p["hazard_rate"]
        lidar_hazard = (rng.random() >= p["lidar_miss"]) if old_h else (rng.random() < p["lidar_fp"])
        lidar_age = rng.uniform(1.0, 10.0)

    # planner: fuses internally, camera-heavy; silently uses camera alone if LiDAR is gone
    if lidar_hazard is None:
        belief = cam_score
    else:
        belief = p["planner_w_cam"] * cam_score + (1 - p["planner_w_cam"]) * (1.0 if lidar_hazard else 0.0)
    proposes = belief < 0.5
    conf = 0.5 + 0.5 * (abs(belief - 0.5) / 0.5) ** 0.5
    if p["conf_tracks_quality"]:
        conf *= 0.4 + 0.6 * cam_health
    conf = _clip(conf + rng.gauss(0.0, p["conf_noise_sd"]))

    return {
        "i": i, "hazard": hazard,
        "cam_score": cam_score, "cam_hazard": cam_hazard, "cam_health": cam_health,
        "cam_fresh": cam_fresh, "cam_age": cam_age,
        "lidar_hazard": lidar_hazard, "lidar_fresh": lidar_fresh,
        "lidar_dropped": lidar_dropped, "lidar_age": lidar_age,
        "proposes": proposes, "conf": conf,
    }


# ---- gates: return "commit", "hold" or "page" for a proposed commit ----

def gate_G0(o, p):
    return "commit"


def gate_G1(o, p):
    return "commit" if o["conf"] >= p["tau"] else "page"


def gate_G2(o, p):
    """Original guard: only healthy, fresh channels count, for veto and authorise alike."""
    cam_ok = o["cam_fresh"] and o["cam_health"] >= p["health_thr"]
    lidar_ok = o["lidar_fresh"]
    if cam_ok and o["cam_hazard"]:
        return "hold"
    if lidar_ok and o["lidar_hazard"]:
        return "hold"
    if lidar_ok:
        return "commit"
    if cam_ok:
        return "page"      # the only healthy evidence is the camera
    return "hold"


def gate_G3(o, p):
    """Ablation: either sensor may veto; no health or freshness checks."""
    if o["cam_hazard"] or o["lidar_hazard"]:
        return "hold"
    return "commit"


def gate_G2b(o, p):
    """Revised guard: any channel may veto, healthy or not; only a fresh, healthy,
    independent channel (LiDAR; the planner leans on the camera) may authorise;
    otherwise hold. Never pages."""
    if o["cam_hazard"] or o["lidar_hazard"]:
        return "hold"
    if o["lidar_fresh"]:
        return "commit"
    return "hold"


def gate_G2bc(o, p):
    """Variant of G2b (not one of the five main gates): a fresh, healthy camera
    may also authorise. Shows the safety-against-availability trade-off."""
    if o["cam_hazard"] or o["lidar_hazard"]:
        return "hold"
    if o["lidar_fresh"] or (o["cam_fresh"] and o["cam_health"] >= p["health_thr"]):
        return "commit"
    return "hold"


GATE_FUNCS = {"G0": gate_G0, "G1": gate_G1, "G2": gate_G2, "G3": gate_G3, "G2b": gate_G2b,
              "G2bc": gate_G2bc}


def guard_with_trace(o, p):
    """G2b decision plus building and hashing (not persisting) its trace record."""
    decision = gate_G2b(o, p)
    record = {
        "trace_id": o["i"],
        "camera": {"health": o["cam_health"], "age_s": o["cam_age"], "fresh": o["cam_fresh"],
                   "verdict": "hazard" if o["cam_hazard"] else "clear"},
        "lidar": {"dropped": o["lidar_dropped"], "age_s": o["lidar_age"], "fresh": o["lidar_fresh"],
                  "verdict": None if o["lidar_hazard"] is None else ("hazard" if o["lidar_hazard"] else "clear")},
        "planner": {"proposes_commit": o["proposes"], "confidence": o["conf"]},
        "decision": decision,
    }
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return decision, digest


class Cusum:
    """Two-sided CUSUM on the camera score, fed only LiDAR-clear frames."""

    def __init__(self, k, h, warmup):
        self.k, self.h, self.warmup = k, h, warmup
        self.buf = []
        self.mu = self.sd = None
        self.hi = self.lo = 0.0
        self.alarms = []          # (decision index, "hi" or "lo")

    def update(self, i, x):
        if self.mu is None:
            self.buf.append(x)
            if len(self.buf) >= self.warmup:
                n = len(self.buf)
                self.mu = sum(self.buf) / n
                self.sd = (sum((v - self.mu) ** 2 for v in self.buf) / (n - 1)) ** 0.5
                self.calibrated_at = i
            return
        z = (x - self.mu) / self.sd
        self.hi = max(0.0, self.hi + z - self.k)
        self.lo = max(0.0, self.lo - z - self.k)
        if self.hi > self.h or self.lo > self.h:
            self.alarms.append((i, "hi" if self.hi > self.h else "lo"))
            self.hi = self.lo = 0.0


def run(condition, seed, overrides=None, gates=GATES, keep_obs=False):
    """One run of n_decisions. All gates see the same draws."""
    p = params_for(condition, overrides)
    rng = random.Random(f"{condition}:{seed}")
    counts = {g: {"unsafe": 0, "false_block": 0, "pages": 0, "commits": 0} for g in gates}
    hazards = clear_proposed = 0
    cusum = Cusum(p["cusum_k"], p["cusum_h"], p["cusum_warmup"])
    obs_list = []
    for i in range(p["n_decisions"]):
        o = observe(rng, p, i)
        if keep_obs:
            obs_list.append(o)
        if o["hazard"]:
            hazards += 1
        if o["lidar_fresh"] and not o["lidar_hazard"]:
            cusum.update(i, o["cam_score"])
        if not o["proposes"]:
            continue
        if not o["hazard"]:
            clear_proposed += 1
        for g in gates:
            d = GATE_FUNCS[g](o, p)
            c = counts[g]
            if d == "commit":
                c["commits"] += 1
                if o["hazard"]:
                    c["unsafe"] += 1
            else:
                if not o["hazard"]:
                    c["false_block"] += 1
                if d == "page":
                    c["pages"] += 1
    n = p["n_decisions"]
    hours = n * p["period_s"] / 3600.0
    metrics = {}
    for g in gates:
        c = counts[g]
        metrics[g] = {
            "unsafe_pct": 100.0 * c["unsafe"] / hazards,
            "false_block_pct": 100.0 * c["false_block"] / clear_proposed,
            "pages_per_hour": c["pages"] / hours,
            "throughput_pct": 100.0 * c["commits"] / n,
        }
    out = {"metrics": metrics, "alarms": cusum.alarms,
           "drift_onset": p["drift_onset"], "period_s": p["period_s"]}
    if keep_obs:
        out["obs"] = obs_list
        out["params"] = p
    return out


def time_guard(obs_list, p):
    """Per-call latency of guard_with_trace in ns (wall clock, perf_counter_ns)."""
    for o in obs_list[:2000]:
        guard_with_trace(o, p)   # warm-up
    samples = []
    for o in obs_list:
        t0 = time.perf_counter_ns()
        guard_with_trace(o, p)
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    return {"median_us": samples[len(samples) // 2] / 1000.0,
            "p99_us": samples[int(len(samples) * 0.99)] / 1000.0,
            "n": len(samples)}


# ---- whole guard cycle: image-quality metrics + decision + durable trace write ----

def make_frame(rng, w=640, h=480):
    """Synthetic 8-bit greyscale frame (not a real image)."""
    return bytes(rng.randrange(256) for _ in range(w * h))


def image_quality(frame, w, h):
    """Mean exposure and variance of the 4-neighbour Laplacian (a blur measure)."""
    exposure = sum(frame) / len(frame)
    n = s1 = s2 = 0
    for y in range(1, h - 1):
        up = frame[(y - 1) * w + 1:y * w - 1]
        row = frame[y * w:(y + 1) * w]
        dn = frame[(y + 1) * w + 1:(y + 2) * w - 1]
        for c, l, r, u, d in zip(row[1:-1], row[:-2], row[2:], up, dn):
            v = 4 * c - l - r - u - d
            s1 += v
            s2 += v * v
        n += w - 2
    return exposure, s2 / n - (s1 / n) ** 2


def image_quality_numpy(frame, w, h):
    """Same two measures with numpy (optional; only used for the timing comparison)."""
    import numpy as np
    a = np.frombuffer(frame, dtype=np.uint8).reshape(h, w).astype(np.int32)
    lap = 4 * a[1:-1, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:] - a[:-2, 1:-1] - a[2:, 1:-1]
    return float(a.mean()), float(lap.var())


def time_cycle(obs_list, p, path, n=100, seed=0, quality=None):
    """Latency of one whole guard cycle, in ms: exposure and blur on a 640x480
    frame in pure Python, the G2b decision with its hashed trace record, and an
    fsync'd append of that record. Sensor I/O and model inference are excluded."""
    import os
    quality = quality or image_quality
    rng = random.Random(f"cycle:{seed}")
    frames = [make_frame(rng) for _ in range(4)]
    samples = []
    with open(path, "a") as f:
        for j in range(n + 5):
            o = dict(obs_list[j])
            t0 = time.perf_counter_ns()
            exposure, blur = quality(frames[j % 4], 640, 480)
            o["cam_health"] = min(1.0, exposure / 128.0) * (1.0 if blur > 100.0 else 0.5)
            decision, digest = guard_with_trace(o, p)
            f.write(digest + " " + decision + "\n")
            f.flush()
            os.fsync(f.fileno())
            if j >= 5:
                samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    return {"median_ms": samples[len(samples) // 2] / 1e6,
            "p99_ms": samples[int(len(samples) * 0.99)] / 1e6, "n": len(samples)}
