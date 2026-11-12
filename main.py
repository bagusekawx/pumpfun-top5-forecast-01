#!/usr/bin/env python3
"""pumpfun-top5-forecast-01: Monte Carlo top-5 forecaster (v9 candidate).

Same calibration as the analytic entry (per-venue vol/drift, per-coin drift
with momentum saturation and mean reversion), but the forecast itself comes
from simulation instead of a Poisson-binomial DP:

* Incumbents: one final log-cap draw per path from their Gaussian, with a
  per-venue death hazard (backtests showed an age-bucketed hazard hurt:
  it penalised the young top coins that actually persist).
* Newcomers: a history-analog field.  Every historical coin that first
  reached log-cap 2.0 contributes a trajectory aligned at that crossing.
  New first-2.0 crossings arrive as a Poisson process (recency-weighted
  rate from the recent window); each newcomer's cap at a read is sampled
  from a random analog trajectory at the same time-since-crossing, with
  drift/vol extrapolation past the trajectory's observed span, plus a
  +0.2 log10 shift: the library cannot exceed the historical maximum cap,
  but the next cohort in a growing market does.
* Per read, N seeded paths give the top 5 (mints + caps); place
  probabilities and size-bin histograms are empirical frequencies,
  averaged over the reads.  Deterministic: fixed seed, no set iteration
  order leaks into the output.
"""
import heapq
import json
import math
import os
import random
import sys
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# ---- tuning constants (priors; the data-driven parts are estimated) ----
GRID_LO, GRID_HI, GRID_STEP = 2.0, 6.0, 0.05   # log10 cap grid
INCUMBENT_CAP = 1.9        # last cap at/above this -> modelled individually
TOP_EVER = 30              # ever ranked this high -> modelled individually
MAX_INCIDENTS = 800        # cap on individually modelled coins
NAME_MIN_P = 0.002         # floor for naming a coin at a place (the greedy
                           # rule below usually stops well above this)
P0_MAX_Q = 0.002           # p=0 only when the place probability is below this
MAX_ROWS_PER_PLACE = 100
DRIFT_TAU_H = 1.5          # momentum saturates over ~this many hours
REVERT_TAU_H = 2.0         # mean-reversion timescale (hours)
REVERT_STRENGTH = 0.9      # fraction of the excess over REVERT_LEVEL lost
REVERT_LEVEL = 2.2         # log10 cap above which the pull applies
VOL_CAP_H = 6.0            # volatility stops growing like sqrt(h) past this
VOL_INFLATE = 1.15         # vol is understated: crashed coins leave the sample
YOUNG_AGE_H = 2.0          # "young crosser" age cutoff for lambda(x)
LAMBDA_TAU_H = 1.0         # recency half-life for the newcomer rate estimate
FLOOR_CAP = 1.46           # untraded curve coins sit at ~1.45; treat as flat
MICRO = 1_000_000

# ---- Monte Carlo constants ----
MC_SEED = 20261112
MC_PATHS = 3000           # paths per read (adapted down for huge inputs)
MC_CONTENDER_CAP = 2.0    # log-cap a coin must reach to join the analog library
MC_NEW_NOISE = 0.08       # extra sd on sampled newcomer caps (smoothing)
MC_MIN_CAP = 1.5          # simulated caps below this cannot make the top 5
MC_NEW_SHIFT = 0.2        # analog caps understate future newcomer caps: the
                          # library cannot exceed the historical maximum, but
                          # in a growing market the next cohort does.  +0.2
                          # log10 on sampled newcomer caps (backtest-tuned:
                          # 0.2 beats 0.1/0.25/0.3 on the representative
                          # truncations; see dev/aggbt.py)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def logcap(row):
    return math.log10(int(row["market_cap_lamports"])) - 9.0


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def norm_pdf(x, mu, sd):
    z = (x - mu) / sd
    return math.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def floor6(p):
    u = math.floor(p * MICRO)
    return u / MICRO if u > 0 else 0.0


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    places = int(task.get("places", 5))
    edges = [float(e) for e in task["size"]["edges"]]
    out_name = task.get("output", "forecast.json")
    reads = [parse_time(s) for s in task.get("reads", [])]
    deadline = parse_time(task["window"]["deadline"]) if task.get("window") else None

    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    if not marks:
        raise SystemExit("the history holds no ok mark")
    marks.sort(key=lambda m: m["at"])
    mark_times = [parse_time(m["at"]) for m in marks]
    t_last = mark_times[-1]
    if not reads:
        base = deadline or t_last
        reads = [base]
    horizons = [max(0.0, (r - t_last).total_seconds() / 3600.0) for r in reads]

    # ---- trajectories: mint -> list of (mark_idx, rank, logcap, venue, created) ----
    traj = {}
    for mi, m in enumerate(marks):
        for row in m["rows"]:
            try:
                lc = logcap(row)
            except (KeyError, ValueError):
                continue
            traj.setdefault(row["mint"], []).append(
                (mi, int(row["rank"]), lc, row.get("venue", "curve"),
                 parse_time(row["created_at"]) if row.get("created_at") else None)
            )
    last_mark_rows = sorted(marks[-1]["rows"], key=lambda r: int(r["rank"]))
    last_rank = {}
    last_cap = {}
    for row in marks[-1]["rows"]:
        last_rank[row["mint"]] = int(row["rank"])
        try:
            last_cap[row["mint"]] = logcap(row)
        except (KeyError, ValueError):
            pass

    # ---- recent window for parameter estimation ----
    span_h = max(0.0, (t_last - mark_times[0]).total_seconds() / 3600.0)
    win_h = max(span_h / 3.0, min(2.0, span_h)) if span_h > 0 else 0.0
    win_start = t_last.timestamp() - win_h * 3600.0
    recent_pairs = [(a, b) for a, b in zip(range(len(marks) - 1), range(1, len(marks)))
                    if mark_times[b].timestamp() >= win_start]
    if not recent_pairs and len(marks) > 1:
        recent_pairs = [(len(marks) - 2, len(marks) - 1)]
    dt_h = []
    for a, b in recent_pairs:
        dt_h.append(max(1e-6, (mark_times[b] - mark_times[a]).total_seconds() / 3600.0))

    # ---- per-venue volatility and drift from active coins ----
    deltas = {"pool": [], "curve": [], "migrating": []}
    for mint, obs in traj.items():
        for (a, b), dh in zip(recent_pairs, dt_h):
            oa = next((o for o in obs if o[0] == a), None)
            ob = next((o for o in obs if o[0] == b), None)
            if oa is None or ob is None:
                continue
            d = ob[2] - oa[2]
            if abs(d) < 1e-9:
                continue  # untraded floor coin: no information
            deltas.setdefault(oa[3], deltas["curve"]).append(d / math.sqrt(max(dh, 1e-6) / (10.0 / 60.0)))
    vol10, drift10 = {}, {}
    for venue in ("pool", "curve", "migrating"):
        v = deltas.get(venue) or []
        if len(v) >= 5:
            mean = sum(v) / len(v)
            var = sum((x - mean) ** 2 for x in v) / len(v)
            vol10[venue] = clamp(math.sqrt(var), 0.05, 0.35)
            drift10[venue] = clamp(mean, -0.10, 0.10)
        else:
            vol10[venue] = 0.13 if venue == "pool" else 0.10
            drift10[venue] = 0.0
    vol10.setdefault("curve", 0.10)
    drift10.setdefault("curve", 0.0)

    # ---- death hazard per venue: P(coin leaves the top-200 next mark) ----
    gone = {"pool": [0, 0], "curve": [0, 0]}
    for a, b in recent_pairs:
        cur = {r["mint"]: r for r in marks[a]["rows"]}
        nxt = {r["mint"] for r in marks[b]["rows"]}
        for mint, r in cur.items():
            g = gone.setdefault(r.get("venue", "curve"), gone["curve"])
            g[1] += 1
            if mint not in nxt:
                g[0] += 1
    lam_death = {}
    for venue in ("pool", "curve", "migrating"):
        g, n = gone.get(venue, [0, 0])
        p10 = (g + 1.0) / (n + 20.0) if n else 0.05
        lam_death[venue] = clamp(-math.log(max(1e-9, 1.0 - p10)) * 6.0, 0.01, 1.0)

    # ---- incumbent selection ----
    incumbents = []
    for mint, obs in traj.items():
        last = obs[-1]
        best_rank = min(o[1] for o in obs)
        last_seen_h = (t_last - mark_times[last[0]]).total_seconds() / 3600.0
        if (last[2] >= INCUMBENT_CAP
                or (best_rank <= TOP_EVER and last_seen_h <= 3.0)
                or last[1] <= TOP_EVER):
            incumbents.append(mint)
    incumbents.sort(key=lambda m: traj[m][-1][2], reverse=True)
    incumbents = incumbents[:MAX_INCIDENTS]
    inc_set = set(incumbents)

    # per-coin drift (own recent slope shrunk to venue drift)
    coin = {}
    for mint in incumbents:
        obs = traj[mint]
        last = obs[-1]
        venue = last[3]
        x = last[2]
        age_h = None
        if last[4] is not None:
            age_h = (mark_times[last[0]] - last[4]).total_seconds() / 3600.0
        # own slope over the last up-to-3 intervals ending at the last obs
        slope, n = 0.0, 0
        tail = obs[-4:]
        for o1, o2 in zip(tail, tail[1:]):
            dh = (mark_times[o2[0]] - mark_times[o1[0]]).total_seconds() / 3600.0
            if dh > 1e-6:
                slope += (o2[2] - o1[2]) / dh
                n += 1
        own = slope / n if n else None
        vdrift_h = drift10.get(venue, 0.0) * 6.0
        if own is None:
            drift_h = vdrift_h
        else:
            w = n / (n + 3.0)
            drift_h = w * clamp(own, -1.0, 1.0) + (1.0 - w) * vdrift_h
        drift_h = clamp(drift_h, -0.6, 0.6)
        sig10 = vol10.get(venue, 0.12) * (1.3 if venue == "curve" else 1.0) * VOL_INFLATE
        coin[mint] = {
            "x": x, "venue": venue, "drift_h": drift_h, "sig10": sig10,
            "lam_d": lam_death.get(venue, 0.1),
            "last_h": (t_last - mark_times[last[0]]).total_seconds() / 3600.0,
            "rank": last[1], "age_h": age_h,
        }

    # ---- newcomer analog field: contender trajectories + crossing rate ----
    # Every historical coin that first reached MC_CONTENDER_CAP contributes
    # a trajectory aligned at that crossing: (minutes since crossing,
    # log-cap).  New crossings arrive as a Poisson process at the
    # recency-weighted rate of first crossings in the recent window.
    analog_trajs = []  # (taus_min list, caps list, venue, cross_time)
    cross_w = 0.0
    for mint, obs in traj.items():
        ci = None
        for j, o in enumerate(obs):
            if o[2] >= MC_CONTENDER_CAP:
                ci = j
                break
        if ci is None:
            continue
        t0 = mark_times[obs[ci][0]]
        if t0.timestamp() >= win_start:
            cross_w += math.exp(-(t_last - t0).total_seconds() / 3600.0
                                / LAMBDA_TAU_H)
        taus, caps = [], []
        for o in obs[ci:]:
            taus.append((mark_times[o[0]] - t0).total_seconds() / 60.0)
            caps.append(o[2])
        analog_trajs.append((taus, caps, obs[-1][3], t0))
    win_span_h = 0.0
    for (a, b), dh in zip(recent_pairs, dt_h):
        win_span_h += (math.exp(-(t_last - mark_times[b]).total_seconds()
                                / 3600.0 / LAMBDA_TAU_H) * dh)
    rate_new = cross_w / win_span_h if win_span_h > 0 else 0.0
    # fallback pool: every contender cap ever observed (used when the
    # sampled trajectory has no point near the requested tau)
    _all_cont_caps = [c for _, caps, _, _ in analog_trajs for c in caps]
    # No simulated newcomer may exceed the historical maximum cap by more
    # than this: a day-old coin 3x the previous max within hours is
    # essentially impossible, and without the cap the max of hundreds of
    # wide extrapolated Gaussians explodes at long horizons.
    _hist_max = max([lc for _, caps, _, _ in analog_trajs for lc in caps]
                    + [c["x"] for c in coin.values()] + [2.0])
    _new_cap_hi = _hist_max + 0.5

    def sample_newcomer_cap(rng, tau_min):
        """A newcomer's log-cap tau_min minutes after its first-2.0
        crossing, from a random analog trajectory; None if it died."""
        if not analog_trajs:
            if not _all_cont_caps:
                return 2.3 + rng.gauss(0.0, 0.6)
            return rng.choice(_all_cont_caps) + rng.gauss(0.0, MC_NEW_NOISE)
        taus, caps, venue, _t0 = rng.choice(analog_trajs)
        if tau_min <= taus[-1]:
            lo, hi = 0, len(taus) - 1
            while lo < hi:
                mid = (lo + hi) >> 1
                if taus[mid] <= tau_min:
                    lo = mid + 1
                else:
                    hi = mid
            j = lo - 1
            if j < 0:
                cap = caps[0]
            elif j >= len(taus) - 1:
                cap = caps[-1]
            else:
                span = taus[j + 1] - taus[j]
                w = 0.0 if span <= 0 else (tau_min - taus[j]) / span
                cap = caps[j] * (1.0 - w) + caps[j + 1] * w
            return min(cap + MC_NEW_SHIFT + rng.gauss(0.0, MC_NEW_NOISE),
                       _new_cap_hi)
        extra_h = (tau_min - taus[-1]) / 60.0
        # Contenders die slower than the average coin in rows (high-cap
        # pool coins leave at ~0.7/h vs ~2/h for curve coins overall),
        # so cap the venue hazard: the trajectory already survived to
        # its last observation, and a 2/h hazard would kill nearly every
        # extrapolated newcomer.
        lam = min(lam_death.get(venue, 0.1), 1.0)
        if rng.random() < 1.0 - math.exp(-lam * extra_h):
            return None
        # Extrapolate with the incumbent dynamics: saturating drift,
        # mean reversion on pumped coins, volatility capped past 6h.
        # (Without the vol cap and reversion, the max of hundreds of
        # wide Gaussians explodes at long horizons.)
        drift_h = drift10.get(venue, 0.0) * 6.0
        sig10 = vol10.get(venue, 0.12) * (1.3 if venue == "curve" else 1.0) \
            * VOL_INFLATE
        mean = (caps[-1]
                + drift_h * DRIFT_TAU_H * (1.0 - math.exp(-extra_h / DRIFT_TAU_H))
                - REVERT_STRENGTH * max(0.0, caps[-1] - REVERT_LEVEL)
                * (1.0 - math.exp(-extra_h / REVERT_TAU_H)))
        sd = max(sig10 * math.sqrt(6.0 * min(extra_h, VOL_CAP_H)), 0.05)
        return min(mean + rng.gauss(0.0, sd) + MC_NEW_SHIFT
                   + rng.gauss(0.0, MC_NEW_NOISE), _new_cap_hi)


    # ---- Monte Carlo simulation per read ----
    n_reads = len(reads)
    debug = os.environ.get("MAKEFUN_DEBUG") == "1"
    rng = random.Random(MC_SEED)
    if len(incumbents) <= 250:
        n_paths = MC_PATHS
    elif len(incumbents) <= 500:
        n_paths = 1500
    else:
        n_paths = 1000
    n_bins = len(edges) + 1
    place_p = {mint: [0.0] * places for mint in incumbents}
    size_acc = [[0.0] * n_bins for _ in range(places)]
    for r, h in enumerate(horizons):
        # incumbent marginals at this read (same as the analytic model,
        # with the hazard evaluated at the coin's mid-horizon age)
        params = []
        for mint in incumbents:
            c = coin[mint]
            hh = h + c["last_h"]
            mean = (c["x"]
                    + c["drift_h"] * DRIFT_TAU_H * (1.0 - math.exp(-hh / DRIFT_TAU_H))
                    - REVERT_STRENGTH * max(0.0, c["x"] - REVERT_LEVEL)
                    * (1.0 - math.exp(-hh / REVERT_TAU_H)))
            sd = max(c["sig10"] * math.sqrt(6.0 * min(hh, VOL_CAP_H)), 0.05)
            surv = math.exp(-c["lam_d"] * hh)
            params.append((mint, mean, sd, surv))
        mu_new = rate_new * h
        place_counts = {mint: [0] * places for mint in incumbents}
        size_counts = [[0] * n_bins for _ in range(places)]
        newcomer_top = [0] * places
        for _ in range(n_paths):
            cands = []
            for mint, mean, sd, surv in params:
                if surv < 1.0 and rng.random() > surv:
                    continue
                cap = rng.gauss(mean, sd)
                if cap >= MC_MIN_CAP:
                    cands.append((cap, mint))
            m_new = _poisson(rng, mu_new)
            for _ in range(m_new):
                tau_min = rng.random() * h * 60.0
                cap = sample_newcomer_cap(rng, tau_min)
                if cap is not None and cap >= MC_MIN_CAP:
                    cands.append((cap, None))
            if not cands:
                for k in range(places):
                    size_counts[k][0] += 1
                continue
            top = heapq.nlargest(places, cands, key=lambda t: t[0])
            for k, (cap, mint) in enumerate(top):
                size_counts[k][_bin_of(cap, edges)] += 1
                if mint is not None:
                    place_counts[mint][k] += 1
                else:
                    newcomer_top[k] += 1
            for k in range(len(top), places):
                size_counts[k][0] += 1
        for mint in incumbents:
            pc = place_counts[mint]
            for k in range(places):
                place_p[mint][k] += pc[k] / n_paths / n_reads
        for k in range(places):
            for b in range(n_bins):
                size_acc[k][b] += size_counts[k][b] / n_paths / n_reads
        if debug:
            print(f"[dbg] read {r} h={h:.2f} paths={n_paths} "
                  f"mu_new={mu_new:.1f} rate_new={rate_new:.1f}/h "
                  f"newcomer_top={[round(c / n_paths, 2) for c in newcomer_top]}",
                  file=sys.stderr)

    # ---- assemble the forecast ----
    # sizes: widen for regime/parameter uncertainty.  No trend shift: the
    # DP's newcomer field already carries the ramp, and a fitted trend
    # double-counts it (v1's shift put the size means ~0.9 too high).
    h_mean = sum(horizons) / max(1, len(horizons))
    n_data = sum(1 for m in marks if m["rows"])
    widen = (0.10 + 0.20 * (1.0 - math.exp(-h_mean / 3.0))
             + 0.10 * max(0.0, 1.0 - n_data / 20.0) * min(1.0, h_mean))

    for k in range(places):
        size_acc[k] = _shift_widen(size_acc[k], edges, 0.0, widen)
    probabilities = []
    # Greedy naming: name a coin at a place only while its probability exceeds
    # the unnamed share it would receive if left unnamed ((1 - sum_named)/D,
    # D ~ 65 chart ids not named).  Naming below that share is strictly worse
    # than leaving the coin unnamed under a proper log score.
    D_EST = 65.0
    for k in range(places):
        cands = sorted(incumbents, key=lambda m: place_p[m][k], reverse=True)
        named = 0
        sum_named = 0.0
        for mint in cands:
            p = place_p[mint][k]
            thresh = max(NAME_MIN_P, (1.0 - sum_named) / D_EST)
            if p < thresh or named >= MAX_ROWS_PER_PLACE:
                break
            pf = floor6(p)
            if pf > 0:
                probabilities.append({"mint": mint, "place": k + 1, "p": pf})
                sum_named += pf
                named += 1
        # p=0 rows: negligible place probability, likely on the answer chart.
        # Naming them removes them from the scorer's unnamed-share
        # denominator.  Candidates: incumbents with q below the threshold,
        # plus untraded floor coins in the last mark's top 100 (their
        # top-5 chance is ~zero, so the -9.21 floor risk is negligible).
        if named < MAX_ROWS_PER_PLACE:
            rest = [m for m in cands
                    if place_p[m][k] < P0_MAX_Q and last_rank.get(m, 999) <= 100]
            rest.sort(key=lambda m: last_rank.get(m, 999))
            floor = [m for m, rk in last_rank.items()
                     if rk <= 100 and last_cap.get(m, 9.0) < 1.6 and m not in inc_set]
            floor.sort(key=lambda m: last_rank.get(m, 999))
            for mint in (rest + floor)[:MAX_ROWS_PER_PLACE - named]:
                probabilities.append({"mint": mint, "place": k + 1, "p": 0.0})
                named += 1
    # top5: greedy by place
    top5 = []
    used = set()
    for k in range(places):
        best, best_p = None, -1.0
        for mint in incumbents:
            if mint in used:
                continue
            if place_p[mint][k] > best_p:
                best, best_p = mint, place_p[mint][k]
        if best is None:
            break
        top5.append(best)
        used.add(best)
    if len(top5) < places:
        for mint in incumbents:
            if mint not in used:
                top5.append(mint)
                used.add(mint)
            if len(top5) >= places:
                break
    while len(top5) < places:
        top5.append("11111111111111111111111111111111")
    # sizes
    sizes = []
    for k in range(places):
        for b, p in enumerate(size_acc[k]):
            pf = floor6(p)
            if pf > 0:
                sizes.append({"place": k + 1, "bin": b, "p": pf})
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, out_name), "w") as f:
        f.write(json.dumps({"probabilities": probabilities, "top5": top5,
                            "sizes": sizes}) + "\n")


def _shift_widen(probs, edges, shift, sd):
    """Shift a binned size distribution by `shift` and convolve with a
    Normal(0, sd), returning new bin probabilities (82 bins)."""
    n_bins = len(edges) + 1
    centers = []
    for b in range(n_bins):
        if b == 0:
            centers.append(edges[0] - 0.05)
        elif b == n_bins - 1:
            centers.append(edges[-1] + 0.05)
        else:
            centers.append(0.5 * (edges[b - 1] + edges[b]))
    out = [0.0] * n_bins
    for b, p in enumerate(probs):
        if p <= 0:
            continue
        mu = centers[b] + shift
        for b2 in range(n_bins):
            lo = -math.inf if b2 == 0 else edges[b2 - 1]
            hi = math.inf if b2 == n_bins - 1 else edges[b2]
            out[b2] += p * (norm_cdf((hi - mu) / sd) - norm_cdf((lo - mu) / sd))
    tot = sum(out)
    if tot > 0:
        out = [v / tot for v in out]
    return out


def _poisson(rng, mu):
    """Poisson draw: Knuth for small means, normal approximation above 30."""
    if mu <= 0:
        return 0
    if mu < 30.0:
        limit = math.exp(-mu)
        k, prod = 0, 1.0
        while True:
            prod *= rng.random()
            if prod <= limit:
                return k
            k += 1
    return max(0, int(round(rng.gauss(mu, math.sqrt(mu)))))


def _bin_of(cap, edges):
    """Size-bin index of a log-cap: 0 below edges[0], len(edges) above."""
    lo, hi = 0, len(edges)
    while lo < hi:
        mid = (lo + hi) >> 1
        if edges[mid] <= cap:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _gi(x):
    return int(round((x - GRID_LO) / GRID_STEP))


def _cdf_at(x, grid, cdf):
    """CDF value at x by nearest grid point (edges lie on the grid)."""
    gi = int(round((x - grid[0]) / GRID_STEP))
    gi = clamp(gi, 0, len(grid) - 1)
    return cdf[gi]


def _leave_one_out(full, p):
    """P(exactly m others above x) for m = 0..4, from the full PMF.

    Full count S = B_i + S_{-i}; P(S=m) = (1-p) P_{-i}(m) + p P_{-i}(m-1).
    """
    out = [0.0] * 5
    if p >= 0.999:
        for m in range(5):
            out[m] = full[m + 1] if m + 1 < len(full) else 0.0
        return out
    q = 1.0 - p
    prev = 0.0
    for m in range(5):
        v = (full[m] - p * prev) / q
        v = clamp(v, 0.0, 1.0)
        out[m] = v
        prev = v
    return out


def _smooth_decreasing(vals):
    """3-point smoothing, then enforce non-increasing."""
    n = len(vals)
    sm = list(vals)
    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n - 1, i + 1)
        sm[i] = sum(vals[lo:hi + 1]) / (hi - lo + 1)
    for i in range(1, n):
        if sm[i] > sm[i - 1]:
            sm[i] = sm[i - 1]
    return sm


def _extrapolate_tail(grid, lam):
    """Log-linear tail above the highest grid point with lam > 0."""
    nz = [i for i, v in enumerate(lam) if v > 0]
    if not nz:
        return lam
    top = nz[-1]
    if top >= len(lam) - 1:
        return lam
    # fit log lam vs x over the last up-to-10 nonzero points
    pts = [(grid[i], math.log(lam[i])) for i in nz[-10:]]
    if len(pts) < 2:
        return lam
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 0:
        return lam
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    slope = clamp(slope, -3.0, -0.5)  # tail must fall, but a cliff fit on
    # a handful of crossings must not annihilate the high tail
    base_x, base_y = pts[-1]
    out = list(lam)
    for i in range(top + 1, len(lam)):
        out[i] = math.exp(base_y + slope * (grid[i] - base_x))
    return out


if __name__ == "__main__":
    main()
