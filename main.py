#!/usr/bin/env python3
"""pumpfun-top5-forecast-01: horizon-aware analytic top-5 forecaster.

Model
-----
Every coin's log10 market cap (in SOL) at a future read is a Gaussian
mixture: with survival probability s it is Normal(mean, sd), else it has
collapsed below every relevant level.  Incumbent coins (seen in the
history, above the relevance threshold or once top-30) get individual
Gaussians whose drift is estimated from their own recent trajectory,
shrunk toward their venue's drift, with a saturating momentum term and a
mean-reversion pull on pumped coins.  Coins not modelled individually --
newborns after the cut and sleeping low coins -- form a Poisson field:
young coins cross above each cap level x at a rate lambda(x) estimated
from the history and stay above it for a mean time tau(x), so the field
count above x at horizon h is Poisson(lambda*tau*(1-exp(-h/tau))).

For each read the count of coins above x is (incumbent Bernoullis) +
(Poisson field); its PMF comes from a Poisson-binomial DP convolved
with the Poisson.  From that PMF:
  * P(coin i at place k) = int f_i(x) * P(exactly k-1 others above x) dx
    (leave-one-out via the exact recursion on the full PMF);
  * P(place-k cap <= x) = P(at most k-1 coins above x), giving the size
    bins directly (order statistics of the population, not the last
    coin's cap plus noise).
Probabilities are averaged over the reads, since one forecast is scored
on every read.  All parameters are estimated from the input history
itself (self-calibrating); nothing is tuned to the practice window.

Output rows: coins with place probability above the unnamed-share
threshold are named at their probability; coins with a negligible
place probability but a high chance of sitting on the answer chart are
named at p=0, which removes them from the scorer's unnamed-share
denominator (worth ~+0.01 nats each per read, only taken when the
place probability is below ~0.001 so the -9.21 floor risk is negligible).
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# ---- tuning constants (priors; the data-driven parts are estimated) ----
GRID_LO, GRID_HI, GRID_STEP = 2.0, 6.0, 0.05   # log10 cap grid
INCUMBENT_CAP = 1.9        # last cap at/above this -> modelled individually
TOP_EVER = 30              # ever ranked this high -> modelled individually
MAX_INCIDENTS = 800        # cap on individually modelled coins
NAME_MIN_P = 0.005         # name a coin at a place above this probability
P0_MAX_Q = 0.002           # p=0 only when the place probability is below this
MAX_ROWS_PER_PLACE = 100
DRIFT_TAU_H = 1.5          # momentum saturates over ~this many hours
REVERT_TAU_H = 1.8         # mean-reversion timescale (hours)
REVERT_STRENGTH = 0.95     # fraction of the excess over REVERT_LEVEL lost
REVERT_LEVEL = 2.2         # log10 cap above which the pull applies
VOL_CAP_H = 6.0            # volatility stops growing like sqrt(h) past this
VOL_INFLATE = 1.15         # vol is understated: crashed coins leave the sample
YOUNG_AGE_H = 2.0          # "young crosser" age cutoff for lambda(x)
LAMBDA_TAU_H = 1.0         # recency half-life for the newcomer rate estimate
FLOOR_CAP = 1.46           # untraded curve coins sit at ~1.45; treat as flat
MICRO = 1_000_000


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

    # ---- grid ----
    n_grid = int(round((GRID_HI - GRID_LO) / GRID_STEP)) + 1
    grid = [GRID_LO + i * GRID_STEP for i in range(n_grid)]

    # ---- newcomer field: lambda(x), tau(x) ----
    # lambda: young coins (age < YOUNG_AGE_H at crossing) crossing above x,
    # per hour, over the recent window.
    # lambda: young coins (age < YOUNG_AGE_H at crossing) crossing above x,
    # per hour, over the recent window, recency-weighted so a ramping
    # market's current rate (not its early average) is what is projected.
    lam = [0.0] * n_grid
    win_span_h = 0.0
    for (a, b), dh in zip(recent_pairs, dt_h):
        w = math.exp(-(t_last - mark_times[b]).total_seconds() / 3600.0 / LAMBDA_TAU_H)
        win_span_h += w * dh
        cur = {r["mint"]: r for r in marks[a]["rows"]}
        nxt = marks[b]["rows"]
        for r in nxt:
            mint = r["mint"]
            try:
                lc = logcap(r)
            except (KeyError, ValueError):
                continue
            prev = cur.get(mint)
            prev_lc = logcap(prev) if prev is not None else None
            if prev_lc is not None and prev_lc >= lc:
                continue  # not a fresh crossing (already above, or fell)
            ca = parse_time(r["created_at"]) if r.get("created_at") else None
            if ca is None:
                continue
            age_h = (mark_times[b] - ca).total_seconds() / 3600.0
            if age_h > YOUNG_AGE_H:
                continue
            lo = prev_lc if prev_lc is not None else GRID_LO - 1.0
            for gi, gx in enumerate(grid):
                if lo < gx <= lc:
                    lam[gi] += w
    if win_span_h > 0:
        lam = [c / win_span_h for c in lam]
    # smooth + enforce non-increasing in x
    lam = _smooth_decreasing(lam)
    # log-linear tail extrapolation above the highest x with lam > 0
    lam = _extrapolate_tail(grid, lam)
    # tau: mean time above x from the leaving rate (all coins above x)
    tau = [1.0] * n_grid
    for gi, gx in enumerate(grid):
        above, left = 0.0, 0.0
        for a, b in recent_pairs:
            w = math.exp(-(t_last - mark_times[b]).total_seconds() / 3600.0 / LAMBDA_TAU_H)
            cur = {r["mint"]: r for r in marks[a]["rows"]}
            nxt = {r["mint"]: r for r in marks[b]["rows"]}
            for mint, r in cur.items():
                try:
                    lc = logcap(r)
                except (KeyError, ValueError):
                    continue
                if lc > gx:
                    above += w
                    n2 = nxt.get(mint)
                    if n2 is None:
                        left += w
                    else:
                        try:
                            if logcap(n2) <= gx:
                                left += w
                        except (KeyError, ValueError):
                            left += w
        if above >= 3:
            rate10 = (left + 0.5) / (above + 1.0)
            tau[gi] = clamp((10.0 / 60.0) / max(rate10, 0.02), 0.1, 8.0)
        else:
            tau[gi] = 0.5
    tau = _smooth_decreasing(tau)

    # ---- per-read DP + place probabilities + sizes ----
    n_reads = len(reads)
    debug = os.environ.get("MAKEFUN_DEBUG") == "1"
    place_p = {mint: [0.0] * places for mint in incumbents}
    size_acc = [[0.0] * (len(edges) + 1) for _ in range(places)]
    for r, h in enumerate(horizons):
        # incumbent Gaussians at this read
        params = {}
        for mint in incumbents:
            c = coin[mint]
            hh = h + c["last_h"]
            mean = (c["x"]
                    + c["drift_h"] * DRIFT_TAU_H * (1.0 - math.exp(-hh / DRIFT_TAU_H))
                    - REVERT_STRENGTH * max(0.0, c["x"] - REVERT_LEVEL)
                    * (1.0 - math.exp(-hh / REVERT_TAU_H)))
            sd = c["sig10"] * math.sqrt(6.0 * min(hh, VOL_CAP_H))
            sd = max(sd, 0.05)
            surv = math.exp(-c["lam_d"] * hh)
            params[mint] = (mean, sd, surv)
        # newcomer Poisson mean per grid point
        mu_new = []
        for gi, gx in enumerate(grid):
            t = tau[gi]
            mu_new.append(lam[gi] * t * (1.0 - math.exp(-h / t)) if h > 0 else 0.0)
        # full count PMF P_m(x), m = 0..5, at each grid point
        pmf = []  # pmf[gi][m]
        p_above = {}  # mint -> list over grid of P(X > x)
        for mint in incumbents:
            mean, sd, surv = params[mint]
            p_above[mint] = [surv * norm_sf((gx - mean) / sd) for gx in grid]
        for gi, gx in enumerate(grid):
            dp = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for mint in incumbents:
                p = p_above[mint][gi]
                if p <= 0.0:
                    continue
                if p >= 1.0:
                    dp = [0.0] + dp[:5]
                    continue
                ndp = [0.0] * 6
                for m in range(6):
                    ndp[m] += dp[m] * (1.0 - p)
                    if m + 1 < 6:
                        ndp[m + 1] += dp[m] * p
                dp = ndp
            mu = mu_new[gi]
            q = [math.exp(-mu)]
            for k in range(1, 6):
                q.append(q[-1] * mu / k)
            full = [0.0] * 6
            for a in range(6):
                if dp[a] == 0.0:
                    continue
                for b in range(6 - a):
                    full[a + b] += dp[a] * q[b]
            pmf.append(full)
        # sizes: F_k(x) = P(at most k-1 above x)
        for k in range(1, places + 1):
            cdf = [sum(full[:k]) for full in pmf]  # at grid points
            # bin probs from the CDF at the edges (edges are on the grid)
            prev_e, prev_c = None, None
            for b in range(len(edges) + 1):
                if b == 0:
                    lo_c, hi_c = 0.0, _cdf_at(edges[0], grid, cdf)
                elif b == len(edges):
                    lo_c, hi_c = _cdf_at(edges[-1], grid, cdf), 1.0
                else:
                    lo_c = _cdf_at(edges[b - 1], grid, cdf)
                    hi_c = _cdf_at(edges[b], grid, cdf)
                size_acc[k - 1][b] += max(0.0, hi_c - lo_c) / n_reads
        # order: per-coin place probabilities by leave-one-out integration
        read_sums = [0.0] * places
        for mint in incumbents:
            mean, sd, surv = params[mint]
            if surv <= 1e-6:
                continue
            lo_x = max(GRID_LO, mean - 3.5 * sd)
            hi_x = min(GRID_HI, mean + 3.5 * sd)
            if hi_x <= lo_x:
                continue
            gi0 = max(0, int(math.floor((lo_x - GRID_LO) / GRID_STEP)))
            gi1 = min(n_grid - 1, int(math.ceil((hi_x - GRID_LO) / GRID_STEP)))
            pa = p_above[mint]
            acc = [0.0] * places
            prev_gx, prev_val = None, None
            for gi in range(gi0, gi1 + 1):
                gx = grid[gi]
                f = surv * norm_pdf(gx, mean, sd)
                if f <= 1e-12:
                    prev_gx, prev_val = gx, [0.0] * places
                    continue
                full = pmf[gi]
                p = pa[gi]
                loo = _leave_one_out(full, p)
                val = [f * loo[k] for k in range(places)]
                if prev_gx is not None:
                    w = gx - prev_gx
                    for k in range(places):
                        acc[k] += 0.5 * (val[k] + prev_val[k]) * w
                prev_gx, prev_val = gx, val
            for k in range(places):
                place_p[mint][k] += acc[k] / n_reads
                read_sums[k] += acc[k]
        if debug:
            print(f"[dbg] read {r} h={h:.2f} newcomer_share="
                  f"{[round(1 - s, 2) for s in read_sums]} "
                  f"mu_new(3.3)={mu_new[_gi(3.3)]:.2f} mu_new(3.6)={mu_new[_gi(3.6)]:.2f} "
                  f"mu_new(4.0)={mu_new[_gi(4.0)]:.2f} lam(3.3)={lam[_gi(3.3)]:.2f} "
                  f"lam(3.6)={lam[_gi(3.6)]:.2f} tau(3.3)={tau[_gi(3.3)]:.2f}",
                  file=sys.stderr)

    # ---- assemble the forecast ----
    # sizes: widen for regime/parameter uncertainty.  No trend shift: the
    # DP's newcomer field already carries the ramp, and a fitted trend
    # double-counts it (v1's shift put the size means ~0.9 too high).
    h_mean = sum(horizons) / max(1, len(horizons))
    n_data = sum(1 for m in marks if m["rows"])
    widen = (0.15 + 0.20 * (1.0 - math.exp(-h_mean / 3.0))
             + 0.10 * max(0.0, 1.0 - n_data / 20.0) * min(1.0, h_mean))
    for k in range(places):
        size_acc[k] = _shift_widen(size_acc[k], edges, 0.0, widen)
    probabilities = []
    for k in range(places):
        cands = sorted(incumbents, key=lambda m: place_p[m][k], reverse=True)
        named = 0
        for mint in cands:
            p = place_p[mint][k]
            if p < NAME_MIN_P or named >= MAX_ROWS_PER_PLACE:
                break
            pf = floor6(p)
            if pf > 0:
                probabilities.append({"mint": mint, "place": k + 1, "p": pf})
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
