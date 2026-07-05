import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import pandas as pd
import math
import uuid
import io

st.set_page_config(page_title="Recomp Planner", page_icon="💪", layout="wide")
st.title("💪 Body Recomposition Planner")
st.caption("Dynamic model — personalized ceiling from frame, rates degrade as you approach it. "
           "Log real weigh-ins to calibrate the model to YOUR response instead of population averages.")

# ══════════════════════════════════════════════════════════════════════════════
# FRAME ENGINE (Casey Butt ceiling + frame-scaled baseline)
# ══════════════════════════════════════════════════════════════════════════════

# Default fraction of the Butt ceiling that an untrained person of the same frame
# sits at. Calibrated so an average frame lands ~FFMI 17 untrained. Because it is
# a fraction of the (frame-derived) ceiling, the baseline scales with frame too.
DEFAULT_BASELINE_K = 0.651

# How hard we pull a user's remembered starting stats toward the frame default.
# 0 = trust their memory fully, 1 = ignore it. Heavy default because starting BF
# from years ago is the least reliable input in the whole model.
STARTING_STATS_SMOOTHING = 0.6

# ── Sex adjustments (distribution-level, honest caveat: the Casey Butt formula
# was derived from male drug-free champions; these ratios adapt its OUTPUT to
# female population data rather than re-deriving the formula) ──────────────────
FEMALE_CEILING_MULT   = 0.86   # female genetic lean ceiling ≈ 86% of male, same frame
FEMALE_BASELINE_RATIO = 0.81   # female untrained lean ≈ 81% of male untrained lean
FEMALE_RATE_MULT      = 0.50   # absolute lean gain rate ≈ half of male

# ── Muscle memory: bounded rebuild bonus when current lean sits below a prior
# peak. Max +50% to the lean gain rate at maximum deficit, decaying linearly to
# zero AT the prior peak. Past the peak, normal first-time rates apply. ────────
MM_MAX_BONUS = 0.50

def casey_butt_ceiling_lean(height_in, wrist_in, ankle_in, bf=10):
    """Max drug-free lean body mass. Verified vs Butt's worked example
    (69in, 7.0, 8.7, 10%bf -> 173.7 lbs)."""
    return (height_in ** 1.5) * (
        math.sqrt(wrist_in) / 22.6670 + math.sqrt(ankle_in) / 17.0104
    ) * (1 + bf / 224.0)

def _height_m(height_in):
    return height_in * 0.0254

def lean_to_norm_ffmi(lean_lbs, height_in):
    hm = _height_m(height_in)
    return (lean_lbs * 0.453592) / (hm ** 2) + 6.1 * (1.8 - hm)

def build_frame(height_in, wrist_in, ankle_in, age, sex="Male",
                start_lean=None, start_lean_weight=None):
    """Compute the personalized ceiling + baseline once.

    ceiling_lean : Casey Butt max lean (age- and sex-adjusted)
    baseline_lean: untrained floor, frame-scaled. Default = K * ceiling.
                   If the user supplies real starting stats, blend their implied
                   untrained lean toward that default (heavy smoothing + clamp).
    rate_mult    : sex multiplier on the lean gain rate curve.
    """
    ceiling = casey_butt_ceiling_lean(height_in, wrist_in, ankle_in, bf=10)
    ceiling *= 1 - max(0, (age - 40) * 0.005)   # mild age tax on the ceiling

    baseline_k = DEFAULT_BASELINE_K
    rate_mult  = 1.0
    if sex == "Female":
        ceiling *= FEMALE_CEILING_MULT
        # Female baseline should land at 0.81x the male untrained lean. Since the
        # ceiling was already scaled by 0.86, the K applied to the FEMALE ceiling
        # is 0.651 * (0.81 / 0.86).
        baseline_k = DEFAULT_BASELINE_K * (FEMALE_BASELINE_RATIO / FEMALE_CEILING_MULT)
        rate_mult  = FEMALE_RATE_MULT

    default_baseline = baseline_k * ceiling

    baseline = default_baseline
    if start_lean_weight is not None and start_lean is not None and start_lean > 0:
        # user's remembered untrained lean, blended toward the frame default
        blended = (STARTING_STATS_SMOOTHING * default_baseline
                   + (1 - STARTING_STATS_SMOOTHING) * start_lean)
        # clamp so an absurd memory can't escape a sane band around the default
        lo = 0.55 * ceiling
        hi = 0.72 * ceiling
        baseline = max(lo, min(hi, blended))

    return {
        "height_in": height_in,
        "ceiling_lean": ceiling,
        "baseline_lean": baseline,
        "ffmi_ceiling": lean_to_norm_ffmi(ceiling, height_in),
        "rate_mult": rate_mult,
    }

# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC PROFILE (rates as a function of proximity to the personalized ceiling)
# ══════════════════════════════════════════════════════════════════════════════

# Decay curve calibrated to the empirical Lyle/Butt schedule
# (year1 +20, year2 +10, year3 +5, year4 +2.5 lbs lean from untrained).
CURVE_MAXRATE_YR = 20.2   # lbs/yr at zero proximity
CURVE_K          = 1.8    # decay exponent

# ── Recomp (slow-cut lean gain) parameters, calibrated to Garthe 2011 ─────────
# Garthe's slow group lost 0.7%/wk of bodyweight and GAINED 2.1% LBM over 8.5wk
# (~+0.33 lbs lean/wk for a ~159 lb athlete); the fast group at 1.0%/wk was flat.
# Both trained 4x/wk. The original model only allowed lean gain below 0.55%/wk
# and capped it at -10% of the cut rate — which (a) showed LOSS at the exact
# 0.7%/wk where Garthe measured GAIN and (b) under-predicted the magnitude ~3x.
# These three constants are tuned so a low-proximity lifter reproduces Garthe's
# ~0.33 lbs/wk at 0.7%/wk, while the (1-prox)^2 gate keeps advanced lifters near
# zero recomp (consistent with Helms: highly trained lifters recomp poorly).
RECOMP_MID   = 0.75   # %/wk sigmoid midpoint of the recomp->loss transition
RECOMP_WIDTH = 0.12   # transition sharpness (smaller = sharper, but always C-inf)
RECOMP_MAG   = 0.605  # peak recomp fraction, tuned to Garthe ~+0.33 lbs/wk @0.6-0.7%/wk
RECOMP_FLOOR = -0.35  # max lean-GAIN fraction of the cut rate (was -0.10)

def muscle_frac_cut_model(cut_pct_wk, current_bf, prox, allow_recomp=True):
    """Share of each lost pound that is lean, as one shared function so the
    profile display and the weekly step can never drift apart. A NEGATIVE value
    means net lean GAIN while losing fat (recomp) — only possible when
    allow_recomp is True.

    Anchors: Helms 2014 (0.5-1%/wk to retain LBM; loss fraction rises with the
    deficit and with leanness) and Garthe 2011 (slow cut ~0.7%/wk -> +2.1% LBM
    in trained-but-not-advanced athletes). The recomp term fades through a
    logistic centered at RECOMP_MID, so there's no kink where it hands off to
    lean loss. The (1-prox)^2 factor reduces recomp for advanced lifters (Helms:
    highly trained lifters recomp poorly) — but that gate is a modeling guess,
    so recomp is OPT-IN: with it off, the best case on a slow cut is ~zero lean
    loss, never lean gain. Note the fast end stays conservative either way: at
    1%/wk this predicts ~20% lean loss for a typical trainee — Garthe's elite
    fast group held LBM, but that reflects optimized training/protein this
    general model doesn't assume. Validated in tests.
    """
    base             = 0.05 + (cut_pct_wk ** 1.5) * 0.16
    leanness_penalty = 0.012 * 0.5 * math.log1p(math.exp(2.0 * (15 - current_bf)))
    if allow_recomp:
        recomp_fade = 1.0 / (1.0 + math.exp((cut_pct_wk - RECOMP_MID) / RECOMP_WIDTH))
        recomp      = -((1 - prox) ** 2) * RECOMP_MAG * recomp_fade
        floor       = RECOMP_FLOOR
    else:
        recomp = 0.0
        floor  = 0.0   # best case with recomp off: hold lean, never gain it
    return max(floor, min(0.55, base + leanness_penalty + recomp - prox * 0.03))

def dynamic_profile(current_lean_lbs, frame, peak_lean_lbs=None, current_bf=15.0):
    ceiling  = frame["ceiling_lean"]
    baseline = frame["baseline_lean"]
    height_in = frame["height_in"]

    # Peak lean (muscle memory) governs PROXIMITY — how close you've ever been
    # to your ceiling. But actual bodyweight-derived numbers (cut rate) use
    # CURRENT lean, so a detrained user isn't assigned rates for a body they
    # don't currently have.
    lean_for_prox = max(current_lean_lbs, peak_lean_lbs) if peak_lean_lbs else current_lean_lbs

    rng = max(1e-6, ceiling - baseline)
    prox = min(1.0, max(0.0, (lean_for_prox - baseline) / rng))

    # Muscle fraction first — needed to convert lean gain to scale weight
    muscle_frac_bulk = round(0.30 + (1 - prox) ** 1.2 * 0.32, 3)
    muscle_frac_bulk = max(0.30, min(0.60, muscle_frac_bulk))

    # Curve gives LEAN gain per year (calibrated to the 20/10/5/2 lean schedule),
    # scaled by sex. Because prox uses PEAK lean, a detrained lifter gets the
    # slower near-peak rate — the muscle-memory bonus below then reflects that
    # rebuilding lost tissue is empirically faster than first-time gain.
    lean_rate_yr = CURVE_MAXRATE_YR * frame.get("rate_mult", 1.0) * (1 - prox) ** CURVE_K
    if peak_lean_lbs and peak_lean_lbs > current_lean_lbs:
        deficit = (peak_lean_lbs - current_lean_lbs) / max(1e-6, peak_lean_lbs - baseline)
        lean_rate_yr *= 1.0 + MM_MAX_BONUS * min(1.0, max(0.0, deficit))

    # Scale-weight bulk rate = lean rate / muscle fraction, since fat rides along.
    bulk_rate_yr      = lean_rate_yr / muscle_frac_bulk
    bulk_rate_weekly  = round(bulk_rate_yr / 52.0, 3)
    bulk_rate_monthly = round(bulk_rate_yr / 12.0, 2)

    # ── Cut rate scales smoothly with CURRENT body fat (leaner = cut slower) ────
    # Continuous exponential, no threshold kinks. Anchored to the muscle-
    # PRESERVATION end of Helms/Garthe: ~0.5%/wk at 15% BF, ~0.8%/wk at 20%,
    # tapering to ~0.3%/wk when very lean, capped at 1.0%/wk at high BF.
    cut_pct_wk = 0.1221 * math.exp(0.0940 * current_bf)
    cut_pct_wk = max(0.30, min(1.00, cut_pct_wk))

    approx_weight    = current_lean_lbs / (1 - current_bf / 100)
    cut_rate_weekly  = round(approx_weight * cut_pct_wk / 100, 3)
    cut_rate_monthly = round(cut_rate_weekly * 4.33, 2)

    muscle_frac_cut = round(muscle_frac_cut_model(cut_pct_wk, current_bf, prox,
                        frame.get("allow_recomp", True)), 3)

    return {
        "ffmi": round(lean_to_norm_ffmi(lean_for_prox, height_in), 2),
        "ffmi_ceiling": round(frame["ffmi_ceiling"], 2),
        "ceiling_pct": round(prox * 100, 1),
        "bulk_rate_weekly": bulk_rate_weekly,
        "bulk_rate_monthly": bulk_rate_monthly,
        "cut_rate_weekly": cut_rate_weekly,
        "cut_rate_monthly": cut_rate_monthly,
        "muscle_frac_bulk": muscle_frac_bulk,
        "muscle_frac_cut": muscle_frac_cut,
    }

# ══════════════════════════════════════════════════════════════════════════════
# WEEKLY STEP (shared by scheduler + simulation)
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_MODS = {"bulk_mult": 1.0, "cut_mult": 1.0}

def step_week(w, lean, fat, peak_lean, action, frame, overrides, mods=None):
    """Advance one week. Returns the new state plus the rate and muscle
    fraction ACTUALLY APPLIED for this week's action (0, 0 for maintain),
    so downstream tables never show a hypothetical bulk rate during a cut.

    mods: multipliers on the DYNAMIC rates (consistency + personal calibration).
    Explicit rate overrides are taken literally and NOT scaled — if the user
    forces a rate, we assume that's the rate they actually achieve.
    """
    if mods is None:
        mods = DEFAULT_MODS
    current_bf = fat / w * 100
    dp = dynamic_profile(lean, frame, peak_lean_lbs=peak_lean, current_bf=current_bf)

    bulk_rate = (overrides["bulk_rate"] if overrides["bulk_rate"] > 0
                 else dp["bulk_rate_weekly"] * mods["bulk_mult"])
    cut_rate  = (overrides["cut_rate"] if overrides["cut_rate"] > 0
                 else dp["cut_rate_weekly"] * mods["cut_mult"])

    if overrides["bulk_muscle"] > 0:
        mfrac_bulk = overrides["bulk_muscle"] / 100
    else:
        mfrac_bulk = dp["muscle_frac_bulk"]

    if overrides["cut_muscle"] > 0:
        mfrac_cut = overrides["cut_muscle"] / 100
    else:
        # recompute from the actual cut rate AND current leanness. Convert the
        # cut rate in use back to % of bodyweight/wk so the lean-loss curve is
        # on the same scale whether the rate is dynamic or overridden.
        cut_pct_wk_eff = cut_rate / w * 100
        mfrac_cut = muscle_frac_cut_model(cut_pct_wk_eff, current_bf,
                                          dp["ceiling_pct"] / 100,
                                          frame.get("allow_recomp", True))

    applied_rate = 0.0
    applied_frac = 0.0

    if action == "bulk":
        lean += bulk_rate * mfrac_bulk
        fat  += bulk_rate * (1 - mfrac_bulk)
        w    += bulk_rate
        applied_rate = bulk_rate
        applied_frac = mfrac_bulk
    elif action == "cut":
        # Essential-fat floor: a natural lifter cannot cut below ~5% BF. When
        # the floor binds, scale the ENTIRE week's loss (fat AND lean) down
        # proportionally — otherwise the model would keep stripping lean mass
        # at full rate while fat holds, simulating impossible pure-muscle loss.
        # Note mfrac_cut may be slightly NEGATIVE (slow-cut recomp): the week
        # then loses a bit more fat than the cut rate and gains a sliver of lean.
        ESSENTIAL_BF = 5.0
        min_fat   = lean * (ESSENTIAL_BF / 100) / (1 - ESSENTIAL_BF / 100)
        fat_loss  = cut_rate * (1 - mfrac_cut)
        lean_loss = cut_rate * mfrac_cut
        if fat - fat_loss < min_fat and fat_loss > 0:
            scale = max(0.0, fat - min_fat) / fat_loss
            fat_loss  *= scale
            lean_loss *= scale
        lean -= lean_loss
        fat  -= fat_loss
        w     = lean + fat
        applied_rate = fat_loss + lean_loss
        applied_frac = mfrac_cut
    else:  # maintain — true flat hold: composition unchanged for the duration.
        # Used in manual mode to model a known off-plan stretch (vacation, deload,
        # work crunch). The auto-scheduler never generates maintain phases, since
        # holding still is never optimal for a minimum-time-to-goal objective.
        pass

    peak_lean = max(peak_lean, lean)
    return w, lean, fat, peak_lean, applied_rate, applied_frac

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION (single source of truth)
# ══════════════════════════════════════════════════════════════════════════════
def simulate_dynamic(start_weight, start_bf, phases, frame, start_date, overrides,
                     mods=None, prior_peak=None):
    w    = start_weight
    lean = start_weight * (1 - start_bf / 100)
    fat  = start_weight * (start_bf / 100)
    peak_lean = max(lean, prior_peak) if prior_peak else lean

    rows = [{
        "week": 0, "date": start_date, "phase": "Start", "phase_type": "start",
        "weight": round(w,1), "lean": round(lean,1), "fat": round(fat,1),
        "bf": round(fat/w*100,1), "change": 0.0, "rate": 0.0, "muscle_frac": 0.0,
    }]

    idx = 1
    current_date = start_date
    for phase in phases:
        ptype   = phase["type"]
        n_weeks = phase.get("weeks", 4)
        for _ in range(n_weeks):
            prev_w = w
            w, lean, fat, peak_lean, rate, mfrac = step_week(
                w, lean, fat, peak_lean, ptype, frame, overrides, mods)
            current_date += timedelta(weeks=1)
            rows.append({
                "week": idx, "date": current_date,
                "phase": phase["name"], "phase_type": ptype,
                "weight": round(w,1), "lean": round(lean,1), "fat": round(fat,1),
                "bf": round(fat/w*100,1), "change": round(w-prev_w,3),
                "rate": round(rate,3), "muscle_frac": round(mfrac,3),
            })
            idx += 1
    return rows

# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO UNCERTAINTY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
# Two orthogonal questions from one ensemble of trajectories:
#   (1) WHEN do I reach goal?   -> spread of goal-crossing weeks  (horizontal CI)
#   (2) WHERE am I at week E?    -> spread of weight/BF at the headline week E
#                                   (vertical CI)
# We fix the plan the user would actually follow (the expected-optimal schedule)
# and perturb the two things we genuinely don't know: the starting BF (a noisy
# measurement) and the individual RATE response (population rates ± personal
# variation). Re-running the full scheduler per sample is ~150x slower and buys
# little — the decision is fixed; what varies is the body's response to it. Once
# a trajectory reaches goal it switches to maintenance (you'd stop cutting), so
# fast responders plateau at goal instead of overshooting.

def _extend_plan_for_mc(phases, max_weeks, max_phase_weeks):
    """Lengthen the expected plan out to max_weeks so slower-than-expected
    responders still have room to reach goal. We extend with a short ALTERNATING
    bulk/cut pattern rather than repeating the tail: a slow responder is short on
    some mix of weight and leanness, and alternation lets them converge from
    whichever side they're on instead of being pushed past goal in one direction.
    Goal-hold (in _simulate_to_goal) freezes each trajectory the moment it
    arrives, so the alternation only acts on samples that haven't gotten there."""
    ext = [dict(p) for p in phases]
    if not ext:
        return ext
    total = sum(p["weeks"] for p in ext)
    nxt = "cut" if ext[-1]["type"] == "bulk" else "bulk"
    chunk = min(4, max_phase_weeks)
    while total < max_weeks:
        add = min(chunk, max_weeks - total)
        ext.append({"name": f"{nxt.capitalize()} (cont.)", "type": nxt, "weeks": add})
        total += add
        nxt = "cut" if nxt == "bulk" else "bulk"
    return ext

def _simulate_to_goal(start_weight, start_bf, phases, frame, overrides, mods,
                      prior_peak, goal_weight, goal_bf,
                      tol_w=0.6, tol_bf=0.4):
    """Simulate the (extended) plan, switching to maintenance once goal is hit.
    Returns (trajectory, crossing_week or None). Trajectory rows are
    (week, weight, bf, lean)."""
    w = start_weight
    lean = start_weight * (1 - start_bf / 100)
    fat = start_weight * (start_bf / 100)
    peak = max(lean, prior_peak) if prior_peak else lean
    traj = [(0, w, fat / w * 100, lean)]
    crossed = None
    idx = 1
    for phase in phases:
        pt = phase["type"]
        for _ in range(phase.get("weeks", 4)):
            action = "maintain" if crossed is not None else pt
            w, lean, fat, peak, _, _ = step_week(w, lean, fat, peak, action,
                                                 frame, overrides, mods)
            bf = fat / w * 100
            traj.append((idx, w, bf, lean))
            if crossed is None and abs(w - goal_weight) <= tol_w and bf <= goal_bf + tol_bf:
                crossed = idx
            idx += 1
    return traj, crossed

def _percentile(sorted_vals, q):
    """Linear-interpolated percentile (q in 0..1) on a pre-sorted list."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

def timeline_ci_full(start_weight, start_bf, goal_weight, goal_bf, bf_ceiling,
                     bf_floor, frame, overrides, mods, prior_peak, max_weeks,
                     max_phase_weeks, n_samples=24, bf_sd=1.0,
                     bulk_cv=0.12, cut_cv=0.12, seed=7):
    """Timeline confidence interval done the honest way: each sample RE-OPTIMIZES
    the whole schedule for its perturbed start-BF and rate response, so every
    sample reaches goal on its own optimal path (no tolerance-box artifacts from
    a fixed plan). Fewer samples than the envelope MC because each full schedule
    is ~150x costlier, but 24 is plenty to characterize a p10-p90 week range.
    Returns p10/p50/p90 weeks-to-goal and the fraction reaching within max_weeks."""
    import random
    rng = random.Random(seed)

    def _clamp(x, lo, hi): return max(lo, min(hi, x))
    weeks = []
    reached_ct = 0
    for _ in range(n_samples):
        sbf = _clamp(rng.gauss(start_bf, bf_sd), 5.0, 40.0)
        bmul = mods["bulk_mult"] * _clamp(rng.gauss(1.0, bulk_cv), 0.45, 1.7)
        cmul = mods["cut_mult"]  * _clamp(rng.gauss(1.0, cut_cv), 0.45, 1.7)
        m = {"bulk_mult": bmul, "cut_mult": cmul}
        ph = auto_schedule_dynamic(start_weight, sbf, goal_weight, goal_bf,
                                   bf_ceiling, bf_floor, frame, overrides,
                                   max_weeks=max_weeks, max_phase_weeks=max_phase_weeks,
                                   mods=m, prior_peak=prior_peak)
        rows = simulate_dynamic(start_weight, sbf, ph, frame, date(2000, 1, 3),
                                overrides, mods=m, prior_peak=prior_peak)
        f = rows[-1]
        if abs(f["weight"] - goal_weight) <= 2 and abs(f["bf"] - goal_bf) <= 1.5:
            weeks.append(f["week"]); reached_ct += 1
    if not weeks:
        return None
    weeks.sort()
    return {
        "p10": _percentile(weeks, 0.10),
        "p50": _percentile(weeks, 0.50),
        "p90": _percentile(weeks, 0.90),
        "frac_reached": reached_ct / n_samples,
        "n": n_samples,
    }

def monte_carlo_bands(start_weight, start_bf, expected_phases, frame, overrides,
                      mods, prior_peak, goal_weight, goal_bf, max_weeks,
                      max_phase_weeks, headline_week,
                      n_samples=200, bf_sd=1.0, bulk_cv=0.12, cut_cv=0.12, seed=42):
    """Run n_samples perturbed simulations of the fixed plan. Returns a dict with
    per-week p10/p50/p90 envelopes for weight/BF/lean, the goal-crossing-week
    distribution (timeline CI), and the weight/BF distribution AT headline_week
    (outcome CI). Start WEIGHT is held fixed (scale is precise); start BF and the
    bulk/cut rate multipliers are the perturbed unknowns."""
    import random
    rng = random.Random(seed)
    ext = _extend_plan_for_mc(expected_phases, max_weeks, max_phase_weeks)
    horizon = sum(p["weeks"] for p in ext)

    # week -> lists of sampled values across trajectories (forward-filled at goal
    # because _simulate_to_goal holds composition once goal is reached)
    wk_weight = [[] for _ in range(horizon + 1)]
    wk_bf     = [[] for _ in range(horizon + 1)]
    wk_lean   = [[] for _ in range(horizon + 1)]
    crossings = []

    def _clamp(x, lo, hi): return max(lo, min(hi, x))

    for _ in range(n_samples):
        sbf = _clamp(rng.gauss(start_bf, bf_sd), 5.0, 40.0)
        bmul = mods["bulk_mult"] * _clamp(rng.gauss(1.0, bulk_cv), 0.45, 1.7)
        cmul = mods["cut_mult"]  * _clamp(rng.gauss(1.0, cut_cv), 0.45, 1.7)
        m = {"bulk_mult": bmul, "cut_mult": cmul}
        traj, crossed = _simulate_to_goal(start_weight, sbf, ext, frame, overrides,
                                          m, prior_peak, goal_weight, goal_bf)
        crossings.append(crossed if crossed is not None else None)
        for (wk, w, bf, lean) in traj:
            wk_weight[wk].append(w)
            wk_bf[wk].append(bf)
            wk_lean[wk].append(lean)

    def _band(series):
        p10, p50, p90 = [], [], []
        for vals in series:
            if not vals:
                p10.append(None); p50.append(None); p90.append(None); continue
            s = sorted(vals)
            p10.append(_percentile(s, 0.10))
            p50.append(_percentile(s, 0.50))
            p90.append(_percentile(s, 0.90))
        return p10, p50, p90

    w10, w50, w90 = _band(wk_weight)
    b10, b50, b90 = _band(wk_bf)
    l10, l50, l90 = _band(wk_lean)

    reached = sorted(c for c in crossings if c is not None)
    frac_reached = len(reached) / max(1, len(crossings))
    timeline = None
    if reached:
        timeline = {
            "p10": _percentile(reached, 0.10),
            "p50": _percentile(reached, 0.50),
            "p90": _percentile(reached, 0.90),
            "frac_reached": frac_reached,
        }

    he = min(max(headline_week, 0), horizon)
    outcome = {
        "week": he,
        "weight_p10": w10[he], "weight_p50": w50[he], "weight_p90": w90[he],
        "bf_p10": b10[he], "bf_p50": b50[he], "bf_p90": b90[he],
    }

    return {
        "weeks": list(range(horizon + 1)),
        "weight": (w10, w50, w90), "bf": (b10, b50, b90), "lean": (l10, l50, l90),
        "timeline": timeline, "outcome": outcome,
        "n_samples": n_samples, "bf_sd": bf_sd, "bulk_cv": bulk_cv, "cut_cv": cut_cv,
    }

# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC LOOK-AHEAD SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════
def _roll_phase(state, action, weeks, frame, overrides, mods=None):
    w, lean, fat, peak = state
    for _ in range(weeks):
        w, lean, fat, peak, *_ = step_week(w, lean, fat, peak, action, frame,
                                           overrides, mods)
    return (w, lean, fat, peak)

def _goal_distance(state, goal_weight, goal_bf):
    w, lean, fat, peak = state
    bf = fat / w * 100
    d = abs(w - goal_weight) + abs(bf - goal_bf)
    # Heavy penalty only for genuinely unhealthy body fat (hard physiological
    # floor), NOT the user's soft preference floor — otherwise, when the goal BF
    # equals the cut floor, the scheduler refuses to cut to goal and stops short.
    HARD_FLOOR = 8.0
    if bf < HARD_FLOOR:
        d += (HARD_FLOOR - bf) * 100
    return d

def auto_schedule_dynamic(start_weight, start_bf, goal_weight, goal_bf,
                          bf_ceiling, bf_floor, frame, overrides,
                          max_weeks=156, max_phase_weeks=20, min_phase_weeks=4,
                          mods=None, prior_peak=None):
    w    = start_weight
    lean = start_weight * (1 - start_bf / 100)
    fat  = start_weight * (start_bf / 100)
    peak = max(lean, prior_peak) if prior_peak else lean

    phases = []
    week_idx = 0
    phase_num = 1
    bf_now = fat / w * 100
    action = "cut" if bf_now >= bf_ceiling else "bulk"
    GOAL_TOL = 0.3

    best_phases   = []
    best_distance = abs(w - goal_weight) + abs(bf_now - goal_bf)
    just_flipped  = False   # guards the action-flip fallback against ping-ponging

    while week_idx < max_weeks:
        bf_now = fat / w * 100
        if abs(w - goal_weight) <= GOAL_TOL and abs(bf_now - goal_bf) <= GOAL_TOL:
            break

        state = (w, lean, fat, peak)
        weeks_left = max_weeks - week_idx
        hi = min(max_phase_weeks, weeks_left)
        near_goal = abs(w - goal_weight) <= 3 and abs(fat/w*100 - goal_bf) <= 2
        lo = 1 if near_goal else min_phase_weeks
        if hi < lo:
            break

        def _search(lo_try):
            """Search phase lengths [lo_try, hi] for the current action.
            Returns (score, length) or None if every length violates a band."""
            best_local = None
            for this_len in range(lo_try, hi + 1):
                s1 = _roll_phase(state, action, this_len, frame, overrides, mods)
                w1, lean1, fat1, peak1 = s1
                bf1 = fat1 / w1 * 100
                if action == "bulk" and bf1 > bf_ceiling + 0.5:
                    break
                if action == "cut" and bf1 < bf_floor - 0.5:
                    break

                next_action = "cut" if action == "bulk" else "bulk"
                weeks_left2 = weeks_left - this_len
                hi2 = min(max_phase_weeks, weeks_left2)
                # Score "stop after this phase" (1-deep) as a valid option, so a
                # finishing phase that lands on goal isn't penalized by a forced
                # follow-up phase it doesn't actually need.
                score_1deep = _goal_distance(s1, goal_weight, goal_bf)
                if hi2 >= min_phase_weeks:
                    best2 = None
                    for next_len in range(min_phase_weeks, hi2 + 1):
                        s2 = _roll_phase(s1, next_action, next_len, frame,
                                         overrides, mods)
                        d = _goal_distance(s2, goal_weight, goal_bf)
                        if best2 is None or d < best2:
                            best2 = d
                    score = min(score_1deep, best2) if best2 is not None else score_1deep
                else:
                    score = score_1deep

                if best_local is None or score < best_local[0]:
                    best_local = (score, this_len)
            return best_local

        best = _search(lo)
        # Fallback 1: if even the minimum phase length violates a BF band
        # (e.g. starting just under the ceiling, where a 4-week bulk overshoots),
        # allow shorter phases down to 1 week.
        if best is None and lo > 1:
            best = _search(1)
        # Fallback 2: if NO length of the current action fits inside the bands,
        # flip the action once (e.g. open with a short corrective cut instead of
        # a bulk). Guarded so two impossible actions terminate instead of looping.
        if best is None:
            if not just_flipped:
                action = "cut" if action == "bulk" else "bulk"
                just_flipped = True
                continue
            break
        just_flipped = False

        chosen_len = best[1]
        new_state = _roll_phase(state, action, chosen_len, frame, overrides, mods)
        w, lean, fat, peak = new_state
        phases.append({
            "name": f"{'Bulk' if action=='bulk' else 'Cut'} {phase_num}",
            "type": action, "weeks": chosen_len,
        })
        if action == "cut":
            phase_num += 1
        action = "cut" if action == "bulk" else "bulk"
        week_idx += chosen_len

        dist_now = abs(w - goal_weight) + abs(fat/w*100 - goal_bf)
        if dist_now < best_distance:
            best_distance = dist_now
            best_phases   = [dict(p) for p in phases]

        if abs(w - goal_weight) <= GOAL_TOL and abs(fat/w*100 - goal_bf) <= GOAL_TOL:
            break

    return best_phases if best_phases else phases

# ══════════════════════════════════════════════════════════════════════════════
# PERSONAL CALIBRATION (actuals vs baseline projection)
# ══════════════════════════════════════════════════════════════════════════════
CALIB_MIN_SAMPLES = 4     # matched weekly deltas per phase type before we trust it
CALIB_CLAMP = (0.4, 2.0)  # sanity bounds on the personal multiplier

def calibrate_from_actuals(actuals, baseline_rows):
    """Estimate personal rate multipliers AND their standard errors from logged
    weigh-ins.

    Method: within each contiguous run of weigh-ins that falls in a single
    phase type, regress OBSERVED weight on PROJECTED weight (with intercept).
    The slope of that line is the personal multiplier for that phase type, and
    the intercept absorbs the anchor offset. Runs are then pooled by
    inverse-variance weighting.

    Why levels and not week-to-week deltas: consecutive deltas of noisy
    weigh-ins have NEGATIVELY CORRELATED errors (the same scale/water error
    enters one delta with + and the next with -), which both destabilizes a
    delta-based fit and makes its textbook SE wildly overstated (verified in
    tests: claimed SE ~3x the true scatter). Level noise is independent
    across days, so OLS-on-levels gives an honest SE — one that shrinks as
    you log more data, which is what lets the app narrow the Monte Carlo
    bands with evidence instead of a guessed CV.

    Returns {bulk_mult, cut_mult, bulk_n, cut_n, bulk_se, cut_se}. Multipliers
    stay 1.0 (se None) until a phase type has a usable run (>= CALIB_MIN_SAMPLES
    points spanning a real projected change).
    """
    out = {"bulk_mult": 1.0, "cut_mult": 1.0, "bulk_n": 0, "cut_n": 0,
           "bulk_se": None, "cut_se": None}
    if len(actuals) < 2 or not baseline_rows:
        return out

    # 1. Match each weigh-in to its nearest projection row (within 10 days)
    matched = []   # (date, obs_weight, phase_type, proj_weight)
    for (d, w) in actuals:
        nearest = min(baseline_rows, key=lambda r: abs((r["date"] - d).days))
        if abs((nearest["date"] - d).days) > 10:
            continue
        if nearest["phase_type"] in ("bulk", "cut"):
            matched.append((d, w, nearest["phase_type"], nearest["weight"]))

    # 2. Split into contiguous same-phase runs (break on phase change or >21d gap)
    runs = []
    cur = []
    for pt_row in matched:
        if cur and (pt_row[2] != cur[-1][2] or (pt_row[0] - cur[-1][0]).days > 21):
            runs.append(cur); cur = []
        cur.append(pt_row)
    if cur:
        runs.append(cur)

    # 3. OLS with intercept per run: obs_w = a + m * proj_w
    fits = {"bulk": [], "cut": []}   # (m, se, n_points)
    for run in runs:
        n = len(run)
        if n < CALIB_MIN_SAMPLES:
            continue
        xs = [r[3] for r in run]; ys = [r[1] for r in run]
        xbar = sum(xs) / n; ybar = sum(ys) / n
        sxx = sum((x - xbar) ** 2 for x in xs)
        if sxx < 0.25:      # projection barely moved in this run; slope is undefined
            continue
        sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
        m = sxy / sxx
        rss = sum((y - ybar - m * (x - xbar)) ** 2 for x, y in zip(xs, ys))
        dof = n - 2
        if dof < 1:
            continue
        se = math.sqrt(rss / dof / sxx)
        if se < 1e-6:
            se = 1e-6   # perfectly collinear synthetic data; avoid div-by-zero
        fits[run[0][2]].append((m, se, n))

    # 4. Pool runs per phase type by inverse variance
    for pt in ("bulk", "cut"):
        if not fits[pt]:
            continue
        wsum = sum(1.0 / se ** 2 for _, se, _ in fits[pt])
        m = sum(mm / se ** 2 for mm, se, _ in fits[pt]) / wsum
        se = math.sqrt(1.0 / wsum)
        out[f"{pt}_n"] = sum(n for _, _, n in fits[pt])
        if m > 0:
            out[f"{pt}_mult"] = max(CALIB_CLAMP[0], min(CALIB_CLAMP[1], m))
            out[f"{pt}_se"] = se
    return out

def _parse_actuals_df(df):
    """DataFrame (date, weight) -> clean sorted list of (date, float) tuples."""
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        d, w = row.get("date"), row.get("weight")
        if pd.isna(d) or pd.isna(w):
            continue
        d = pd.to_datetime(d).date()
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        if 50 <= w <= 500:
            out.append((d, w))
    out.sort(key=lambda t: t[0])
    return out

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📋 Starting Stats")
    start_weight = st.number_input("Start Weight (lbs)", 90.0, 300.0, 145.0, 0.5)
    start_bf     = st.number_input("Start Body Fat %",   5.0, 40.0, 15.5, 0.5)

    st.divider()
    st.header("👤 Profile")
    sex           = st.radio("Sex", ["Male", "Female"], horizontal=True,
                             help="Adjusts muscle-potential and gain-rate "
                                  "estimates. On average, women build roughly "
                                  "half the muscle per year of men at the same "
                                  "training level, with a lower total ceiling "
                                  "for the same frame.")
    height_ft     = st.number_input("Height (ft)", 4, 7, 5)
    height_in_rem = st.number_input("Height (in)", 0, 11, 8)
    wrist_in      = st.number_input("Wrist circumference (in)", 4.0, 10.0, 6.25, 0.25)
    ankle_in      = st.number_input("Ankle circumference (in)", 6.0, 14.0, 8.0, 0.25)
    age           = st.number_input("Age", 16, 80, 22)

    # ── optional: history (collapsed) ──────────────────────────────────────────
    with st.expander("📦 Training history (optional)", expanded=False):
        st.caption("Roughly what you weighed and your BF% when you FIRST started "
                   "lifting. Used to personalize your untrained baseline. Heavily "
                   "smoothed toward the frame default, since old-memory BF is noisy. "
                   "Leave at 0 to use the frame default.")
        first_weight = st.number_input("Starting weight when you began (lbs)", 0.0, 300.0, 0.0, 1.0)
        first_bf     = st.number_input("Starting BF% when you began", 0.0, 40.0, 0.0, 0.5)
        st.caption("If you were once MORE muscular than today, enter your highest "
                   "ever lean mass. Rebuilding lost muscle is faster than gaining "
                   "it the first time — the model applies a bounded rebuild bonus "
                   "(up to +50% rate) that decays to zero at your prior peak.")
        prior_peak_in = st.number_input("Highest lean mass ever held (lbs, 0 = skip)",
                                        0.0, 250.0, 0.0, 0.5)

    st.divider()
    st.header("🎯 Goal")
    goal_weight = st.number_input("Goal Weight (lbs)", 90.0, 300.0, 155.0, 0.5)
    goal_bf     = st.number_input("Goal Body Fat %",   5.0, 40.0, 15.0, 0.5)
    bf_ceiling  = st.number_input("Max BF% (ceiling)", 8.0, 35.0, 17.0, 0.5)
    bf_floor    = st.number_input("Min BF% (floor)",   4.0, 25.0, 14.5, 0.5)

    st.divider()
    st.header("📊 Adherence")
    consistency = st.slider("Consistency (% of days on plan)", 50, 100, 100, 5,
        help="How many days you'll realistically stick to the plan. Off-plan "
             "days hurt roughly twice as much during a cut as during a bulk — "
             "a cheat day while cutting erases deficit, not just pauses it. "
             "At 100% this has no effect.")

    allow_recomp = st.checkbox(
        "Allow slight muscle gain on slow cuts", value=False,
        help="Research on athletes (Garthe 2011) found newer lifters can gain "
             "a little muscle while dieting slowly. Leave this OFF unless "
             "you're new to lifting or coming back from a long break — for "
             "experienced lifters, holding muscle on a slow cut is the "
             "realistic best case.")

    with st.expander("🧪 Advanced settings", expanded=False):
        st.caption("The shaded ranges come from running your plan hundreds of "
                   "times with realistic variation in two things nobody knows "
                   "exactly: your true starting body fat, and how fast your body "
                   "responds compared to average.")
        bf_sd_ui   = st.slider("Body-fat reading accuracy (±%)", 0.5, 4, 1.0, 0.25,
                               help="How trustworthy your starting body-fat number "
                                    "is. 1.0 if it came from a DEXA scan; 2.0+ if "
                                    "it's a smart-scale reading or an eyeball "
                                    "estimate.")
        rate_cv_ui = st.slider("Person-to-person variation (%)", 5, 25, 12, 1,
                               help="How differently individuals respond to the "
                                    "same plan. 12% is a sensible default. Once "
                                    "you log weigh-ins and apply calibration, "
                                    "your real measured variation replaces this.")

    st.divider()
    st.header("⚙️ Mode")
    mode = st.radio("Planning mode", ["🤖 Auto Schedule", "🔧 Manual Phases"])

    # ── build the frame (ceiling + baseline) ───────────────────────────────────
    height_in_total = (height_ft * 12) + height_in_rem
    start_lean_from_first = None
    if first_weight > 0 and first_bf > 0:
        start_lean_from_first = first_weight * (1 - first_bf / 100)

    frame = build_frame(height_in_total, wrist_in, ankle_in, age, sex=sex,
                        start_lean=start_lean_from_first,
                        start_lean_weight=first_weight if first_weight > 0 else None)
    frame["allow_recomp"] = allow_recomp

    prior_peak = prior_peak_in if prior_peak_in > 0 else None
    start_lean = start_weight * (1 - start_bf / 100)
    dp_start = dynamic_profile(start_lean, frame, peak_lean_lbs=prior_peak,
                               current_bf=start_bf)

    st.divider()
    st.header("📐 Your Profile (starting)")
    st.caption("Rates degrade over the plan as you approach your ceiling.")
    st.metric("FFMI", f"{dp_start['ffmi']}")
    st.metric("Ceiling (lean lbs)", f"{frame['ceiling_lean']:.0f}",
              help=f"Casey Butt max ({sex.lower()}-adjusted), "
                   f"normalized FFMI {frame['ffmi_ceiling']:.1f}")
    st.metric("Baseline (untrained lbs)", f"{frame['baseline_lean']:.0f}")
    st.metric("% to Ceiling", f"{dp_start['ceiling_pct']}%")
    st.metric("Start Bulk Rate",
              f"{dp_start['bulk_rate_monthly']} lbs/mo ({dp_start['bulk_rate_weekly']} lbs/wk)")
    st.metric("Start Muscle % (Bulk)", f"{dp_start['muscle_frac_bulk']*100:.0f}%")
    st.metric("Start Cut Rate",
              f"{dp_start['cut_rate_monthly']} lbs/mo ({dp_start['cut_rate_weekly']} lbs/wk)",
              help="Research target is 0.5-1%/wk of bodyweight for muscle retention; "
                   "this sits at the conservative end and rises slightly as you advance.")
    st.metric("Start Muscle Loss % (Cut)", f"{dp_start['muscle_frac_cut']*100:.0f}%",
              help="Share of each lost pound that is lean. Scales with cut "
                   "aggressiveness, per Helms/Garthe: faster cuts lose more muscle. "
                   "A NEGATIVE value means a very slow cut is projected to add a "
                   "sliver of lean while losing fat (Garthe recomp effect — "
                   "novices and returning lifters only).")

    st.divider()
    st.header("📅 Schedule Length")
    max_weeks = st.slider("Maximum total weeks", 26, 260, 156, 2,
                          help="52 = 1 yr, 104 = 2 yr, 156 = 3 yr.")
    max_phase_weeks = st.slider("Max weeks per phase", 4, 32, 20, 1)

    # ── optional: manual overrides (collapsed) ─────────────────────────────────
    with st.expander("⚙️ Rate overrides (optional)", expanded=False):
        st.caption("Want to force a specific weekly rate? Set it here. Leave "
                   "at 0 and the app picks sensible rates for you. Forced rates "
                   "are taken exactly as entered — the consistency and "
                   "calibration adjustments won't touch them.")
        bulk_rate_override   = st.number_input("Bulk rate (lbs/week)", 0.0, 2.0, 0.0, 0.05)
        cut_rate_override    = st.number_input("Cut rate (lbs/week)",  0.0, 2.0, 0.0, 0.05)
        bulk_muscle_override = st.number_input("Muscle % on bulk", 0, 100, 0, 1)
        cut_muscle_override  = st.number_input("Muscle loss % on cut", 0, 100, 0, 1)

    # ── optional: start date (collapsed) ───────────────────────────────────────
    with st.expander("🗓️ Start date (optional)", expanded=False):
        start_date = st.date_input("Start date", value=date(2026, 6, 13))

    # ── actuals log (feeds personal calibration + chart overlay) ───────────────
    with st.expander("📉 Log your weigh-ins", expanded=False):
        st.caption("Log real weigh-ins (or import a CSV with `date,weight` columns "
                   "— MacroFactor exports work). Data lives in this browser "
                   "session only; use Download to keep it between sessions.")
        up = st.file_uploader("Import CSV", type="csv", key="actuals_csv")
        if up is not None and st.session_state.get("actuals_csv_name") != up.name:
            try:
                raw = pd.read_csv(up)
                raw.columns = [c.strip().lower() for c in raw.columns]
                if "date" in raw.columns and "weight" in raw.columns:
                    imp = raw[["date", "weight"]].copy()
                    imp["date"] = pd.to_datetime(imp["date"], errors="coerce")
                    imp["weight"] = pd.to_numeric(imp["weight"], errors="coerce")
                    imp = imp.dropna()
                    st.session_state.actuals_df = imp
                    st.session_state.actuals_csv_name = up.name
                else:
                    st.error("CSV needs `date` and `weight` columns.")
            except Exception as e:
                st.error(f"Couldn't parse CSV: {e}")

        if "actuals_df" not in st.session_state:
            st.session_state.actuals_df = pd.DataFrame(
                {"date": pd.Series(dtype="datetime64[ns]"),
                 "weight": pd.Series(dtype="float")})

        edited_actuals = st.data_editor(
            st.session_state.actuals_df,
            num_rows="dynamic", hide_index=True, key="actuals_editor",
            column_config={
                "date": st.column_config.DateColumn("Date"),
                "weight": st.column_config.NumberColumn("Weight (lbs)",
                                                        min_value=50.0, max_value=500.0,
                                                        step=0.1, format="%.1f"),
            })
        st.session_state.actuals_df = edited_actuals

        actuals = _parse_actuals_df(edited_actuals)
        if actuals:
            csv_buf = io.StringIO()
            pd.DataFrame(actuals, columns=["date", "weight"]).to_csv(csv_buf, index=False)
            st.download_button("⬇️ Download actuals CSV", csv_buf.getvalue(),
                               "actuals.csv", "text/csv")
        apply_calibration = st.checkbox(
            "Apply personal calibration", value=False,
            help="Adjusts the plan to how YOUR body has actually been "
                 "responding, based on your logged weigh-ins. Needs a few "
                 "weeks of data before it kicks in.")

overrides = {
    "bulk_rate":   bulk_rate_override,
    "cut_rate":    cut_rate_override,
    "bulk_muscle": bulk_muscle_override,
    "cut_muscle":  cut_muscle_override,
}

# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULE + BAND (cached)
# ══════════════════════════════════════════════════════════════════════════════
def _frame_from_tuple(t):
    return {"height_in": t[0], "ceiling_lean": t[1], "baseline_lean": t[2],
            "ffmi_ceiling": t[3], "rate_mult": t[4], "allow_recomp": t[5]}

def _ov_from_tuple(t):
    return {"bulk_rate": t[0], "cut_rate": t[1], "bulk_muscle": t[2], "cut_muscle": t[3]}

def _mods_from_tuple(t):
    return {"bulk_mult": t[0], "cut_mult": t[1]}

@st.cache_data(show_spinner="Optimizing schedule…")
def cached_schedule(sw, sbf, gw, gbf, ceil, floor, frame_tuple, ov_tuple, mw, mpw,
                    mods_tuple, prior_peak):
    return auto_schedule_dynamic(
        sw, sbf, gw, gbf, ceil, floor,
        _frame_from_tuple(frame_tuple), _ov_from_tuple(ov_tuple),
        max_weeks=mw, max_phase_weeks=mpw,
        mods=_mods_from_tuple(mods_tuple), prior_peak=prior_peak)

@st.cache_data(show_spinner="Estimating timeline uncertainty (re-optimizing 24 scenarios)…")
def cached_timeline_ci(sw, sbf, gw, gbf, ceil, floor, frame_tuple, ov_tuple, mw, mpw,
                       mods_tuple, prior_peak, bf_sd, bulk_cv, cut_cv):
    """Timeline CI via full re-optimization per sample. The horizon is
    DELIBERATELY extended beyond the user's max-weeks setting (+104 wks) so the
    high end of the CI reflects genuinely slow response scenarios instead of
    being censored at the planning window."""
    horizon = mw + 104
    return timeline_ci_full(sw, sbf, gw, gbf, ceil, floor,
                            _frame_from_tuple(frame_tuple), _ov_from_tuple(ov_tuple),
                            _mods_from_tuple(mods_tuple), prior_peak,
                            horizon, mpw, bf_sd=bf_sd, bulk_cv=bulk_cv, cut_cv=cut_cv)

@st.cache_data(show_spinner="Simulating outcome envelope (200 scenarios)…")
def cached_envelope(sw, sbf, phases_key, frame_tuple, ov_tuple, mods_tuple,
                    prior_peak, gw, gbf, mw, mpw, headline, bf_sd, bulk_cv, cut_cv):
    """Per-week p10/p50/p90 envelope from 200 fixed-plan simulations, run out to
    an extended horizon (max weeks + 104) so band edges aren't clipped by the
    planning window either."""
    phases = [{"name": f"{t.capitalize()}", "type": t, "weeks": wk}
              for (t, wk) in phases_key]
    horizon = mw + 104
    return monte_carlo_bands(sw, sbf, phases, _frame_from_tuple(frame_tuple),
                             _ov_from_tuple(ov_tuple), _mods_from_tuple(mods_tuple),
                             prior_peak, gw, gbf, horizon, mpw, headline,
                             n_samples=200, bf_sd=bf_sd, bulk_cv=bulk_cv, cut_cv=cut_cv)

frame_tuple = (frame["height_in"], frame["ceiling_lean"], frame["baseline_lean"],
               frame["ffmi_ceiling"], frame["rate_mult"], frame["allow_recomp"])
ov_tuple = (bulk_rate_override, cut_rate_override, bulk_muscle_override, cut_muscle_override)

# ── Baseline (uncalibrated, 100%-consistency) projection: the fixed reference
# that personal calibration measures against ──────────────────────────────────
baseline_phases = cached_schedule(start_weight, start_bf, goal_weight, goal_bf,
                                  bf_ceiling, bf_floor, frame_tuple, ov_tuple,
                                  max_weeks, max_phase_weeks, (1.0, 1.0), prior_peak)
baseline_rows = simulate_dynamic(start_weight, start_bf, baseline_phases, frame,
                                 start_date, overrides, mods=DEFAULT_MODS,
                                 prior_peak=prior_peak)

calib = calibrate_from_actuals(actuals, baseline_rows)

# ── Assemble effective modifiers: consistency x personal calibration ──────────
a = consistency / 100.0
cons_bulk = a                                   # missed surplus days just slow gain
cons_cut  = max(0.1, 1.6 * a - 0.6)             # off-days on a cut erase deficit harder
calib_bulk = calib["bulk_mult"] if apply_calibration else 1.0
calib_cut  = calib["cut_mult"]  if apply_calibration else 1.0
mods = {"bulk_mult": cons_bulk * calib_bulk, "cut_mult": cons_cut * calib_cut}
mods_tuple = (round(mods["bulk_mult"], 4), round(mods["cut_mult"], 4))

auto_phases = cached_schedule(
    start_weight, start_bf, goal_weight, goal_bf, bf_ceiling, bf_floor,
    frame_tuple, ov_tuple, max_weeks, max_phase_weeks, mods_tuple, prior_peak)

# ══════════════════════════════════════════════════════════════════════════════
# MODE UI
# ══════════════════════════════════════════════════════════════════════════════
def _with_ids(phases):
    """Attach a stable unique id to each phase so widget keys survive
    insertions/deletions (index-based keys corrupt neighbors on delete)."""
    return [{**p, "id": uuid.uuid4().hex[:8]} for p in phases]

if mode == "🤖 Auto Schedule":
    st.subheader("🤖 Auto-Generated Schedule")
    st.caption("Phases chosen automatically to reach your goal as fast as your "
               "body-fat limits allow.")
    if not auto_phases:
        st.error("The scheduler couldn't build a plan inside your BF ceiling/floor "
                 "band with these inputs. Widen the band or adjust the goal.")
    active_phases = auto_phases
else:
    st.subheader("🔧 Manual Phase Builder")
    st.caption("Seeded from the auto schedule when you switch in — switching back "
               "to Auto and returning will RESET your edits. Edit freely after.")

    if "phases" not in st.session_state or st.session_state.get("last_auto") != auto_phases:
        if mode == "🔧 Manual Phases" and st.session_state.get("mode_prev") != mode:
            st.session_state.phases = _with_ids(auto_phases)
        elif "phases" not in st.session_state:
            st.session_state.phases = _with_ids(auto_phases)
        st.session_state.last_auto = auto_phases
    st.session_state.mode_prev = mode

    phases_to_delete = []
    for i, phase in enumerate(st.session_state.phases):
        pid = phase["id"]
        c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
        with c1:
            st.session_state.phases[i]["name"] = st.text_input(
                f"P{i+1}", value=phase["name"], key=f"name_{pid}", label_visibility="collapsed")
        with c2:
            st.session_state.phases[i]["type"] = st.selectbox(
                "Type", ["bulk", "cut", "maintain"],
                index=["bulk","cut","maintain"].index(phase["type"]),
                key=f"type_{pid}", label_visibility="collapsed")
        with c3:
            st.session_state.phases[i]["weeks"] = st.number_input(
                "Weeks", 1, 104, value=phase.get("weeks", 8),
                key=f"weeks_{pid}", label_visibility="collapsed")
        with c4:
            if st.button("🗑️", key=f"del_{pid}"):
                phases_to_delete.append(i)
    if phases_to_delete:
        for i in sorted(phases_to_delete, reverse=True):
            st.session_state.phases.pop(i)
        st.rerun()

    ca, cr, _ = st.columns([1, 1, 4])
    with ca:
        if st.button("➕ Add Phase"):
            st.session_state.phases.append({
                "name": f"Phase {len(st.session_state.phases)+1}",
                "type": "bulk", "weeks": 8,
                "id": uuid.uuid4().hex[:8],
            })
            st.rerun()
    with cr:
        if st.button("↺ Reset to Auto"):
            st.session_state.phases = _with_ids(auto_phases)
            st.rerun()

    active_phases = st.session_state.phases

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION + SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
data = simulate_dynamic(start_weight, start_bf, active_phases, frame, start_date,
                        overrides, mods=mods, prior_peak=prior_peak)

final = data[-1]
start = data[0]
total_weeks = final["week"]
lean_gained = round(final["lean"] - start["lean"], 1)
peak_bf_row = max(data, key=lambda r: r["bf"])
on_track = abs(final["weight"] - goal_weight) <= 2 and abs(final["bf"] - goal_bf) <= 1.5

if mode == "🤖 Auto Schedule":
    cA, cB, cC = st.columns(3)
    cA.metric("Total Weeks", f"{total_weeks}")
    cB.metric("Final Weight (dynamic)", f"{final['weight']} lbs")
    cC.metric("Final BF (dynamic)", f"{final['bf']}%")
    st.markdown("**Phases:**")
    h = st.columns([3, 2, 1]); h[0].markdown("**Phase**"); h[1].markdown("**Type**"); h[2].markdown("**Weeks**")
    for p in auto_phases:
        r = st.columns([3, 2, 1]); r[0].write(p["name"]); r[1].write(p["type"].capitalize()); r[2].write(p["weeks"])

st.divider()
st.subheader("📊 Projection Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Final Weight", f"{final['weight']} lbs", f"{round(final['weight']-start['weight'],1):+.1f} lbs")
c2.metric("Final BF%", f"{final['bf']}%", f"{round(final['bf']-start['bf'],1):+.1f}%", delta_color="inverse")
c3.metric("Lean Mass Added", f"+{lean_gained} lbs")
c4.metric("Peak BF%", f"{peak_bf_row['bf']}%", f"Week {peak_bf_row['week']}")
with c5:
    st.metric("Total Weeks", f"{total_weeks} wks")
    st.caption(f"≈ {round(total_weeks/4.33, 1)} months")

# ── Uncertainty: timeline CI + outcome-at-headline CI ─────────────────────────
# ── Effective per-channel rate CVs: the slider is the PRIOR (belief before
# data); once calibration is applied and a channel has enough samples, its CV
# is REPLACED by the measured relative standard error of your multiplier —
# so logging more weigh-ins genuinely narrows the bands for that channel.
# Floored at 4% (weekly noise never lets certainty go to zero) and capped at
# 30% (a terrible early fit shouldn't explode the bands beyond usefulness).
CV_FLOOR, CV_CAP = 0.04, 0.30
def _effective_cv(mult, se, n, slider_cv):
    if apply_calibration and se is not None and n >= CALIB_MIN_SAMPLES and mult > 0:
        return max(CV_FLOOR, min(CV_CAP, se / mult))
    return slider_cv

slider_cv = rate_cv_ui / 100.0
bulk_cv_eff = _effective_cv(calib["bulk_mult"], calib["bulk_se"], calib["bulk_n"], slider_cv)
cut_cv_eff  = _effective_cv(calib["cut_mult"],  calib["cut_se"],  calib["cut_n"],  slider_cv)

tl_ci = cached_timeline_ci(start_weight, start_bf, goal_weight, goal_bf,
                           bf_ceiling, bf_floor, frame_tuple, ov_tuple,
                           max_weeks, max_phase_weeks, mods_tuple, prior_peak,
                           bf_sd_ui, bulk_cv_eff, cut_cv_eff)
phases_key = tuple((p["type"], p.get("weeks", 4)) for p in active_phases)
envelope = cached_envelope(start_weight, start_bf, phases_key, frame_tuple,
                           ov_tuple, mods_tuple, prior_peak, goal_weight, goal_bf,
                           max_weeks, max_phase_weeks, total_weeks,
                           bf_sd_ui, bulk_cv_eff, cut_cv_eff)
oc = envelope["outcome"]

if tl_ci:
    over_note = ""
    if tl_ci["p90"] > max_weeks:
        over_note = (f" The slow end runs past your {max_weeks}-week planning "
                     f"window — that's fine, it just means a slower-than-average "
                     f"response would take longer than the plan shown.")
    frac_note = ""
    if tl_ci["frac_reached"] < 0.98:
        frac_note = (f" (A few slow-response cases don't get there even with "
                     f"plenty of extra time — about {100 - tl_ci['frac_reached']*100:.0f}%.)")
    st.info(f"⏱️ **Time to goal: most likely {tl_ci['p10']:.0f}–{tl_ci['p90']:.0f} weeks**, "
            f"typically around {tl_ci['p50']:.0f}. Everyone responds a little "
            f"differently — this range covers the realistic middle 80% of outcomes."
            + over_note + frac_note)
else:
    st.warning("⚠️ Couldn't estimate a time-to-goal range — the goal looks "
               "out of reach with these settings. Try adjusting the goal or "
               "the body-fat ceiling/floor.")

st.markdown(f"**If you follow this plan for {total_weeks} weeks, here's the "
            f"realistic range of where you'd land:**")
u1, u2 = st.columns(2)
u1.metric(f"Weight @ wk {total_weeks}",
          f"{oc['weight_p50']:.1f} lbs",
          f"likely {oc['weight_p10']:.1f}–{oc['weight_p90']:.1f} lbs",
          delta_color="off")
u2.metric(f"Body Fat @ wk {total_weeks}",
          f"{oc['bf_p50']:.1f}%",
          f"likely {oc['bf_p10']:.1f}–{oc['bf_p90']:.1f}%",
          delta_color="off")
st.caption("Respond faster than expected and you'd simply arrive early and "
           "hold there — that's why the shaded ranges on the chart narrow in "
           "toward your goal instead of flying past it.")

# ── Personal calibration status ────────────────────────────────────────────────
if actuals:
    st.subheader("📉 Your Data vs. the Model")
    k1, k2, k3 = st.columns(3)
    k1.metric("Logged weigh-ins", f"{len(actuals)}")
    bulk_ready = calib["bulk_n"] >= CALIB_MIN_SAMPLES
    cut_ready  = calib["cut_n"]  >= CALIB_MIN_SAMPLES
    k2.metric("Your bulk rate vs model",
              f"×{calib['bulk_mult']:.2f} ± {calib['bulk_se']:.2f}" if bulk_ready and calib['bulk_se'] is not None
              else ("×%.2f" % calib['bulk_mult'] if bulk_ready else "—"),
              f"{calib['bulk_n']}/{CALIB_MIN_SAMPLES} samples" if not bulk_ready
              else f"{calib['bulk_n']} samples")
    k3.metric("Your cut rate vs model",
              f"×{calib['cut_mult']:.2f} ± {calib['cut_se']:.2f}" if cut_ready and calib['cut_se'] is not None
              else ("×%.2f" % calib['cut_mult'] if cut_ready else "—"),
              f"{calib['cut_n']}/{CALIB_MIN_SAMPLES} samples" if not cut_ready
              else f"{calib['cut_n']} samples")
    if apply_calibration and (bulk_ready or cut_ready):
        parts = []
        if bulk_ready and calib["bulk_se"] is not None:
            parts.append(f"bulk ±{bulk_cv_eff*100:.0f}% (measured)")
        else:
            parts.append(f"bulk ±{bulk_cv_eff*100:.0f}% (slider)")
        if cut_ready and calib["cut_se"] is not None:
            parts.append(f"cut ±{cut_cv_eff*100:.0f}% (measured)")
        else:
            parts.append(f"cut ±{cut_cv_eff*100:.0f}% (slider)")
        st.success("✅ Your logged weigh-ins are now steering the plan — "
                   f"{', '.join(parts)}. The more you log, the tighter these "
                   "ranges get, because the app learns how YOUR body actually "
                   "responds instead of guessing from averages.")
    elif apply_calibration:
        st.warning("Calibration is on, but it needs a few more weeks of "
                   "weigh-ins before it changes anything. Keep logging.")
    else:
        st.caption("Your data has been analyzed but isn't steering the plan yet — "
                   "tick 'Apply personal calibration' in the sidebar to use it.")

gw_gap = round(final["weight"] - goal_weight, 1)
gbf_gap = round(final["bf"] - goal_bf, 1)
if on_track:
    st.success(f"✅ On track — projected finish: {final['weight']} lbs @ {final['bf']}% BF")
else:
    st.warning(f"⚠️ Closest the model reaches: {final['weight']} lbs @ {final['bf']}% BF "
               f"(off by {gw_gap:+.1f} lbs, {gbf_gap:+.1f}% BF). "
               f"This is the honest limit given your rates — extend max weeks or adjust goal.")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📈 Charts")
COLORS = {"bulk": "#10b981", "cut": "#ef4444", "maintain": "#f59e0b", "start": "#6366f1"}
dates   = [r["date"] for r in data]
weights = [r["weight"] for r in data]
leans   = [r["lean"] for r in data]
bfs     = [r["bf"] for r in data]
plist   = [r["phase"] for r in data]

fig = make_subplots(rows=2, cols=1,
    subplot_titles=("Scale Weight & Lean Mass (lbs)", "Body Fat %"),
    vertical_spacing=0.12, row_heights=[0.6, 0.4])

# ── 80% CI bands (drawn first so projection lines render on top). The band is
# allowed to run PAST the deterministic plan end, out to the later of the plan
# end and the p90 goal-arrival week — slow scenarios don't get visually clipped.
ew10, ew50, ew90 = envelope["weight"]
eb10, eb50, eb90 = envelope["bf"]
band_end = total_weeks
if tl_ci:
    band_end = max(band_end, int(math.ceil(tl_ci["p90"])) + 4)
band_end = min(band_end, len(envelope["weeks"]) - 1)
band_dates = [start_date + timedelta(weeks=i) for i in range(band_end + 1)]

fig.add_trace(go.Scatter(x=band_dates, y=ew90[:band_end+1], mode="lines",
    line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
fig.add_trace(go.Scatter(x=band_dates, y=ew10[:band_end+1], mode="lines",
    line=dict(width=0), fill="tonexty", fillcolor="rgba(16,185,129,0.13)",
    name="Likely weight range", hoverinfo="skip"), row=1, col=1)
fig.add_trace(go.Scatter(x=band_dates, y=eb90[:band_end+1], mode="lines",
    line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)
fig.add_trace(go.Scatter(x=band_dates, y=eb10[:band_end+1], mode="lines",
    line=dict(width=0), fill="tonexty", fillcolor="rgba(245,158,11,0.13)",
    name="Likely body-fat range", hoverinfo="skip"), row=2, col=1)

if tl_ci:
    arrival = start_date + timedelta(weeks=tl_ci["p50"])
    for row in (1, 2):
        fig.add_vline(x=arrival, line_dash="dashdot",
                      line_color="rgba(148,163,184,0.55)", row=row, col=1)
    fig.add_annotation(x=arrival, yref="paper", y=1.06, showarrow=False,
                       text=f"typical goal arrival (wk {tl_ci['p50']:.0f})",
                       font=dict(size=11, color="#94a3b8"))

fig.add_trace(go.Scatter(x=dates, y=weights, mode="lines+markers", name="Projected Weight",
    line=dict(color="#10b981", width=2.5), marker=dict(size=3),
    hovertemplate="<b>%{text}</b><br>Weight: %{y} lbs<extra></extra>", text=plist), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=leans, mode="lines", name="Lean Mass",
    line=dict(color="#6366f1", width=1.8, dash="dot"),
    hovertemplate="<b>%{text}</b><br>Lean: %{y} lbs<extra></extra>", text=plist), row=1, col=1)
if actuals:
    fig.add_trace(go.Scatter(
        x=[d for d, _ in actuals], y=[w for _, w in actuals],
        mode="markers", name="Actual Weigh-ins",
        marker=dict(symbol="diamond", size=7, color="#f8fafc",
                    line=dict(color="#0ea5e9", width=1.5)),
        hovertemplate="Actual: %{y} lbs<br>%{x}<extra></extra>"), row=1, col=1)
fig.add_hline(y=goal_weight, line_dash="dash", line_color="rgba(255,255,255,0.27)",
    annotation_text=f"Goal: {goal_weight} lbs", row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=bfs, mode="lines+markers", name="Body Fat %",
    line=dict(color="#f59e0b", width=2.5), marker=dict(size=3),
    hovertemplate="<b>%{text}</b><br>BF: %{y}%<extra></extra>", text=plist), row=2, col=1)
fig.add_hline(y=goal_bf, line_dash="dash", line_color="rgba(255,255,255,0.27)",
    annotation_text=f"Goal: {goal_bf}%", row=2, col=1)
fig.add_hline(y=bf_ceiling, line_dash="dot", line_color="rgba(239,68,68,0.4)",
    annotation_text=f"Ceiling: {bf_ceiling}%", row=2, col=1)
fig.add_hline(y=bf_floor, line_dash="dot", line_color="rgba(16,185,129,0.4)",
    annotation_text=f"Floor: {bf_floor}%", row=2, col=1)

phase_starts = {}
for r in data:
    if r["phase"] not in phase_starts:
        phase_starts[r["phase"]] = (r["date"], r["phase_type"])
pnames = list(phase_starts.keys())
for i, pn in enumerate(pnames):
    pdate, ptype = phase_starts[pn]
    end = phase_starts[pnames[i+1]][0] if i+1 < len(pnames) else dates[-1]
    for row in [1, 2]:
        fig.add_vrect(x0=pdate, x1=end, fillcolor=COLORS.get(ptype, "#888"), opacity=0.07, line_width=0, row=row, col=1)

fig.update_layout(height=600, paper_bgcolor="#0f1624", plot_bgcolor="#080b14",
    font=dict(color="#e2e8f0"),
    legend=dict(bgcolor="#0f1624", bordercolor="#1e293b", borderwidth=1),
    hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10))
fig.update_xaxes(gridcolor="#1e293b", showgrid=True)
fig.update_yaxes(gridcolor="#1e293b", showgrid=True)
st.plotly_chart(fig, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📅 Week-by-Week Breakdown")
with st.expander("Show full table", expanded=False):
    df = pd.DataFrame(data)
    df["date"] = df["date"].astype(str)
    df["change"] = df["change"].apply(lambda x: f"+{x:.2f}" if x > 0 else (f"{x:.2f}" if x != 0 else "—"))
    df = df[["week","date","phase","weight","lean","fat","bf","change","rate"]]
    df.columns = ["Week","Date","Phase","Weight","Lean","Fat","BF%","ΔWt","Rate/wk"]
    st.dataframe(df, width="stretch", hide_index=True)
