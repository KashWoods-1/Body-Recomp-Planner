import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import pandas as pd

st.set_page_config(page_title="Recomp Planner", page_icon="💪", layout="wide")
st.title("💪 Body Recomposition Planner")
st.caption("Dynamic model — rates degrade as you approach your genetic ceiling.")

# ── Static profile: computed once from unchanging inputs ────────────────────────
def static_profile(height_in, wrist_in, age):
    height_m = height_in * 0.0254
    if wrist_in < 6.5:
        ffmi_ceiling = 21.5
    elif wrist_in < 7.5:
        ffmi_ceiling = 23.5
    else:
        ffmi_ceiling = 25.0
    ffmi_ceiling -= max(0, (age - 40) * 0.1)
    return {"height_m": height_m, "ffmi_ceiling": ffmi_ceiling}

# ── Dynamic profile: recomputed each week from current lean mass ────────────────
def dynamic_profile(current_lean_lbs, static, peak_lean_lbs=None):
    height_m     = static["height_m"]
    ffmi_ceiling = static["ffmi_ceiling"]
    ffmi_floor   = 16.0

    lean_for_calc = max(current_lean_lbs, peak_lean_lbs) if peak_lean_lbs else current_lean_lbs
    lean_kg   = lean_for_calc * 0.453592
    ffmi      = lean_kg / (height_m ** 2)
    ffmi_norm = ffmi + 6.1 * (1.8 - height_m)

    prox = min(1.0, max(0.0, (ffmi_norm - ffmi_floor) / (ffmi_ceiling - ffmi_floor)))

    EXP, SCALE = 1.2, 2.078
    remaining = (1 - prox) ** EXP

    bulk_rate_monthly = round(0.3 + remaining * SCALE, 2)
    bulk_rate_weekly  = round(bulk_rate_monthly / 4.33, 3)

    muscle_frac_bulk = round(0.30 + remaining * 0.55, 3)
    rate_penalty     = max(0, (bulk_rate_monthly - 0.75) * 0.08)
    muscle_frac_bulk = round(max(0.30, muscle_frac_bulk - rate_penalty), 3)

    cut_rate_monthly = round(0.75 + prox * 0.5, 2)
    cut_rate_weekly  = round(cut_rate_monthly / 4.33, 3)
    base_cut_loss    = 0.04 + (cut_rate_weekly ** 1.4) * 0.16
    muscle_frac_cut  = round(max(0.06, min(0.55, base_cut_loss - prox * 0.03)), 3)

    return {
        "ffmi": round(ffmi_norm, 2),
        "ffmi_ceiling": round(ffmi_ceiling, 2),
        "ceiling_pct": round(prox * 100, 1),
        "bulk_rate_weekly": bulk_rate_weekly,
        "bulk_rate_monthly": bulk_rate_monthly,
        "cut_rate_weekly": cut_rate_weekly,
        "cut_rate_monthly": cut_rate_monthly,
        "muscle_frac_bulk": muscle_frac_bulk,
        "muscle_frac_cut": muscle_frac_cut,
    }

# ── Core weekly step (shared by scheduler and simulation) ───────────────────────
def step_week(w, lean, fat, peak_lean, action, static,
              overrides):
    dp = dynamic_profile(lean, static, peak_lean_lbs=peak_lean)

    bulk_rate = overrides["bulk_rate"] if overrides["bulk_rate"] > 0 else dp["bulk_rate_weekly"]
    cut_rate  = overrides["cut_rate"]  if overrides["cut_rate"]  > 0 else dp["cut_rate_weekly"]

    if overrides["bulk_muscle"] > 0:
        mfrac_bulk = overrides["bulk_muscle"] / 100
    else:
        mfrac_bulk = dp["muscle_frac_bulk"]

    if overrides["cut_muscle"] > 0:
        mfrac_cut = overrides["cut_muscle"] / 100
    else:
        # recompute from actual cut rate so overrides drive lean loss honestly
        mfrac_cut = max(0.06, min(0.55, 0.04 + (cut_rate ** 1.4) * 0.16 - dp["ceiling_pct"]/100 * 0.03))

    if action == "bulk":
        lean += bulk_rate * mfrac_bulk
        fat  += bulk_rate * (1 - mfrac_bulk)
        w    += bulk_rate
    elif action == "cut":
        lean -= cut_rate * mfrac_cut
        fat  -= cut_rate * (1 - mfrac_cut)
        w    -= cut_rate
    else:  # maintain
        lean += 0.012; fat += 0.012; w += 0.023

    peak_lean = max(peak_lean, lean)
    return w, lean, fat, peak_lean, bulk_rate, mfrac_bulk, cut_rate, mfrac_cut

# ── Simulation: the single source of truth for all reported numbers ─────────────
def simulate_dynamic(start_weight, start_bf, phases, static, start_date, overrides):
    w    = start_weight
    lean = start_weight * (1 - start_bf / 100)
    fat  = start_weight * (start_bf / 100)
    peak_lean = lean

    rows = [{
        "week": 0, "date": start_date, "phase": "Start", "phase_type": "start",
        "weight": round(w,1), "lean": round(lean,1), "fat": round(fat,1),
        "bf": round(fat/w*100,1), "change": 0.0, "bulk_rate": 0.0, "muscle_frac": 0.0,
    }]

    idx = 1
    current_date = start_date
    for phase in phases:
        ptype   = phase["type"]
        n_weeks = phase.get("weeks", 4)
        for _ in range(n_weeks):
            prev_w = w
            w, lean, fat, peak_lean, br, mfb, cr, mfc = step_week(
                w, lean, fat, peak_lean, ptype, static, overrides)
            current_date += timedelta(weeks=1)
            rows.append({
                "week": idx, "date": current_date,
                "phase": phase["name"], "phase_type": ptype,
                "weight": round(w,1), "lean": round(lean,1), "fat": round(fat,1),
                "bf": round(fat/w*100,1), "change": round(w-prev_w,3),
                "bulk_rate": round(br,3), "muscle_frac": round(mfb,3),
            })
            idx += 1
    return rows

# ── Dynamic look-ahead scheduler ────────────────────────────────────────────────
# Bounded 2-phase look-ahead. Min phase = 4 weeks. Uses the same dynamic engine.
def _roll_phase(state, action, weeks, static, overrides):
    """Advance a copy of state through `weeks` of `action`. Returns new state."""
    w, lean, fat, peak = state
    for _ in range(weeks):
        w, lean, fat, peak, *_ = step_week(w, lean, fat, peak, action, static, overrides)
    return (w, lean, fat, peak)

def _goal_distance(state, goal_weight, goal_bf):
    w, lean, fat, peak = state
    bf = fat / w * 100
    # weight and BF errors weighted roughly equally
    return abs(w - goal_weight) + abs(bf - goal_bf) * 1.0

def auto_schedule_dynamic(start_weight, start_bf, goal_weight, goal_bf,
                          bf_ceiling, bf_floor, static, overrides,
                          max_weeks=156, max_phase_weeks=20, min_phase_weeks=4):
    w    = start_weight
    lean = start_weight * (1 - start_bf / 100)
    fat  = start_weight * (start_bf / 100)
    peak = lean

    phases = []
    week_idx = 0
    phase_num = 1

    bf_now = fat / w * 100
    action = "cut" if bf_now >= bf_ceiling else "bulk"

    GOAL_TOL = 1.0

    while week_idx < max_weeks:
        bf_now = fat / w * 100
        # stop if at goal
        if abs(w - goal_weight) <= GOAL_TOL and abs(bf_now - goal_bf) <= GOAL_TOL:
            break

        state = (w, lean, fat, peak)
        weeks_left = max_weeks - week_idx
        hi = min(max_phase_weeks, weeks_left)
        if hi < min_phase_weeks:
            break

        # Candidate lengths for THIS phase, bounded by BF guardrails
        best = None  # (score, this_len)
        for this_len in range(min_phase_weeks, hi + 1):
            s1 = _roll_phase(state, action, this_len, static, overrides)
            w1, lean1, fat1, peak1 = s1
            bf1 = fat1 / w1 * 100

            # respect guardrails: don't blow past ceiling on bulk / floor on cut
            if action == "bulk" and bf1 > bf_ceiling + 0.5:
                break  # longer will only be worse
            if action == "cut" and bf1 < bf_floor - 0.5:
                break

            # 2nd phase look-ahead (opposite action)
            next_action = "cut" if action == "bulk" else "bulk"
            weeks_left2 = weeks_left - this_len
            hi2 = min(max_phase_weeks, weeks_left2)
            if hi2 >= min_phase_weeks:
                best2 = None
                for next_len in range(min_phase_weeks, hi2 + 1):
                    s2 = _roll_phase(s1, next_action, next_len, static, overrides)
                    d = _goal_distance(s2, goal_weight, goal_bf)
                    if best2 is None or d < best2:
                        best2 = d
                score = best2 if best2 is not None else _goal_distance(s1, goal_weight, goal_bf)
            else:
                score = _goal_distance(s1, goal_weight, goal_bf)

            if best is None or score < best[0]:
                best = (score, this_len)

        if best is None:
            break

        chosen_len = best[1]
        # commit chosen phase
        new_state = _roll_phase(state, action, chosen_len, static, overrides)
        w, lean, fat, peak = new_state
        phases.append({
            "name": f"{'Bulk' if action=='bulk' else 'Cut'} {phase_num}",
            "type": action,
            "weeks": chosen_len,
        })
        if action == "cut":
            phase_num += 1
        action = "cut" if action == "bulk" else "bulk"
        week_idx += chosen_len

        # safety: if we're basically at goal weight, allow loop's goal check to end it
        if abs(w - goal_weight) <= GOAL_TOL and abs(fat/w*100 - goal_bf) <= GOAL_TOL:
            break

    return phases

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Starting Stats")
    start_weight = st.number_input("Start Weight (lbs)", 90.0, 300.0, 145.0, 0.5)
    start_bf     = st.number_input("Start Body Fat %",   5.0, 40.0, 15.5, 0.5)

    st.divider()
    st.header("👤 Profile")
    height_ft     = st.number_input("Height (ft)", 4, 7, 5)
    height_in_rem = st.number_input("Height (in)", 0, 11, 8)
    wrist_in      = st.number_input("Wrist circumference (in)", 4.0, 10.0, 6.25, 0.25)
    age           = st.number_input("Age", 16, 80, 22)

    st.divider()
    st.header("🎯 Goal")
    goal_weight = st.number_input("Goal Weight (lbs)", 90.0, 300.0, 155.0, 0.5)
    goal_bf     = st.number_input("Goal Body Fat %",   5.0, 40.0, 15.0, 0.5)
    bf_ceiling  = st.number_input("Max BF% (ceiling)", 8.0, 35.0, 17.0, 0.5)
    bf_floor    = st.number_input("Min BF% (floor)",   4.0, 25.0, 15.0, 0.5)

    st.divider()
    st.header("⚙️ Mode")
    mode = st.radio("Planning mode", ["🤖 Auto Schedule", "🔧 Manual Phases"])

    height_in_total = (height_ft * 12) + height_in_rem
    static = static_profile(height_in_total, wrist_in, age)

    # starting-point dynamic profile for the sidebar display
    start_lean = start_weight * (1 - start_bf / 100)
    dp_start = dynamic_profile(start_lean, static)

    st.divider()
    st.header("📐 Your Profile (starting)")
    st.caption("These rates degrade over the plan as you approach your ceiling.")
    st.metric("FFMI", f"{dp_start['ffmi']}")
    st.metric("FFMI Ceiling", f"{dp_start['ffmi_ceiling']}")
    st.metric("% to Ceiling", f"{dp_start['ceiling_pct']}%")
    st.metric("Start Bulk Rate", f"{dp_start['bulk_rate_monthly']} lbs/mo ({dp_start['bulk_rate_weekly']} lbs/wk)")
    st.metric("Start Cut Rate", f"{dp_start['cut_rate_monthly']} lbs/mo ({dp_start['cut_rate_weekly']} lbs/wk)")
    st.metric("Start Muscle % (Bulk)", f"{dp_start['muscle_frac_bulk']*100:.0f}%")

    st.divider()
    st.header("⚙️ Overrides")
    st.caption("Optional — leave at 0 to use dynamic values.")
    bulk_rate_override   = st.number_input("Bulk rate (lbs/week)", 0.0, 2.0, 0.0, 0.05)
    cut_rate_override    = st.number_input("Cut rate (lbs/week)",  0.0, 2.0, 0.0, 0.05)
    bulk_muscle_override = st.number_input("Muscle % on bulk", 0, 100, 0, 1)
    cut_muscle_override  = st.number_input("Muscle loss % on cut", 0, 100, 0, 1)

    st.divider()
    st.header("⚙️ Auto Schedule Settings")
    max_weeks = st.slider("Maximum total weeks", 26, 260, 156, 2,
                          help="52 = 1 yr, 104 = 2 yr, 156 = 3 yr.")
    max_phase_weeks = st.slider("Max weeks per phase", 4, 32, 20, 1)

    st.divider()
    st.header("🗓️ Start Date")
    start_date = st.date_input("Start date", value=date(2026, 6, 13))

overrides = {
    "bulk_rate":   bulk_rate_override,
    "cut_rate":    cut_rate_override,
    "bulk_muscle": bulk_muscle_override,
    "cut_muscle":  cut_muscle_override,
}

# ── Build the schedule (cached so live reruns stay snappy) ───────────────────────
@st.cache_data(show_spinner="Optimizing schedule…")
def cached_schedule(sw, sbf, gw, gbf, ceil, floor, static_tuple, ov_tuple, mw, mpw):
    static_d = {"height_m": static_tuple[0], "ffmi_ceiling": static_tuple[1]}
    ov = {"bulk_rate": ov_tuple[0], "cut_rate": ov_tuple[1],
          "bulk_muscle": ov_tuple[2], "cut_muscle": ov_tuple[3]}
    return auto_schedule_dynamic(sw, sbf, gw, gbf, ceil, floor, static_d, ov,
                                 max_weeks=mw, max_phase_weeks=mpw)

auto_phases = cached_schedule(
    start_weight, start_bf, goal_weight, goal_bf, bf_ceiling, bf_floor,
    (static["height_m"], static["ffmi_ceiling"]),
    (bulk_rate_override, cut_rate_override, bulk_muscle_override, cut_muscle_override),
    max_weeks, max_phase_weeks,
)

# ── Mode UI ──────────────────────────────────────────────────────────────────────
if mode == "🤖 Auto Schedule":
    st.subheader("🤖 Auto-Generated Schedule")
    st.caption("Phases chosen by 2-phase look-ahead using the dynamic engine. "
               "All numbers below come from the dynamic simulation.")
    active_phases = auto_phases
else:
    st.subheader("🔧 Manual Phase Builder")
    st.caption("Seeded from the auto schedule when you switch in. Edit freely after.")

    if "phases" not in st.session_state or st.session_state.get("last_auto") != auto_phases:
        if mode == "🔧 Manual Phases" and st.session_state.get("mode_prev") != mode:
            st.session_state.phases = [dict(p) for p in auto_phases]
        elif "phases" not in st.session_state:
            st.session_state.phases = [dict(p) for p in auto_phases]
        st.session_state.last_auto = auto_phases
    st.session_state.mode_prev = mode

    phases_to_delete = []
    for i, phase in enumerate(st.session_state.phases):
        c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
        with c1:
            st.session_state.phases[i]["name"] = st.text_input(
                f"P{i+1}", value=phase["name"], key=f"name_{i}", label_visibility="collapsed")
        with c2:
            st.session_state.phases[i]["type"] = st.selectbox(
                "Type", ["bulk", "cut", "maintain"],
                index=["bulk","cut","maintain"].index(phase["type"]),
                key=f"type_{i}", label_visibility="collapsed")
        with c3:
            st.session_state.phases[i]["weeks"] = st.number_input(
                "Weeks", 1, 104, value=phase.get("weeks", 8),
                key=f"weeks_{i}", label_visibility="collapsed")
        with c4:
            if st.button("🗑️", key=f"del_{i}"):
                phases_to_delete.append(i)
    for i in sorted(phases_to_delete, reverse=True):
        st.session_state.phases.pop(i)

    ca, cr, _ = st.columns([1, 1, 4])
    with ca:
        if st.button("➕ Add Phase"):
            st.session_state.phases.append({"name": f"Phase {len(st.session_state.phases)+1}", "type": "bulk", "weeks": 8})
            st.rerun()
    with cr:
        if st.button("↺ Reset to Auto"):
            st.session_state.phases = [dict(p) for p in auto_phases]
            st.rerun()

    active_phases = st.session_state.phases

# ── Single dynamic simulation feeds everything ──────────────────────────────────
data = simulate_dynamic(start_weight, start_bf, active_phases, static, start_date, overrides)

final = data[-1]
start = data[0]
total_weeks = final["week"]
lean_gained = round(final["lean"] - start["lean"], 1)
peak_bf_row = max(data, key=lambda r: r["bf"])
on_track = abs(final["weight"] - goal_weight) <= 2 and abs(final["bf"] - goal_bf) <= 1.5

# Auto-schedule summary (now pulled from the dynamic sim — single source of truth)
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

gw_gap = round(final["weight"] - goal_weight, 1)
gbf_gap = round(final["bf"] - goal_bf, 1)
if on_track:
    st.success(f"✅ On track — projected finish: {final['weight']} lbs @ {final['bf']}% BF")
else:
    st.warning(f"⚠️ Closest the model reaches: {final['weight']} lbs @ {final['bf']}% BF "
               f"(off by {gw_gap:+.1f} lbs, {gbf_gap:+.1f}% BF). "
               f"This is the honest limit given your rates — extend max weeks or adjust goal.")

# ── Charts ──────────────────────────────────────────────────────────────────────
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

fig.add_trace(go.Scatter(x=dates, y=weights, mode="lines+markers", name="Scale Weight",
    line=dict(color="#10b981", width=2.5), marker=dict(size=3),
    hovertemplate="<b>%{text}</b><br>Weight: %{y} lbs<extra></extra>", text=plist), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=leans, mode="lines", name="Lean Mass",
    line=dict(color="#6366f1", width=1.8, dash="dot"),
    hovertemplate="<b>%{text}</b><br>Lean: %{y} lbs<extra></extra>", text=plist), row=1, col=1)
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
st.plotly_chart(fig, use_container_width=True)

# ── Table ────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📅 Week-by-Week Breakdown")
with st.expander("Show full table", expanded=False):
    df = pd.DataFrame(data)
    df["date"] = df["date"].astype(str)
    df["change"] = df["change"].apply(lambda x: f"+{x:.2f}" if x > 0 else (f"{x:.2f}" if x != 0 else "—"))
    df = df[["week","date","phase","weight","lean","fat","bf","change","bulk_rate"]]
    df.columns = ["Week","Date","Phase","Weight","Lean","Fat","BF%","ΔWt","BulkRate/wk"]
    st.dataframe(df, use_container_width=True, hide_index=True)
