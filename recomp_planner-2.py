import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recomp Planner",
    page_icon="💪",
    layout="wide",
)

st.title("💪 Body Recomposition Planner")
st.caption("Model your bulk/cut cycles and see week-by-week projections.")

# ── Profile Calculator ─────────────────────────────────────────────────────────
def calculate_profile(height_in, weight_lbs, bf_pct, wrist_in, age, training_years):
    lean_lbs  = weight_lbs * (1 - bf_pct / 100)
    lean_kg   = lean_lbs * 0.453592
    height_m  = height_in * 0.0254
    ffmi      = lean_kg / (height_m ** 2)
    ffmi_norm = ffmi + 6.1 * (1.8 - height_m)

    if wrist_in < 6.5:
        ffmi_ceiling = 21.5
    elif wrist_in < 7.5:
        ffmi_ceiling = 23.5
    else:
        ffmi_ceiling = 25.0

    age_penalty  = max(0, (age - 35) * 0.1)
    ffmi_ceiling -= age_penalty
    ffmi_floor   = 16.0
    ceiling_pct  = min(1.0, max(0.0, (ffmi_norm - ffmi_floor) / (ffmi_ceiling - ffmi_floor)))

    if training_years < 1:
        experience_factor = 1.0
    elif training_years < 2:
        experience_factor = 0.8
    elif training_years < 4:
        experience_factor = 0.65
    elif training_years < 7:
        experience_factor = 0.50
    else:
        experience_factor = 0.38

    combined_factor  = (ceiling_pct * 0.6) + ((1 - experience_factor) * 0.4)

    # Convert monthly rates to weekly
    bulk_rate_monthly = round(1.6 - (combined_factor * 1.1), 2)
    bulk_rate_monthly = max(0.4, min(2.0, bulk_rate_monthly))
    bulk_rate_weekly  = round(bulk_rate_monthly / 4.33, 3)

    base_muscle_frac = 0.62 - (combined_factor * 0.28)
    rate_penalty     = max(0, (bulk_rate_monthly - 0.75) * 0.08)
    muscle_frac_bulk = round(max(0.30, base_muscle_frac - rate_penalty), 2)

    cut_rate_monthly = round(0.75 + (combined_factor * 0.5), 2)
    cut_rate_monthly = max(0.5, min(1.5, cut_rate_monthly))
    cut_rate_weekly  = round(cut_rate_monthly / 4.33, 3)

    muscle_frac_cut  = round(0.14 - (experience_factor * 0.06), 2)
    muscle_frac_cut  = max(0.06, min(0.20, muscle_frac_cut))

    return {
        "ffmi":              round(ffmi_norm, 1),
        "ffmi_ceiling":      round(ffmi_ceiling, 1),
        "ceiling_pct":       round(ceiling_pct * 100, 1),
        "bulk_rate_weekly":  bulk_rate_weekly,
        "bulk_rate_monthly": bulk_rate_monthly,
        "cut_rate_weekly":   cut_rate_weekly,
        "cut_rate_monthly":  cut_rate_monthly,
        "muscle_frac_bulk":  muscle_frac_bulk,
        "muscle_frac_cut":   muscle_frac_cut,
        "experience_factor": round(experience_factor, 2),
    }

# ── Auto Scheduler ─────────────────────────────────────────────────────────────
def auto_schedule(start_weight, start_bf, goal_weight, goal_bf,
                  bf_ceiling, bf_floor, profile,
                  max_weeks=156, max_phase_weeks=20):

    bulk_rate        = profile["bulk_rate_weekly"]
    cut_rate         = profile["cut_rate_weekly"]
    muscle_frac_bulk = profile["muscle_frac_bulk"]
    muscle_frac_cut  = profile["muscle_frac_cut"]

    w    = start_weight
    lean = start_weight * (1 - start_bf / 100)
    fat  = start_weight * (start_bf / 100)

    phases    = []
    week_idx  = 0
    phase_num = 1

    current_bf = fat / w * 100
    action     = "cut" if current_bf >= bf_ceiling else "bulk"

    while week_idx < max_weeks:
        at_goal_weight = abs(w - goal_weight) <= 1.0
        at_goal_bf     = abs(current_bf - goal_bf) <= 1.0
        overshooting   = (action == "cut" and current_bf < goal_bf - 0.3)
        if (at_goal_weight and at_goal_bf) or overshooting:
            break

        current_bf = fat / w * 100

        if action == "bulk" and current_bf >= bf_ceiling:
            action = "cut"
        elif action == "cut" and current_bf <= bf_floor:
            action = "bulk"
        if action == "bulk" and w >= goal_weight + 1.5:
            action = "cut"

        phase_weeks = 0
        phase_name  = f"{'Bulk' if action == 'bulk' else 'Cut'} {phase_num}"

        while week_idx < max_weeks:
            current_bf    = fat / w * 100
            should_switch = False

            if action == "bulk" and current_bf >= bf_ceiling:
                should_switch = True
            elif action == "cut" and current_bf <= bf_floor:
                should_switch = True
            elif action == "bulk" and w >= goal_weight + 1.5:
                should_switch = True
            elif phase_weeks >= max_phase_weeks:
                should_switch = True

            at_goal_weight = abs(w - goal_weight) <= 0.5
            at_goal_bf     = abs(current_bf - goal_bf) <= 0.5
            if at_goal_weight and at_goal_bf:
                break

            if should_switch and phase_weeks > 0:
                break

            if action == "bulk":
                lean += bulk_rate * muscle_frac_bulk
                fat  += bulk_rate * (1 - muscle_frac_bulk)
                w    += bulk_rate
            else:
                lean -= cut_rate * muscle_frac_cut
                fat  -= cut_rate * (1 - muscle_frac_cut)
                w    -= cut_rate

            phase_weeks += 1
            week_idx    += 1

        if phase_weeks > 0:
            phases.append({
                "name":  phase_name,
                "type":  action,
                "weeks": phase_weeks,
            })
            if action == "cut":
                phase_num += 1
            action = "cut" if action == "bulk" else "bulk"

    return phases, round(w, 1), round(fat / w * 100, 1)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Starting Stats")
    start_weight   = st.number_input("Start Weight (lbs)", min_value=90.0,  max_value=300.0, value=145.0, step=0.5)
    start_bf       = st.number_input("Start Body Fat %",   min_value=5.0,   max_value=40.0,  value=15.5,  step=0.5)

    st.divider()
    st.header("👤 Profile")
    height_ft      = st.number_input("Height (ft)",         min_value=4,     max_value=7,     value=5)
    height_in_rem  = st.number_input("Height (in)",         min_value=0,     max_value=11,    value=8)
    wrist_in       = st.number_input("Wrist circumference (in)", min_value=4.0, max_value=10.0, value=6.25, step=0.25)
    age            = st.number_input("Age",                 min_value=16,    max_value=80,    value=22)
    training_years = st.number_input("Years of serious training", min_value=0.0, max_value=30.0, value=3.5, step=0.5)

    st.divider()
    st.header("🎯 Goal")
    goal_weight    = st.number_input("Goal Weight (lbs)",   min_value=90.0,  max_value=300.0, value=155.0, step=0.5)
    goal_bf        = st.number_input("Goal Body Fat %",     min_value=5.0,   max_value=40.0,  value=15.0,  step=0.5)
    bf_ceiling     = st.number_input("Max BF% you'll tolerate (ceiling)", min_value=8.0, max_value=35.0, value=17.0, step=0.5)
    bf_floor       = st.number_input("Min BF% you'll cut to (floor)",     min_value=4.0, max_value=25.0, value=15.0, step=0.5)

    st.divider()
    st.header("⚙️ Mode")
    mode = st.radio("Planning mode", ["🤖 Auto Schedule", "🔧 Manual Phases"])

    height_in_total = (height_ft * 12) + height_in_rem
    profile = calculate_profile(
        height_in      = height_in_total,
        weight_lbs     = start_weight,
        bf_pct         = start_bf,
        wrist_in       = wrist_in,
        age            = age,
        training_years = training_years,
    )

    st.divider()
    st.header("📐 Your Profile")
    st.metric("FFMI",            f"{profile['ffmi']}")
    st.metric("FFMI Ceiling",    f"{profile['ffmi_ceiling']}")
    st.metric("% to Ceiling",    f"{profile['ceiling_pct']}%")
    st.metric("Rec. Bulk Rate",  f"{profile['bulk_rate_monthly']} lbs/mo  ({profile['bulk_rate_weekly']} lbs/wk)")
    st.metric("Rec. Cut Rate",   f"{profile['cut_rate_monthly']} lbs/mo  ({profile['cut_rate_weekly']} lbs/wk)")
    st.metric("Muscle % (Bulk)", f"{profile['muscle_frac_bulk']*100:.0f}%")
    st.metric("Muscle % (Cut)",  f"{profile['muscle_frac_cut']*100:.0f}%")

    st.divider()
    st.header("⚙️ Override Rates")
    st.caption("Optional — leave at 0 to use calculated rates.")
    bulk_rate_override = st.number_input("Bulk rate (lbs/week)", min_value=0.0, max_value=2.0, value=0.0, step=0.05)
    cut_rate_override  = st.number_input("Cut rate (lbs/week)",  min_value=0.0, max_value=2.0, value=0.0, step=0.05)

    bulk_rate_final = bulk_rate_override if bulk_rate_override > 0 else profile["bulk_rate_weekly"]
    cut_rate_final  = cut_rate_override  if cut_rate_override  > 0 else profile["cut_rate_weekly"]

    st.divider()
    st.header("🗓️ Start Date")
    start_date = st.date_input("Start date", value=date(2026, 6, 13))

# ── Phase Builder or Auto Schedule ────────────────────────────────────────────
if mode == "🤖 Auto Schedule":
    st.subheader("🤖 Auto-Generated Schedule")
    st.caption("Phases calculated automatically based on your profile and BF guardrails.")

    max_phase_weeks = st.slider("Max weeks per phase", min_value=4, max_value=32, value=20, step=1)

    auto_phases, auto_final_weight, auto_final_bf = auto_schedule(
        start_weight     = start_weight,
        start_bf         = start_bf,
        goal_weight      = goal_weight,
        goal_bf          = goal_bf,
        bf_ceiling       = bf_ceiling,
        bf_floor         = bf_floor,
        profile          = {
            "bulk_rate_weekly":  bulk_rate_final,
            "cut_rate_weekly":   cut_rate_final,
            "muscle_frac_bulk":  profile["muscle_frac_bulk"],
            "muscle_frac_cut":   profile["muscle_frac_cut"],
        },
        max_phase_weeks  = max_phase_weeks,
    )

    total_weeks = sum(p["weeks"] for p in auto_phases)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Weeks",    f"{total_weeks}")
    col2.metric("Est. Final Weight", f"{auto_final_weight} lbs")
    col3.metric("Est. Final BF",  f"{auto_final_bf}%")

    st.markdown("**Generated phases:**")
    header_cols = st.columns([3, 2, 1])
    header_cols[0].markdown("**Phase**")
    header_cols[1].markdown("**Type**")
    header_cols[2].markdown("**Weeks**")
    for p in auto_phases:
        row_cols = st.columns([3, 2, 1])
        row_cols[0].write(p["name"])
        row_cols[1].write(p["type"].capitalize())
        row_cols[2].write(p["weeks"])

    active_phases = auto_phases

else:
    st.subheader("🔧 Manual Phase Builder")
    st.caption("Add phases in order. Each phase is a bulk, cut, or maintenance period.")
    
    if "phases" not in st.session_state:
        st.session_state.phases = [
            {"name": "Lean Bulk I",   "type": "bulk",     "weeks": 18},
            {"name": "Mini Cut I",    "type": "cut",      "weeks": 8},
            {"name": "Lean Bulk II",  "type": "bulk",     "weeks": 20},
            {"name": "Mini Cut II",   "type": "cut",      "weeks": 8},
            {"name": "Lean Bulk III", "type": "bulk",     "weeks": 20},
            {"name": "Final Cut",     "type": "cut",      "weeks": 16},
            {"name": "Maintain",      "type": "maintain", "weeks": 8},
        ]

    phases_to_delete = []
    for i, phase in enumerate(st.session_state.phases):
        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
        with col1:
            st.session_state.phases[i]["name"] = st.text_input(
                f"Phase {i+1} Name", value=phase["name"], key=f"name_{i}", label_visibility="collapsed"
            )
        with col2:
            st.session_state.phases[i]["type"] = st.selectbox(
                "Type", options=["bulk", "cut", "maintain"],
                index=["bulk", "cut", "maintain"].index(phase["type"]),
                key=f"type_{i}", label_visibility="collapsed"
            )
        with col3:
            st.session_state.phases[i]["weeks"] = st.number_input(
                "Weeks", min_value=1, max_value=104, value=phase.get("weeks", phase.get("months", 8)),
                key=f"weeks_{i}", label_visibility="collapsed"
            )
        with col4:
            if st.button("🗑️", key=f"del_{i}", help="Delete this phase"):
                phases_to_delete.append(i)

    for i in sorted(phases_to_delete, reverse=True):
        st.session_state.phases.pop(i)

    col_add, col_reset, _ = st.columns([1, 1, 4])
    with col_add:
        if st.button("➕ Add Phase"):
            st.session_state.phases.append({"name": f"Phase {len(st.session_state.phases)+1}", "type": "bulk", "weeks": 8})
            st.rerun()
    with col_reset:
        if st.button("↺ Reset Phases"):
            del st.session_state.phases
            st.rerun()

    active_phases = st.session_state.phases

# ── Simulate ───────────────────────────────────────────────────────────────────
def simulate(start_weight, start_bf, phases, bulk_rate, bulk_muscle_pct,
             cut_rate, cut_muscle_pct, start_date):
    muscle_frac_bulk = bulk_muscle_pct / 100
    muscle_frac_cut  = cut_muscle_pct  / 100

    w    = start_weight
    lean = start_weight * (1 - start_bf / 100)
    fat  = start_weight * (start_bf / 100)

    rows = [{
        "week": 0, "date": start_date, "phase": "Start", "phase_type": "start",
        "weight": round(w, 1), "lean": round(lean, 1),
        "fat": round(fat, 1), "bf": round(fat / w * 100, 1), "change": 0.0,
    }]

    idx          = 1
    current_date = start_date
    for phase in phases:
        ptype   = phase["type"]
        n_weeks = phase.get("weeks", phase.get("months", 4))
        for _ in range(n_weeks):
            prev_w = w
            if ptype == "bulk":
                lean += bulk_rate * muscle_frac_bulk
                fat  += bulk_rate * (1 - muscle_frac_bulk)
                w    += bulk_rate
            elif ptype == "cut":
                lean -= cut_rate * muscle_frac_cut
                fat  -= cut_rate * (1 - muscle_frac_cut)
                w    -= cut_rate
            else:
                lean += 0.012
                fat  += 0.012
                w    += 0.023

            current_date = current_date + timedelta(weeks=1)
            rows.append({
                "week":       idx,
                "date":       current_date,
                "phase":      phase["name"],
                "phase_type": ptype,
                "weight":     round(w, 1),
                "lean":       round(lean, 1),
                "fat":        round(fat, 1),
                "bf":         round(fat / w * 100, 1),
                "change":     round(w - prev_w, 3),
            })
            idx += 1

    return rows

data = simulate(
    start_weight,
    start_bf,
    active_phases,
    bulk_rate_final,
    profile["muscle_frac_bulk"] * 100,
    cut_rate_final,
    profile["muscle_frac_cut"] * 100,
    start_date,
)

# ── Summary Stats ──────────────────────────────────────────────────────────────
final        = data[-1]
start        = data[0]
total_weeks  = final["week"]
lean_gained  = round(final["lean"] - start["lean"], 1)
peak_bf_row  = max(data, key=lambda r: r["bf"])
on_track     = abs(final["weight"] - goal_weight) <= 2 and abs(final["bf"] - goal_bf) <= 1.5

st.divider()
st.subheader("📊 Projection Summary")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Final Weight",     f"{final['weight']} lbs", f"{round(final['weight']-start['weight'],1):+.1f} lbs")
c2.metric("Final BF%",        f"{final['bf']}%",        f"{round(final['bf']-start['bf'],1):+.1f}%")
c3.metric("Lean Mass Added",  f"+{lean_gained} lbs")
c4.metric("Peak BF%",         f"{peak_bf_row['bf']}%",  f"Week {peak_bf_row['week']}")
with c5:
    st.metric("Total Weeks", f"{total_weeks} wks")
    st.caption(f"≈ {round(total_weeks/4.33, 1)} months")

goal_weight_gap = round(final["weight"] - goal_weight, 1)
goal_bf_gap     = round(final["bf"] - goal_bf, 1)
if on_track:
    st.success(f"✅ On track — projected finish: {final['weight']} lbs @ {final['bf']}% BF")
else:
    st.warning(
        f"⚠️ Misses goal by **{goal_weight_gap:+.1f} lbs** and **{goal_bf_gap:+.1f}% BF**. "
        f"Adjust phases or rates."
    )

# ── Charts ─────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Charts")

COLORS = {"bulk": "#10b981", "cut": "#ef4444", "maintain": "#f59e0b", "start": "#6366f1"}

dates       = [r["date"]   for r in data]
weights     = [r["weight"] for r in data]
leans       = [r["lean"]   for r in data]
bfs         = [r["bf"]     for r in data]
phases_list = [r["phase"]  for r in data]

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Scale Weight & Lean Mass (lbs)", "Body Fat %"),
    vertical_spacing=0.12,
    row_heights=[0.6, 0.4],
)

fig.add_trace(go.Scatter(
    x=dates, y=weights, mode="lines+markers",
    name="Scale Weight",
    line=dict(color="#10b981", width=2.5),
    marker=dict(size=3),
    hovertemplate="<b>%{text}</b><br>Weight: %{y} lbs<extra></extra>",
    text=phases_list,
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=dates, y=leans, mode="lines",
    name="Lean Mass",
    line=dict(color="#6366f1", width=1.8, dash="dot"),
    hovertemplate="<b>%{text}</b><br>Lean: %{y} lbs<extra></extra>",
    text=phases_list,
), row=1, col=1)

fig.add_hline(y=goal_weight, line_dash="dash", line_color="rgba(255,255,255,0.27)",
              annotation_text=f"Goal: {goal_weight} lbs", row=1, col=1)

fig.add_trace(go.Scatter(
    x=dates, y=bfs, mode="lines+markers",
    name="Body Fat %",
    line=dict(color="#f59e0b", width=2.5),
    marker=dict(size=3),
    hovertemplate="<b>%{text}</b><br>BF: %{y}%<extra></extra>",
    text=phases_list,
), row=2, col=1)

fig.add_hline(y=goal_bf, line_dash="dash", line_color="rgba(255,255,255,0.27)",
              annotation_text=f"Goal: {goal_bf}%", row=2, col=1)

fig.add_hline(y=bf_ceiling, line_dash="dot", line_color="rgba(239,68,68,0.4)",
              annotation_text=f"BF Ceiling: {bf_ceiling}%", row=2, col=1)

fig.add_hline(y=bf_floor, line_dash="dot", line_color="rgba(16,185,129,0.4)",
              annotation_text=f"BF Floor: {bf_floor}%", row=2, col=1)

phase_starts = {}
for r in data:
    if r["phase"] not in phase_starts:
        phase_starts[r["phase"]] = (r["date"], r["phase_type"])

phase_names = list(phase_starts.keys())
for idx_p, pname in enumerate(phase_names):
    pdate, ptype = phase_starts[pname]
    end_date = phase_starts[phase_names[idx_p+1]][0] if idx_p+1 < len(phase_names) else dates[-1]
    color = COLORS.get(ptype, "#888888")
    for row in [1, 2]:
        fig.add_vrect(
            x0=pdate, x1=end_date,
            fillcolor=color, opacity=0.07, line_width=0,
            row=row, col=1,
        )

fig.update_layout(
    height=600,
    paper_bgcolor="#0f1624",
    plot_bgcolor="#080b14",
    font=dict(color="#e2e8f0"),
    legend=dict(bgcolor="#0f1624", bordercolor="#1e293b", borderwidth=1),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=40, b=10),
)
fig.update_xaxes(gridcolor="#1e293b", showgrid=True)
fig.update_yaxes(gridcolor="#1e293b", showgrid=True)

st.plotly_chart(fig, use_container_width=True)

# ── Week Table ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📅 Week-by-Week Breakdown")

with st.expander("Show full table", expanded=False):
    df = pd.DataFrame(data)
    df["date"]   = df["date"].astype(str)
    df["change"] = df["change"].apply(lambda x: f"+{x:.2f}" if x > 0 else (f"{x:.2f}" if x != 0 else "—"))
    df.columns   = ["Week", "Date", "Phase", "Type", "Weight (lbs)", "Lean (lbs)", "Fat (lbs)", "BF%", "Δ Weight"]
    df = df.drop(columns=["Type"])
    st.dataframe(df, use_container_width=True, hide_index=True)
