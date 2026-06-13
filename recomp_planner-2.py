import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
from dateutil.relativedelta import relativedelta

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recomp Planner",
    page_icon="💪",
    layout="wide",
)

st.title("💪 Body Recomposition Planner")
st.caption("Model your bulk/cut cycles and see month-by-month projections.")

# ── Sidebar: Global Settings ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Starting Stats")
    start_weight = st.number_input("Start Weight (lbs)", min_value=90.0, max_value=300.0, value=145.0, step=0.5)
    start_bf     = st.number_input("Start Body Fat %",   min_value=5.0,  max_value=40.0,  value=15.5, step=0.5)

    st.divider()
    st.header("🎯 Goal")
    goal_weight = st.number_input("Goal Weight (lbs)", min_value=90.0, max_value=300.0, value=155.0, step=0.5)
    goal_bf     = st.number_input("Goal Body Fat %",   min_value=5.0,  max_value=40.0,  value=15.0, step=0.5)

    st.divider()
    st.header("⚙️ Assumptions")
    st.markdown("**Bulk**")
    bulk_muscle_pct = st.slider(
        "% of bulk gain that is muscle",
        min_value=20, max_value=80, value=47, step=1,
        help="Beginners ~60%, intermediates ~45-50%, advanced ~35-45%"
    )
    bulk_rate = st.number_input(
        "Monthly scale gain during bulk (lbs)",
        min_value=0.1, max_value=5.0, value=1.1, step=0.1,
        help="0.5–1.0 = lean bulk. Above 1.5 = aggressive bulk."
    )

    st.markdown("**Cut**")
    cut_muscle_pct = st.slider(
        "% of cut loss that is muscle",
        min_value=5, max_value=40, value=10, step=1,
        help="With high protein + training: 8-12%. Aggressive cut: up to 20%."
    )
    cut_rate = st.number_input(
        "Monthly scale loss during cut (lbs)",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="0.5–1.0 = conservative. Above 1.5 = aggressive."
    )

    st.divider()
    st.header("🗓️ Start Date")
    start_date = st.date_input("Start date", value=date(2026, 6, 13))

# ── Phase Builder ──────────────────────────────────────────────────────────────
st.subheader("🔧 Build Your Phase Plan")
st.caption("Add phases in order. Each phase is either a bulk, cut, or maintenance period.")

# Store phases in session state so they persist between reruns
if "phases" not in st.session_state:
    st.session_state.phases = [
        {"name": "Lean Bulk I",   "type": "bulk",     "months": 4},
        {"name": "Mini Cut I",    "type": "cut",      "months": 2},
        {"name": "Lean Bulk II",  "type": "bulk",     "months": 5},
        {"name": "Mini Cut II",   "type": "cut",      "months": 2},
        {"name": "Lean Bulk III", "type": "bulk",     "months": 5},
        {"name": "Final Cut",     "type": "cut",      "months": 4},
        {"name": "Maintain",      "type": "maintain", "months": 2},
    ]

# Display and edit each phase
phases_to_delete = []
for i, phase in enumerate(st.session_state.phases):
    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
    with col1:
        st.session_state.phases[i]["name"] = st.text_input(
            f"Phase {i+1} Name", value=phase["name"], key=f"name_{i}", label_visibility="collapsed"
        )
    with col2:
        st.session_state.phases[i]["type"] = st.selectbox(
            "Type", options=["bulk", "cut", "maintain"], index=["bulk","cut","maintain"].index(phase["type"]),
            key=f"type_{i}", label_visibility="collapsed"
        )
    with col3:
        st.session_state.phases[i]["months"] = st.number_input(
            "Months", min_value=1, max_value=24, value=phase["months"],
            key=f"months_{i}", label_visibility="collapsed"
        )
    with col4:
        if st.button("🗑️", key=f"del_{i}", help="Delete this phase"):
            phases_to_delete.append(i)

for i in sorted(phases_to_delete, reverse=True):
    st.session_state.phases.pop(i)

# Add phase button
col_add, col_reset, _ = st.columns([1, 1, 4])
with col_add:
    if st.button("➕ Add Phase"):
        st.session_state.phases.append({"name": f"Phase {len(st.session_state.phases)+1}", "type": "bulk", "months": 3})
        st.rerun()
with col_reset:
    if st.button("↺ Reset Phases"):
        del st.session_state.phases
        st.rerun()

# ── Simulation ─────────────────────────────────────────────────────────────────
def simulate(start_weight, start_bf, phases, bulk_rate, bulk_muscle_pct,
             cut_rate, cut_muscle_pct, start_date):
    muscle_frac_bulk = bulk_muscle_pct / 100
    muscle_frac_cut  = cut_muscle_pct  / 100

    w    = start_weight
    lean = start_weight * (1 - start_bf / 100)
    fat  = start_weight * (start_bf / 100)

    rows = [{
        "month": 0, "date": start_date, "phase": "Start", "phase_type": "start",
        "weight": round(w, 1), "lean": round(lean, 1),
        "fat": round(fat, 1),  "bf": round(fat / w * 100, 1), "change": 0.0,
    }]

    idx = 1
    current_date = start_date
    for phase in phases:
        ptype   = phase["type"]
        n_months = phase["months"]
        for _ in range(n_months):
            prev_w = w
            if ptype == "bulk":
                gain = bulk_rate
                lean += gain * muscle_frac_bulk
                fat  += gain * (1 - muscle_frac_bulk)
                w    += gain
            elif ptype == "cut":
                loss = cut_rate
                lean -= loss * muscle_frac_cut
                fat  -= loss * (1 - muscle_frac_cut)
                w    -= loss
            else:  # maintain
                lean += 0.05
                fat  += 0.05
                w    += 0.1

            current_date = current_date + relativedelta(months=1)
            rows.append({
                "month":      idx,
                "date":       current_date,
                "phase":      phase["name"],
                "phase_type": ptype,
                "weight":     round(w, 1),
                "lean":       round(lean, 1),
                "fat":        round(fat, 1),
                "bf":         round(fat / w * 100, 1),
                "change":     round(w - prev_w, 2),
            })
            idx += 1

    return rows

data = simulate(
    start_weight, start_bf,
    st.session_state.phases,
    bulk_rate, bulk_muscle_pct,
    cut_rate, cut_muscle_pct,
    start_date,
)

# ── Summary Stats ──────────────────────────────────────────────────────────────
final  = data[-1]
start  = data[0]
total_months = final["month"]
lean_gained  = round(final["lean"] - start["lean"], 1)
fat_change   = round(final["fat"]  - start["fat"],  1)
peak_bf_row  = max(data, key=lambda r: r["bf"])

goal_lean = goal_weight * (1 - goal_bf / 100)
projected_lean = final["lean"]
on_track = abs(final["weight"] - goal_weight) <= 2 and abs(final["bf"] - goal_bf) <= 1.5

st.divider()
st.subheader("📊 Projection Summary")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Final Weight",    f"{final['weight']} lbs",  f"{round(final['weight']-start['weight'],1):+.1f} lbs")
c2.metric("Final BF%",      f"{final['bf']}%",          f"{round(final['bf']-start['bf'],1):+.1f}%")
c3.metric("Lean Mass Added", f"+{lean_gained} lbs")
c4.metric("Peak BF%",       f"{peak_bf_row['bf']}%",    f"Month {peak_bf_row['month']}")
c5.metric("Total Months",   f"{total_months}")

# Goal gap
goal_weight_gap = round(final["weight"] - goal_weight, 1)
goal_bf_gap     = round(final["bf"] - goal_bf, 1)
if on_track:
    st.success(f"✅ On track to hit your goal. Final: {final['weight']} lbs @ {final['bf']}% BF")
else:
    st.warning(
        f"⚠️ Projection misses goal by **{goal_weight_gap:+.1f} lbs** weight "
        f"and **{goal_bf_gap:+.1f}%** body fat. Adjust your phases or rates in the sidebar."
    )

# ── Charts ─────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📈 Charts")

COLORS = {"bulk": "#10b981", "cut": "#ef4444", "maintain": "#f59e0b", "start": "#6366f1"}

dates   = [r["date"]   for r in data]
weights = [r["weight"] for r in data]
leans   = [r["lean"]   for r in data]
bfs     = [r["bf"]     for r in data]
phases_list = [r["phase"] for r in data]

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Scale Weight & Lean Mass (lbs)", "Body Fat %"),
    vertical_spacing=0.12,
    row_heights=[0.6, 0.4],
)

# Weight line
fig.add_trace(go.Scatter(
    x=dates, y=weights, mode="lines+markers",
    name="Scale Weight",
    line=dict(color="#10b981", width=2.5),
    marker=dict(size=5),
    hovertemplate="<b>%{text}</b><br>Weight: %{y} lbs<extra></extra>",
    text=phases_list,
), row=1, col=1)

# Lean mass line
fig.add_trace(go.Scatter(
    x=dates, y=leans, mode="lines",
    name="Lean Mass",
    line=dict(color="#6366f1", width=1.8, dash="dot"),
    hovertemplate="<b>%{text}</b><br>Lean: %{y} lbs<extra></extra>",
    text=phases_list,
), row=1, col=1)

# Goal weight line
fig.add_hline(y=goal_weight, line_dash="dash", line_color="rgba(255,255,255,0.27)",
              annotation_text=f"Goal: {goal_weight} lbs", row=1, col=1)

# BF line
fig.add_trace(go.Scatter(
    x=dates, y=bfs, mode="lines+markers",
    name="Body Fat %",
    line=dict(color="#f59e0b", width=2.5),
    marker=dict(size=5),
    hovertemplate="<b>%{text}</b><br>BF: %{y}%<extra></extra>",
    text=phases_list,
), row=2, col=1)

# Goal BF line
fig.add_hline(y=goal_bf, line_dash="dash", line_color="rgba(255,255,255,0.27)",
              annotation_text=f"Goal: {goal_bf}%", row=2, col=1)

# Phase background shading
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

# ── Month Table ────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📅 Month-by-Month Breakdown")

with st.expander("Show full table", expanded=False):
    import pandas as pd
    df = pd.DataFrame(data)
    df["date"]   = df["date"].astype(str)
    df["change"] = df["change"].apply(lambda x: f"+{x:.1f}" if x > 0 else (f"{x:.1f}" if x != 0 else "—"))
    df.columns   = ["Month", "Date", "Phase", "Type", "Weight (lbs)", "Lean (lbs)", "Fat (lbs)", "BF%", "Δ Weight"]
    df = df.drop(columns=["Type"])
    st.dataframe(df, use_container_width=True, hide_index=True)
