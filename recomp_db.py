"""Database layer for the recomp planner — works against BOTH:

  - SQLite (a local file, zero setup) when no DATABASE_URL is configured
  - Postgres (e.g. a free Neon database) when DATABASE_URL is set in
    Streamlit Cloud secrets — this is what makes the deployed app's data
    survive redeploys.

Built on SQLAlchemy Core so the same table definitions and functions run on
both engines. Schema design notes:

  - `weigh_ins.date` UNIQUE: one weigh-in per day; re-saving full-syncs the
    table (the app editor is the source of truth, deletions propagate).
  - `predictions` composite primary key (plan_id, week_num): a week of a plan
    is identified by which plan it belongs to plus its position.
  - Foreign keys cascade: deleting a plan removes its phases/predictions.
    (SQLite ships with FK enforcement OFF; the connect hook below turns it on.)
"""

from datetime import datetime, date as date_type

from sqlalchemy import (create_engine, event, text, MetaData, Table, Column,
                        Integer, Float, Text, Date, DateTime, ForeignKey)

metadata = MetaData()

weigh_ins = Table(
    "weigh_ins", metadata,
    Column("id", Integer, primary_key=True),
    Column("date", Date, nullable=False, unique=True),
    Column("weight_lbs", Float, nullable=False),
    Column("bf", Float, nullable=True),        # optional: DEXA/measurement days
    Column("waist_in", Float, nullable=True),  # optional: weekly tape measurement
    Column("source", Text, nullable=False, server_default="app"),
)

plans = Table(
    "plans", metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime, nullable=False),
    Column("start_weight", Float, nullable=False),
    Column("start_bf", Float, nullable=False),
    Column("goal_weight", Float, nullable=False),
    Column("goal_bf", Float, nullable=False),
    Column("bf_ceiling", Float, nullable=False),
    Column("bf_floor", Float, nullable=False),
    Column("consistency", Integer, nullable=False),
    Column("recomp_allowed", Integer, nullable=False),
    Column("headline_weeks", Integer, nullable=False),
)

plan_phases = Table(
    "plan_phases", metadata,
    Column("plan_id", Integer,
           ForeignKey("plans.id", ondelete="CASCADE"), primary_key=True),
    Column("seq", Integer, primary_key=True),
    Column("phase_type", Text, nullable=False),
    Column("weeks", Integer, nullable=False),
)

predictions = Table(
    "predictions", metadata,
    Column("plan_id", Integer,
           ForeignKey("plans.id", ondelete="CASCADE"), primary_key=True),
    Column("week_num", Integer, primary_key=True),
    Column("pred_date", Date, nullable=False),
    Column("phase_type", Text, nullable=False),
    Column("weight", Float, nullable=False),
    Column("lean", Float, nullable=False),
    Column("fat", Float, nullable=False),
    Column("bf", Float, nullable=False),
)

calibration_events = Table(
    "calibration_events", metadata,
    Column("id", Integer, primary_key=True),
    Column("run_at", DateTime, nullable=False),
    Column("plan_id", Integer,
           ForeignKey("plans.id", ondelete="SET NULL"), nullable=True),
    Column("phase_type", Text, nullable=False),
    Column("multiplier", Float, nullable=False),
    Column("standard_error", Float, nullable=True),
    Column("n_points", Integer, nullable=False),
)


def get_engine(url):
    """url: 'sqlite:///path/to/file.db' or a Postgres URL.
    Neon hands out 'postgres://' or 'postgresql://' strings; SQLAlchemy 2.x
    needs the explicit driver, so we normalize. pool_pre_ping revives
    connections that Neon's autosuspend has quietly closed."""
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    eng = create_engine(url, pool_pre_ping=True)
    if eng.dialect.name == "sqlite":
        @event.listens_for(eng, "connect")
        def _enable_fk(dbapi_con, _):
            dbapi_con.execute("PRAGMA foreign_keys=ON")
    return eng


def init_db(engine):
    """Create any missing tables, and apply tiny in-place migrations for
    columns added after a table already exists (create_all never alters
    existing tables). Safe to call on every app start."""
    metadata.create_all(engine)
    from sqlalchemy import inspect
    cols = {c["name"] for c in inspect(engine).get_columns("weigh_ins")}
    with engine.begin() as con:
        if "bf" not in cols:
            con.execute(text("ALTER TABLE weigh_ins ADD COLUMN bf REAL"))
        if "waist_in" not in cols:
            con.execute(text("ALTER TABLE weigh_ins ADD COLUMN waist_in REAL"))


def backend_name(engine):
    """'sqlite' or 'postgresql' — the app shows this so the user always knows
    whether saves are going somewhere durable."""
    return engine.dialect.name


def load_weigh_ins(engine):
    """Return list of (date, weight) sorted by date."""
    init_db(engine)
    with engine.connect() as con:
        rows = con.execute(
            weigh_ins.select().order_by(weigh_ins.c.date)).fetchall()
    return [(r.date, r.weight_lbs, r.bf, r.waist_in) for r in rows]


def save_weigh_ins(engine, entries, source="app"):
    """Full-sync weigh_ins to `entries` (list of (date, weight)) in one
    transaction — deletions in the app editor propagate here."""
    init_db(engine)
    payload = []
    for row in entries:
        d, w = row[0], row[1]
        bf = row[2] if len(row) > 2 else None
        waist = row[3] if len(row) > 3 else None
        if isinstance(d, str):
            d = date_type.fromisoformat(d[:10])
        elif hasattr(d, "date") and not isinstance(d, date_type):
            d = d.date()   # pandas Timestamp / datetime -> date
        payload.append({"date": d, "weight_lbs": float(w),
                        "bf": (float(bf) if bf is not None else None),
                        "waist_in": (float(waist) if waist is not None else None),
                        "source": source})
    with engine.begin() as con:
        con.execute(weigh_ins.delete())
        if payload:
            con.execute(weigh_ins.insert(), payload)
    return len(payload)


def save_plan_snapshot(engine, inputs, phases, prediction_rows, calib=None):
    """Persist one generated plan: inputs, phases, full weekly projection, and
    (optionally) the calibration state at save time. Returns the plan id."""
    init_db(engine)
    now = datetime.now().replace(microsecond=0)
    with engine.begin() as con:
        res = con.execute(plans.insert().values(
            created_at=now,
            start_weight=inputs["start_weight"], start_bf=inputs["start_bf"],
            goal_weight=inputs["goal_weight"], goal_bf=inputs["goal_bf"],
            bf_ceiling=inputs["bf_ceiling"], bf_floor=inputs["bf_floor"],
            consistency=inputs["consistency"],
            recomp_allowed=int(inputs["recomp_allowed"]),
            headline_weeks=inputs["headline_weeks"]))
        plan_id = res.inserted_primary_key[0]

        con.execute(plan_phases.insert(), [
            {"plan_id": plan_id, "seq": i,
             "phase_type": p["type"], "weeks": p.get("weeks", 4)}
            for i, p in enumerate(phases)])

        con.execute(predictions.insert(), [
            {"plan_id": plan_id, "week_num": r["week"], "pred_date": r["date"],
             "phase_type": r["phase_type"], "weight": r["weight"],
             "lean": r["lean"], "fat": r["fat"], "bf": r["bf"]}
            for r in prediction_rows])

        if calib is not None:
            for pt in ("bulk", "cut"):
                if calib.get(f"{pt}_n", 0) > 0:
                    con.execute(calibration_events.insert().values(
                        run_at=now, plan_id=plan_id, phase_type=pt,
                        multiplier=calib[f"{pt}_mult"],
                        standard_error=calib.get(f"{pt}_se"),
                        n_points=calib[f"{pt}_n"]))
    return plan_id


def count_plans(engine):
    init_db(engine)
    with engine.connect() as con:
        return con.execute(text("SELECT COUNT(*) FROM plans")).scalar()


def run_query(engine, sql, params=None):
    """For analysis scripts: returns (column_names, rows)."""
    with engine.connect() as con:
        cur = con.execute(text(sql), params or {})
        cols = list(cur.keys()) if cur.returns_rows else []
        rows = cur.fetchall() if cur.returns_rows else []
        con.commit()
    return cols, rows
