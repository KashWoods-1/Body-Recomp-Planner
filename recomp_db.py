"""SQLite persistence for the recomp planner.

Design notes (the WHY, since this doubles as a learning artifact):

- Five tables. `weigh_ins` is ground truth. `plans` snapshots the INPUTS of a
  generated plan; `plan_phases` and `predictions` are its children (what the
  model said would happen, week by week). `calibration_events` records each
  fit of your personal multipliers. Predicted-vs-actual analysis is then a
  JOIN between `predictions` and `weigh_ins` — the whole point of the schema.

- `weigh_ins.date` has a UNIQUE constraint: one weigh-in per day. Re-saving
  replaces the table contents inside a transaction (the app's editor is the
  source of truth, so deletions propagate).

- `predictions` uses a composite primary key (plan_id, week_num): a week of a
  plan is identified by which plan it belongs to plus its position. No
  surrogate id needed — the natural key IS the identity.

- Dates are stored as ISO-8601 TEXT ('2026-07-05'). SQLite has no DATE type;
  ISO strings sort correctly and julianday()/date() functions parse them.

- Foreign keys are ON per-connection (SQLite default is OFF, a classic trap).
"""

import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS weigh_ins (
    id          INTEGER PRIMARY KEY,
    date        TEXT NOT NULL UNIQUE,          -- ISO 'YYYY-MM-DD'
    weight_lbs  REAL NOT NULL CHECK (weight_lbs BETWEEN 50 AND 500),
    source      TEXT NOT NULL DEFAULT 'app'    -- 'app' | 'csv_import' | ...
);

CREATE TABLE IF NOT EXISTS plans (
    id              INTEGER PRIMARY KEY,
    created_at      TEXT NOT NULL,             -- ISO timestamp
    start_weight    REAL NOT NULL,
    start_bf        REAL NOT NULL,
    goal_weight     REAL NOT NULL,
    goal_bf         REAL NOT NULL,
    bf_ceiling      REAL NOT NULL,
    bf_floor        REAL NOT NULL,
    consistency     INTEGER NOT NULL,
    recomp_allowed  INTEGER NOT NULL,          -- 0/1 (SQLite has no BOOLEAN)
    headline_weeks  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS plan_phases (
    plan_id     INTEGER NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,              -- 0-based position in the plan
    phase_type  TEXT NOT NULL CHECK (phase_type IN ('bulk','cut','maintain')),
    weeks       INTEGER NOT NULL,
    PRIMARY KEY (plan_id, seq)
);

CREATE TABLE IF NOT EXISTS predictions (
    plan_id     INTEGER NOT NULL REFERENCES plans(id) ON DELETE CASCADE,
    week_num    INTEGER NOT NULL,
    pred_date   TEXT NOT NULL,
    phase_type  TEXT NOT NULL,
    weight      REAL NOT NULL,
    lean        REAL NOT NULL,
    fat         REAL NOT NULL,
    bf          REAL NOT NULL,
    PRIMARY KEY (plan_id, week_num)
);

CREATE TABLE IF NOT EXISTS calibration_events (
    id              INTEGER PRIMARY KEY,
    run_at          TEXT NOT NULL,
    plan_id         INTEGER REFERENCES plans(id) ON DELETE SET NULL,
    phase_type      TEXT NOT NULL CHECK (phase_type IN ('bulk','cut')),
    multiplier      REAL NOT NULL,
    standard_error  REAL,
    n_points        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(pred_date);
"""


def _connect(db_path):
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db(db_path):
    """Create tables if absent. Safe to call every app start."""
    with _connect(db_path) as con:
        con.executescript(SCHEMA)


def load_weigh_ins(db_path):
    """Return list of (iso_date_str, weight) sorted by date. [] if none/no db."""
    if not Path(db_path).exists():
        return []
    with _connect(db_path) as con:
        rows = con.execute(
            "SELECT date, weight_lbs FROM weigh_ins ORDER BY date").fetchall()
    return rows


def save_weigh_ins(db_path, entries, source="app"):
    """Full-sync the weigh_ins table to `entries` (list of (date, weight)).
    The app's editor is the source of truth: rows deleted there disappear
    here too. One transaction, so a crash can't leave a half-synced table."""
    init_db(db_path)
    with _connect(db_path) as con:
        con.execute("DELETE FROM weigh_ins")
        con.executemany(
            "INSERT INTO weigh_ins (date, weight_lbs, source) VALUES (?, ?, ?)",
            [(d.isoformat() if hasattr(d, "isoformat") else str(d),
              float(w), source) for d, w in entries])
    return len(entries)


def save_plan_snapshot(db_path, inputs, phases, prediction_rows, calib=None):
    """Persist one generated plan: inputs, phase list, full weekly projection,
    and (optionally) the calibration state at the moment of saving.
    Returns the new plan id."""
    init_db(db_path)
    with _connect(db_path) as con:
        cur = con.execute(
            """INSERT INTO plans (created_at, start_weight, start_bf,
                 goal_weight, goal_bf, bf_ceiling, bf_floor, consistency,
                 recomp_allowed, headline_weeks)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now().isoformat(timespec="seconds"),
             inputs["start_weight"], inputs["start_bf"],
             inputs["goal_weight"], inputs["goal_bf"],
             inputs["bf_ceiling"], inputs["bf_floor"],
             inputs["consistency"], int(inputs["recomp_allowed"]),
             inputs["headline_weeks"]))
        plan_id = cur.lastrowid

        con.executemany(
            "INSERT INTO plan_phases (plan_id, seq, phase_type, weeks) VALUES (?,?,?,?)",
            [(plan_id, i, p["type"], p.get("weeks", 4))
             for i, p in enumerate(phases)])

        con.executemany(
            """INSERT INTO predictions
                 (plan_id, week_num, pred_date, phase_type, weight, lean, fat, bf)
               VALUES (?,?,?,?,?,?,?,?)""",
            [(plan_id, r["week"], r["date"].isoformat(), r["phase_type"],
              r["weight"], r["lean"], r["fat"], r["bf"])
             for r in prediction_rows])

        if calib is not None:
            now = datetime.now().isoformat(timespec="seconds")
            for pt in ("bulk", "cut"):
                if calib.get(f"{pt}_n", 0) > 0:
                    con.execute(
                        """INSERT INTO calibration_events
                             (run_at, plan_id, phase_type, multiplier,
                              standard_error, n_points)
                           VALUES (?,?,?,?,?,?)""",
                        (now, plan_id, pt, calib[f"{pt}_mult"],
                         calib.get(f"{pt}_se"), calib[f"{pt}_n"]))
    return plan_id


def count_plans(db_path):
    if not Path(db_path).exists():
        return 0
    with _connect(db_path) as con:
        return con.execute("SELECT COUNT(*) FROM plans").fetchone()[0]


def run_query(db_path, sql, params=()):
    """Convenience for analysis scripts: returns (column_names, rows)."""
    with _connect(db_path) as con:
        cur = con.execute(sql, params)
        cols = [c[0] for c in cur.description] if cur.description else []
        return cols, cur.fetchall()
