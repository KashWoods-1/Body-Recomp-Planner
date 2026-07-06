-- ═══════════════════════════════════════════════════════════════════════════
-- QUERY 1: 7-day rolling average weight (the trend line, rebuilt in SQL)
-- ═══════════════════════════════════════════════════════════════════════════
-- Concepts: window functions, RANGE frames, julianday() date math.
--
-- Why RANGE and not ROWS: ROWS BETWEEN 6 PRECEDING counts *rows*, so if you
-- skip weigh-in days it silently averages over more calendar time than a week.
-- RANGE over julianday(date) is calendar-aware: "everything within the last 7
-- days," however many weigh-ins that is. This distinction is a classic
-- interview probe.
--
-- julianday() turns the ISO date into a number so RANGE can do arithmetic on
-- it (SQLite RANGE frames need a numeric ORDER BY expression).

SELECT
    date,
    weight_lbs,
    ROUND(
        AVG(weight_lbs) OVER (
            ORDER BY julianday(date)
            RANGE BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 2)                                   AS rolling_7d_avg,
    COUNT(*) OVER (
            ORDER BY julianday(date)
            RANGE BETWEEN 6 PRECEDING AND CURRENT ROW
        )                                       AS days_in_window
FROM weigh_ins
ORDER BY date;

-- Your turn next (Phase 2 ladder, in order):
--   2. Weekly rate of change per phase        (LAG, GROUP BY, date math)
--   3. Logging streaks & gaps                 (gaps-and-islands)
--   4. Predicted vs actual                    (JOIN predictions↔weigh_ins, CTEs)
--   5. Calibration confidence over time       (calibration_events)
--   6. Drift detection                        (rolling residuals, flags)
