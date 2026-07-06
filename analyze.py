"""Run a .sql file against the recomp database and pretty-print results.

Usage:
    python analyze.py              # runs queries.sql
    python analyze.py myfile.sql   # runs another query file

Deliberately tiny: the point of Phase 2 is writing SQL in .sql files, not
building tooling. Resist the urge to grow this script.
"""
import sys
from pathlib import Path
import recomp_db

DB = Path(__file__).parent / "recomp_data.db"

def main():
    sql_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("queries.sql")
    sql = sql_file.read_text()
    # split on semicolons, keep only statements that actually select something
    stmts = [q.strip() for q in sql.split(";") if q.strip()
             and not all(ln.strip().startswith("--") for ln in q.strip().splitlines())]
    for i, stmt in enumerate(stmts, 1):
        cols, rows = recomp_db.run_query(DB, stmt)
        print(f"\n── statement {i} ── ({len(rows)} rows)")
        if cols:
            widths = [max(len(str(c)), *(len(str(r[j])) for r in rows)) if rows
                      else len(str(c)) for j, c in enumerate(cols)]
            print("  " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)))
            print("  " + "-+-".join("-" * w for w in widths))
            for r in rows:
                print("  " + " | ".join(str(v).ljust(w) for v, w in zip(r, widths)))

if __name__ == "__main__":
    main()
