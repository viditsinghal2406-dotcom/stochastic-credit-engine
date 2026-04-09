"""
STPA - Database Layer
======================
SQLite persistence for borrower profiles and simulation results.

Tables:
    borrowers          — Borrower profiles and parameters
    simulation_results — STPA risk output per run
    stress_tests       — Stress test scenario results
    audit_log          — API request/response audit trail
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

DB_PATH = Path(__file__).parent / "stpa.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS borrowers (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            borrower_id     TEXT    NOT NULL UNIQUE,
            health_score    REAL    NOT NULL,
            long_run_mean   REAL    NOT NULL,
            reversion_speed REAL    NOT NULL,
            volatility      REAL    NOT NULL,
            initial_state   TEXT    NOT NULL,
            risk_tier       TEXT    NOT NULL,
            created_at      TEXT    DEFAULT (datetime('now')),
            updated_at      TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS simulation_results (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            borrower_id          TEXT    NOT NULL,
            pd_score             REAL    NOT NULL,
            pd_probability       REAL    NOT NULL,
            risk_tier            TEXT    NOT NULL,
            expected_ttd_months  REAL,
            survival_12m         REAL,
            survival_24m         REAL,
            base_pd              REAL,
            stressed_pd          REAL,
            n_simulations        INTEGER,
            horizon_months       INTEGER,
            key_drivers          TEXT,   -- JSON array
            survival_curve       TEXT,   -- JSON array
            run_at               TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (borrower_id) REFERENCES borrowers(borrower_id)
        );

        CREATE TABLE IF NOT EXISTS stress_tests (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            borrower_id TEXT NOT NULL,
            scenario    TEXT NOT NULL,
            pd_score    REAL NOT NULL,
            risk_tier   TEXT NOT NULL,
            run_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint    TEXT NOT NULL,
            borrower_id TEXT,
            request     TEXT,   -- JSON
            response    TEXT,   -- JSON (summary only)
            status_code INTEGER,
            duration_ms REAL,
            logged_at   TEXT DEFAULT (datetime('now'))
        );
    """)

    conn.commit()
    conn.close()


def save_borrower(profile: Dict[str, Any]) -> bool:
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO borrowers
                (borrower_id, health_score, long_run_mean, reversion_speed,
                 volatility, initial_state, risk_tier)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(borrower_id) DO UPDATE SET
                health_score=excluded.health_score,
                long_run_mean=excluded.long_run_mean,
                updated_at=datetime('now')
        """, (
            profile["borrower_id"],
            profile["health_score"],
            profile["long_run_mean"],
            profile["reversion_speed"],
            profile["volatility"],
            profile["initial_state"],
            profile["risk_tier"],
        ))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def save_simulation_result(result: Dict[str, Any]) -> int:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO simulation_results
                (borrower_id, pd_score, pd_probability, risk_tier,
                 expected_ttd_months, survival_12m, survival_24m,
                 base_pd, stressed_pd, n_simulations, horizon_months,
                 key_drivers, survival_curve)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result["borrower_id"],
            result["pd_score"],
            result["pd_probability"],
            result["risk_tier"],
            result.get("expected_ttd_months"),
            result.get("survival_12m"),
            result.get("survival_24m"),
            result.get("base_pd"),
            result.get("stressed_pd"),
            result.get("n_simulations"),
            result.get("horizon_months"),
            json.dumps(result.get("key_drivers", [])),
            json.dumps(result.get("survival_curve", [])),
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_borrower_history(borrower_id: str) -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM simulation_results
        WHERE borrower_id = ?
        ORDER BY run_at DESC
    """, (borrower_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_portfolio_summary() -> List[Dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT borrower_id,
               pd_score,
               risk_tier,
               run_at
        FROM simulation_results
        WHERE id IN (
            SELECT MAX(id) FROM simulation_results GROUP BY borrower_id
        )
        ORDER BY pd_score DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Initialize on import
init_db()
