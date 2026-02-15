"""SQLite helpers for pre-compute benchmark results."""

from __future__ import annotations

import sqlite3
from typing import Any

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS phspectra (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    beta         REAL NOT NULL,
    n_spectra    INTEGER NOT NULL,
    total_time_s REAL NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS phspectra_components (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    INTEGER NOT NULL REFERENCES phspectra(run_id),
    xpos      INTEGER NOT NULL,
    ypos      INTEGER NOT NULL,
    amplitude REAL NOT NULL,
    mean      REAL NOT NULL,
    stddev    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS phspectra_pixels (
    run_id       INTEGER NOT NULL REFERENCES phspectra(run_id),
    xpos         INTEGER NOT NULL,
    ypos         INTEGER NOT NULL,
    n_components INTEGER NOT NULL,
    rms          REAL NOT NULL,
    time_s       REAL NOT NULL,
    PRIMARY KEY (run_id, xpos, ypos)
);

CREATE TABLE IF NOT EXISTS gausspyplus (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    alpha1       REAL NOT NULL,
    alpha2       REAL NOT NULL,
    phase        TEXT NOT NULL,
    n_spectra    INTEGER NOT NULL,
    total_time_s REAL NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS gausspyplus_components (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    INTEGER NOT NULL REFERENCES gausspyplus(run_id),
    xpos      INTEGER NOT NULL,
    ypos      INTEGER NOT NULL,
    amplitude REAL NOT NULL,
    mean      REAL NOT NULL,
    stddev    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS gausspyplus_pixels (
    run_id       INTEGER NOT NULL REFERENCES gausspyplus(run_id),
    xpos         INTEGER NOT NULL,
    ypos         INTEGER NOT NULL,
    n_components INTEGER NOT NULL,
    rms          REAL NOT NULL,
    time_s       REAL NOT NULL,
    PRIMARY KEY (run_id, xpos, ypos)
);
"""


def create_db(path: str) -> sqlite3.Connection:
    """Create (or open) a pre-compute SQLite database and ensure schema exists."""
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    return conn


def insert_phspectra_run(
    conn: sqlite3.Connection,
    beta: float,
    n_spectra: int,
    total_time_s: float,
) -> int:
    """Insert a phspectra run row and return the run_id."""
    cur = conn.execute(
        "INSERT INTO phspectra (beta, n_spectra, total_time_s) VALUES (?, ?, ?)",
        (beta, n_spectra, total_time_s),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def insert_gausspyplus_run(
    conn: sqlite3.Connection,
    alpha1: float,
    alpha2: float,
    phase: str,
    n_spectra: int,
    total_time_s: float,
) -> int:
    """Insert a gausspyplus run row and return the run_id."""
    cur = conn.execute(
        "INSERT INTO gausspyplus (alpha1, alpha2, phase, n_spectra, total_time_s) VALUES (?, ?, ?, ?, ?)",
        (alpha1, alpha2, phase, n_spectra, total_time_s),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def insert_components(
    conn: sqlite3.Connection,
    table: str,
    run_id: int,
    xpos: int,
    ypos: int,
    components: list[tuple[float, float, float]],
) -> None:
    """Insert component rows (amplitude, mean, stddev) for one pixel."""
    conn.executemany(
        f"INSERT INTO {table} (run_id, xpos, ypos, amplitude, mean, stddev) "  # noqa: S608
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(run_id, xpos, ypos, a, m, s) for a, m, s in components],
    )


def insert_pixels(
    conn: sqlite3.Connection,
    table: str,
    run_id: int,
    rows: list[tuple[int, int, int, float, float]],
) -> None:
    """Insert pixel summary rows (xpos, ypos, n_components, rms, time_s)."""
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} (run_id, xpos, ypos, n_components, rms, time_s) "  # noqa: S608
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(run_id, x, y, n, r, t) for x, y, n, r, t in rows],
    )


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def load_run(
    db_path: str,
    tool: str,
    run_id: int | None = None,
) -> dict[str, Any]:
    """Load run metadata for phspectra or gausspyplus.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    tool : str
        ``"phspectra"`` or ``"gausspyplus"``.
    run_id : int or None
        Specific run to load.  ``None`` returns the latest.

    Returns
    -------
    dict
        Row as a dict with column names as keys.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if run_id is not None:
        row = conn.execute(f"SELECT * FROM {tool} WHERE run_id = ?", (run_id,)).fetchone()  # noqa: S608
    else:
        row = conn.execute(f"SELECT * FROM {tool} ORDER BY run_id DESC LIMIT 1").fetchone()  # noqa: S608
    conn.close()
    if row is None:
        raise ValueError(f"No {tool} run found in {db_path}")
    return dict(row)


def load_pixels(
    db_path: str,
    tool: str,
    run_id: int | None = None,
) -> list[dict[str, Any]]:
    """Load pixel summaries from ``{tool}_pixels``.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    tool : str
        ``"phspectra"`` or ``"gausspyplus"``.
    run_id : int or None
        Specific run.  ``None`` uses the latest.

    Returns
    -------
    list[dict]
        One dict per pixel row.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if run_id is None:
        rid_row = conn.execute(f"SELECT run_id FROM {tool} ORDER BY run_id DESC LIMIT 1").fetchone()  # noqa: S608
        if rid_row is None:
            conn.close()
            raise ValueError(f"No {tool} run found in {db_path}")
        run_id = rid_row["run_id"]
    rows = conn.execute(
        f"SELECT * FROM {tool}_pixels WHERE run_id = ?",  # noqa: S608
        (run_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_components(
    db_path: str,
    tool: str,
    run_id: int | None = None,
) -> dict[tuple[int, int], list[tuple[float, float, float]]]:
    """Load components grouped by pixel.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    tool : str
        ``"phspectra"`` or ``"gausspyplus"``.
    run_id : int or None
        Specific run.  ``None`` uses the latest.

    Returns
    -------
    dict[tuple[int, int], list[tuple[float, float, float]]]
        ``{(xpos, ypos): [(amplitude, mean, stddev), ...]}``
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if run_id is None:
        rid_row = conn.execute(f"SELECT run_id FROM {tool} ORDER BY run_id DESC LIMIT 1").fetchone()  # noqa: S608
        if rid_row is None:
            conn.close()
            raise ValueError(f"No {tool} run found in {db_path}")
        run_id = rid_row["run_id"]
    rows = conn.execute(
        f"SELECT xpos, ypos, amplitude, mean, stddev FROM {tool}_components WHERE run_id = ?",  # noqa: S608
        (run_id,),
    ).fetchall()
    conn.close()
    result: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    for r in rows:
        key = (r["xpos"], r["ypos"])
        result.setdefault(key, []).append((r["amplitude"], r["mean"], r["stddev"]))
    return result
