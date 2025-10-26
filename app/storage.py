# app/storage.py
from __future__ import annotations
import os
import json
import time
import pickle
import contextlib
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from datetime import datetime

# Backends posibles: duckdb (default) o sqlite
BACKEND = os.getenv("STORAGE_BACKEND", "duckdb").lower()
DB_PATH = os.getenv("DB_PATH", "risk.duckdb" if BACKEND == "duckdb" else "risk.sqlite")

if BACKEND == "duckdb":
    import duckdb as db
else:
    import sqlite3 as db  # type: ignore


# ----------------------------
# Conexión & utilidades básicas
# ----------------------------
def _connect():
    if BACKEND == "duckdb":
        # duckdb.connect crea el archivo si no existe
        con = db.connect(DB_PATH)
        # Mejores defaults
        con.execute("PRAGMA threads=4")
        return con
    else:
        # sqlite
        con = db.connect(DB_PATH, detect_types=db.PARSE_DECLTYPES)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con


@contextlib.contextmanager
def get_conn():
    con = _connect()
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ----------------------------
# Inicialización de esquema
# ----------------------------
DDL = {
    "raw_transactions": """
    CREATE TABLE IF NOT EXISTS raw_transactions (
        customer_id   TEXT NOT NULL,
        account_id    TEXT NOT NULL,
        type          TEXT NOT NULL,             -- purchase | withdrawal | transfer | deposit
        merchant_id   TEXT,
        amount        DOUBLE NOT NULL,
        currency      TEXT,
        lat           DOUBLE,
        lon           DOUBLE,
        timestamp     TIMESTAMP NOT NULL,
        channel       TEXT,                      -- atm | branch | online | present | unknown
        mcc           TEXT,
        raw_json      TEXT
    );
    """,
    "raw_bills": """
    CREATE TABLE IF NOT EXISTS raw_bills (
        customer_id   TEXT NOT NULL,
        bill_id       TEXT NOT NULL,
        merchant_id   TEXT,
        status        TEXT,
        amount        DOUBLE,
        due_date      TIMESTAMP,
        raw_json      TEXT,
        PRIMARY KEY (bill_id)
    );
    """,
    "merchants": """
    CREATE TABLE IF NOT EXISTS merchants (
        merchant_id   TEXT NOT NULL,
        name          TEXT,
        category      TEXT,
        lat           DOUBLE,
        lon           DOUBLE,
        raw_json      TEXT,
        PRIMARY KEY (merchant_id)
    );
    """,
    "models": """
    CREATE TABLE IF NOT EXISTS models (
        customer_id   TEXT NOT NULL,
        blob          BLOB NOT NULL,
        trained_at    TIMESTAMP NOT NULL,
        params_json   TEXT
    );
    """,
    "globals": """
    CREATE TABLE IF NOT EXISTS globals (
        model_blob    BLOB NOT NULL,
        trained_at    TIMESTAMP NOT NULL,
        params_json   TEXT
    );
    """,
    "feedback": """
    CREATE TABLE IF NOT EXISTS feedback (
        customer_id   TEXT NOT NULL,
        txn_id        TEXT,
        label         TEXT NOT NULL,            -- not_me | fraud | legit | unknown
        created_at    TIMESTAMP NOT NULL
    );
    """,
    # Índices útiles
    "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_rt_customer_time ON raw_transactions(customer_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_rt_account_time ON raw_transactions(account_id, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_rt_type ON raw_transactions(type)",
        "CREATE INDEX IF NOT EXISTS idx_rt_merchant ON raw_transactions(merchant_id)",
        "CREATE INDEX IF NOT EXISTS idx_bills_customer ON raw_bills(customer_id)",
        "CREATE INDEX IF NOT EXISTS idx_feedback_customer ON feedback(customer_id)"
    ]
}


def init_db() -> None:
    """Crea tablas e índices si no existen."""
    with get_conn() as con:
        for name, sql in DDL.items():
            if name == "indexes":
                continue
            con.execute(sql)
        for idx_sql in DDL["indexes"]:
            con.execute(idx_sql)


# ----------------------------
# Inserts (singular y bulk)
# ----------------------------
def insert_raw_transaction(
    customer_id: str,
    account_id: str,
    type: str,
    amount: float,
    timestamp: Union[str, datetime],
    merchant_id: Optional[str] = None,
    currency: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    channel: Optional[str] = None,
    mcc: Optional[str] = None,
    raw_json: Optional[Union[str, Dict[str, Any]]] = None
) -> None:
    if isinstance(timestamp, str):
        ts = timestamp
    else:
        ts = timestamp.isoformat()

    raw = raw_json if isinstance(raw_json, str) or raw_json is None else json.dumps(raw_json)

    sql = """
        INSERT INTO raw_transactions
        (customer_id, account_id, type, merchant_id, amount, currency, lat, lon, timestamp, channel, mcc, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with get_conn() as con:
        con.execute(sql, (customer_id, account_id, type, merchant_id, amount, currency, lat, lon, ts, channel, mcc, raw))


def insert_raw_transactions_bulk(rows: Iterable[Dict[str, Any]]) -> int:
    """
    rows: iterable de dicts con las llaves de insert_raw_transaction.
    Retorna número de filas insertadas.
    """
    sql = """
        INSERT INTO raw_transactions
        (customer_id, account_id, type, merchant_id, amount, currency, lat, lon, timestamp, channel, mcc, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    cnt = 0
    with get_conn() as con:
        cur = con.cursor()
        for r in rows:
            ts = r.get("timestamp")
            if isinstance(ts, datetime):
                ts = ts.isoformat()
            raw = r.get("raw_json")
            if raw is not None and not isinstance(raw, str):
                raw = json.dumps(raw)
            cur.execute(sql, (
                r.get("customer_id"),
                r.get("account_id"),
                r.get("type"),
                r.get("merchant_id"),
                float(r.get("amount", 0.0)),
                r.get("currency"),
                r.get("lat"),
                r.get("lon"),
                ts,
                r.get("channel"),
                r.get("mcc"),
                raw
            ))
            cnt += 1
    return cnt


def insert_raw_bill(
    customer_id: str,
    bill_id: str,
    merchant_id: Optional[str],
    status: Optional[str],
    amount: Optional[float],
    due_date: Optional[Union[str, datetime]],
    raw_json: Optional[Union[str, Dict[str, Any]]] = None
) -> None:
    dd = None
    if isinstance(due_date, datetime):
        dd = due_date.isoformat()
    elif isinstance(due_date, str):
        dd = due_date

    raw = raw_json if isinstance(raw_json, str) or raw_json is None else json.dumps(raw_json)

    sql = """
        INSERT OR REPLACE INTO raw_bills
        (customer_id, bill_id, merchant_id, status, amount, due_date, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    # DuckDB soporta INSERT OR REPLACE (desde 0.10+). Si tu versión no lo soporta,
    # se puede cambiar por try INSERT -> on error UPDATE.
    with get_conn() as con:
        con.execute(sql, (customer_id, bill_id, merchant_id, status, amount, dd, raw))


def insert_raw_bills_bulk(rows: Iterable[Dict[str, Any]]) -> int:
    sql = """
        INSERT OR REPLACE INTO raw_bills
        (customer_id, bill_id, merchant_id, status, amount, due_date, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    cnt = 0
    with get_conn() as con:
        cur = con.cursor()
        for r in rows:
            dd = r.get("due_date")
            if isinstance(dd, datetime):
                dd = dd.isoformat()
            raw = r.get("raw_json")
            if raw is not None and not isinstance(raw, str):
                raw = json.dumps(raw)
            cur.execute(sql, (
                r.get("customer_id"),
                r.get("bill_id"),
                r.get("merchant_id"),
                r.get("status"),
                r.get("amount"),
                dd,
                raw
            ))
            cnt += 1
    return cnt


def upsert_merchant(
    merchant_id: str,
    name: Optional[str],
    category: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
    raw_json: Optional[Union[str, Dict[str, Any]]] = None
) -> None:
    raw = raw_json if isinstance(raw_json, str) or raw_json is None else json.dumps(raw_json)
    insert_sql = """
        INSERT INTO merchants (merchant_id, name, category, lat, lon, raw_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    update_sql = """
        UPDATE merchants
           SET name = ?, category = ?, lat = ?, lon = ?, raw_json = ?
         WHERE merchant_id = ?
    """
    with get_conn() as con:
        try:
            con.execute(insert_sql, (merchant_id, name, category, lat, lon, raw))
        except Exception:
            # Ya existía → actualiza
            con.execute(update_sql, (name, category, lat, lon, raw, merchant_id))


def upsert_merchants_bulk(rows: Iterable[Dict[str, Any]]) -> int:
    cnt = 0
    for r in rows:
        upsert_merchant(
            merchant_id=r.get("merchant_id") or r.get("_id") or r.get("id"),
            name=r.get("name"),
            category=r.get("category"),
            lat=r.get("lat"),
            lon=r.get("lon"),
            raw_json=r.get("raw_json") or r
        )
        cnt += 1
    return cnt


# ----------------------------
# Modelos (persistencia blob)
# ----------------------------
def insert_customer_model(customer_id: str, model_obj: Any, params: Optional[Dict[str, Any]] = None) -> None:
    blob = pickle.dumps(model_obj)
    trained_at = _now_iso()
    params_json = json.dumps(params or {})
    sql = "INSERT INTO models (customer_id, blob, trained_at, params_json) VALUES (?, ?, ?, ?)"
    with get_conn() as con:
        con.execute(sql, (customer_id, db.Binary(blob) if BACKEND == "sqlite" else blob, trained_at, params_json))


def get_latest_customer_model(customer_id: str) -> Optional[Tuple[Any, Dict[str, Any], str]]:
    sql = """
        SELECT blob, params_json, trained_at
          FROM models
         WHERE customer_id = ?
      ORDER BY trained_at DESC
         LIMIT 1
    """
    with get_conn() as con:
        cur = con.cursor()
        row = cur.execute(sql, (customer_id,)).fetchone()
        if not row:
            return None
        blob = row[0]
        params_json = row[1] or "{}"
        trained_at = row[2]
        model_obj = pickle.loads(bytes(blob))
        return model_obj, json.loads(params_json), trained_at


def insert_global_model(model_obj: Any, params: Optional[Dict[str, Any]] = None) -> None:
    blob = pickle.dumps(model_obj)
    trained_at = _now_iso()
    params_json = json.dumps(params or {})
    sql = "INSERT INTO globals (model_blob, trained_at, params_json) VALUES (?, ?, ?)"
    with get_conn() as con:
        con.execute(sql, (db.Binary(blob) if BACKEND == "sqlite" else blob, trained_at, params_json))


def get_latest_global_model() -> Optional[Tuple[Any, Dict[str, Any], str]]:
    sql = """
        SELECT model_blob, params_json, trained_at
          FROM globals
      ORDER BY trained_at DESC
         LIMIT 1
    """
    with get_conn() as con:
        cur = con.cursor()
        row = cur.execute(sql).fetchone()
        if not row:
            return None
        blob = row[0]
        params_json = row[1] or "{}"
        trained_at = row[2]
        model_obj = pickle.loads(bytes(blob))
        return model_obj, json.loads(params_json), trained_at


# ----------------------------
# Feedback
# ----------------------------
def insert_feedback(customer_id: str, label: str, txn_id: Optional[str] = None, created_at: Optional[Union[str, datetime]] = None) -> None:
    ts = created_at or _now_iso()
    if isinstance(ts, datetime):
        ts = ts.isoformat() + "Z"
    sql = "INSERT INTO feedback (customer_id, txn_id, label, created_at) VALUES (?, ?, ?, ?)"
    with get_conn() as con:
        con.execute(sql, (customer_id, txn_id, label, ts))


# ----------------------------
# Lecturas comunes
# ----------------------------
def read_transactions_by_customer(customer_id: str, since: Optional[str] = None, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    since: ISO-8601 opcional para filtrar por timestamp >= since
    """
    with get_conn() as con:
        cur = con.cursor()
        if since:
            sql = """
                SELECT customer_id, account_id, type, merchant_id, amount, currency, lat, lon,
                       timestamp, channel, mcc, raw_json
                  FROM raw_transactions
                 WHERE customer_id = ? AND timestamp >= ?
                 ORDER BY timestamp ASC
                 LIMIT ?
            """
            rows = cur.execute(sql, (customer_id, since, limit)).fetchall()
        else:
            sql = """
                SELECT customer_id, account_id, type, merchant_id, amount, currency, lat, lon,
                       timestamp, channel, mcc, raw_json
                  FROM raw_transactions
                 WHERE customer_id = ?
                 ORDER BY timestamp ASC
                 LIMIT ?
            """
            rows = cur.execute(sql, (customer_id, limit)).fetchall()

    cols = ["customer_id", "account_id", "type", "merchant_id", "amount", "currency", "lat", "lon",
            "timestamp", "channel", "mcc", "raw_json"]
    out: List[Dict[str, Any]] = []
    for r in rows:
        rec = {k: r[i] for i, k in enumerate(cols)}
        # raw_json como dict si es posible
        try:
            rec["raw_json"] = json.loads(rec["raw_json"]) if rec["raw_json"] else None
        except Exception:
            pass
        out.append(rec)
    return out


def read_bills_by_customer(customer_id: str) -> List[Dict[str, Any]]:
    with get_conn() as con:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT customer_id, bill_id, merchant_id, status, amount, due_date, raw_json "
            "FROM raw_bills WHERE customer_id = ? ORDER BY due_date ASC", (customer_id,)
        ).fetchall()
    cols = ["customer_id", "bill_id", "merchant_id", "status", "amount", "due_date", "raw_json"]
    out: List[Dict[str, Any]] = []
    for r in rows:
        rec = {k: r[i] for i, k in enumerate(cols)}
        try:
            rec["raw_json"] = json.loads(rec["raw_json"]) if rec["raw_json"] else None
        except Exception:
            pass
        out.append(rec)
    return out


def read_merchants(limit: int = 10000) -> List[Dict[str, Any]]:
    with get_conn() as con:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT merchant_id, name, category, lat, lon, raw_json FROM merchants LIMIT ?", (limit,)
        ).fetchall()
    cols = ["merchant_id", "name", "category", "lat", "lon", "raw_json"]
    out: List[Dict[str, Any]] = []
    for r in rows:
        rec = {k: r[i] for i, k in enumerate(cols)}
        try:
            rec["raw_json"] = json.loads(rec["raw_json"]) if rec["raw_json"] else None
        except Exception:
            pass
        out.append(rec)
    return out


def read_feedback_by_customer(customer_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
    with get_conn() as con:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT customer_id, txn_id, label, created_at FROM feedback WHERE customer_id = ? ORDER BY created_at DESC LIMIT ?",
            (customer_id, limit)
        ).fetchall()
    cols = ["customer_id", "txn_id", "label", "created_at"]
    return [{k: r[i] for i, k in enumerate(cols)} for r in rows]


# ----------------------------
# Helper de arranque rápido
# ----------------------------
def bootstrap() -> None:
    """
    Crea la base si no existe y valida que las tablas estén listas.
    Útil para llamar desde app.main al iniciar.
    """
    init_db()
