# app/ingest.py
from __future__ import annotations
import asyncio
import argparse
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .nessie_client import NessieClient, NessieAPIError
from . import storage

# -----------------------------
# Normalización de utilidades
# -----------------------------
def _to_iso(ts: Optional[str]) -> Optional[str]:
    if not ts:
        return None
    # Si ya parece ISO, regresa tal cual
    try:
        # intenta parsear con datetime.fromisoformat (quita 'Z' si está)
        iso = ts.rstrip("Z")
        # fromisoformat no acepta 'Z', pero sí '+00:00'; intentamos directo
        try:
            _ = datetime.fromisoformat(iso)
            return ts if ts.endswith("Z") else iso
        except ValueError:
            pass
        # formatos comunes Nessie (YYYY-MM-DD / YYYY-MM-DDTHH:MM:SSZ)
        # si solo viene fecha, añade 'T00:00:00Z'
        if len(ts) == 10 and ts[4] == "-" and ts[7] == "-":
            return ts + "T00:00:00Z"
        # último recurso: regresa original
        return ts
    except Exception:
        return ts


def _norm_channel(tx: Dict[str, Any]) -> str:
    # Busca en 'medium', 'channel', 'type' u otros hints
    raw = (tx.get("medium") or tx.get("channel") or tx.get("type") or "").lower()
    if "atm" in raw:
        return "atm"
    if "branch" in raw or "teller" in raw:
        return "branch"
    if "online" in raw or "ecom" in raw or "web" in raw:
        return "online"
    if "pos" in raw or "present" in raw or "card" in raw:
        return "present"
    return "unknown"


def _norm_amount(tx: Dict[str, Any]) -> float:
    # Campos típicos: amount, transaction_amount
    amt = tx.get("amount", tx.get("transaction_amount", 0.0))
    try:
        return float(amt)
    except Exception:
        return 0.0


def _norm_currency(tx: Dict[str, Any]) -> Optional[str]:
    return tx.get("currency") or tx.get("iso_currency_code") or "USD"


def _norm_mcc(tx: Dict[str, Any]) -> Optional[str]:
    # Algunos datasets exponen MCC en merchant.category_code o 'mcc'
    if isinstance(tx.get("merchant"), dict):
        mcc = tx["merchant"].get("category_code") or tx["merchant"].get("mcc")
        if mcc:
            return str(mcc)
    return tx.get("mcc")


def _norm_lat_lon_from_tx(tx: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    # Algunos tx incluyen lat/lon directos; otros vienen dentro de merchant->geocode
    lat = tx.get("lat")
    lon = tx.get("lon") or tx.get("lng")
    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon)
        except Exception:
            pass

    mer = tx.get("merchant") or {}
    geo = mer.get("geocode") or {}
    lat2 = geo.get("lat")
    lon2 = geo.get("lon") or geo.get("lng")
    try:
        return (float(lat2) if lat2 is not None else None,
                float(lon2) if lon2 is not None else None)
    except Exception:
        return (None, None)


def _norm_timestamp_from_tx(tx: Dict[str, Any]) -> Optional[str]:
    # Nombres comunes
    for key in ("timestamp", "transaction_time", "transaction_date", "purchase_date", "date"):
        if key in tx and tx[key]:
            return _to_iso(str(tx[key]))
    return None


def _extract_merchant_id(tx: Dict[str, Any]) -> Optional[str]:
    # Prioriza merchant_id explícito; si no, del objeto merchant
    mid = tx.get("merchant_id") or tx.get("payee_id") or tx.get("counterparty_id")
    if mid:
        return str(mid)
    m = tx.get("merchant")
    if isinstance(m, dict):
        return str(m.get("id") or m.get("_id") or m.get("merchant_id") or "")
    return None


def _norm_tx_base(
    customer_id: str,
    account_id: str,
    tx: Dict[str, Any],
    tx_type: str
) -> Dict[str, Any]:
    merchant_id = _extract_merchant_id(tx)
    amount = _norm_amount(tx)
    currency = _norm_currency(tx)
    lat, lon = _norm_lat_lon_from_tx(tx)
    ts = _norm_timestamp_from_tx(tx) or _to_iso(tx.get("created_at") or tx.get("updated_at") or None)
    channel = _norm_channel(tx)
    mcc = _norm_mcc(tx)

    return {
        "customer_id": customer_id,
        "account_id": account_id,
        "type": tx_type,  # purchase | withdrawal | transfer | deposit
        "merchant_id": merchant_id,
        "amount": amount,
        "currency": currency,
        "lat": lat,
        "lon": lon,
        "timestamp": ts or _to_iso(datetime.utcnow().isoformat() + "Z"),
        "channel": channel,
        "mcc": mcc,
        "raw_json": tx
    }


def _norm_bill(
    customer_id: str,
    bill: Dict[str, Any]
) -> Dict[str, Any]:
    # Campos comunes: id/_id, status, payment_amount/amount, payee/merchant_id, due_date
    bill_id = bill.get("id") or bill.get("_id") or bill.get("bill_id")
    merchant_id = bill.get("merchant_id")
    if not merchant_id and isinstance(bill.get("payee"), dict):
        merchant_id = bill["payee"].get("id") or bill["payee"].get("_id")
    status = bill.get("status") or bill.get("payment_status")
    amount = bill.get("amount") or bill.get("payment_amount")
    due = bill.get("due_date") or bill.get("payment_date") or bill.get("scheduled_date")
    return {
        "customer_id": customer_id,
        "bill_id": str(bill_id) if bill_id else None,
        "merchant_id": str(merchant_id) if merchant_id else None,
        "status": status,
        "amount": float(amount) if amount is not None else None,
        "due_date": _to_iso(str(due)) if due else None,
        "raw_json": bill
    }


def _norm_merchant(mer: Dict[str, Any]) -> Dict[str, Any]:
    mid = mer.get("merchant_id") or mer.get("id") or mer.get("_id")
    name = mer.get("name") or mer.get("legal_name")
    category = mer.get("category") or mer.get("mcc") or (mer.get("tags", [])[:1] if isinstance(mer.get("tags"), list) else None)
    lat = mer.get("lat")
    lon = mer.get("lon") or mer.get("lng")
    if lat is None or lon is None:
        geo = mer.get("geocode") or {}
        lat = lat or geo.get("lat")
        lon = lon or geo.get("lon") or geo.get("lng")
    try:
        lat = float(lat) if lat is not None else None
    except Exception:
        lat = None
    try:
        lon = float(lon) if lon is not None else None
    except Exception:
        lon = None
    return {
        "merchant_id": str(mid) if mid else None,
        "name": name,
        "category": category if isinstance(category, str) else (category[0] if isinstance(category, list) and category else None),
        "lat": lat,
        "lon": lon,
        "raw_json": mer
    }


# -----------------------------
# Ingesta principal
# -----------------------------
async def ingest_customer(customer_id: str) -> Dict[str, int]:
    """
    Descarga accounts, purchases, withdrawals, transfers, deposits, bills y merchants.
    Inserta en tablas raw_* y merchants. Devuelve métricas simples.
    """
    storage.init_db()  # asegura tablas

    client = NessieClient()
    accounts = await client.get_accounts_by_customer(customer_id)
    # Normaliza posibles formas de respuesta
    if isinstance(accounts, dict) and "accounts" in accounts:
        accounts = accounts["accounts"]
    if not isinstance(accounts, list):
        accounts = []

    # --- Merchants (carga full catálogo si está disponible) ---
    merchants_upserted = 0
    try:
        merchants = await client.get_merchants()
        if isinstance(merchants, dict) and "merchants" in merchants:
            merchants = merchants["merchants"]
        if isinstance(merchants, list):
            merchant_rows = [_norm_merchant(m) for m in merchants]
            # filtra los que no tengan ID
            merchant_rows = [m for m in merchant_rows if m.get("merchant_id")]
            merchants_upserted = storage.upsert_merchants_bulk(merchant_rows)
    except NessieAPIError:
        # Algunos despliegues no exponen /merchants; continúa sin bloquear
        merchants_upserted = 0

    transactions_inserted = 0
    bills_inserted = 0

    # --- Recorre cuentas ---
    for acct in accounts:
        account_id = str(acct.get("id") or acct.get("_id") or acct.get("account_id") or "")
        if not account_id:
            continue

        # Transacciones
        purchases = await client.get_purchases(account_id) or []
        withdrawals = await client.get_withdrawals(account_id) or []
        transfers = await client.get_transfers(account_id) or []
        deposits = await client.get_deposits(account_id) or []

        tx_rows: List[Dict[str, Any]] = []
        for tx in purchases:
            tx_rows.append(_norm_tx_base(customer_id, account_id, tx, "purchase"))
        for tx in withdrawals:
            tx_rows.append(_norm_tx_base(customer_id, account_id, tx, "withdrawal"))
        for tx in transfers:
            tx_rows.append(_norm_tx_base(customer_id, account_id, tx, "transfer"))
        for tx in deposits:
            tx_rows.append(_norm_tx_base(customer_id, account_id, tx, "deposit"))

        if tx_rows:
            transactions_inserted += storage.insert_raw_transactions_bulk(tx_rows)

        # Bills por cuenta
        try:
            bills = await client.get_bills_by_account(account_id) or []
        except NessieAPIError:
            bills = []
        bill_rows = [_norm_bill(customer_id, b) for b in bills if b]
        # filtra bill_id vacío
        bill_rows = [b for b in bill_rows if b.get("bill_id")]
        if bill_rows:
            bills_inserted += storage.insert_raw_bills_bulk(bill_rows)

    await client.close()

    metrics = {
        "accounts_found": len(accounts),
        "transactions_inserted": transactions_inserted,
        "bills_inserted": bills_inserted,
        "merchants_upserted": merchants_upserted
    }
    return metrics


# -----------------------------
# CLI
# -----------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest data from Nessie into local storage")
    p.add_argument("--customer", required=True, help="Customer ID")
    return p.parse_args()


async def _amain():
    args = _parse_args()
    metrics = await ingest_customer(args.customer)
    print("[INGEST] Done:", metrics)


if __name__ == "__main__":
    asyncio.run(_amain())
