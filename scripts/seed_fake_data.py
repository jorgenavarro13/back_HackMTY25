# scripts/seed_fake_data.py
from datetime import datetime, timedelta
from random import randint, choice
from app import storage

def iso(dt):
    return dt.replace(microsecond=0).isoformat() + "Z"

def main():
    storage.init_db()
    print("DB inicializada / tablas creadas.")

    cust = "cust_demo_1"
    now = datetime.utcnow()

    merchants = [
        {"merchant_id": "m_stream_1", "name": "StreamFlix", "category": "streaming", "lat": 25.674, "lon": -100.31, "raw_json": {"sample":"s1"}},
        {"merchant_id": "m_coffee_1", "name": "Café Mono", "category": "cafes", "lat": 25.675, "lon": -100.312, "raw_json": {}},
        {"merchant_id": "m_grocery_1", "name": "La Tiendita", "category": "grocery", "lat": 25.676, "lon": -100.305, "raw_json": {}},
        {"merchant_id": "m_atm_1", "name": "ATM BancoX", "category": "atm", "lat": 25.672, "lon": -100.315, "raw_json": {}},
        {"merchant_id": "m_far_city", "name": "FarCity Shop", "category": "electronics", "lat": 19.4326, "lon": -99.1332, "raw_json": {}},
    ]
    storage.upsert_merchants_bulk(merchants)
    print(f"Inserted/updated {len(merchants)} merchants")

    txs = []

    for d in range(10, 3, -1):
        t = now - timedelta(days=d, hours=randint(8,20))
        txs.append({
            "customer_id": cust,
            "account_id": "acc_001",
            "type": "purchase",
            "merchant_id": choice(["m_coffee_1","m_grocery_1"]),
            "amount": round(choice([2.5, 5.0, 12.0, 20.0]),2),
            "currency": "USD",
            "lat": 25.675 + (randint(-5,5)/10000),
            "lon": -100.31 + (randint(-5,5)/10000),
            "timestamp": iso(t),
            "channel": "present",
            "mcc": None,
            "raw_json": {}
        })

    for k in range(3):
        t = now - timedelta(days=30*(k+1))
        txs.append({
            "customer_id": cust,
            "account_id": "acc_001",
            "type": "purchase",
            "merchant_id": "m_stream_1",
            "amount": 9.99,
            "currency": "USD",
            "lat": 25.674, "lon": -100.31,
            "timestamp": iso(t),
            "channel": "online",
            "mcc": None,
            "raw_json": {"description":"monthly subscription"}
        })

    txs.append({
        "customer_id": cust, "account_id": "acc_001", "type": "purchase",
        "merchant_id": "m_far_city", "amount": 1200.00, "currency": "USD",
        "lat": 19.4326, "lon": -99.1332, "timestamp": iso(now - timedelta(minutes=10)),
        "channel": "online", "mcc": None, "raw_json": {}
    })

    txs.append({
        "customer_id": cust, "account_id": "acc_001", "type": "purchase",
        "merchant_id": "m_coffee_1", "amount": 3.5, "currency": "USD",
        "lat": 25.675, "lon": -100.311, "timestamp": iso(now - timedelta(minutes=12)),
        "channel": "present", "mcc": None, "raw_json": {}
    })

    base = now - timedelta(hours=2)
    for i, a in enumerate([100, 150, 200]):
        t = base + timedelta(minutes=5*i)
        txs.append({
            "customer_id": cust, "account_id": "acc_001", "type": "withdrawal",
            "merchant_id": "m_atm_1", "amount": a, "currency": "USD",
            "lat": 25.672 + (i*0.0001), "lon": -100.315 + (i*0.0001),
            "timestamp": iso(t), "channel": "atm", "mcc": None, "raw_json": {}
        })

    for i in range(6):
        t = now - timedelta(days=i, hours=randint(9,21))
        txs.append({
            "customer_id": cust, "account_id": "acc_002", "type": "purchase",
            "merchant_id": "m_grocery_1", "amount": round(choice([15.0, 30.0, 45.0]),2),
            "currency": "USD", "lat": 25.676, "lon": -100.305,
            "timestamp": iso(t), "channel": "present", "mcc": None, "raw_json": {}
        })

    inserted = storage.insert_raw_transactions_bulk(txs)
    print(f"Inserted {inserted} transactions")

    bills = [
        {"customer_id": cust, "bill_id": "bill_001", "merchant_id": "m_stream_1", "status": "due", "amount": 9.99, "due_date": iso(now + timedelta(days=10)), "raw_json": {}},
        {"customer_id": cust, "bill_id": "bill_002", "merchant_id": "m_grocery_1", "status": "paid", "amount": 45.0, "due_date": iso(now - timedelta(days=2)), "raw_json": {}}
    ]
    bcount = storage.insert_raw_bills_bulk(bills)
    print(f"Inserted {bcount} bills")

    print("✅ Poblado finalizado. Ya puedes usar /train o /score desde /docs")

if __name__ == "__main__":
    main()
