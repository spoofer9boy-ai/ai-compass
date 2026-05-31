import sqlite3
import random
from datetime import datetime, timedelta


def seed_db(path: str = "data/raw_transactions.db", n_users: int = 100, n_tx: int = 10_000):
    import os
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            tx_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            amount REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    c.execute("DELETE FROM transactions")
    users = [f"user_{i:04d}" for i in range(n_users)]
    base = datetime(2024, 1, 1)
    for i in range(n_tx):
        tx_id = f"tx_{i:08d}"
        user_id = random.choice(users)
        amount = round(random.uniform(5.0, 500.0), 2)
        ts = base + timedelta(minutes=random.randint(0, 60 * 24 * 7))
        c.execute(
            "INSERT INTO transactions VALUES (?, ?, ?, ?)",
            (tx_id, user_id, amount, ts.isoformat()),
        )
    conn.commit()
    conn.close()
    print(f"Seeded {n_tx} transactions for {n_users} users.")


if __name__ == "__main__":
    seed_db()
