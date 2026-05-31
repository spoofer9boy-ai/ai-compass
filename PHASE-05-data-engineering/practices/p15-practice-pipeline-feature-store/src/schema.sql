CREATE TABLE IF NOT EXISTS transactions (
    tx_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    amount REAL NOT NULL,
    timestamp TEXT NOT NULL
);
