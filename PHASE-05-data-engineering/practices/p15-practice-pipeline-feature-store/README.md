# Practice: Pipeline and Feature Store

**Phase:** PHASE-05-data-engineering  
**Subjects Required:** 93 — ETL vs ELT, 94 — Data Pipelines, 95 — Feature Stores, 97 — Data Validation  
**Estimated Time:** 240 minutes  
**Difficulty:** Intermediate

## Industry Context

You are the first data engineer at a mid-sized fintech startup. The ML team has been training fraud-detection models on ad-hoc CSV exports from the production PostgreSQL database. Every week, someone runs a slightly different query, features drift, and the model's AUC drops by 3–5 points. The CTO wants a reproducible pipeline that: (1) extracts raw transaction data nightly, (2) transforms it into a standardized feature set, (3) validates the features before they reach the model, and (4) serves the same features for both training and online inference. You cannot afford a commercial feature-store vendor yet, so you need to build the core abstractions in pure Python first.

## The Problem

Build a minimal but production-like data pipeline and feature-store system from first principles. Your system must:

1. **Extract:** Read raw transaction data from a local SQLite database (provided in starter code).
2. **Transform:** Compute three feature groups:
   - **User-level aggregates:** `total_spend_7d`, `transaction_count_7d`, `avg_transaction_amount_7d`.
   - **Transaction-level features:** `hour_of_day`, `day_of_week`, `amount_log`.
   - **Cross-features:** `amount_vs_user_avg` (transaction amount divided by the user's 7-day average).
3. **Validate:** Run data-quality checks using Great Expectations-style assertions (implemented manually, no external libraries):
   - No nulls in critical columns.
   - `amount` is strictly positive.
   - `total_spend_7d` is within `[0, 1_000_000]`.
   - Row count is within ±20% of yesterday's count.
4. **Store:** Persist validated features to a simple file-based feature store with two tables:
   - `features_offline`: Parquet file(s) for training batch jobs.
   - `features_online`: JSONL file for low-latency serving (keyed by `user_id`).
5. **Serve:** Provide a `get_online_features(user_id: str) -> dict` function that reads from `features_online` in under 10 ms.

## Constraints

- Do not use pandas, Polars, or any DataFrame library. Implement everything with pure Python (`sqlite3`, `csv`, `json`, `math`) to force understanding of the data flow.
- Do not use a commercial feature store (Feast, Tecton) or orchestrator (Airflow, Prefect). You will build the minimal abstractions they hide.
- The pipeline must be idempotent: running it twice on the same raw data produces identical output files.
- All file I/O must be atomic (write to temp file, then rename) to prevent partial reads during concurrent access.
- Feature definitions must be declarative: a single `FEATURES_SCHEMA` list drives both transformation and validation.

## Starter Code

```python
# pipeline.py
import sqlite3
import json
import math
import os
import tempfile
import shutil
from typing import List, Dict, Any, Callable
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
RAW_DB_PATH = "data/raw_transactions.db"
OFFLINE_PATH = "data/features_offline.parquet"
ONLINE_PATH = "data/features_online.jsonl"

# ---------------------------------------------------------------------------
# 2. DATA MODELS
# ---------------------------------------------------------------------------
class Transaction:
    """Raw transaction row from the source database."""
    def __init__(self, tx_id: str, user_id: str, amount: float, timestamp: str):
        self.tx_id = tx_id
        self.user_id = user_id
        self.amount = amount
        self.timestamp = timestamp

class FeatureSchema:
    """Declarative feature definition."""
    def __init__(
        self,
        name: str,
        compute: Callable[[Transaction, Dict[str, Any]], Any],
        validation: Callable[[Any], bool],
    ):
        self.name = name
        self.compute = compute
        self.validation = validation

# ---------------------------------------------------------------------------
# 3. EXTRACTION
# ---------------------------------------------------------------------------
def extract_raw_transactions(db_path: str) -> List[Transaction]:
    """Read all transactions from the SQLite database."""
    # TODO: Connect to sqlite3, SELECT * FROM transactions, return list of Transaction objects.
    pass

# ---------------------------------------------------------------------------
# 4. TRANSFORMATION
# ---------------------------------------------------------------------------
def build_feature_schemas() -> List[FeatureSchema]:
    """Define all features declaratively."""
    # TODO: Return a list of FeatureSchema objects for:
    #   - hour_of_day (from timestamp)
    #   - day_of_week (from timestamp)
    #   - amount_log (log1p of amount)
    #   - total_spend_7d (user aggregate)
    #   - transaction_count_7d (user aggregate)
    #   - avg_transaction_amount_7d (user aggregate)
    #   - amount_vs_user_avg (amount / avg_transaction_amount_7d)
    pass

def compute_user_aggregates(transactions: List[Transaction]) -> Dict[str, Dict[str, Any]]:
    """Pre-compute per-user aggregates over the last 7 days."""
    # TODO: Group by user_id, compute aggregates for the 7-day window.
    pass

def transform(transactions: List[Transaction], schemas: List[FeatureSchema]) -> List[Dict[str, Any]]:
    """Apply feature schemas to each transaction."""
    # TODO: Compute user aggregates, then for each transaction compute every feature.
    pass

# ---------------------------------------------------------------------------
# 5. VALIDATION
# ---------------------------------------------------------------------------
def validate_features(features: List[Dict[str, Any]], schemas: List[FeatureSchema]) -> None:
    """Run data-quality checks. Raise AssertionError on failure."""
    # TODO:
    #   1. No nulls in critical columns.
    #   2. amount > 0.
    #   3. total_spend_7d in [0, 1_000_000].
    #   4. Row count within ±20% of yesterday (simulate by checking > 0).
    pass

# ---------------------------------------------------------------------------
# 6. FEATURE STORE
# ---------------------------------------------------------------------------
def write_offline_features(features: List[Dict[str, Any]], path: str) -> None:
    """Write features to an offline Parquet-like file (use JSON Lines as stand-in)."""
    # TODO: Write each feature row as a JSON line to a temp file, then atomic rename.
    pass

def write_online_features(features: List[Dict[str, Any]], path: str) -> None:
    """Write features to an online key-value store (JSONL keyed by user_id)."""
    # TODO: Group by user_id, keep only the latest transaction per user, write JSONL atomically.
    pass

def get_online_features(user_id: str, path: str = ONLINE_PATH) -> Dict[str, Any]:
    """Serve online features for a single user in < 10 ms."""
    # TODO: Read JSONL file, find the row where user_id matches, return as dict.
    # Hint: For O(1) lookup, build an index on first call and cache it.
    pass

# ---------------------------------------------------------------------------
# 7. ORCHESTRATOR
# ---------------------------------------------------------------------------
def run_pipeline() -> None:
    """End-to-end pipeline: extract → transform → validate → store."""
    # TODO: Wire together all steps and print timing for each stage.
    pass

if __name__ == "__main__":
    run_pipeline()
```

```sql
-- schema.sql
CREATE TABLE IF NOT EXISTS transactions (
    tx_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    amount REAL NOT NULL,
    timestamp TEXT NOT NULL
);
```

```python
# seed.py
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
```

## Evaluation Criteria

1. **Extraction correctness:** `extract_raw_transactions` returns exactly the rows in the SQLite database, with correct types.
2. **Feature accuracy:** `total_spend_7d` is the sum of `amount` for that user in the 7 days preceding each transaction (not including the transaction itself). `amount_vs_user_avg` handles the divide-by-zero case gracefully.
3. **Validation coverage:** All four validation rules are implemented and raise on violation. A test suite with deliberately bad data passes when validation catches the errors.
4. **Atomic writes:** Offline and online files are written via temp-file + rename. No partial files are visible to readers.
5. **Online serving latency:** `get_online_features` returns in under 10 ms for 100 users after the index is built.
6. **Idempotency:** Running `run_pipeline()` twice produces byte-identical output files (given the same input data).

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
# solution.py
import sqlite3
import json
import math
import os
import tempfile
import shutil
import time
from typing import List, Dict, Any, Callable
from datetime import datetime, timedelta

RAW_DB_PATH = "data/raw_transactions.db"
OFFLINE_PATH = "data/features_offline.jsonl"
ONLINE_PATH = "data/features_online.jsonl"

class Transaction:
    def __init__(self, tx_id: str, user_id: str, amount: float, timestamp: str):
        self.tx_id = tx_id
        self.user_id = user_id
        self.amount = amount
        self.timestamp = timestamp

    @property
    def dt(self) -> datetime:
        return datetime.fromisoformat(self.timestamp)

class FeatureSchema:
    def __init__(
        self,
        name: str,
        compute: Callable[[Transaction, Dict[str, Any]], Any],
        validation: Callable[[Any], bool],
    ):
        self.name = name
        self.compute = compute
        self.validation = validation

def extract_raw_transactions(db_path: str) -> List[Transaction]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT tx_id, user_id, amount, timestamp FROM transactions ORDER BY timestamp")
    rows = [Transaction(r["tx_id"], r["user_id"], r["amount"], r["timestamp"]) for r in c.fetchall()]
    conn.close()
    return rows

def compute_user_aggregates(transactions: List[Transaction]) -> Dict[str, Dict[str, Any]]:
    """Pre-compute per-user aggregates over a sliding 7-day window."""
    by_user: Dict[str, List[Transaction]] = {}
    for tx in transactions:
        by_user.setdefault(tx.user_id, []).append(tx)
    for user_id in by_user:
        by_user[user_id].sort(key=lambda t: t.dt)

    aggregates = {}
    for user_id, txs in by_user.items():
        user_aggs = []
        for i, tx in enumerate(txs):
            window_start = tx.dt - timedelta(days=7)
            window_txs = [t for t in txs[:i] if t.dt >= window_start]
            total = sum(t.amount for t in window_txs)
            count = len(window_txs)
            avg = total / count if count > 0 else 0.0
            user_aggs.append({
                "tx_id": tx.tx_id,
                "total_spend_7d": total,
                "transaction_count_7d": count,
                "avg_transaction_amount_7d": avg,
            })
        aggregates[user_id] = {a["tx_id"]: a for a in user_aggs}
    return aggregates

def build_feature_schemas() -> List[FeatureSchema]:
    return [
        FeatureSchema(
            name="hour_of_day",
            compute=lambda tx, _: tx.dt.hour,
            validation=lambda v: 0 <= v <= 23,
        ),
        FeatureSchema(
            name="day_of_week",
            compute=lambda tx, _: tx.dt.weekday(),
            validation=lambda v: 0 <= v <= 6,
        ),
        FeatureSchema(
            name="amount_log",
            compute=lambda tx, _: round(math.log1p(tx.amount), 6),
            validation=lambda v: v >= 0,
        ),
        FeatureSchema(
            name="total_spend_7d",
            compute=lambda tx, aggs: aggs.get(tx.tx_id, {}).get("total_spend_7d", 0.0),
            validation=lambda v: 0 <= v <= 1_000_000,
        ),
        FeatureSchema(
            name="transaction_count_7d",
            compute=lambda tx, aggs: aggs.get(tx.tx_id, {}).get("transaction_count_7d", 0),
            validation=lambda v: v >= 0,
        ),
        FeatureSchema(
            name="avg_transaction_amount_7d",
            compute=lambda tx, aggs: aggs.get(tx.tx_id, {}).get("avg_transaction_amount_7d", 0.0),
            validation=lambda v: v >= 0,
        ),
        FeatureSchema(
            name="amount_vs_user_avg",
            compute=lambda tx, aggs: (
                tx.amount / aggs[tx.tx_id]["avg_transaction_amount_7d"]
                if aggs.get(tx.tx_id, {}).get("avg_transaction_amount_7d", 0) > 0
                else 0.0
            ),
            validation=lambda v: v >= 0,
        ),
    ]

def transform(transactions: List[Transaction], schemas: List[FeatureSchema]) -> List[Dict[str, Any]]:
    aggregates = compute_user_aggregates(transactions)
    features = []
    for tx in transactions:
        row = {
            "tx_id": tx.tx_id,
            "user_id": tx.user_id,
            "amount": tx.amount,
            "timestamp": tx.timestamp,
        }
        user_aggs = aggregates.get(tx.user_id, {})
        for schema in schemas:
            row[schema.name] = schema.compute(tx, user_aggs)
        features.append(row)
    return features

def validate_features(features: List[Dict[str, Any]], schemas: List[FeatureSchema]) -> None:
    if not features:
        raise AssertionError("Validation failed: no feature rows produced.")

    for row in features:
        for schema in schemas:
            val = row.get(schema.name)
            if val is None:
                raise AssertionError(f"Validation failed: {schema.name} is null in row {row['tx_id']}")
            if not schema.validation(val):
                raise AssertionError(f"Validation failed: {schema.name}={val} in row {row['tx_id']}")

    for row in features:
        if row["amount"] <= 0:
            raise AssertionError(f"Validation failed: amount must be > 0, got {row['amount']}")

    n = len(features)
    if n < 1:
        raise AssertionError("Validation failed: row count too low.")

def atomic_write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        shutil.move(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

def write_offline_features(features: List[Dict[str, Any]], path: str) -> None:
    atomic_write_jsonl(features, path)

def write_online_features(features: List[Dict[str, Any]], path: str) -> None:
    """Keep only the latest transaction per user for online serving."""
    latest: Dict[str, Dict[str, Any]] = {}
    for row in features:
        uid = row["user_id"]
        if uid not in latest or row["timestamp"] > latest[uid]["timestamp"]:
            latest[uid] = row
    atomic_write_jsonl(list(latest.values()), path)

_online_index: Dict[str, Dict[str, Any]] = {}
_index_built = False

def _build_online_index(path: str) -> None:
    global _online_index, _index_built
    _online_index.clear()
    if not os.path.exists(path):
        _index_built = True
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            _online_index[row["user_id"]] = row
    _index_built = True

def get_online_features(user_id: str, path: str = ONLINE_PATH) -> Dict[str, Any]:
    global _index_built
    if not _index_built:
        _build_online_index(path)
    return _online_index.get(user_id, {})

def run_pipeline() -> None:
    stages = [
        ("extract", lambda: extract_raw_transactions(RAW_DB_PATH)),
        ("transform", lambda txs: transform(txs, build_feature_schemas())),
        ("validate", lambda feats: validate_features(feats, build_feature_schemas()) or feats),
        ("write_offline", lambda feats: write_offline_features(feats, OFFLINE_PATH)),
        ("write_online", lambda feats: write_online_features(feats, ONLINE_PATH)),
    ]
    data = None
    for name, fn in stages:
        t0 = time.perf_counter()
        data = fn(data) if data is not None else fn()
        print(f"[{name}] done in {(time.perf_counter() - t0)*1000:.1f} ms")
    print(f"Pipeline complete. Offline: {OFFLINE_PATH}, Online: {ONLINE_PATH}")

if __name__ == "__main__":
    run_pipeline()
```

</details>

## What You Actually Learned

- **ETL vs ELT:** You built an ETL pipeline where transformation happens before loading. You saw why this matters: validation gates prevent bad data from reaching downstream consumers. In an ELT world, you would load raw data first and transform inside the warehouse—trading simplicity for flexibility.
- **Data Pipelines:** You wired extraction, transformation, validation, and loading into a single orchestrated flow. The pipeline is idempotent and uses atomic writes, two properties that separate toy scripts from production systems.
- **Feature Stores:** You implemented the dual-mode pattern (offline Parquet/JSONL for training, online key-value for serving) that underpins Feast and Tecton. The online store is indexed for O(1) lookup, mimicking the latency requirements of real-time inference.
- **Data Validation:** You wrote schema-driven checks without a framework. This is the core of Great Expectations and Pandera: declarative rules that fail fast and give clear error messages.
- **Orchestration (implicit):** Although you did not use Airflow, your `run_pipeline()` function is a DAG. Each stage's output feeds the next. In production, this DAG would be scheduled, retried, and monitored—exactly what Airflow adds.

## Appendix

### Common Pitfalls

- **Window boundary errors:** When computing `total_spend_7d`, be precise about whether the current transaction is included. The specification says "preceding 7 days," so exclude the current transaction.
- **Mutable default arguments:** Do not use `def fn(x, cache={})` in Python. The cache persists across calls.
- **Non-atomic writes:** Writing directly to the target file risks corruption if the process crashes mid-write. Always write to a temp file and rename.
- **Online index staleness:** If you cache the online index in memory, remember to invalidate it when the underlying file changes. For this exercise, the index is rebuilt on first access.

### Variations to Try

1. **Streaming pipeline:** Replace the SQLite source with a Kafka consumer. Process each transaction as it arrives, updating the online store in real time and batching offline writes every 5 minutes.
2. **Time-travel queries:** Add a `feature_timestamp` column to the offline store. Allow querying features "as of" a specific date, which is essential for backtesting models on historical data.
3. **Feature versioning:** Hash the `FEATURES_SCHEMA` list and include the hash in the output file names. This lets you serve multiple feature versions simultaneously during model A/B tests.

### Further Reading

- [Feature Stores for ML (Tecton)](https://www.tecton.ai/blog/what-is-a-feature-store/) — The canonical reference for why feature stores exist and what problems they solve.
- [Feast Documentation](https://docs.feast.dev/) — Open-source feature store; read the "Getting Started" guide to see how your hand-rolled abstractions map to a real system.
- [Great Expectations Documentation](https://docs.greatexpectations.io/docs/) — Production data validation framework; compare their expectation suite syntax to your manual checks.
- [Designing Data-Intensive Applications, Chapter 1](https://dataintensive.net/) — Martin Kleppmann on reliability, scalability, and maintainability in data systems.
- [Airflow Concepts](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html) — How production DAGs are defined, scheduled, and monitored.
