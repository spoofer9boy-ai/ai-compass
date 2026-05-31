"""Tests for the Pipeline and Feature Store practice."""

import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta

import pytest

from pipeline import (
    Transaction,
    FeatureSchema,
    extract_raw_transactions,
    compute_user_aggregates,
    build_feature_schemas,
    transform,
    validate_features,
    atomic_write_jsonl,
    write_offline_features,
    write_online_features,
    get_online_features,
    run_pipeline,
)


@pytest.fixture
def sample_transactions():
    base = datetime(2024, 1, 15, 12, 0, 0)
    return [
        Transaction("tx_1", "user_a", 100.0, base.isoformat()),
        Transaction("tx_2", "user_a", 200.0, (base + timedelta(hours=1)).isoformat()),
        Transaction("tx_3", "user_a", 50.0, (base + timedelta(hours=2)).isoformat()),
        Transaction("tx_4", "user_b", 300.0, base.isoformat()),
        Transaction("tx_5", "user_b", 400.0, (base + timedelta(days=1)).isoformat()),
    ]


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE transactions (
            tx_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            amount REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    base = datetime(2024, 1, 15)
    rows = [
        (f"tx_{i:03d}", f"user_{i % 10:02d}", float((i % 50) + 10), (base + timedelta(minutes=i)).isoformat())
        for i in range(100)
    ]
    c.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return path


class TestExtraction:
    def test_extract_returns_all_rows(self, db_path):
        txs = extract_raw_transactions(db_path)
        assert len(txs) == 100

    def test_extract_types(self, db_path):
        txs = extract_raw_transactions(db_path)
        assert all(isinstance(t.amount, float) for t in txs)
        assert all(isinstance(t.timestamp, str) for t in txs)


class TestTransformation:
    def test_user_aggregates_structure(self, sample_transactions):
        aggs = compute_user_aggregates(sample_transactions)
        assert "user_a" in aggs
        assert "tx_1" in aggs["user_a"]
        assert "total_spend_7d" in aggs["user_a"]["tx_1"]

    def test_total_spend_excludes_current(self, sample_transactions):
        """The 7-day window should exclude the current transaction."""
        aggs = compute_user_aggregates(sample_transactions)
        # tx_1 is the first transaction for user_a, so no preceding transactions
        assert aggs["user_a"]["tx_1"]["total_spend_7d"] == 0.0
        # tx_2 has tx_1 in its window
        assert aggs["user_a"]["tx_2"]["total_spend_7d"] == 100.0

    def test_feature_schemas_count(self):
        schemas = build_feature_schemas()
        assert len(schemas) == 7

    def test_transform_output(self, sample_transactions):
        schemas = build_feature_schemas()
        features = transform(sample_transactions, schemas)
        assert len(features) == len(sample_transactions)
        assert all("hour_of_day" in f for f in features)
        assert all("amount_log" in f for f in features)

    def test_amount_vs_user_avg_divide_by_zero(self, sample_transactions):
        """When avg is 0, amount_vs_user_avg should be 0.0, not inf."""
        schemas = build_feature_schemas()
        features = transform(sample_transactions, schemas)
        # tx_1 for user_a has avg=0 (no preceding transactions)
        tx_1_features = next(f for f in features if f["tx_id"] == "tx_1")
        assert tx_1_features["amount_vs_user_avg"] == 0.0


class TestValidation:
    def test_valid_features_pass(self, sample_transactions):
        schemas = build_feature_schemas()
        features = transform(sample_transactions, schemas)
        validate_features(features, schemas)  # should not raise

    def test_null_value_fails(self, sample_transactions):
        schemas = build_feature_schemas()
        features = transform(sample_transactions, schemas)
        features[0]["hour_of_day"] = None
        with pytest.raises(AssertionError, match="hour_of_day is null"):
            validate_features(features, schemas)

    def test_negative_amount_fails(self, sample_transactions):
        schemas = build_feature_schemas()
        features = transform(sample_transactions, schemas)
        features[0]["amount"] = -10.0
        with pytest.raises(AssertionError, match="amount must be > 0"):
            validate_features(features, schemas)

    def test_total_spend_out_of_range_fails(self, sample_transactions):
        schemas = build_feature_schemas()
        features = transform(sample_transactions, schemas)
        features[0]["total_spend_7d"] = 2_000_000
        with pytest.raises(AssertionError, match="total_spend_7d"):
            validate_features(features, schemas)


class TestFeatureStore:
    def test_atomic_write_creates_file(self, tmp_path):
        path = str(tmp_path / "out.jsonl")
        atomic_write_jsonl([{"a": 1}, {"b": 2}], path)
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_offline_write(self, tmp_path):
        path = str(tmp_path / "offline.jsonl")
        features = [{"tx_id": "tx_1", "amount": 100.0}]
        write_offline_features(features, path)
        with open(path) as f:
            row = json.loads(f.readline())
        assert row["tx_id"] == "tx_1"

    def test_online_keeps_latest_per_user(self, tmp_path):
        path = str(tmp_path / "online.jsonl")
        features = [
            {"user_id": "u1", "timestamp": "2024-01-01T00:00:00", "amount": 100},
            {"user_id": "u1", "timestamp": "2024-01-02T00:00:00", "amount": 200},
            {"user_id": "u2", "timestamp": "2024-01-01T00:00:00", "amount": 50},
        ]
        write_online_features(features, path)
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) == 2
        u1 = next(r for r in rows if r["user_id"] == "u1")
        assert u1["amount"] == 200

    def test_online_serve_latency(self, tmp_path):
        path = str(tmp_path / "online.jsonl")
        features = [{"user_id": f"user_{i:03d}", "timestamp": "2024-01-01T00:00:00", "amount": float(i)} for i in range(100)]
        write_online_features(features, path)

        # Reset global index
        import pipeline
        pipeline._index_built = False
        pipeline._online_index.clear()

        t0 = time.perf_counter()
        result = get_online_features("user_050", path)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert result["user_id"] == "user_050"
        assert elapsed_ms < 10.0


class TestIdempotency:
    def test_pipeline_idempotent(self, db_path, tmp_path):
        import pipeline as p
        p.RAW_DB_PATH = db_path
        p.OFFLINE_PATH = str(tmp_path / "offline.jsonl")
        p.ONLINE_PATH = str(tmp_path / "online.jsonl")

        run_pipeline()
        with open(p.OFFLINE_PATH, "rb") as f:
            first = f.read()
        with open(p.ONLINE_PATH, "rb") as f:
            first_online = f.read()

        run_pipeline()
        with open(p.OFFLINE_PATH, "rb") as f:
            second = f.read()
        with open(p.ONLINE_PATH, "rb") as f:
            second_online = f.read()

        assert first == second
        assert first_online == second_online
