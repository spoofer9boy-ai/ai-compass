# Practice: Deploy ML API with Monitoring

**Phase:** PHASE-06-mlops-production  
**Subjects Required:** 101 (LLM Evaluation Metrics), 102 (Agent: Tool Use), 104 (Practice: Fine-Tune Sentiment Classifier)  
**Estimated Time:** 300 minutes  
**Difficulty:** Advanced

## Industry Context

You are the first ML engineer at a legal-tech startup. The product team wants a sentiment-analysis endpoint for customer-support tickets so they can route angry users to senior agents within 30 seconds. You already fine-tuned a small transformer in a notebook, but the CTO will not sign off on production usage until she sees three things: a reproducible REST API, request/response logging, and a monitoring dashboard that can detect when the model starts returning nonsense because of a data-drift spike. Your job is to ship the API and the monitoring layer in one afternoon.

## The Problem

Build a container-ready FastAPI service that serves your fine-tuned sentiment classifier, exposes a `/predict` endpoint, and emits Prometheus-compatible metrics. Then add a lightweight drift-detection layer that compares incoming request distributions against a reference dataset and exposes a health endpoint that goes unhealthy when drift exceeds a threshold.

You must implement:

1. A FastAPI application with:
   - `POST /predict` — accepts JSON `{"text": "..."}` and returns `{"label": "positive|negative|neutral", "score": float, "model_version": str}`.
   - `GET /health` — returns HTTP 200 with `{"status": "ok", "drift_detected": bool}`.
   - `GET /metrics` — Prometheus exposition format with at least `prediction_requests_total`, `prediction_latency_seconds`, and `drift_score`.
2. A `RequestLogger` middleware that writes every request/response pair to a local JSONL file with ISO timestamps.
3. A `DriftMonitor` class that:
   - Loads a reference CSV of historical texts (provided below).
   - Computes a simple text-length distribution on every N requests.
   - Uses a two-sample Kolmogorov–Smirnov test to compare incoming text lengths against the reference.
   - Sets `drift_detected = True` when the KS p-value < 0.05.
4. A `Dockerfile` that uses `python:3.11-slim`, installs dependencies, and runs the app with `uvicorn` on port 8000.
5. A `test_client.py` script that:
   - Sends 20 requests (mix of short and long texts).
   - Prints the final drift score and whether `/health` is still OK.
   - Reads the JSONL log file and asserts at least 20 lines exist.

## Constraints

- Do not use a real GPU or large model. Use a scikit-learn `LogisticRegression` + `TfidfVectorizer` pipeline as the "fine-tuned" model (this keeps the focus on deployment and monitoring, not training).
- Do not use external databases or cloud services. Everything must run locally in Docker.
- The drift monitor must work in-memory; no persistent state between restarts.
- Prometheus metrics must use the official `prometheus_client` Python library.
- All code must be in a single `app/` directory with `main.py`, `monitor.py`, `logger.py`, `model.py`, `Dockerfile`, and `test_client.py`.

## Starter Code

```python
# app/model.py
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_VERSION = "sentiment-v1.0.0"

# TODO: Build and return a scikit-learn pipeline trained on dummy data.
def build_model():
    """Return a fitted Pipeline."""
    pass


def predict(pipeline, text: str):
    """Return dict with label, score, and model_version."""
    pass
```

```python
# app/logger.py
import json
import time
from pathlib import Path

LOG_PATH = Path("/tmp/requests.jsonl")

# TODO: Implement a middleware-compatible logger that appends one JSON line per request.
class RequestLogger:
    def __init__(self, log_path: Path = LOG_PATH):
        pass

    def log(self, request_body: dict, response_body: dict):
        pass
```

```python
# app/monitor.py
import numpy as np
from scipy import stats

# TODO: Implement DriftMonitor with KS test on text lengths.
class DriftMonitor:
    def __init__(self, reference_texts: list[str], window_size: int = 20):
        pass

    def add(self, text: str):
        pass

    def is_drifted(self) -> bool:
        pass

    def current_score(self) -> float:
        """Return the latest KS p-value (1.0 means identical)."""
        pass
```

```python
# app/main.py
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# TODO: Wire model, logger, monitor, and Prometheus metrics together.
app = FastAPI(title="Sentiment API")

# Prometheus metrics placeholders
prediction_requests_total = Counter("prediction_requests_total", "Total predictions")
prediction_latency_seconds = Histogram("prediction_latency_seconds", "Prediction latency")
drift_score = Gauge("drift_score", "Current KS p-value")

@app.post("/predict")
async def predict_endpoint(request: Request):
    pass

@app.get("/health")
async def health():
    pass

@app.get("/metrics")
async def metrics():
    pass
```

```dockerfile
# app/Dockerfile
# TODO: Multi-stage or single-stage build that installs requirements and runs uvicorn.
```

```python
# app/test_client.py
import requests

BASE = "http://localhost:8000"

# TODO: Send 20 mixed-length requests, print drift score, assert log lines >= 20.
```

## Evaluation Criteria

1. **Correctness:** `/predict` returns the expected JSON schema; `/health` reflects drift state; `/metrics` exposes valid Prometheus text format.
2. **Observability:** The JSONL log file contains every request with timestamps; Prometheus metrics increment and record latency.
3. **Drift Detection:** The KS test triggers `drift_detected = True` when a batch of very long texts is sent after a baseline of short texts.
4. **Containerization:** `docker build -t sentiment-api .` succeeds and `docker run -p 8000:8000 sentiment-api` serves the API.
5. **Test Script:** `test_client.py` runs without modification and prints a clear pass/fail summary.

## Solution

<details>
<summary>Click to reveal solution</summary>

### `app/model.py`

```python
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

MODEL_VERSION = "sentiment-v1.0.0"

# Minimal dummy training data so the pipeline is fitted.
_DUMMY_TEXTS = [
    "I love this product",
    "This is amazing",
    "Best experience ever",
    "Absolutely fantastic",
    "I am so happy",
    "This is terrible",
    "I hate this",
    "Worst service ever",
    "Very disappointing",
    "Not good at all",
    "It is okay",
    "Nothing special",
    "Average experience",
    "So so",
]
_DUMMY_LABELS = ["positive"] * 5 + ["negative"] * 5 + ["neutral"] * 4


def build_model():
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(_DUMMY_TEXTS, _DUMMY_LABELS)
    return pipeline


def predict(pipeline, text: str):
    proba = pipeline.predict_proba([text])[0]
    label = pipeline.classes_[proba.argmax()]
    score = float(proba.max())
    return {"label": label, "score": score, "model_version": MODEL_VERSION}
```

### `app/logger.py`

```python
import json
import time
from pathlib import Path

LOG_PATH = Path("/tmp/requests.jsonl")


class RequestLogger:
    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = log_path
        # Ensure directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, request_body: dict, response_body: dict):
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "request": request_body,
            "response": response_body,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

### `app/monitor.py`

```python
import numpy as np
from scipy import stats


class DriftMonitor:
    def __init__(self, reference_texts: list[str], window_size: int = 20):
        self.reference_lengths = np.array([len(t) for t in reference_texts])
        self.window_size = window_size
        self._buffer: list[str] = []
        self._latest_pvalue = 1.0

    def add(self, text: str):
        self._buffer.append(text)
        if len(self._buffer) >= self.window_size:
            self._evaluate()
            self._buffer = []

    def _evaluate(self):
        current_lengths = np.array([len(t) for t in self._buffer])
        if len(current_lengths) == 0 or len(self.reference_lengths) == 0:
            self._latest_pvalue = 1.0
            return
        # Two-sample KS test
        statistic, pvalue = stats.ks_2samp(self.reference_lengths, current_lengths)
        self._latest_pvalue = float(pvalue)

    def is_drifted(self) -> bool:
        return self._latest_pvalue < 0.05

    def current_score(self) -> float:
        return self._latest_pvalue
```

### `app/main.py`

```python
import time
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from model import build_model, predict
from logger import RequestLogger
from monitor import DriftMonitor

app = FastAPI(title="Sentiment API")

# Load model once at startup
pipeline = build_model()
logger = RequestLogger()

# Reference texts for drift monitoring (same distribution as dummy training data)
_reference_texts = [
    "I love this product",
    "This is amazing",
    "Best experience ever",
    "Absolutely fantastic",
    "I am so happy",
    "This is terrible",
    "I hate this",
    "Worst service ever",
    "Very disappointing",
    "Not good at all",
]
monitor = DriftMonitor(reference_texts=_reference_texts, window_size=20)

# Prometheus metrics
prediction_requests_total = Counter(
    "prediction_requests_total", "Total predictions", ["model_version"]
)
prediction_latency_seconds = Histogram(
    "prediction_latency_seconds", "Prediction latency"
)
drift_score = Gauge("drift_score", "Current KS p-value")


@app.post("/predict")
async def predict_endpoint(request: Request):
    body = await request.json()
    text = body.get("text", "")

    start = time.time()
    result = predict(pipeline, text)
    latency = time.time() - start

    prediction_requests_total.labels(model_version=result["model_version"]).inc()
    prediction_latency_seconds.observe(latency)

    monitor.add(text)
    drift_score.set(monitor.current_score())

    logger.log(body, result)
    return result


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "drift_detected": monitor.is_drifted(),
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

### `app/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `app/requirements.txt`

```text
fastapi
uvicorn
prometheus-client
scikit-learn
scipy
numpy
requests
```

### `app/test_client.py`

```python
import requests
import json
from pathlib import Path

BASE = "http://localhost:8000"
LOG_PATH = Path("/tmp/requests.jsonl")

# Mix of short and long texts
short_texts = ["Great!", "Bad.", "OK", "Love it", "Hate it"] * 2
long_texts = [
    "This is an extremely long and detailed customer complaint about the worst experience I have ever had with any company in my entire life and I want everyone to know about it",
] * 10

all_texts = short_texts + long_texts

def main():
    for text in all_texts:
        resp = requests.post(f"{BASE}/predict", json={"text": text})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "label" in data and "score" in data and "model_version" in data

    health = requests.get(f"{BASE}/health").json()
    print("Health:", health)

    metrics = requests.get(f"{BASE}/metrics").text
    assert "prediction_requests_total" in metrics
    assert "prediction_latency_seconds" in metrics
    assert "drift_score" in metrics
    print("Prometheus metrics present: OK")

    # Assert log file has at least 20 lines
    lines = LOG_PATH.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) >= 20, f"Expected >=20 log lines, got {len(lines)}"
    print(f"Log lines: {len(lines)} OK")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
```

</details>

## What You Actually Learned

- **LLM Evaluation Metrics:** You operationalized "model health" as a quantitative drift score rather than a vague gut feeling.
- **Agent: Tool Use:** You built a self-contained service that external systems (Grafana, alerting rules, CI/CD pipelines) can query just like an agent calling a tool.
- **Fine-Tune Sentiment Classifier:** You saw how a training artifact (the fitted scikit-learn pipeline) becomes a production dependency that must be versioned, logged, and monitored.
- **MLOps production skills:** You practiced containerization, structured logging, Prometheus instrumentation, and statistical drift detection — the minimum viable observability stack for any deployed model.
