# Practice: API Data Pipeline

**Phase:** PHASE-01-foundations  
**Subjects Required:** 28-NumPy Fundamentals, 29-Pandas Fundamentals, 33-Building REST APIs  
**Estimated Time:** 150 minutes  
**Difficulty:** Intermediate

## Industry Context

You are a data engineer at a mid-sized logistics company. Every morning at 06:00, a partner API publishes a JSON feed of overnight shipment events (pickups, scans, delays, deliveries). Your job is to fetch that feed, validate the schema, clean malformed rows, compute a simple delay metric, and expose the results as a paginated REST endpoint so the operations dashboard can render a daily summary. The pipeline must be robust: the API is occasionally down, fields are sometimes missing, and the dashboard expects a stable JSON contract. You cannot use a full orchestration framework (Airflow, Prefect) because the team is not there yet; the solution must be plain Python, NumPy, Pandas, and a lightweight HTTP server.

## The Problem

Build a complete data pipeline with the following stages:

1. **Fetch** a remote JSON dataset from a mock API endpoint.
2. **Parse** the JSON into a Pandas DataFrame.
3. **Validate** the schema: every record must contain `tracking_id` (string), `event_type` (string), `timestamp` (ISO-8601), and `location` (string). Records missing any of these fields must be dropped.
4. **Clean** the `timestamp` column to proper `datetime64[ns]`, the `event_type` column to lowercase, and strip whitespace from `location`.
5. **Engineer** a feature: `delay_minutes` — the time difference in minutes between the current event timestamp and the first event timestamp for that `tracking_id`. For the first event of each tracking ID, `delay_minutes` is `0`.
6. **Aggregate** the cleaned data: group by `event_type` and compute the mean `delay_minutes`.
7. **Serve** the aggregated result via a REST API endpoint `GET /summary` that returns JSON with the group means, and a `GET /events` endpoint that returns the full cleaned DataFrame as a JSON list, paginated with `page` and `per_page` query parameters.

The entire pipeline (fetch → clean → aggregate → serve) must be runnable from a single Python script.

## Constraints

- Use only **NumPy**, **Pandas**, and the **Python standard library** (`urllib`, `http.server`, `json`). No FastAPI, Flask, or external HTTP clients like `requests`.
- The fetch step must handle HTTP errors gracefully: if the mock API is unreachable, print a clear error and exit with code `1`.
- All datetime operations must use Pandas `to_datetime` or `DatetimeIndex`, not raw string manipulation.
- Pagination must default to `page=1` and `per_page=20`, and must return a `400 Bad Request` if `per_page > 100`.
- Do not use any ML libraries (scikit-learn, PyTorch, TensorFlow).

## Starter Code

```python
#!/usr/bin/env python3
"""
API Data Pipeline — Starter Code
Prerequisites: NumPy Fundamentals, Pandas Fundamentals, Building REST APIs
"""

import json
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
API_URL = "https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv"
# NOTE: The dataset above is a placeholder. For the real exercise you will
# point to a local mock server or a JSON feed.  The starter code below
# assumes a JSON array of objects shaped like:
#
#   [
#     {"tracking_id": "TRK-001", "event_type": "PICKUP",   "timestamp": "2024-06-01T08:00:00Z", "location": " NYC "},
#     {"tracking_id": "TRK-001", "event_type": "SCAN",     "timestamp": "2024-06-01T14:30:00Z", "location": "Chicago"},
#     {"tracking_id": "TRK-002", "event_type": "DELIVERY", "timestamp": "2024-06-01T09:15:00Z", "location": "LA"},
#     ...
#   ]
#
# For local testing you can spin up a tiny JSON server or read from a file.

# ---------------------------------------------------------------------------
# 2. FETCH
# ---------------------------------------------------------------------------
def fetch_data(url: str) -> list[dict]:
    """Fetch JSON array from *url* and return it as a Python list of dicts."""
    # TODO: Use urllib.request to fetch the JSON feed.
    # TODO: Handle urllib.error.URLError and return an empty list on failure.
    pass

# ---------------------------------------------------------------------------
# 3. CLEAN & VALIDATE
# ---------------------------------------------------------------------------
def clean_and_validate(raw_records: list[dict]) -> pd.DataFrame:
    """
    Drop rows missing required fields, normalise types, and engineer
    the *delay_minutes* feature.
    """
    REQUIRED = {"tracking_id", "event_type", "timestamp", "location"}

    # TODO: Filter raw_records so every record contains all REQUIRED keys.
    # TODO: Build a DataFrame from the filtered records.
    # TODO: Convert 'timestamp' to datetime64[ns] with pd.to_datetime.
    # TODO: Normalise 'event_type' to lowercase and strip 'location'.
    # TODO: Compute delay_minutes per tracking_id (first event = 0).
    pass

# ---------------------------------------------------------------------------
# 4. AGGREGATE
# ---------------------------------------------------------------------------
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with mean delay_minutes grouped by event_type."""
    # TODO: groupby + mean, then reset_index.
    pass

# ---------------------------------------------------------------------------
# 5. REST API
# ---------------------------------------------------------------------------
class PipelineHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that serves /summary and /events."""

    # The cleaned DataFrame and the summary DataFrame will be injected here
    # before the server starts.
    cleaned_df: pd.DataFrame | None = None
    summary_df: pd.DataFrame | None = None

    def do_GET(self):
        # TODO: Parse the path.  If /summary → return summary_df as JSON.
        # TODO: If /events → return paginated slice of cleaned_df as JSON.
        # TODO: Validate page/per_page query params; cap per_page at 100.
        pass

    def _send_json(self, status: int, payload):
        """Helper to send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, format, *args):
        # Suppress default logging noise
        pass

# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    print("Fetching data...")
    raw = fetch_data(API_URL)
    if not raw:
        print("ERROR: Could not fetch data. Exiting.")
        exit(1)

    print("Cleaning & validating...")
    df = clean_and_validate(raw)

    print("Aggregating...")
    summary = aggregate(df)

    # Inject data into the handler class
    PipelineHandler.cleaned_df = df
    PipelineHandler.summary_df = summary

    print("Starting server on http://localhost:8000")
    print("  GET /summary   → aggregated means")
    print("  GET /events    → paginated cleaned events")
    server = HTTPServer(("", 8000), PipelineHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")

if __name__ == "__main__":
    main()
```

## Evaluation Criteria

1. **Correctness:**
   - `fetch_data` returns a list of dicts when the URL is reachable.
   - `clean_and_validate` drops every record missing a required field, converts timestamps correctly, normalises text, and computes `delay_minutes` accurately (first event per `tracking_id` is `0`).
   - `aggregate` returns a DataFrame whose rows are `event_type` and `mean_delay_minutes`.
   - `GET /summary` returns the aggregated JSON exactly.
   - `GET /events?page=2&per_page=5` returns the correct slice; page 1 is the default.

2. **Robustness:**
   - If the remote URL is unreachable, the script prints a clear message and exits with code `1`.
   - If `per_page > 100`, the server responds with HTTP `400` and a JSON error body.
   - Empty `cleaned_df` is handled gracefully (`/events` returns `[]`, `/summary` returns `[]`).

3. **Efficiency:**
   - The pipeline runs in under 2 seconds on 10,000 records on a single CPU core.
   - No explicit Python loops over rows; use vectorised Pandas/NumPy operations.

4. **Style:**
   - All functions are typed and documented.
   - No external HTTP libraries (requests, httpx, aiohttp).

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
#!/usr/bin/env python3
"""
API Data Pipeline — Solution
Prerequisites: NumPy Fundamentals, Pandas Fundamentals, Building REST APIs
"""

import json
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
# For the exercise, point this to a local JSON file or a mock server.
# Example local test data (save as mock_data.json and set API_URL to its path):
# [
#   {"tracking_id": "TRK-001", "event_type": "PICKUP",   "timestamp": "2024-06-01T08:00:00Z", "location": " NYC "},
#   {"tracking_id": "TRK-001", "event_type": "SCAN",     "timestamp": "2024-06-01T14:30:00Z", "location": "Chicago"},
#   {"tracking_id": "TRK-002", "event_type": "DELIVERY", "timestamp": "2024-06-01T09:15:00Z", "location": "LA"},
#   {"tracking_id": "TRK-002", "event_type": "PICKUP",   "timestamp": "2024-06-01T07:00:00Z", "location": "Seattle"},
#   {"tracking_id": "TRK-003", "event_type": "DELAY",    "timestamp": "2024-06-01T12:00:00Z", "location": "Miami"}
# ]
API_URL = "http://localhost:9000/mock_data.json"  # adjust as needed

# ---------------------------------------------------------------------------
# 2. FETCH
# ---------------------------------------------------------------------------
def fetch_data(url: str) -> list[dict]:
    """Fetch JSON array from *url* and return it as a Python list of dicts."""
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = response.read().decode("utf-8")
            return json.loads(data)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        print(f"Fetch failed: {exc}")
        return []

# ---------------------------------------------------------------------------
# 3. CLEAN & VALIDATE
# ---------------------------------------------------------------------------
def clean_and_validate(raw_records: list[dict]) -> pd.DataFrame:
    """
    Drop rows missing required fields, normalise types, and engineer
    the *delay_minutes* feature.
    """
    REQUIRED = {"tracking_id", "event_type", "timestamp", "location"}

    # Keep only records that contain every required key
    filtered = [r for r in raw_records if REQUIRED.issubset(r.keys())]
    if not filtered:
        return pd.DataFrame(columns=[
            "tracking_id", "event_type", "timestamp", "location", "delay_minutes"
        ])

    df = pd.DataFrame(filtered)

    # Normalise text
    df["event_type"] = df["event_type"].str.lower().str.strip()
    df["location"] = df["location"].str.strip()

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # Drop rows where timestamp could not be parsed
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

    # Compute delay_minutes per tracking_id
    # Sort so the first chronological event per ID gets delay 0
    df = df.sort_values(["tracking_id", "timestamp"]).reset_index(drop=True)
    first_ts = df.groupby("tracking_id")["timestamp"].transform("min")
    df["delay_minutes"] = (df["timestamp"] - first_ts).dt.total_seconds() / 60.0

    return df

# ---------------------------------------------------------------------------
# 4. AGGREGATE
# ---------------------------------------------------------------------------
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with mean delay_minutes grouped by event_type."""
    if df.empty:
        return pd.DataFrame(columns=["event_type", "mean_delay_minutes"])
    grouped = (
        df.groupby("event_type", as_index=False)["delay_minutes"]
        .mean()
        .rename(columns={"delay_minutes": "mean_delay_minutes"})
    )
    return grouped

# ---------------------------------------------------------------------------
# 5. REST API
# ---------------------------------------------------------------------------
class PipelineHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that serves /summary and /events."""

    cleaned_df: pd.DataFrame | None = None
    summary_df: pd.DataFrame | None = None

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/summary":
            self._handle_summary()
        elif path == "/events":
            self._handle_events(params)
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_summary(self):
        if self.summary_df is None or self.summary_df.empty:
            self._send_json(200, [])
            return
        payload = self.summary_df.to_dict(orient="records")
        self._send_json(200, payload)

    def _handle_events(self, params: dict):
        try:
            page = int(params.get("page", ["1"])[0])
            per_page = int(params.get("per_page", ["20"])[0])
        except ValueError:
            self._send_json(400, {"error": "page and per_page must be integers"})
            return

        if per_page > 100:
            self._send_json(400, {"error": "per_page must be <= 100"})
            return

        if page < 1 or per_page < 1:
            self._send_json(400, {"error": "page and per_page must be >= 1"})
            return

        if self.cleaned_df is None or self.cleaned_df.empty:
            self._send_json(200, [])
            return

        start = (page - 1) * per_page
        end = start + per_page
        slice_df = self.cleaned_df.iloc[start:end]
        payload = slice_df.to_dict(orient="records")
        self._send_json(200, payload)

    def _send_json(self, status: int, payload):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload, default=str).encode())

    def log_message(self, format, *args):
        pass

# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    print("Fetching data...")
    raw = fetch_data(API_URL)
    if not raw:
        print("ERROR: Could not fetch data. Exiting.")
        exit(1)

    print("Cleaning & validating...")
    df = clean_and_validate(raw)

    print("Aggregating...")
    summary = aggregate(df)

    PipelineHandler.cleaned_df = df
    PipelineHandler.summary_df = summary

    print("Starting server on http://localhost:8000")
    print("  GET /summary   → aggregated means")
    print("  GET /events    → paginated cleaned events")
    server = HTTPServer(("", 8000), PipelineHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")

if __name__ == "__main__":
    main()
```

</details>

## What You Actually Learned

- **NumPy Fundamentals:** You used NumPy implicitly through Pandas (`pd.to_datetime` relies on NumPy datetime64 under the hood, and vectorised arithmetic uses NumPy ufuncs). You also reasoned about `NaT` (Not-a-Time) values when parsing failed.
- **Pandas Fundamentals:** You practiced DataFrame construction from raw JSON, schema validation via set operations, `groupby` aggregation, datetime parsing with `to_datetime`, and row slicing with `iloc`. The `delay_minutes` feature required `groupby().transform("min")` — a core Pandas pattern for windowed computations.
- **Building REST APIs:** You built a production-grade HTTP server using only the Python standard library (`http.server`, `urllib`). You handled routing, query-parameter parsing, pagination, and error responses (400, 404) without any framework magic. This is exactly the skill you need when you are asked to "just expose a CSV as an API" on a machine where you cannot install dependencies.

This pipeline is a microcosm of real data engineering: fetch, validate, clean, feature-engineer, aggregate, and serve. The next time you see an ETL job, you will recognise the same stages — just with bigger tools.
