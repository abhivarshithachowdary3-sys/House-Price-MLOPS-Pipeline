# MLOps Task 0 — Rolling-Mean Signal Batch Job

A minimal, reproducible MLOps-style batch job. It loads a YAML config,
reads an OHLCV CSV, computes a rolling mean on `close`, derives a binary
signal, and writes structured metrics (JSON) plus a detailed run log.

## Files

| File | Purpose |
|---|---|
| `run.py` | Main pipeline (CLI entry point) |
| `config.yaml` | Run configuration (`seed`, `window`, `version`) |
| `data.csv` | Input OHLCV dataset (10,000 rows) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build for one-command reproducible runs |
| `test_run.py` | *(bonus)* smoke tests for validation, determinism, error paths |
| `metrics.json` | Sample output from a successful run |
| `run.log` | Sample log from a successful run |

> **Note on `data.csv`:** the original dataset lives in a Google Sheet
> linked from the task doc. My sandbox environment could not reach
> `docs.google.com` (network egress is allow-listed to a fixed set of
> package/dev domains), so the `data.csv` included here is a
> **deterministically generated synthetic OHLCV series** (10,000 rows,
> same `timestamp/open/high/low/close/volume` schema, seeded with
> `numpy.random.default_rng(42)`). `run.py` itself makes no assumption
> beyond "a `close` column exists" — drop the real `data.csv` in place
> of this one and everything works identically without any code
> changes.

## Local run

```bash
pip install -r requirements.txt
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

No paths are hard-coded — all four are CLI arguments, so you can point
this at any config/CSV combination.

## Docker build & run

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

The container bundles `data.csv` and `config.yaml`, runs the pipeline
with the required CLI exactly as above, writes `metrics.json` and
`run.log` inside the container, prints the final metrics JSON to
**stdout** (and only the JSON — logs go to `run.log`, not stdout, so
stdout stays machine-parseable), and exits `0` on success / non-zero on
failure.

## Design decisions

- **Determinism:** `numpy.random.seed(config["seed"])` is set right
  after config validation. The pipeline itself is deterministic
  arithmetic on the input (no randomness is actually consumed by the
  rolling-mean/signal logic), so repeated runs on the same input
  produce identical `rows_processed`, `signal_rate`, and `seed` — only
  `latency_ms` varies, since it's a genuine wall-clock measurement.
- **Rolling-mean warm-up (first `window - 1` rows):** pandas'
  `rolling(window, min_periods=window).mean()` yields `NaN` for rows
  without a full window of history. Those rows are **excluded** from
  signal generation and from `signal_rate` (not treated as `0` or `1`,
  since neither is really "true"). `rows_processed` still reports the
  full row count read from the input, so it reflects total throughput.
- **Validation:** config and input are validated independently before
  processing (missing file, empty file, invalid CSV, missing `close`
  column, missing/invalid config keys), each raising a specific,
  descriptive error that is caught by a single top-level handler.
- **Metrics always written:** a `try/except` around the whole pipeline
  guarantees `metrics.json` is written in both the success and error
  cases, matching the required schemas exactly.

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4906,
  "latency_ms": 13,
  "seed": 42,
  "status": "success"
}
```

## Example `metrics.json` (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found in input. Available columns: ['a', 'b', 'c']"
}
```

## Bonus: tests

```bash
pip install pytest
pytest test_run.py -v
```

Covers: successful run + schema shape, determinism across repeated
runs, missing input file, missing `close` column, empty input file, and
invalid/incomplete config — 6 tests, all passing.
