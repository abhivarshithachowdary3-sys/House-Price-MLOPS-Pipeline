#!/usr/bin/env python3
"""
run.py - Minimal MLOps-style batch job.

Loads config from YAML, reads an OHLCV CSV, computes a rolling mean on
`close`, derives a binary signal (close > rolling_mean), and writes
structured metrics (JSON) plus detailed logs.

Usage:
    python run.py --input data.csv --config config.yaml \
                   --output metrics.json --log-file run.log

Design notes:
    - Determinism: the config `seed` is applied via numpy.random.seed()
      before any computation. The pipeline itself is purely arithmetic
      on the input data (no randomness is actually consumed), so results
      are bit-for-bit reproducible across runs given the same input.
    - Rolling-mean warm-up: the first (window - 1) rows have no full
      window of history, so pandas' rolling().mean() yields NaN there.
      Those rows are EXCLUDED from signal generation and from
      `signal_rate` (they are not "0" or "1" -- they are undefined).
      `rows_processed` still reports the full row count read from the
      input file so the metric reflects total throughput.
    - Metrics file is written in BOTH success and error paths, per spec.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REQUIRED_CONFIG_FIELDS = ("seed", "window", "version")
REQUIRED_COLUMN = "close"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal MLOps batch job: rolling-mean signal pipeline."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV (OHLCV data).")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output", required=True, help="Path to write metrics JSON.")
    parser.add_argument("--log-file", required=True, help="Path to write the run log.")
    return parser.parse_args()


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_task")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Logs are written to the log file only. stdout is reserved exclusively
    # for the final metrics JSON (per the Docker requirement), so it stays
    # machine-parseable for automated grading.
    return logger


def write_metrics(output_path: str, payload: dict, logger: logging.Logger) -> None:
    """Metrics file must be written in both success and error cases."""
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Metrics written to {output_path}: {json.dumps(payload)}")


def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file '{config_path}': {e}")

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid config structure in '{config_path}': expected a mapping/object."
        )

    missing = [field for field in REQUIRED_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Config missing required field(s): {missing}")

    if not isinstance(config["seed"], int):
        raise ValueError("Config field 'seed' must be an integer.")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError("Config field 'window' must be a positive integer.")
    if not isinstance(config["version"], str) or not config["version"]:
        raise ValueError("Config field 'version' must be a non-empty string.")

    logger.info(
        "Config loaded + validated: "
        f"seed={config['seed']}, window={config['window']}, version={config['version']}"
    )
    return config


def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file has no parseable data: {input_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV format in '{input_path}': {e}")

    if df.empty:
        raise ValueError(f"Input file contains no rows: {input_path}")

    if REQUIRED_COLUMN not in df.columns:
        raise ValueError(
            f"Required column '{REQUIRED_COLUMN}' not found in input. "
            f"Available columns: {list(df.columns)}"
        )

    if not pd.api.types.is_numeric_dtype(df[REQUIRED_COLUMN]):
        try:
            df[REQUIRED_COLUMN] = pd.to_numeric(df[REQUIRED_COLUMN])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Column '{REQUIRED_COLUMN}' contains non-numeric values: {e}")

    logger.info(f"Rows loaded: {len(df)} (columns: {list(df.columns)})")
    return df


def compute_signal(df: pd.DataFrame, window: int, logger: logging.Logger):
    df = df.copy()
    df["rolling_mean"] = df[REQUIRED_COLUMN].rolling(window=window, min_periods=window).mean()
    logger.info(f"Rolling mean computed on '{REQUIRED_COLUMN}' with window={window}")

    df["signal"] = np.where(
        df["rolling_mean"].isna(),
        np.nan,
        (df[REQUIRED_COLUMN] > df["rolling_mean"]).astype(float),
    )
    valid_signals = df["signal"].dropna()
    logger.info(
        "Signal generated: "
        f"{len(valid_signals)} valid rows (first {window - 1} rows excluded as warm-up NaNs)"
    )
    return df, valid_signals


def main():
    args = parse_args()
    logger = setup_logging(args.log_file)
    start_time = time.perf_counter()
    logger.info("Job start")
    logger.info(f"Args: input={args.input}, config={args.config}, "
                f"output={args.output}, log_file={args.log_file}")

    version_for_error = "unknown"

    try:
        config = load_config(args.config, logger)
        version_for_error = config.get("version", "unknown")

        np.random.seed(config["seed"])
        logger.info(f"Random seed set: {config['seed']}")

        df = load_dataset(args.input, logger)
        df, valid_signals = compute_signal(df, config["window"], logger)

        rows_processed = len(df)
        signal_rate = float(valid_signals.mean()) if len(valid_signals) > 0 else 0.0
        latency_ms = int(round((time.perf_counter() - start_time) * 1000))

        metrics = {
            "version": config["version"],
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": latency_ms,
            "seed": config["seed"],
            "status": "success",
        }

        logger.info(
            "Metrics summary: "
            f"rows_processed={rows_processed}, signal_rate={metrics['value']}, "
            f"latency_ms={latency_ms}"
        )
        write_metrics(args.output, metrics, logger)
        logger.info("Job end | status=success")
        print(json.dumps(metrics))
        sys.exit(0)

    except Exception as e:
        latency_ms = int(round((time.perf_counter() - start_time) * 1000))
        logger.exception(f"Validation/processing error: {e}")

        error_payload = {
            "version": version_for_error,
            "status": "error",
            "error_message": str(e),
        }
        write_metrics(args.output, error_payload, logger)
        logger.info(f"Job end | status=error | latency_ms={latency_ms}")
        print(json.dumps(error_payload))
        sys.exit(1)


if __name__ == "__main__":
    main()
