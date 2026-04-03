"""
AlphaForge — Data & Model Drift Detection
Uses Evidently AI to detect:
  - Data drift: feature distribution shifts (PSI-based)
  - Model drift: prediction distribution changes
  - Target drift: label distribution changes

Triggers automatic retraining when PSI > threshold.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd

from src.config import get_settings
from src.data.storage import get_engine, load_features
from src.features.technical import FEATURE_COLUMNS
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

DRIFT_REPORTS_DIR = Path("/tmp/alphaforge/drift_reports")
DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── PSI Computation (no Evidently dependency for core logic) ─────────────────

def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.1  → No significant change
    PSI 0.1-0.2 → Moderate change, monitor
    PSI > 0.2  → Significant drift → retrain
    """
    # Use expected distribution to define bins
    eps = 1e-6
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_counts / (len(expected) + eps)
    actual_pct = actual_counts / (len(actual) + eps)

    # Clip to avoid log(0)
    expected_pct = np.clip(expected_pct, eps, 1)
    actual_pct = np.clip(actual_pct, eps, 1)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> dict[str, float]:
    """
    Compute PSI for each feature column.
    Returns {feature_name: psi_score}.
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    available = [c for c in feature_cols if c in reference_df.columns and c in current_df.columns]

    psi_scores: dict[str, float] = {}
    for col in available:
        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values
        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue
        psi_scores[col] = compute_psi(ref_vals, cur_vals)

    return psi_scores


# ─── Evidently Report Generation ──────────────────────────────────────────────

def generate_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_name: str = "drift_report",
    feature_cols: list[str] | None = None,
) -> tuple[Path, dict]:
    """
    Generate Evidently HTML drift report.
    Falls back to PSI-only report if Evidently is unavailable.
    Returns (report_path, summary_dict).
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    available = [c for c in feature_cols if c in reference_df.columns]

    # Always compute PSI (our core implementation)
    psi_scores = compute_feature_drift(reference_df, current_df, available)
    drifted_features = [f for f, psi in psi_scores.items() if psi > settings.psi_threshold]

    summary = {
        "n_features_tested": len(psi_scores),
        "n_features_drifted": len(drifted_features),
        "drifted_features": drifted_features,
        "mean_psi": float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0,
        "max_psi": float(max(psi_scores.values())) if psi_scores else 0.0,
        "drift_detected": len(drifted_features) > 0,
        "psi_by_feature": psi_scores,
    }

    # Try to generate Evidently HTML report
    report_path = DRIFT_REPORTS_DIR / f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    try:
        from evidently import ColumnMapping
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset

        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])

        ref = reference_df[available].fillna(0)
        cur = current_df[available].fillna(0)
        report.run(reference_data=ref, current_data=cur)
        report.save_html(str(report_path))
        logger.info("evidently_report_saved", path=str(report_path))

    except ImportError:
        # Write a minimal HTML summary
        html = _build_fallback_html(summary, psi_scores)
        report_path.write_text(html)
        logger.info("psi_report_saved", path=str(report_path))

    except Exception as e:
        logger.warning("evidently_report_failed", error=str(e))
        html = _build_fallback_html(summary, psi_scores)
        report_path.write_text(html)

    return report_path, summary


def _build_fallback_html(summary: dict, psi_scores: dict[str, float]) -> str:
    """Minimal HTML drift report when Evidently is unavailable."""
    rows = "".join(
        f"<tr><td>{feat}</td><td>{psi:.4f}</td>"
        f"<td style='color:{'red' if psi > 0.2 else 'orange' if psi > 0.1 else 'green'}'>"
        f"{'🔴 DRIFT' if psi > 0.2 else '🟡 WARN' if psi > 0.1 else '🟢 OK'}</td></tr>"
        for feat, psi in sorted(psi_scores.items(), key=lambda x: x[1], reverse=True)
    )
    return f"""<!DOCTYPE html><html><head>
    <title>AlphaForge Drift Report</title>
    <style>body{{font-family:monospace;background:#050810;color:#e2e8f0;padding:2rem}}
    table{{border-collapse:collapse;width:100%}}
    th,td{{border:1px solid #1e2d45;padding:.5rem 1rem;text-align:left}}
    th{{background:#111827}}h1{{color:#00d4ff}}</style></head><body>
    <h1>AlphaForge — Drift Report</h1>
    <p>Generated: {datetime.now().isoformat()}</p>
    <p>Drift Detected: <strong style="color:{'red' if summary['drift_detected'] else 'green'}">
    {summary['drift_detected']}</strong></p>
    <p>Features Drifted: {summary['n_features_drifted']} / {summary['n_features_tested']}</p>
    <p>Mean PSI: {summary['mean_psi']:.4f} | Max PSI: {summary['max_psi']:.4f}</p>
    <table><tr><th>Feature</th><th>PSI</th><th>Status</th></tr>{rows}</table>
    </body></html>"""


# ─── Drift Pipeline ───────────────────────────────────────────────────────────

def run_drift_detection(
    asset: str | None = None,
    timeframe: str | None = None,
    baseline_window_days: int = 90,
    current_window_days: int = 30,
) -> dict:
    """
    Full drift detection pipeline:
    1. Load reference (baseline) features
    2. Load current features
    3. Compute PSI per feature
    4. Generate Evidently HTML report
    5. Write results to TimescaleDB
    6. Log to MLflow
    Returns summary dict.
    """
    timeframe = timeframe or settings.crypto_timeframe
    assets = [asset] if asset else settings.all_assets

    now = datetime.utcnow()
    baseline_start = now - timedelta(days=baseline_window_days + current_window_days)
    baseline_end = now - timedelta(days=current_window_days)
    current_start = baseline_end
    current_end = now

    all_summaries = []

    for ast in assets:
        try:
            ref_df = load_features(ast, timeframe, start=baseline_start, end=baseline_end)
            cur_df = load_features(ast, timeframe, start=current_start, end=current_end)

            if ref_df.empty or cur_df.empty:
                logger.warning("insufficient_data_for_drift", asset=ast)
                continue

            report_path, summary = generate_evidently_report(
                ref_df, cur_df,
                report_name=f"drift_{ast.replace('/', '_')}",
            )
            summary["asset"] = ast
            all_summaries.append(summary)

            _write_drift_report(ast, summary, str(report_path))

            if summary["drift_detected"]:
                logger.warning(
                    "DRIFT_DETECTED",
                    asset=ast,
                    n_drifted=summary["n_features_drifted"],
                    max_psi=summary["max_psi"],
                    drifted_features=summary["drifted_features"][:5],
                )

        except Exception as e:
            logger.error("drift_detection_failed", asset=ast, error=str(e))

    return {"assets": all_summaries, "any_drift": any(s.get("drift_detected") for s in all_summaries)}


def _write_drift_report(asset: str, summary: dict, report_path: str) -> None:
    """Persist drift report metadata to TimescaleDB."""
    from sqlalchemy import text

    row = {
        "report_date": datetime.utcnow().date(),
        "asset": asset,
        "psi_score": summary.get("max_psi", 0.0),
        "drift_detected": summary.get("drift_detected", False),
        "affected_features": summary.get("drifted_features", []),
        "report_path": report_path,
    }

    sql = text("""
        INSERT INTO drift_reports
            (report_date, asset, psi_score, drift_detected, affected_features, report_path)
        VALUES
            (:report_date, :asset, :psi_score, :drift_detected, :affected_features, :report_path)
    """)

    try:
        with get_engine().begin() as conn:
            conn.execute(sql, row)
    except Exception as e:
        logger.warning("drift_db_write_failed", error=str(e))
