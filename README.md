# AlphaForge ⚡

> **Production-grade Algorithmic Trading Signal Engine** — End-to-end ML pipeline for multi-asset price direction prediction with full MLOps observability.

[![CI](https://github.com/yourusername/alphaforge/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/alphaforge/actions/workflows/ci.yml)
[![CD](https://github.com/yourusername/alphaforge/actions/workflows/cd.yml/badge.svg)](https://github.com/yourusername/alphaforge/actions/workflows/cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│               Apache Airflow (DAG Orchestration)              │
│    ingest_dag → feature_dag → train_dag → backtest_dag       │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│  DATA LAYER                                                   │
│  Binance (ccxt) · Yahoo Finance → TimescaleDB hypertables     │
│  Great Expectations validation · Feast feature store          │
├──────────────────────────────────────────────────────────────┤
│  FEATURE ENGINEERING                                          │
│  RSI · MACD · Bollinger · ATR · OBV · ADX                    │
│  Multi-horizon momentum · Cross-sectional rank                │
│  Realized vol · Order flow imbalance · Amihud illiquidity     │
├──────────────────────────────────────────────────────────────┤
│  MODEL LAYER                                                  │
│  Temporal Fusion Transformer (primary) · LSTM (baseline)      │
│  Purged K-Fold CV · Optuna HPO · MLflow tracking+registry    │
├──────────────────────────────────────────────────────────────┤
│  EVALUATION                                                   │
│  AUC-ROC · F1 · ECE · Information Coefficient                │
│  Sharpe · Sortino · Max Drawdown · Calmar                     │
│  Backtrader simulation · Evidently drift detection · SHAP     │
├──────────────────────────────────────────────────────────────┤
│  SERVING                                                      │
│  FastAPI · Redis cache · Prometheus · Grafana · Loki          │
└──────────────────────────────────────────────────────────────┘
```

---

## Quickstart (< 5 minutes)

```bash
# 1. Clone
git clone https://github.com/yourusername/alphaforge.git && cd alphaforge

# 2. Start all services (TimescaleDB, Redis, MLflow, Airflow, API, Grafana, Prometheus)
make up

# 3. Run full end-to-end demo
make demo

# 4. Explore
open http://localhost:8000/docs    # Signal API (Swagger)
open http://localhost:5000         # MLflow Experiment Tracker
open http://localhost:8080         # Airflow (admin/admin)
open http://localhost:3000         # Grafana Dashboards (admin/alphaforge)
```

---

## Services

| Service | URL | Credentials |
|---|---|---|
| **Signal API** | http://localhost:8000/docs | — |
| **MLflow** | http://localhost:5000 | — |
| **Airflow** | http://localhost:8080 | admin / admin |
| **Grafana** | http://localhost:3000 | admin / alphaforge |
| **Prometheus** | http://localhost:9090 | — |
| **TimescaleDB** | localhost:5433 | alphaforge / alphaforge |

---

## Pipeline Stages

```bash
make ingest          # Fetch OHLCV: Binance (BTC/ETH/SOL) + Yahoo (SPY/QQQ)
make features        # Engineer 24 alpha signals across all assets
make train-baseline  # Train LSTM + XGBoost with Purged K-Fold CV
make train           # Train TFT (Temporal Fusion Transformer)
make hparam-search   # Optuna hyperparameter search (50 trials)
make backtest        # Backtest production model (Backtrader)
make evaluate        # Full evaluation suite + SHAP + drift report
make serve           # Start Signal API locally
```

---

## Key Design Decisions

### Why Temporal Fusion Transformer?
TFT (Lim et al., 2021) natively supports multi-horizon forecasting, handles static vs. time-varying covariates, and provides interpretable attention weights over time. This maps directly to financial time series where regime changes and asset-specific context matter.

### Why Purged K-Fold CV?
Standard K-Fold leaks future information into training in time-series settings because labels at adjacent timestamps overlap. Purged K-Fold (López de Prado, *Advances in Financial Machine Learning*) removes training samples whose label windows overlap with the test period, and adds an embargo gap afterward.

```
Standard K-Fold:  ████████░░░░████████░░░░  ← leakage at boundaries
Purged K-Fold:    ████████  ░░░░████████    ← purge + embargo gap
```

### Why Sharpe > Accuracy?
A model with 51% accuracy and Sharpe 1.8 is more valuable than one with 65% accuracy and Sharpe 0.3. The model promotion gate requires **both** `Sharpe ≥ 0.8` and `Max Drawdown ≤ 20%` — not just AUC-ROC.

### Why TimescaleDB?
Full SQL support (unlike InfluxDB) with automatic time-series partitioning, native Feast integration, and hypertable compression. Complex feature engineering queries that would require stream processing in InfluxDB are simple SQL here.

---

## Feature Registry

| Category | Features |
|---|---|
| **Technical** | RSI-14, MACD, MACD Signal, MACD Histogram, Bollinger %B, ATR-14, OBV, ADX-14 |
| **Momentum** | 1/5/20-bar returns, cross-sectional momentum rank, distance from SMA-20, price range position |
| **Volatility** | Realized vol (5/20-bar), vol ratio (short/long), return autocorrelation |
| **Microstructure** | Order flow imbalance (OFI), OFI-5 MA, Amihud illiquidity, volume surprise |
| **Labels** | Forward returns (1/5/20 bars), binary direction labels |

---

## Evaluation Metrics

### ML Metrics
| Metric | Target | Description |
|---|---|---|
| AUC-ROC | > 0.55 | Discrimination across confidence thresholds |
| Information Coefficient | > 0.03 | Spearman rank correlation: predicted vs actual returns |
| ECE | < 0.05 | Expected Calibration Error — probability calibration |
| F1-Score | > 0.52 | Precision-recall balance |

### Financial Metrics (Backtest)
| Metric | Target | Description |
|---|---|---|
| **Sharpe Ratio** | **≥ 0.8** | **Primary promotion gate** |
| Sortino Ratio | > 1.0 | Downside-adjusted return |
| Max Drawdown | ≤ 20% | Peak-to-trough loss |
| Calmar Ratio | > 1.0 | CAGR / Max Drawdown |
| Win Rate | > 48% | % profitable trades |

### Operational Metrics (Prometheus/Grafana)
| Metric | Alert Threshold |
|---|---|
| P99 Prediction Latency | > 200ms → alert |
| Data Drift (PSI) | > 0.2 → trigger retrain |
| Prediction Error Rate | > 5% → critical alert |

---

## MLOps Pipeline

```
Code Push → GitHub Actions CI
           ├── ruff lint
           ├── mypy type check
           ├── pytest (70%+ coverage gate)
           └── Docker build validation

Merge to Main → GitHub Actions CD
               ├── Push to GHCR
               ├── Smoke tests
               └── Deploy notification

Weekly (Airflow) → train_dag
                  ├── Train LSTM baseline (Purged K-Fold)
                  ├── Train TFT primary
                  ├── Backtest with slippage + commission
                  ├── Promotion gate (Sharpe + Max DD)
                  └── MLflow registry → Production

Daily (Airflow) → ingest_dag → feature_dag
                               └── drift detection (PSI)
                                   └── if drift > 0.2 → trigger train_dag
```

---

## Stack

| Layer | Tools |
|---|---|
| **Orchestration** | Apache Airflow 2.8 |
| **Data** | yfinance, ccxt (Binance), TimescaleDB, Feast, Great Expectations |
| **Features** | pandas, numpy, pandas-ta |
| **Models** | PyTorch 2.1, PyTorch Lightning, pytorch-forecasting (TFT), XGBoost |
| **HPO** | Optuna (TPE + Hyperband) |
| **MLOps** | MLflow (tracking + registry), Evidently AI, SHAP |
| **Backtesting** | Backtrader, vectorbt, pyfolio |
| **Serving** | FastAPI, Uvicorn, Redis, Pydantic v2 |
| **Observability** | Prometheus, Grafana, Loki, structlog |
| **CI/CD** | GitHub Actions, Docker, GHCR |

**Infrastructure cost: $0** — all services run locally via Docker Compose.

---

## Project Structure

```
alphaforge/
├── .github/workflows/     # CI (lint+test+build) + CD (push+deploy)
├── airflow/dags/          # ingest / feature / train / backtest DAGs
├── src/
│   ├── data/              # ingestion (Binance + yfinance) + storage (TimescaleDB ORM)
│   ├── features/          # technical + momentum + microstructure + pipeline
│   ├── models/            # TFT + LSTM + XGBoost
│   ├── training/          # trainer + Purged K-Fold CV + Optuna search
│   ├── evaluation/        # ML metrics + backtesting + drift + SHAP
│   └── serving/           # FastAPI + Redis cache + Pydantic schemas
├── monitoring/            # Prometheus config + Grafana dashboards + alerts
├── tests/                 # pytest unit + integration tests
├── docker-compose.yml     # Full 10-service stack
├── Makefile               # make up / demo / train / backtest / test
└── pyproject.toml         # Poetry dependency management
```

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. — Purged K-Fold CV, feature importance via SHAP
- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. Int. Journal of Forecasting.
- Amihud, Y. (2002). *Illiquidity and stock returns*. Journal of Financial Markets.
- Corwin & Schultz (2012). *A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices*.

---

*Built with Python 3.11 · PyTorch 2.1 · Zero cloud spend*
