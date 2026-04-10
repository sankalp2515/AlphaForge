# AlphaForge ⚡

> **Production-grade Algorithmic Trading Signal Engine** — End-to-end ML pipeline for multi-asset price direction prediction with full MLOps observability.

[![CI](https://github.com/yourusername/alphaforge/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/alphaforge/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│               Apache Airflow (Python process)                 │
│    ingest_dag → feature_dag → train_dag → backtest_dag       │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│  DATA LAYER                                                   │
│  Binance (ccxt) · Yahoo Finance → TimescaleDB hypertables     │
│  Great Expectations validation · PyArrow Parquet feature store│
├──────────────────────────────────────────────────────────────┤
│  FEATURE ENGINEERING  (24 alpha signals)                      │
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
│  Backtrader simulation · Evidently drift · SHAP               │
├──────────────────────────────────────────────────────────────┤
│  SERVING  (Python/uvicorn process)                            │
│  FastAPI · Redis cache · Prometheus metrics · structlog       │
└──────────────────────────────────────────────────────────────┘
```

---

## Infrastructure Design

AlphaForge uses a **hybrid approach** — only services that truly need containers run in Docker. Everything else runs as plain Python processes in a virtual environment. This keeps memory usage under **1.5 GB** instead of 15 GB.

| Service | How it runs | Why |
|---|---|---|
| **TimescaleDB** | Docker | Needs PostgreSQL extension — easiest via image |
| **Redis** | Docker | System service — cleanest in a container |
| **MLflow** | Docker | Needs a persistent server process |
| **Airflow** | Python process | `airflow webserver` + `airflow scheduler` directly |
| **Signal API** | Python process | `uvicorn src.serving.api:app` directly |
| **Training** | Python process | `python -m src.training.trainer` directly |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/yourusername/alphaforge.git
cd alphaforge

# 2. Create virtual environment and install all dependencies
make venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Start only the infrastructure containers (~1.5 GB)
make infra-up

# 4. Initialise the database schema
make db-init

# 5. Run the full pipeline
make demo
```

After `make demo` completes:

```bash
make serve          # Signal API → http://localhost:8000/docs
make airflow        # Airflow UI → http://localhost:8080 (admin/admin)
```

---

## All Commands

```bash
# Infrastructure
make infra-up        # Start TimescaleDB + Redis + MLflow (Docker)
make infra-down      # Stop containers
make db-init         # Create TimescaleDB schema (run once)

# Pipeline
make ingest          # Fetch OHLCV: Binance (BTC/ETH/SOL) + Yahoo (SPY/QQQ)
make features        # Engineer 24 alpha signals
make train-baseline  # Train LSTM baseline (Purged K-Fold CV, MLflow)
make train           # Train TFT primary model
make hparam-search   # Optuna HPO (50 trials)
make backtest        # Backtest production model (Backtrader)
make evaluate        # Metrics + SHAP + drift report

# Serving
make serve           # Signal API with hot-reload (dev)
make serve-prod      # Signal API, 2 workers (production mode)

# Airflow
make airflow-init    # Init Airflow DB + create admin (run once)
make airflow         # Start webserver + scheduler as background processes
make airflow-stop    # Stop Airflow

# Quality
make test            # pytest with coverage
make lint            # ruff + mypy
make format          # ruff format
make clean           # Remove caches
make clean-all       # Full reset (venv + volumes)
```

---

## Services & Ports

| Service | URL / Port | Credentials |
|---|---|---|
| **Signal API** | http://localhost:8000/docs | — |
| **MLflow** | http://localhost:5000 | — |
| **Airflow** | http://localhost:8080 | admin / admin |
| **TimescaleDB** | localhost:5433 | alphaforge / alphaforge |
| **Redis** | localhost:6379 | — |

---

## Key Design Decisions

### Why Purged K-Fold CV?
Standard K-Fold leaks future information into training in time-series because labels at adjacent timestamps overlap. Purged K-Fold (López de Prado, *Advances in Financial Machine Learning*) removes training samples whose label windows overlap with the test period, and adds an embargo gap afterward.

```
Standard K-Fold:  ████████░░░░████████  ← leakage at boundaries
Purged K-Fold:    ████████  ░░████████  ← purge + embargo gap
```

### Why Sharpe > Accuracy?
The model promotion gate requires `Sharpe ≥ 0.8` AND `Max Drawdown ≤ 20%`, not just AUC-ROC. A model with 51% accuracy and Sharpe 1.8 outperforms one with 65% accuracy and Sharpe 0.3.

### Why `ta` instead of `pandas-ta`?
`pandas-ta 0.3.14b0` is a pre-release that Poetry cannot resolve. The `ta` library provides identical indicators (RSI, MACD, Bollinger Bands, ATR, OBV, ADX), is fully stable, and has zero dependency conflicts.

### Why no Feast?
Feast requires `fastapi<0.100` and `numpy<1.25`, both incompatible with PyTorch 2.x and FastAPI 0.104+. Feature store is implemented with PyArrow Parquet files — same concept (offline store = Parquet, online store = Redis), zero dependency conflicts.

---

## Feature Registry

| Category | Features |
|---|---|
| **Technical** | RSI-14, MACD, MACD Signal, MACD Histogram, Bollinger %B, ATR-14, OBV, ADX-14 |
| **Momentum** | 1/5/20-bar returns, cross-sectional rank, distance from SMA-20, price range position |
| **Volatility** | Realized vol (5/20-bar), vol ratio, return autocorrelation |
| **Microstructure** | Order flow imbalance (OFI), OFI-5 MA, Amihud illiquidity, volume surprise |
| **Labels** | Forward returns (1/5/20 bars), binary direction labels |

---

## Evaluation Metrics

| Category | Metric | Target |
|---|---|---|
| **ML** | AUC-ROC | > 0.55 |
| **ML** | Information Coefficient (IC) | > 0.03 |
| **ML** | Expected Calibration Error | < 0.05 |
| **Financial** | **Sharpe Ratio** | **≥ 0.8** (promotion gate) |
| **Financial** | Max Drawdown | ≤ 20% (promotion gate) |
| **Financial** | Sortino Ratio | > 1.0 |
| **Ops** | P99 API Latency | < 200ms |
| **Ops** | Data Drift (PSI) | < 0.2 before alert |

---

## Stack

| Layer | Tools |
|---|---|
| **Orchestration** | Apache Airflow 2.8 (Python process) |
| **Data** | yfinance, ccxt (Binance), TimescaleDB, PyArrow Parquet |
| **Features** | pandas, numpy, `ta` library |
| **Models** | PyTorch 2.1, PyTorch Lightning, pytorch-forecasting (TFT), XGBoost |
| **HPO** | Optuna (TPE + Hyperband) |
| **MLOps** | MLflow (tracking + registry), Evidently AI, SHAP |
| **Backtesting** | Backtrader, pyfolio-reloaded |
| **Serving** | FastAPI, uvicorn, Redis, Pydantic v2 |
| **Observability** | Prometheus-client, structlog |
| **CI/CD** | GitHub Actions |

**Docker used only for:** TimescaleDB · Redis · MLflow (~1.5 GB total)

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*.
- Amihud, Y. (2002). *Illiquidity and stock returns*. Journal of Financial Markets.
