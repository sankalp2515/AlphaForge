.PHONY: help up down ingest features train backtest serve test lint format clean demo

PYTHON := python
DOCKER_COMPOSE := docker compose

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Infrastructure ───
up: ## Start all services (Airflow, MLflow, TimescaleDB, Redis, API, Monitoring)
	$(DOCKER_COMPOSE) up -d
	@echo "⏳ Waiting for services to be ready..."
	@sleep 15
	@echo "✅ Services ready:"
	@echo "   Airflow:    http://localhost:8080  (admin/admin)"
	@echo "   MLflow:     http://localhost:5000"
	@echo "   API:        http://localhost:8000/docs"
	@echo "   Grafana:    http://localhost:3000  (admin/alphaforge)"
	@echo "   Prometheus: http://localhost:9090"

down: ## Stop all services
	$(DOCKER_COMPOSE) down

restart: ## Restart all services
	$(DOCKER_COMPOSE) restart

logs: ## Tail logs for all services
	$(DOCKER_COMPOSE) logs -f

# ─── Pipeline Stages (run locally) ───
ingest: ## Ingest latest market data (crypto + equities)
	$(PYTHON) -m src.data.ingestion --assets BTC/USDT,ETH/USDT,SOL/USDT --source crypto
	$(PYTHON) -m src.data.ingestion --assets SPY,QQQ --source equity

features: ## Compute features for all assets
	$(PYTHON) -m src.features.pipeline --run-all

train: ## Train TFT model (logs to MLflow)
	$(PYTHON) -m src.training.trainer --model tft --experiment alphaforge-v1

train-baseline: ## Train LSTM + XGBoost baselines
	$(PYTHON) -m src.training.trainer --model lstm --experiment alphaforge-baselines
	$(PYTHON) -m src.training.trainer --model xgboost --experiment alphaforge-baselines

hparam-search: ## Run Optuna hyperparameter search
	$(PYTHON) -m src.training.optuna_search --n-trials 50 --model tft

backtest: ## Run backtest on best registered model
	$(PYTHON) -m src.evaluation.backtest --model-stage Production

evaluate: ## Full evaluation suite (ML metrics + backtest + drift + SHAP)
	$(PYTHON) -m src.evaluation.metrics --model-stage Production
	$(PYTHON) -m src.evaluation.drift --baseline-window 30d
	$(PYTHON) -m src.evaluation.explainability --n-samples 500

serve: ## Start Signal API locally (outside Docker)
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

# ─── Full Demo ───
demo: ## Run end-to-end demo pipeline (ingest → features → train → backtest → serve)
	@echo "🚀 Running AlphaForge end-to-end demo..."
	$(MAKE) ingest
	$(MAKE) features
	$(MAKE) train-baseline
	$(MAKE) train
	$(MAKE) backtest
	@echo "✅ Demo complete. Check MLflow at http://localhost:5000"

# ─── Quality ───
test: ## Run full test suite with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast: ## Run tests excluding slow integration tests
	pytest tests/ -v -m "not slow" --cov=src

lint: ## Run ruff linter + mypy type check
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format: ## Auto-format code with ruff
	ruff format src/ tests/

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

# ─── Database ───
db-init: ## Initialize TimescaleDB schema
	$(PYTHON) -m scripts.init_db

db-shell: ## Open TimescaleDB shell
	$(DOCKER_COMPOSE) exec timescaledb psql -U alphaforge -d alphaforge

# ─── Cleanup ───
clean: ## Remove Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ dist/ build/

clean-data: ## Remove all local data (careful!)
	$(DOCKER_COMPOSE) down -v
	@echo "⚠️  All volumes deleted. Run 'make up' to restart."
