# AlphaForge Makefile — Windows + Mac/Linux compatible

ifeq ($(OS),Windows_NT)
    PY      := .venv\Scripts\python.exe
    PYTEST  := .venv\Scripts\python.exe -m pytest
    RUFF    := .venv\Scripts\python.exe -m ruff
    MYPY    := .venv\Scripts\python.exe -m mypy
    UVICORN := .venv\Scripts\python.exe -m uvicorn
else
    PY      := .venv/bin/python
    PYTEST  := .venv/bin/python -m pytest
    RUFF    := .venv/bin/python -m ruff
    MYPY    := .venv/bin/python -m mypy
    UVICORN := .venv/bin/python -m uvicorn
endif

# Always call tools via 'python -m <tool>' — works on Windows and Linux identically.
# Never call .venv/bin/pip or .venv\Scripts\pip.exe directly (breaks on Windows).

.PHONY: help venv infra-up infra-down db-init airflow-init airflow \
        ingest features train train-baseline backtest serve demo test lint format clean

help:
	@echo AlphaForge — make targets
	@echo   make venv           Create .venv and install deps
	@echo   make infra-up       Start Docker services
	@echo   make infra-down     Stop Docker services
	@echo   make db-init        Init TimescaleDB schema
	@echo   make airflow-init   Init Airflow (run once)
	@echo   make airflow        Start Airflow webserver
	@echo   make ingest         Fetch OHLCV data
	@echo   make features       Compute alpha features
	@echo   make train          Train TFT model
	@echo   make train-baseline Train LSTM baseline
	@echo   make backtest       Run backtest
	@echo   make serve          Start Signal API
	@echo   make demo           Full pipeline
	@echo   make test           Run tests
	@echo   make lint           Ruff + mypy
	@echo   make format         Auto-format
	@echo   make clean          Remove caches

# ── Virtual Environment ───────────────────────────────────────────────────────
venv:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@echo Done. Activate with: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Mac/Linux)

# ── Infrastructure ────────────────────────────────────────────────────────────
infra-up:
	docker compose up -d

infra-down:
	docker compose down

infra-logs:
	docker compose logs -f

# ── Database ──────────────────────────────────────────────────────────────────
db-init:
ifeq ($(OS),Windows_NT)
	docker exec -i alphaforge-timescaledb psql -U alphaforge -d alphaforge < scripts/init_db.sql
else
	PGPASSWORD=alphaforge psql -h localhost -p 5433 -U alphaforge -d alphaforge -f scripts/init_db.sql
endif

# ── Airflow ───────────────────────────────────────────────────────────────────
airflow-init:
	set AIRFLOW__CORE__DAGS_FOLDER=airflow/dags && \
	set AIRFLOW__CORE__LOAD_EXAMPLES=False && \
	set AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:///airflow.db && \
	$(PY) -m airflow db migrate
	$(PY) -m airflow users create \
		--username admin --password admin \
		--firstname Admin --lastname User \
		--role Admin --email admin@alphaforge.io

airflow:
	$(PY) -m airflow webserver --port 8080

airflow-scheduler:
	$(PY) -m airflow scheduler

# ── Pipeline ──────────────────────────────────────────────────────────────────
ingest:
	$(PY) -m src.data.ingestion

features:
	$(PY) -m src.features.pipeline

train:
	$(PY) -m src.training.trainer --model tft

train-baseline:
	$(PY) -m src.training.trainer --model lstm

backtest:
	$(PY) -m src.evaluation.backtest

# ── Serving ───────────────────────────────────────────────────────────────────
serve:
	$(UVICORN) src.serving.api:app --host 0.0.0.0 --port 8000 --reload

# ── Demo ─────────────────────────────────────────────────────────────────────
demo: infra-up db-init ingest features train-baseline train backtest
	@echo Demo complete. Run make serve to start the API.

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

lint:
	$(RUFF) check src/ tests/
	$(MYPY) src/ --ignore-missing-imports

format:
	$(RUFF) format src/ tests/

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
ifeq ($(OS),Windows_NT)
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
	del /s /q *.pyc 2>nul
else
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/
endif