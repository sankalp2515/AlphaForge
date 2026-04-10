@echo off
REM AlphaForge — Local Development Runner (no Docker required)
REM Usage: setup <command>

setlocal enabledelayedexpansion

set PY=.venv\Scripts\python.exe
set AIRFLOW=.venv\Scripts\airflow.exe

if "%1"=="" goto :help
if "%1"=="help"              goto :help
if "%1"=="venv"              goto :venv
if "%1"=="init"              goto :init
if "%1"=="airflow-init"      goto :airflow-init
if "%1"=="airflow"           goto :airflow
if "%1"=="airflow-scheduler" goto :airflow-scheduler
if "%1"=="mlflow"            goto :mlflow
if "%1"=="ingest"            goto :ingest
if "%1"=="features"          goto :features
if "%1"=="train"             goto :train
if "%1"=="train-baseline"    goto :train-baseline
if "%1"=="backtest"          goto :backtest
if "%1"=="serve"             goto :serve
if "%1"=="demo"              goto :demo
if "%1"=="test"              goto :test
if "%1"=="clean"             goto :clean
echo Unknown command: %1
goto :help

:help
echo.
echo  AlphaForge — Local Dev Commands (no Docker)
echo  =============================================
echo  setup venv              STEP 1: Create virtualenv + install deps
echo  setup init              STEP 2: Create local DB + data folders
echo  setup ingest            STEP 3: Fetch OHLCV market data
echo  setup features          STEP 4: Compute alpha features
echo  setup train-baseline    STEP 5: Train LSTM baseline
echo  setup train             STEP 6: Train TFT model
echo  setup backtest          STEP 7: Run backtest
echo  setup serve             STEP 8: Start API -^> http://localhost:8000/docs
echo  -----------------------------------------------
echo  setup mlflow            View MLflow UI -^> http://localhost:5000
echo  setup airflow-init      Init Airflow (optional, run once)
echo  setup airflow           Start Airflow webserver (optional)
echo  setup airflow-scheduler Start Airflow scheduler (optional)
echo  setup demo              Run steps 3-7 in sequence
echo  setup test              Run test suite
echo  setup clean             Remove caches
echo.
goto :end

REM ─── venv (STEP 1) ─────────────────────────────────────────────────────────
:venv
echo.
echo [1/2] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Python not found. Download from https://python.org
    goto :end
)
echo [2/2] Installing dependencies...
%PY% -m pip install --upgrade pip --quiet
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Some packages failed to install. See errors above.
    goto :end
)
echo.
echo  Done!
echo  Next: setup init
goto :end

REM ─── init (STEP 2) ─────────────────────────────────────────────────────────
:init
echo Creating local data directories...
if not exist data mkdir data
if not exist data\features mkdir data\features
if not exist mlruns mkdir mlruns
if not exist logs mkdir logs
echo Initialising local SQLite database...
REM Storage auto-creates tables on first import — just touch the DB
%PY% -c "from src.data.storage import get_engine; get_engine(); print('DB ready at data/alphaforge.db')"
echo.
echo  Local environment ready:
echo    Database    -^>  data\alphaforge.db  (SQLite)
echo    Features    -^>  data\features\      (Parquet)
echo    MLflow      -^>  mlruns\             (local files)
echo.
echo  Next: setup ingest
goto :end

REM ─── mlflow UI ─────────────────────────────────────────────────────────────
:mlflow
echo Starting MLflow UI on http://localhost:5000
echo Press Ctrl+C to stop.
%PY% -m mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000
goto :end

REM ─── airflow ───────────────────────────────────────────────────────────────
:airflow-init
echo Initialising Airflow...
set AIRFLOW__CORE__DAGS_FOLDER=airflow/dags
set AIRFLOW__CORE__LOAD_EXAMPLES=False
set AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:///airflow.db
set ALPHAFORGE_DB_URL=sqlite:///data/alphaforge.db
set MLFLOW_TRACKING_URI=mlruns
%AIRFLOW% db migrate
%AIRFLOW% users create ^
    --username admin --password admin ^
    --firstname Admin --lastname User ^
    --role Admin --email admin@alphaforge.io
echo  Airflow ready. Run: setup airflow
goto :end

:airflow
echo Starting Airflow webserver on http://localhost:8080  (admin / admin)
echo Press Ctrl+C to stop.
set AIRFLOW__CORE__DAGS_FOLDER=airflow/dags
set AIRFLOW__CORE__LOAD_EXAMPLES=False
set AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:///airflow.db
set ALPHAFORGE_DB_URL=sqlite:///data/alphaforge.db
set MLFLOW_TRACKING_URI=mlruns
%AIRFLOW% webserver --port 8080
goto :end

:airflow-scheduler
echo Starting Airflow scheduler (run in a separate terminal)
set AIRFLOW__CORE__DAGS_FOLDER=airflow/dags
set AIRFLOW__CORE__LOAD_EXAMPLES=False
set AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:///airflow.db
set ALPHAFORGE_DB_URL=sqlite:///data/alphaforge.db
set MLFLOW_TRACKING_URI=mlruns
%AIRFLOW% scheduler
goto :end

REM ─── pipeline ──────────────────────────────────────────────────────────────
:ingest
echo Fetching OHLCV data (Binance + Yahoo Finance)...
set ALPHAFORGE_DB_URL=sqlite:///data/alphaforge.db
set MLFLOW_TRACKING_URI=mlruns
%PY% -m src.data.ingestion
goto :end

:features
echo Computing alpha features...
set ALPHAFORGE_DB_URL=sqlite:///data/alphaforge.db
set MLFLOW_TRACKING_URI=mlruns
%PY% -m src.features.pipeline
goto :end

:train
echo Training TFT model (logs to mlruns/)...
set ALPHAFORGE_DB_URL=sqlite:///data/alphaforge.db
set MLFLOW_TRACKING_URI=mlruns
%PY% -m src.training.trainer --model tft
goto :end

:train-baseline
echo Training LSTM baseline (logs to mlruns/)...
set ALPHAFORGE_DB_URL=sqlite:///data/alphaforge.db
set MLFLOW_TRACKING_URI=mlruns
%PY% -m src.training.trainer --model lstm
goto :end

:backtest
echo Running backtest...
set ALPHAFORGE_DB_URL=sqlite:///data/alphaforge.db
set MLFLOW_TRACKING_URI=mlruns
%PY% -m src.evaluation.backtest
goto :end

REM ─── serve ─────────────────────────────────────────────────────────────────
:serve
echo.
echo  Starting Signal API on http://localhost:8000/docs
echo  Press Ctrl+C to stop.
echo.
set ALPHAFORGE_DB_URL=sqlite:///data/alphaforge.db
set MLFLOW_TRACKING_URI=mlruns
set REDIS_URL=
set ENVIRONMENT=development
%PY% -m uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
goto :end

REM ─── demo ──────────────────────────────────────────────────────────────────
:demo
echo.
echo  AlphaForge — Full Pipeline Demo
echo  ==================================
call %0 init
call %0 ingest
if errorlevel 1 goto :end
call %0 features
call %0 train-baseline
call %0 train
call %0 backtest
echo.
echo  Demo complete!
echo    View experiments:  setup mlflow
echo    Start API:         setup serve
goto :end

REM ─── test ──────────────────────────────────────────────────────────────────
:test
set ALPHAFORGE_DB_URL=sqlite:///data/test_alphaforge.db
set MLFLOW_TRACKING_URI=mlruns_test
set ENVIRONMENT=development
%PY% -m pytest tests/ -v --cov=src --cov-report=term-missing
goto :end

REM ─── clean ─────────────────────────────────────────────────────────────────
:clean
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
del /s /q *.pyc 2>nul
if exist .pytest_cache rd /s /q .pytest_cache
if exist htmlcov rd /s /q htmlcov
if exist .coverage del .coverage
echo Cleaned.
goto :end

:end
endlocal