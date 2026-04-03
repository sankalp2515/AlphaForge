"""
AlphaForge — Backtesting Engine
Uses Backtrader to simulate signal-based trading with:
  - Transaction cost modeling (slippage + commission)
  - Long/short/flat signals from ML model
  - Walk-forward evaluation across multiple windows
  - pyfolio tearsheet generation
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import backtrader as bt
import mlflow
import numpy as np
import pandas as pd

from src.config import get_settings
from src.evaluation.metrics import FinancialMetrics, compute_financial_metrics
from src.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ─── Signal Strategy ──────────────────────────────────────────────────────────

class MLSignalStrategy(bt.Strategy):
    """
    Backtrader strategy driven by ML model signals.
    - signal > 0.55 → LONG (buy)
    - signal < 0.45 → SHORT (sell / go flat)
    - else → FLAT (no position)

    Includes realistic transaction cost modeling.
    """

    params = (
        ("signals", None),          # pd.Series of predicted probabilities
        ("long_threshold", 0.55),
        ("short_threshold", 0.45),
        ("position_size", 0.95),    # fraction of portfolio to deploy
        ("commission", 0.001),      # 0.1% per trade (Binance-like)
        ("slippage", 0.0005),       # 0.05% slippage
        ("verbose", False),
    )

    def __init__(self) -> None:
        self.signal_iter = iter(self.params.signals)
        self.current_signal: float = 0.5
        self.trade_log: list[dict] = []

    def next(self) -> None:
        try:
            self.current_signal = next(self.signal_iter)
        except StopIteration:
            return

        current_pos = self.getposition().size
        price = self.data.close[0]
        portfolio_value = self.broker.getvalue()
        target_size = int((portfolio_value * self.params.position_size) / price)

        if self.current_signal > self.params.long_threshold:
            # Go LONG
            if current_pos <= 0:
                if current_pos < 0:
                    self.close()  # close short first
                self.buy(size=target_size)
                if self.params.verbose:
                    logger.debug("signal_long", price=price, signal=self.current_signal)

        elif self.current_signal < self.params.short_threshold:
            # Go FLAT (we don't short in this baseline — extend for pairs trading)
            if current_pos > 0:
                self.close()
                if self.params.verbose:
                    logger.debug("signal_flat", price=price, signal=self.current_signal)

    def notify_trade(self, trade: bt.Trade) -> None:
        if trade.isclosed:
            self.trade_log.append({
                "open_date": bt.num2date(trade.dtopen),
                "close_date": bt.num2date(trade.dtclose),
                "pnl": trade.pnl,
                "pnl_net": trade.pnlcomm,
                "commission": trade.commission,
                "size": trade.size,
            })


# ─── Backtest Runner ──────────────────────────────────────────────────────────

def run_backtest(
    price_df: pd.DataFrame,
    signals: pd.Series,
    initial_cash: float = 100_000,
    commission: float = 0.001,
    slippage: float = 0.0005,
) -> tuple[FinancialMetrics, list[dict], pd.Series]:
    """
    Run a single backtest pass.

    Args:
        price_df: OHLCV DataFrame with DatetimeIndex
        signals: Predicted probabilities aligned with price_df
        initial_cash: Starting portfolio value
        commission: Per-trade commission rate
        slippage: Slippage rate

    Returns:
        (FinancialMetrics, trade_log, portfolio_returns)
    """
    cerebro = bt.Cerebro()

    # Data feed
    data = bt.feeds.PandasData(
        dataname=price_df,
        datetime=None,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1,
    )
    cerebro.adddata(data)

    # Strategy
    cerebro.addstrategy(
        MLSignalStrategy,
        signals=signals,
        commission=commission,
        slippage=slippage,
    )

    # Broker settings
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.broker.set_slippage_perc(slippage)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        riskfreerate=0.05, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")

    logger.info("running_backtest", initial_cash=initial_cash, n_bars=len(price_df))

    results = cerebro.run()
    strat = results[0]

    # Extract returns series for pyfolio
    time_return = strat.analyzers.time_return.get_analysis()
    portfolio_returns = pd.Series(time_return).sort_index()

    # Compute our custom financial metrics
    fin_metrics = compute_financial_metrics(
        portfolio_returns,
        periods_per_year=252 if len(portfolio_returns) < 1000 else 8760,
    )

    # Override with Backtrader's built-in analyzers (more accurate)
    try:
        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio", None)
        if sharpe is not None and not np.isnan(sharpe):
            fin_metrics.sharpe_ratio = float(sharpe)

        drawdown = strat.analyzers.drawdown.get_analysis()
        fin_metrics.max_drawdown = -float(drawdown.get("max", {}).get("drawdown", 0)) / 100

        trade_analysis = strat.analyzers.trades.get_analysis()
        fin_metrics.total_trades = int(trade_analysis.get("total", {}).get("closed", 0))
        won = trade_analysis.get("won", {}).get("total", 0)
        lost = trade_analysis.get("lost", {}).get("total", 0)
        if (won + lost) > 0:
            fin_metrics.win_rate = float(won / (won + lost))

    except Exception as e:
        logger.warning("analyzer_extraction_failed", error=str(e))

    final_value = cerebro.broker.getvalue()
    fin_metrics.total_return = (final_value - initial_cash) / initial_cash

    logger.info(
        "backtest_complete",
        total_return=fin_metrics.total_return,
        sharpe=fin_metrics.sharpe_ratio,
        max_dd=fin_metrics.max_drawdown,
        win_rate=fin_metrics.win_rate,
        trades=fin_metrics.total_trades,
    )

    return fin_metrics, strat.trade_log, portfolio_returns


def run_walk_forward_backtest(
    price_df: pd.DataFrame,
    signals: pd.Series,
    n_windows: int = 4,
) -> list[FinancialMetrics]:
    """
    Walk-forward backtest: splits data into N windows,
    runs individual backtests, returns per-window metrics.
    Simulates realistic out-of-sample performance evaluation.
    """
    window_size = len(price_df) // n_windows
    all_metrics = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size if i < n_windows - 1 else len(price_df)

        window_prices = price_df.iloc[start:end]
        window_signals = signals.iloc[start:end]

        logger.info("walk_forward_window", window=i,
                    start=window_prices.index[0], end=window_prices.index[-1])

        try:
            metrics, _, _ = run_backtest(window_prices, window_signals)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error("window_backtest_failed", window=i, error=str(e))

    return all_metrics


def log_backtest_to_mlflow(
    run_id: str,
    fin_metrics: FinancialMetrics,
    model_version: str,
    strategy: str = "ml_signal_strategy",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> None:
    """Save backtest results to MLflow and TimescaleDB."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(fin_metrics.to_dict())

    # Also write to TimescaleDB
    from sqlalchemy import text
    from src.data.storage import get_engine

    row = {
        "run_id": run_id,
        "model_version": model_version,
        "strategy": strategy,
        "start_date": start_date or datetime.now().date(),
        "end_date": end_date or datetime.now().date(),
        "total_return": fin_metrics.total_return,
        "sharpe_ratio": fin_metrics.sharpe_ratio,
        "sortino_ratio": fin_metrics.sortino_ratio,
        "max_drawdown": fin_metrics.max_drawdown,
        "calmar_ratio": fin_metrics.calmar_ratio,
        "win_rate": fin_metrics.win_rate,
        "total_trades": fin_metrics.total_trades,
        "turnover": fin_metrics.turnover,
    }

    sql = text("""
        INSERT INTO backtest_results
            (run_id, model_version, strategy, start_date, end_date,
             total_return, sharpe_ratio, sortino_ratio, max_drawdown,
             calmar_ratio, win_rate, total_trades, turnover)
        VALUES
            (:run_id, :model_version, :strategy, :start_date, :end_date,
             :total_return, :sharpe_ratio, :sortino_ratio, :max_drawdown,
             :calmar_ratio, :win_rate, :total_trades, :turnover)
    """)

    with get_engine().begin() as conn:
        conn.execute(sql, row)

    logger.info("backtest_logged", run_id=run_id, sharpe=fin_metrics.sharpe_ratio)
