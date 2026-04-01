import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# =========================
# 0. 全局配置
# =========================
DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(exist_ok=True)

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)


# =========================
# 1. 数据获取（带缓存）
# =========================
def get_cache_file(symbol, start, end, interval="1d", auto_adjust=True):
    adj_flag = "adj" if auto_adjust else "raw"
    fname = f"{symbol}_{start}_{end}_{interval}_{adj_flag}.csv"
    return DATA_DIR / fname


def normalize_columns(df):
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    }
    df = df.rename(columns=rename_map)
    return df


def ensure_series(x, name="value"):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0].rename(name)
        raise ValueError(f"{name} 不是单列 Series，而是多列 DataFrame: columns={list(x.columns)}")
    return x.rename(name) if hasattr(x, "rename") else x


def get_data(symbol="QQQ", start="2015-01-01", end="2025-01-01", interval="1d", auto_adjust=True):
    cache_file = get_cache_file(symbol, start, end, interval, auto_adjust)

    if cache_file.exists():
        print(f"[缓存] 读取本地数据: {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df = normalize_columns(df)
        df = df.sort_index().dropna()
        return df

    print(f"[下载] 本地无缓存，开始下载: {symbol}")
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False
    )

    if df.empty:
        raise ValueError("下载数据为空，请检查 ticker、日期区间或网络连接。")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache_file)
    print(f"[缓存] 已保存到: {cache_file}")

    df = normalize_columns(df)
    df = df.sort_index().dropna()
    return df


# =========================
# 2. 技术指标
# =========================
def calc_rsi(series, window=14):
    series = ensure_series(series, "series")
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_atr(df, window=14):
    high = ensure_series(df["high"], "high")
    low = ensure_series(df["low"], "low")
    close = ensure_series(df["close"], "close")

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(window).mean()
    return atr


def add_indicators(df):
    out = df.copy()
    close = ensure_series(out["close"], "close")

    out["ma10"] = close.rolling(10).mean()
    out["ma20"] = close.rolling(20).mean()
    out["ma50"] = close.rolling(50).mean()
    out["ma100"] = close.rolling(100).mean()
    out["ma200"] = close.rolling(200).mean()

    out["rsi14"] = calc_rsi(close, 14)
    out["atr14"] = calc_atr(out, 14)

    out["roll_high_20"] = close.rolling(20).max()
    out["roll_low_10"] = close.rolling(10).min()

    return out


# =========================
# 3. 策略函数
# signal:
#   1 = 持有多头
#   0 = 空仓
# =========================
def strategy_buy_and_hold(df, **kwargs):
    out = add_indicators(df)
    out["signal"] = 1
    return out


def strategy_ma_cross(df, short_window=20, long_window=50, **kwargs):
    out = add_indicators(df)
    close = ensure_series(out["close"], "close")

    out["ma_short_custom"] = close.rolling(short_window).mean()
    out["ma_long_custom"] = close.rolling(long_window).mean()

    out["signal"] = 0
    out.loc[out["ma_short_custom"] > out["ma_long_custom"], "signal"] = 1
    return out


def strategy_rsi_reversion(df, rsi_window=14, buy_threshold=30, sell_threshold=70, **kwargs):
    out = add_indicators(df)
    out["rsi_custom"] = calc_rsi(out["close"], rsi_window)

    out["raw_signal"] = np.nan
    out.loc[out["rsi_custom"] < buy_threshold, "raw_signal"] = 1
    out.loc[out["rsi_custom"] > sell_threshold, "raw_signal"] = 0
    out["signal"] = out["raw_signal"].ffill().fillna(0)
    return out


def strategy_trend_pullback(df, rsi_buy=35, rsi_sell=60, **kwargs):
    """
    适合 QQQ 的思路：
    只在大趋势向上时，逢回调买入
    入场：
      - close > ma200
      - ma50 > ma200
      - RSI < rsi_buy
    出场：
      - RSI > rsi_sell
      - 或 close < ma50
    """
    out = add_indicators(df)
    close = ensure_series(out["close"], "close")

    trend_ok = (close > out["ma200"]) & (out["ma50"] > out["ma200"])
    buy_cond = trend_ok & (out["rsi14"] < rsi_buy)
    sell_cond = (out["rsi14"] > rsi_sell) | (close < out["ma50"])

    out["raw_signal"] = np.nan
    out.loc[buy_cond, "raw_signal"] = 1
    out.loc[sell_cond, "raw_signal"] = 0
    out["signal"] = out["raw_signal"].ffill().fillna(0)
    return out


def strategy_breakout_trend(df, breakout_window=20, exit_window=10, **kwargs):
    """
    趋势突破：
    入场：
      - close > 前20日最高
      - ma50 > ma200
    出场：
      - close < 前10日最低
      - 或 ma50 < ma200
    """
    out = add_indicators(df)
    close = ensure_series(out["close"], "close")

    entry_level = close.shift(1).rolling(breakout_window).max()
    exit_level = close.shift(1).rolling(exit_window).min()

    buy_cond = (close > entry_level) & (out["ma50"] > out["ma200"])
    sell_cond = (close < exit_level) | (out["ma50"] < out["ma200"])

    out["raw_signal"] = np.nan
    out.loc[buy_cond, "raw_signal"] = 1
    out.loc[sell_cond, "raw_signal"] = 0
    out["signal"] = out["raw_signal"].ffill().fillna(0)
    return out


# =========================
# 4. 策略调度器
# =========================
def apply_strategy(df, strategy_name, **params):
    strategies = {
        "buy_and_hold": strategy_buy_and_hold,
        "ma_cross": strategy_ma_cross,
        "rsi_reversion": strategy_rsi_reversion,
        "trend_pullback": strategy_trend_pullback,
        "breakout_trend": strategy_breakout_trend,
    }

    if strategy_name not in strategies:
        raise ValueError(f"未知策略: {strategy_name}，可选: {list(strategies.keys())}")

    return strategies[strategy_name](df, **params)


# =========================
# 5. 带风控的回测核心
# =========================
def backtest(
    df,
    strategy_name="ma_cross",
    initial_capital=10000,
    trading_cost=0.0005,
    stop_loss_pct=0.08,
    trailing_stop_pct=0.12,
    max_holding_days=None,
    **strategy_params
):
    data = apply_strategy(df, strategy_name, **strategy_params).copy()

    close = ensure_series(data["close"], "close")
    data["return"] = close.pct_change().fillna(0)

    # 用循环处理止损 / 移动止损 / 最大持仓天数
    position = []
    strategy_return = []
    trade = []

    in_position = 0
    entry_price = np.nan
    highest_price = np.nan
    holding_days = 0
    prev_position = 0

    for i in range(len(data)):
        today_close = close.iloc[i]
        today_signal = data["signal"].iloc[i]

        # 默认今天仓位延续昨天
        current_position = in_position

        # 先处理已有持仓的风控
        if in_position == 1:
            holding_days += 1
            highest_price = max(highest_price, today_close)

            stop_trigger = False if stop_loss_pct is None else (today_close <= entry_price * (1 - stop_loss_pct))
            trail_trigger = False if trailing_stop_pct is None else (
                        today_close <= highest_price * (1 - trailing_stop_pct))
            time_trigger = (max_holding_days is not None) and (holding_days >= max_holding_days)
            signal_exit = today_signal == 0

            if stop_trigger or trail_trigger or time_trigger or signal_exit:
                current_position = 0
                in_position = 0
                entry_price = np.nan
                highest_price = np.nan
                holding_days = 0

        # 空仓时，根据信号决定是否开仓
        if in_position == 0 and today_signal == 1:
            current_position = 1
            in_position = 1
            entry_price = today_close
            highest_price = today_close
            holding_days = 0

        position.append(current_position)

        today_trade = abs(current_position - prev_position)
        trade.append(today_trade)
        prev_position = current_position

    data["position"] = pd.Series(position, index=data.index).shift(1).fillna(0)
    data["trade"] = pd.Series(trade, index=data.index).fillna(0)

    data["strategy_return"] = data["position"] * data["return"] - data["trade"] * trading_cost
    data["benchmark_equity"] = initial_capital * (1 + data["return"]).cumprod()
    data["strategy_equity"] = initial_capital * (1 + data["strategy_return"]).cumprod()

    return data


# =========================
# 6. 交易明细提取
# =========================
def extract_trades(result):
    close = ensure_series(result["close"], "close")
    pos = ensure_series(result["position"], "position")

    trades = []
    entry_date = None
    entry_price = None

    for i in range(1, len(result)):
        prev_pos = pos.iloc[i - 1]
        curr_pos = pos.iloc[i]
        date = result.index[i]

        # 开仓
        if prev_pos == 0 and curr_pos == 1:
            entry_date = date
            entry_price = close.iloc[i]

        # 平仓
        elif prev_pos == 1 and curr_pos == 0 and entry_date is not None:
            exit_date = date
            exit_price = close.iloc[i]
            ret = exit_price / entry_price - 1
            hold_days = (exit_date - entry_date).days

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "return": ret,
                "hold_days": hold_days
            })

            entry_date = None
            entry_price = None

    trades_df = pd.DataFrame(trades)
    return trades_df


# =========================
# 7. 绩效指标
# =========================
def calculate_metrics(equity_series, return_series, trading_days=252):
    equity_series = ensure_series(equity_series, "equity_series").dropna()
    return_series = ensure_series(return_series, "return_series").dropna()

    total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
    n = len(return_series)

    annual_return = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (trading_days / max(n, 1)) - 1

    running_max = equity_series.cummax()
    drawdown = equity_series / running_max - 1
    max_drawdown = drawdown.min()

    vol = return_series.std() * np.sqrt(trading_days)
    sharpe = np.nan
    if pd.notna(vol) and vol != 0:
        sharpe = (return_series.mean() * trading_days) / vol

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
    }


def calculate_trade_stats(trades_df):
    if trades_df.empty:
        return {
            "trade_count": 0,
            "win_rate": np.nan,
            "avg_trade_return": np.nan,
            "avg_hold_days": np.nan,
        }

    wins = (trades_df["return"] > 0).sum()
    trade_count = len(trades_df)

    return {
        "trade_count": int(trade_count),
        "win_rate": float(wins / trade_count),
        "avg_trade_return": float(trades_df["return"].mean()),
        "avg_hold_days": float(trades_df["hold_days"].mean()),
    }


def print_metrics(name, metrics, trade_stats=None):
    print(f"\n===== {name} =====")
    print(f"总收益:       {metrics['total_return']:.2%}")
    print(f"年化收益:     {metrics['annual_return']:.2%}")
    print(f"最大回撤:     {metrics['max_drawdown']:.2%}")
    print(f"Sharpe:       {metrics['sharpe']:.2f}")

    if trade_stats is not None:
        print(f"交易次数:     {trade_stats['trade_count']}")
        if pd.notna(trade_stats["win_rate"]):
            print(f"胜率:         {trade_stats['win_rate']:.2%}")
            print(f"平均单笔收益: {trade_stats['avg_trade_return']:.2%}")
            print(f"平均持仓天数: {trade_stats['avg_hold_days']:.1f}")


# =========================
# 8. 单策略运行
# =========================
def run_backtest(
    symbol="QQQ",
    start="2015-01-01",
    end="2025-01-01",
    interval="1d",
    strategy_name="trend_pullback",
    initial_capital=10000,
    trading_cost=0.0005,
    auto_adjust=True,
    plot=False,
    export_trades=True,
    stop_loss_pct=0.08,
    trailing_stop_pct=0.12,
    max_holding_days=None,
    **strategy_params
):
    df = get_data(
        symbol=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust
    )

    result = backtest(
        df=df,
        strategy_name=strategy_name,
        initial_capital=initial_capital,
        trading_cost=trading_cost,
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
        max_holding_days=max_holding_days,
        **strategy_params
    )

    strategy_metrics = calculate_metrics(
        equity_series=result["strategy_equity"],
        return_series=result["strategy_return"]
    )

    benchmark_metrics = calculate_metrics(
        equity_series=result["benchmark_equity"],
        return_series=result["return"]
    )

    trades_df = extract_trades(result)
    trade_stats = calculate_trade_stats(trades_df)

    print(f"\nTicker: {symbol}")
    print(f"策略: {strategy_name}")
    print(f"区间: {start} -> {end}")
    print(f"参数: {strategy_params}")
    print(f"风控: stop_loss={stop_loss_pct}, trailing_stop={trailing_stop_pct}, max_holding_days={max_holding_days}")

    print_metrics("策略表现", strategy_metrics, trade_stats)
    print_metrics("买入持有", benchmark_metrics)

    if export_trades and not trades_df.empty:
        trades_file = RESULT_DIR / f"{symbol}_{strategy_name}_trades.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"[导出] 交易明细已保存: {trades_file}")

    if plot:
        plot_result(result, title=f"{symbol} - {strategy_name}")

    return result, strategy_metrics, benchmark_metrics, trades_df


# =========================
# 9. 跑所有策略
# =========================
def run_all_strategies(
    symbol="QQQ",
    start="2015-01-01",
    end="2025-01-01",
    interval="1d",
    initial_capital=10000,
    trading_cost=0.0005,
    auto_adjust=True,
    plot=True
):
    df = get_data(
        symbol=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust
    )

    strategy_configs = {
        "buy_and_hold": {},
        "ma_cross": {
            "short_window": 20,
            "long_window": 50
        },
        "rsi_reversion": {
            "rsi_window": 14,
            "buy_threshold": 30,
            "sell_threshold": 70
        },
        "trend_pullback": {
            "rsi_buy": 35,
            "rsi_sell": 60
        },
        "breakout_trend": {
            "breakout_window": 20,
            "exit_window": 10
        }
    }

    summary = []
    results = {}

    close = ensure_series(df["close"], "close")
    benchmark_returns = close.pct_change().fillna(0)
    benchmark_equity = initial_capital * (1 + benchmark_returns).cumprod()
    benchmark_metrics = calculate_metrics(benchmark_equity, benchmark_returns)

    for strategy_name, params in strategy_configs.items():
        result = backtest(
            df=df,
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            trading_cost=trading_cost,
            stop_loss_pct=0.08,
            trailing_stop_pct=0.12,
            max_holding_days=80,
            **params
        )

        metrics = calculate_metrics(
            equity_series=result["strategy_equity"],
            return_series=result["strategy_return"]
        )

        trades_df = extract_trades(result)
        trade_stats = calculate_trade_stats(trades_df)

        results[strategy_name] = result
        summary.append({
            "strategy": strategy_name,
            "total_return": metrics["total_return"],
            "annual_return": metrics["annual_return"],
            "max_drawdown": metrics["max_drawdown"],
            "sharpe": metrics["sharpe"],
            "trade_count": trade_stats["trade_count"],
            "win_rate": trade_stats["win_rate"],
            "avg_hold_days": trade_stats["avg_hold_days"],
        })

    summary_df = pd.DataFrame(summary).sort_values("annual_return", ascending=False).reset_index(drop=True)

    print(f"\n=== {symbol} 所有策略回测结果 ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 买入持有基准 ===")
    print(pd.DataFrame([{
        "strategy": "benchmark_buy_and_hold",
        "total_return": benchmark_metrics["total_return"],
        "annual_return": benchmark_metrics["annual_return"],
        "max_drawdown": benchmark_metrics["max_drawdown"],
        "sharpe": benchmark_metrics["sharpe"],
    }]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary_file = RESULT_DIR / f"{symbol}_strategy_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"[导出] 汇总结果已保存: {summary_file}")

    if plot:
        plot_all_results(results, df, initial_capital, symbol)

    return summary_df, results, benchmark_metrics


# =========================
# 10. 画图
# =========================
def plot_result(result, title="Backtest Result"):
    plt.figure(figsize=(12, 6))
    plt.plot(result.index, result["strategy_equity"], label="Strategy Equity")
    plt.plot(result.index, result["benchmark_equity"], label="Buy & Hold Equity")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_all_results(results, df, initial_capital=10000, symbol="QQQ"):
    close = ensure_series(df["close"], "close")
    benchmark_returns = close.pct_change().fillna(0)
    benchmark_equity = initial_capital * (1 + benchmark_returns).cumprod()

    plt.figure(figsize=(14, 7))
    plt.plot(benchmark_equity.index, benchmark_equity.values, label="benchmark_buy_and_hold", linewidth=2)

    for strategy_name, result in results.items():
        plt.plot(result.index, result["strategy_equity"], label=strategy_name)

    plt.title(f"{symbol} - All Strategy Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================
# 11. 可选：参数扫描
# =========================
def optimize_trend_pullback(
    symbol="QQQ",
    start="2015-01-01",
    end="2025-01-01",
    interval="1d",
    rsi_buy_list=(30, 35, 40),
    rsi_sell_list=(55, 60, 65),
):
    df = get_data(symbol=symbol, start=start, end=end, interval=interval, auto_adjust=True)
    rows = []

    for rsi_buy in rsi_buy_list:
        for rsi_sell in rsi_sell_list:
            result = backtest(
                df=df,
                strategy_name="trend_pullback",
                initial_capital=10000,
                trading_cost=0.0005,
                stop_loss_pct=0.08,
                trailing_stop_pct=0.12,
                max_holding_days=80,
                rsi_buy=rsi_buy,
                rsi_sell=rsi_sell
            )

            metrics = calculate_metrics(result["strategy_equity"], result["strategy_return"])
            trades_df = extract_trades(result)
            trade_stats = calculate_trade_stats(trades_df)

            rows.append({
                "rsi_buy": rsi_buy,
                "rsi_sell": rsi_sell,
                "annual_return": metrics["annual_return"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe": metrics["sharpe"],
                "trade_count": trade_stats["trade_count"],
                "win_rate": trade_stats["win_rate"],
            })

    opt_df = pd.DataFrame(rows).sort_values(["sharpe", "annual_return"], ascending=False).reset_index(drop=True)
    out_file = RESULT_DIR / f"{symbol}_trend_pullback_optimization.csv"
    opt_df.to_csv(out_file, index=False)
    print(opt_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"[导出] 参数优化结果已保存: {out_file}")
    return opt_df


# =========================
# 12. 清除缓存（可选）
# =========================
def delete_cache(symbol="QQQ", start="2015-01-01", end="2025-01-01", interval="1d", auto_adjust=True):
    cache_file = get_cache_file(symbol, start, end, interval, auto_adjust)
    if cache_file.exists():
        cache_file.unlink()
        print(f"[删除] 缓存已删除: {cache_file}")
    else:
        print(f"[提示] 缓存不存在: {cache_file}")

from itertools import product
from pathlib import Path

# =========================
# 14. 时间切分
# =========================
def split_train_test(df, train_start="2015-01-01", train_end="2022-12-31",
                     test_start="2023-01-01", test_end="2026-12-31"):
    train_df = df.loc[train_start:train_end].copy()
    test_df = df.loc[test_start:test_end].copy()

    if train_df.empty:
        raise ValueError("训练集为空，请检查时间范围。")
    if test_df.empty:
        raise ValueError("测试集为空，请检查时间范围。")

    return train_df, test_df


# =========================
# 15. 评价指标增强
# =========================
def calculate_extended_metrics(result, trades_df=None, trading_days=252):
    metrics = calculate_metrics(
        equity_series=result["strategy_equity"],
        return_series=result["strategy_return"],
        trading_days=trading_days
    )

    annual_return = metrics["annual_return"]
    max_drawdown = abs(metrics["max_drawdown"]) if metrics["max_drawdown"] < 0 else np.nan
    calmar = np.nan
    if pd.notna(max_drawdown) and max_drawdown != 0:
        calmar = annual_return / max_drawdown

    sortino = np.nan
    downside = result["strategy_return"].copy()
    downside = downside[downside < 0]
    if len(downside) > 0:
        downside_std = downside.std() * np.sqrt(trading_days)
        if pd.notna(downside_std) and downside_std != 0:
            sortino = (result["strategy_return"].mean() * trading_days) / downside_std

    if trades_df is None:
        trades_df = extract_trades(result)

    trade_stats = calculate_trade_stats(trades_df)

    profit_factor = np.nan
    if not trades_df.empty:
        gross_profit = trades_df.loc[trades_df["return"] > 0, "return"].sum()
        gross_loss = -trades_df.loc[trades_df["return"] < 0, "return"].sum()
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss

    out = {
        **metrics,
        "calmar": float(calmar) if pd.notna(calmar) else np.nan,
        "sortino": float(sortino) if pd.notna(sortino) else np.nan,
        "profit_factor": float(profit_factor) if pd.notna(profit_factor) else np.nan,
        "trade_count": trade_stats["trade_count"],
        "win_rate": trade_stats["win_rate"],
        "avg_trade_return": trade_stats["avg_trade_return"],
        "avg_hold_days": trade_stats["avg_hold_days"],
    }
    return out


# =========================
# 16. 参数网格定义
# 你可以继续加密一点，但注意组合数会爆炸
# =========================
def get_param_grid(strategy_name):
    grids = {
        "ma_cross": {
            "short_window": [5, 10, 15, 20, 25, 30, 40, 50],
            "long_window":  [40, 50, 60, 80, 100, 120, 150, 200],
        },
        "rsi_reversion": {
            "rsi_window":     [7, 10, 14, 21],
            "buy_threshold":  [20, 25, 30, 35, 40],
            "sell_threshold": [55, 60, 65, 70, 75, 80],
        },
        "trend_pullback": {
            "rsi_buy":  [25, 30, 35, 40, 45, 50],
            "rsi_sell": [55, 60, 65, 70, 75, 80],
        },
        "breakout_trend": {
            "breakout_window": [10, 15, 20, 30, 40, 55, 80, 100],
            "exit_window":     [3, 5, 10, 15, 20, 30],
        }
    }

    if strategy_name not in grids:
        raise ValueError(f"未定义参数网格: {strategy_name}")

    return grids[strategy_name]


def generate_param_combinations(strategy_name):
    grid = get_param_grid(strategy_name)
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    combos = []
    for vals in product(*values):
        params = dict(zip(keys, vals))

        # 合法性约束
        if strategy_name == "ma_cross":
            if params["short_window"] >= params["long_window"]:
                continue

        if strategy_name == "rsi_reversion":
            if params["buy_threshold"] >= params["sell_threshold"]:
                continue

        if strategy_name == "trend_pullback":
            if params["rsi_buy"] >= params["rsi_sell"]:
                continue

        if strategy_name == "breakout_trend":
            if params["exit_window"] >= params["breakout_window"]:
                continue

        combos.append(params)

    return combos


# =========================
# 17. 参数邻域稳健性
# 逻辑：
# 一个参数点如果自己不错，且附近参数也不错，就更稳健
# =========================
def param_distance(p1, p2):
    keys = sorted(list(p1.keys()))
    dist = 0
    for k in keys:
        dist += abs(float(p1[k]) - float(p2[k]))
    return dist


def add_robustness_score(df_result):
    """
    输入:
        df_result 需要至少包含:
        - params
        - annual_return
        - sharpe
        - calmar
        - max_drawdown
        - trade_count
    输出:
        增加:
        - base_score
        - neighbor_score
        - robustness_score
    """
    out = df_result.copy()

    # 先做基础分
    # 这里不直接迷信收益，综合看 Sharpe / Calmar / 年化 / 回撤 / 交易数
    def zscore(s):
        s = s.astype(float)
        std = s.std()
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    out["z_annual"] = zscore(out["annual_return"])
    out["z_sharpe"] = zscore(out["sharpe"])
    out["z_calmar"] = zscore(out["calmar"].fillna(out["calmar"].median()))
    out["z_mdd"] = zscore(-out["max_drawdown"].abs())  # 回撤越小越好
    out["z_trade"] = zscore(np.minimum(out["trade_count"], 80))  # 过多交易不额外加分

    out["base_score"] = (
        0.25 * out["z_annual"] +
        0.30 * out["z_sharpe"] +
        0.25 * out["z_calmar"] +
        0.15 * out["z_mdd"] +
        0.05 * out["z_trade"]
    )

    # 邻域分：看距离最近的若干个参数点平均表现
    neighbor_scores = []
    params_list = out["params"].tolist()
    base_scores = out["base_score"].tolist()

    for i, p in enumerate(params_list):
        dists = []
        for j, q in enumerate(params_list):
            if i == j:
                continue
            d = param_distance(p, q)
            dists.append((j, d))

        dists = sorted(dists, key=lambda x: x[1])

        # 取最近的 8 个邻居
        neighbors = [idx for idx, _ in dists[:8]]
        if len(neighbors) == 0:
            neighbor_scores.append(base_scores[i])
        else:
            neighbor_scores.append(np.mean([base_scores[j] for j in neighbors]))

    out["neighbor_score"] = neighbor_scores
    out["robustness_score"] = 0.6 * out["base_score"] + 0.4 * out["neighbor_score"]

    return out


# =========================
# 18. 单策略参数扫描（训练集）
# =========================
def scan_strategy_parameters(
    df_train,
    strategy_name,
    initial_capital=10000,
    trading_cost=0.0005,
    stop_loss_pct=0.08,
    trailing_stop_pct=0.12,
    max_holding_days=80,
    min_trade_count=5,
    result_dir="results",
):
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)

    combos = generate_param_combinations(strategy_name)
    rows = []

    total = len(combos)
    print(f"\n[开始扫描] {strategy_name}，参数组合数: {total}")

    for idx, params in enumerate(combos, start=1):
        if idx % 100 == 0 or idx == total:
            print(f"  进度: {idx}/{total}")

        result = backtest(
            df=df_train,
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            trading_cost=trading_cost,
            stop_loss_pct=stop_loss_pct,
            trailing_stop_pct=trailing_stop_pct,
            max_holding_days=max_holding_days,
            **params
        )

        trades_df = extract_trades(result)
        ext = calculate_extended_metrics(result, trades_df=trades_df)

        # 过滤太少交易的参数点
        if ext["trade_count"] < min_trade_count and strategy_name != "ma_cross":
            # ma_cross 有时本来交易就少，所以放松
            pass

        row = {
            "strategy": strategy_name,
            "params": params,
            **ext
        }
        rows.append(row)

    df_result = pd.DataFrame(rows)
    df_result = add_robustness_score(df_result)

    # 导出完整扫描表
    scan_file = result_dir / f"{strategy_name}_train_scan.csv"
    export_df = df_result.copy()
    export_df["params"] = export_df["params"].astype(str)
    export_df.to_csv(scan_file, index=False)
    print(f"[导出] 训练集扫描结果已保存: {scan_file}")

    # 只看更稳健的前若干个
    df_sorted = df_result.sort_values(
        ["robustness_score", "base_score", "sharpe", "annual_return"],
        ascending=False
    ).reset_index(drop=True)

    return df_sorted


# =========================
# 19. 训练集选参数，测试集验证
# =========================
def validate_best_params_on_test(
    df_train,
    df_test,
    strategy_name,
    top_n=10,
    initial_capital=10000,
    trading_cost=0.0005,
    stop_loss_pct=0.08,
    trailing_stop_pct=0.12,
    max_holding_days=80,
    result_dir="results",
):
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)

    train_scan = scan_strategy_parameters(
        df_train=df_train,
        strategy_name=strategy_name,
        initial_capital=initial_capital,
        trading_cost=trading_cost,
        stop_loss_pct=stop_loss_pct,
        trailing_stop_pct=trailing_stop_pct,
        max_holding_days=max_holding_days,
        result_dir=result_dir,
    )

    top_candidates = train_scan.head(top_n).copy()

    test_rows = []
    for _, row in top_candidates.iterrows():
        params = row["params"]

        result_test = backtest(
            df=df_test,
            strategy_name=strategy_name,
            initial_capital=initial_capital,
            trading_cost=trading_cost,
            stop_loss_pct=stop_loss_pct,
            trailing_stop_pct=trailing_stop_pct,
            max_holding_days=max_holding_days,
            **params
        )

        trades_test = extract_trades(result_test)
        ext_test = calculate_extended_metrics(result_test, trades_df=trades_test)

        test_rows.append({
            "strategy": strategy_name,
            "params": params,
            "train_robustness_score": row["robustness_score"],
            "train_annual_return": row["annual_return"],
            "train_sharpe": row["sharpe"],
            "train_calmar": row["calmar"],
            "train_max_drawdown": row["max_drawdown"],
            "train_trade_count": row["trade_count"],
            "test_annual_return": ext_test["annual_return"],
            "test_sharpe": ext_test["sharpe"],
            "test_calmar": ext_test["calmar"],
            "test_max_drawdown": ext_test["max_drawdown"],
            "test_trade_count": ext_test["trade_count"],
            "test_win_rate": ext_test["win_rate"],
        })

    df_test_eval = pd.DataFrame(test_rows)

    # 给测试集也做一个综合分
    def zscore(s):
        s = s.astype(float)
        std = s.std()
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / std

    df_test_eval["test_score"] = (
        0.30 * zscore(df_test_eval["test_sharpe"]) +
        0.25 * zscore(df_test_eval["test_calmar"].fillna(df_test_eval["test_calmar"].median())) +
        0.25 * zscore(df_test_eval["test_annual_return"]) +
        0.20 * zscore(-df_test_eval["test_max_drawdown"].abs())
    )

    df_test_eval = df_test_eval.sort_values(
        ["test_score", "test_sharpe", "test_annual_return"],
        ascending=False
    ).reset_index(drop=True)

    test_file = result_dir / f"{strategy_name}_train_test_validation.csv"
    export_df = df_test_eval.copy()
    export_df["params"] = export_df["params"].astype(str)
    export_df.to_csv(test_file, index=False)
    print(f"[导出] 训练/测试验证结果已保存: {test_file}")

    return train_scan, df_test_eval


# =========================
# 20. 扫全部策略
# =========================
def run_full_parameter_research(
    symbol="QQQ",
    start="2015-01-01",
    end="2026-12-31",
    interval="1d",
    train_start="2015-01-01",
    train_end="2022-12-31",
    test_start="2023-01-01",
    test_end="2026-12-31",
    strategies=("ma_cross", "rsi_reversion", "trend_pullback", "breakout_trend"),
    top_n=10,
    initial_capital=10000,
    trading_cost=0.0005,
    stop_loss_pct=0.08,
    trailing_stop_pct=0.12,
    max_holding_days=80,
    result_dir="results",
):
    df = get_data(
        symbol=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True
    )

    df_train, df_test = split_train_test(
        df,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end
    )

    all_best_rows = []

    for strategy_name in strategies:
        print("\n" + "=" * 80)
        print(f"策略研究: {strategy_name}")
        print("=" * 80)

        train_scan, test_eval = validate_best_params_on_test(
            df_train=df_train,
            df_test=df_test,
            strategy_name=strategy_name,
            top_n=top_n,
            initial_capital=initial_capital,
            trading_cost=trading_cost,
            stop_loss_pct=stop_loss_pct,
            trailing_stop_pct=trailing_stop_pct,
            max_holding_days=max_holding_days,
            result_dir=result_dir,
        )

        best_row = test_eval.iloc[0].copy()
        all_best_rows.append(best_row)

        print("\n[训练集最稳健前5]")
        show_train = train_scan.head(5)[[
            "params", "annual_return", "sharpe", "calmar", "max_drawdown",
            "trade_count", "robustness_score"
        ]].copy()
        show_train["params"] = show_train["params"].astype(str)
        print(show_train.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print("\n[测试集表现前5]")
        show_test = test_eval.head(5)[[
            "params", "train_robustness_score",
            "test_annual_return", "test_sharpe", "test_calmar",
            "test_max_drawdown", "test_trade_count", "test_win_rate", "test_score"
        ]].copy()
        show_test["params"] = show_test["params"].astype(str)
        print(show_test.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary = pd.DataFrame(all_best_rows)
    summary = summary.sort_values(
        ["test_score", "test_sharpe", "test_annual_return"],
        ascending=False
    ).reset_index(drop=True)

    summary_file = Path(result_dir) / f"{symbol}_all_strategy_best_params_summary.csv"
    export_df = summary.copy()
    export_df["params"] = export_df["params"].astype(str)
    export_df.to_csv(summary_file, index=False)

    print("\n" + "#" * 80)
    print("全部策略最优稳健参数汇总（按测试集排序）")
    print("#" * 80)

    show_summary = summary[[
        "strategy", "params", "train_robustness_score",
        "test_annual_return", "test_sharpe", "test_calmar",
        "test_max_drawdown", "test_trade_count", "test_win_rate", "test_score"
    ]].copy()
    show_summary["params"] = show_summary["params"].astype(str)
    print(show_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n[导出] 全策略汇总已保存: {summary_file}")

    return summary



if __name__ == "__main__":
    summary = run_full_parameter_research(
        symbol="QQQ",
        start="2015-01-01",
        end="2026-12-31",
        interval="1d",

        train_start="2015-01-01",
        train_end="2022-12-31",
        test_start="2023-01-01",
        test_end="2026-12-31",

        strategies=("ma_cross", "rsi_reversion", "trend_pullback", "breakout_trend"),
        top_n=10,

        initial_capital=10000,
        trading_cost=0.0005,

        stop_loss_pct=0.08,
        trailing_stop_pct=0.12,
        max_holding_days=80,

        result_dir="results"
    )