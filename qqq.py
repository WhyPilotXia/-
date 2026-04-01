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


# =========================
# 1. 数据获取（带缓存）
# 仅当缓存不存在时下载
# =========================
def get_cache_file(symbol, start, end, interval="1d", auto_adjust=True):
    adj_flag = "adj" if auto_adjust else "raw"
    fname = f"{symbol}_{start}_{end}_{interval}_{adj_flag}.csv"
    return DATA_DIR / fname


def normalize_columns(df):
    # 兼容 yfinance 返回列名
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
    """
    确保输入是 pandas Series。
    如果是单列 DataFrame，就转成 Series；
    如果不是单列，直接报错。
    """
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

    # 如果是 MultiIndex 列，拍平成单层
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
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# =========================
# 3. 策略函数
# signal:
#   1 = 持有多头
#   0 = 空仓
# =========================
def strategy_buy_and_hold(df, **kwargs):
    out = df.copy()
    out["signal"] = 1
    return out


def strategy_ma_cross(df, short_window=20, long_window=50, **kwargs):
    out = df.copy()
    out["ma_short"] = out["close"].rolling(short_window).mean()
    out["ma_long"] = out["close"].rolling(long_window).mean()

    out["signal"] = 0
    out.loc[out["ma_short"] > out["ma_long"], "signal"] = 1
    return out


def strategy_rsi_reversion(df, rsi_window=14, buy_threshold=30, sell_threshold=70, **kwargs):
    out = df.copy()
    out["rsi"] = calc_rsi(out["close"], window=rsi_window)

    out["raw_signal"] = np.nan
    out.loc[out["rsi"] < buy_threshold, "raw_signal"] = 1
    out.loc[out["rsi"] > sell_threshold, "raw_signal"] = 0

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
    }

    if strategy_name not in strategies:
        raise ValueError(f"未知策略: {strategy_name}，可选: {list(strategies.keys())}")

    return strategies[strategy_name](df, **params)


# =========================
# 5. 回测核心
# =========================
def backtest(
    df,
    strategy_name="ma_cross",
    initial_capital=10000,
    trading_cost=0.0005,   # 单边交易成本
    **strategy_params
):
    data = apply_strategy(df, strategy_name, **strategy_params).copy()

    close = ensure_series(data["close"], "close")
    data["return"] = close.pct_change().fillna(0)

    # 用昨天信号决定今天仓位，避免未来函数
    data["position"] = data["signal"].shift(1).fillna(0)

    # 交易次数变化
    data["trade"] = data["position"].diff().abs().fillna(0)

    # 策略收益
    data["strategy_return"] = data["position"] * data["return"] - data["trade"] * trading_cost

    # 净值
    data["benchmark_equity"] = initial_capital * (1 + data["return"]).cumprod()
    data["strategy_equity"] = initial_capital * (1 + data["strategy_return"]).cumprod()

    return data


# =========================
# 6. 绩效指标
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


def print_metrics(name, metrics):
    print(f"\n===== {name} =====")
    print(f"总收益:     {metrics['total_return']:.2%}")
    print(f"年化收益:   {metrics['annual_return']:.2%}")
    print(f"最大回撤:   {metrics['max_drawdown']:.2%}")
    print(f"Sharpe:     {metrics['sharpe']:.2f}")


# =========================
# 7. 单策略运行
# =========================
def run_backtest(
    symbol="QQQ",
    start="2015-01-01",
    end="2025-01-01",
    interval="1d",
    strategy_name="ma_cross",
    initial_capital=10000,
    trading_cost=0.0005,
    auto_adjust=True,
    plot=False,
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

    print(f"\nTicker: {symbol}")
    print(f"策略: {strategy_name}")
    print(f"区间: {start} -> {end}")
    print(f"参数: {strategy_params}")

    print_metrics("策略表现", strategy_metrics)
    print_metrics("买入持有", benchmark_metrics)

    if plot:
        plot_result(result, title=f"{symbol} - {strategy_name}")

    return result, strategy_metrics, benchmark_metrics


# =========================
# 8. 跑所有策略
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
        }
    }

    summary = []
    results = {}

    # 基准只算一次
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
            **params
        )

        metrics = calculate_metrics(
            equity_series=result["strategy_equity"],
            return_series=result["strategy_return"]
        )

        results[strategy_name] = result
        summary.append({
            "strategy": strategy_name,
            "total_return": metrics["total_return"],
            "annual_return": metrics["annual_return"],
            "max_drawdown": metrics["max_drawdown"],
            "sharpe": metrics["sharpe"],
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

    if plot:
        plot_all_results(results, df, initial_capital, symbol)

    return summary_df, results, benchmark_metrics


# =========================
# 9. 画图
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
    benchmark_returns = df["close"].pct_change().fillna(0)
    benchmark_equity = initial_capital * (1 + benchmark_returns).cumprod()

    plt.figure(figsize=(14, 7))
    plt.plot(benchmark_equity.index, benchmark_equity.values, label="benchmark_buy_and_hold")

    for strategy_name, result in results.items():
        plt.plot(result.index, result["strategy_equity"], label=strategy_name)

    plt.title(f"{symbol} - All Strategy Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================
# 10. 清除某个缓存文件（可选）
# =========================
def delete_cache(symbol="QQQ", start="2015-01-01", end="2025-01-01", interval="1d", auto_adjust=True):
    cache_file = get_cache_file(symbol, start, end, interval, auto_adjust)
    if cache_file.exists():
        cache_file.unlink()
        print(f"[删除] 缓存已删除: {cache_file}")
    else:
        print(f"[提示] 缓存不存在: {cache_file}")


# =========================
# 11. 主程序
# =========================
if __name__ == "__main__":
    # 跑所有策略
    summary_df, results, benchmark_metrics = run_all_strategies(
        symbol="QQQ",
        start="2015-01-01",
        end="2025-01-01",
        interval="1d",
        initial_capital=10000,
        trading_cost=0.0005,
        auto_adjust=True,
        plot=True
    )

    # 单独跑某个策略示例
    # result, strategy_metrics, benchmark_metrics = run_backtest(
    #     symbol="QQQ",
    #     start="2015-01-01",
    #     end="2025-01-01",
    #     strategy_name="ma_cross",
    #     short_window=20,
    #     long_window=50,
    #     trading_cost=0.0005,
    #     plot=True
    # )