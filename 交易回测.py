import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import jqdatasdk as jq

# ==============================
# 一、聚宽 API 登录
# ==============================
jq.auth('15684243649', 'Lxm052596')
assert jq.is_auth(), "聚宽 API 登录失败"

# ==============================
# 二、全局参数
# ==============================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 16

initial_capital = 2000000
contract_size = 1000
stop_loss_ratio = 0.02
threshold = 0.01
risk_free_rate = 0.02
models_to_use = ['BiLSTM-Attention', 'Transformer']

# Excel文件及sheet
file_path = r"E:\lw\代码\prediction_results_all_windows.xlsx"
sheet_names = ['Window_1', 'Window_5', 'Window_15', 'Window_30']
window_labels = ['Window=1', 'Window=5', 'Window=15', 'Window=30']
symbol = "SC9999.XINE"  # 上海原油连续合约

# ==============================
# 三、函数定义
# ==============================

def load_sheet_data(file_path, sheet_name):
    """读取单个sheet并预处理"""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.rename(columns={'trade_date': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def fetch_real_prices(start_date, end_date):
    """从聚宽获取收盘价"""
    price_df = jq.get_price(
        symbol,
        start_date=start_date,
        end_date=end_date,
        frequency='daily',
        fields=["close"],
        skip_paused=True
    )
    price_df.rename(columns={'close':'Actual'}, inplace=True)
    return price_df

def generate_trading_signals(df, model_col, threshold=0.01):
    """生成交易信号，1=买入, -1=卖出, 0=持仓"""
    df_signal = df.copy()
    df_signal['prev_actual'] = df_signal['Actual'].shift(1)
    df_signal['pred_return'] = (df_signal[model_col] / df_signal['prev_actual']) - 1
    df_signal['signal'] = 0
    df_signal.loc[df_signal['pred_return'] > threshold, 'signal'] = 1
    df_signal.loc[df_signal['pred_return'] < -threshold, 'signal'] = -1
    df_signal.loc[0, 'signal'] = 0
    return df_signal

def backtest_strategy(df_signal, initial_capital=2000000, contract_size=1000,
                      stop_loss_ratio=0.02, risk_free_rate=0.02):
    """策略回测"""
    cash = initial_capital
    position = 0
    trades = []
    daily_value = []
    max_portfolio_value = initial_capital
    drawdowns = []

    for idx, row in df_signal.iterrows():
        current_price = row['Actual']
        signal = row['signal']

        # 买入
        if signal == 1 and position == 0:
            cost = current_price * contract_size
            if cash >= cost:
                position = 1
                cash -= cost
                entry_price = current_price
                trades.append({'date': row['Date'], 'action': 'buy', 'price': current_price})

        # 卖出平仓
        elif signal == -1 and position == 1:
            revenue = current_price * contract_size
            cash += revenue
            position = 0
            trades.append({'date': row['Date'], 'action': 'sell', 'price': current_price})

        # 止损
        if position == 1:
            loss_ratio = (current_price - entry_price) / entry_price
            if loss_ratio < -stop_loss_ratio:
                revenue = current_price * contract_size
                cash += revenue
                position = 0
                trades.append({'date': row['Date'], 'action': 'stop_loss_sell', 'price': current_price})

        portfolio_value = cash + (position * current_price * contract_size)
        daily_value.append(portfolio_value)
        if portfolio_value > max_portfolio_value:
            max_portfolio_value = portfolio_value
        drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value
        drawdowns.append(drawdown)

    final_portfolio_value = cash + (position * df_signal.iloc[-1]['Actual'] * contract_size)
    total_return = (final_portfolio_value - initial_capital) / initial_capital
    trading_days = len(df_signal)
    annual_return = (1 + total_return) ** (252 / trading_days) - 1
    trade_count = len(trades)
    daily_returns = np.diff(daily_value) / daily_value[:-1] if len(daily_value) > 1 else []
    sharpe_ratio = (np.mean(daily_returns)*252 - risk_free_rate) / (np.std(daily_returns)*np.sqrt(252)) if len(daily_returns) > 0 else 0
    max_drawdown = max(drawdowns) if drawdowns else 0

    return {
        'final_portfolio_value': final_portfolio_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'trade_count': trade_count,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'daily_value': daily_value,
        'trades': trades
    }

# ==============================
# 四、回测主程序
# ==============================
all_results = {model: [] for model in models_to_use}

for sheet, label in zip(sheet_names, window_labels):
    df = load_sheet_data(file_path, sheet)
    # 从聚宽获取真实收盘价
    start = df['Date'].min().strftime('%Y-%m-%d')
    end = df['Date'].max().strftime('%Y-%m-%d')
    real_prices = fetch_real_prices(start, end)
    df = df.set_index('Date').join(real_prices, how='inner').reset_index()

    for model in models_to_use:
        df_signal = generate_trading_signals(df, model, threshold)
        results = backtest_strategy(df_signal, initial_capital, contract_size, stop_loss_ratio, risk_free_rate)
        all_results[model].append(results)
        print(f"\n{label} - {model} 回测完成：")
        print(f"最终净值: {results['final_portfolio_value']:.2f}, "
              f"总收益率: {results['total_return']*100:.2f}%, "
              f"年化收益率: {results['annual_return']*100:.2f}%, "
              f"交易次数: {results['trade_count']}, "
              f"夏普比率: {results['sharpe_ratio']:.2f}")

# ==============================
# 五、绘图保存
# ==============================
x = np.arange(len(window_labels))
width = 0.3
pattern_colors = ['#C82423', '#2878B5']
patterns = ['.', '\\\\']
line_colors = ['#C82423', '#2878B5']
line_styles = ['-', '--']
markers = ['o', '^']

fig, ax1 = plt.subplots(figsize=(16,8))
ax2 = ax1.twinx()

# 交易次数柱状图
for i, model in enumerate(models_to_use):
    trades_count = [all_results[model][j]['trade_count'] for j in range(len(window_labels))]
    offset = (i - 0.5) * width
    ax1.bar(x + offset, trades_count, width, color='white', edgecolor=pattern_colors[i],
            linewidth=2, hatch=patterns[i]*4)

# 年化收益率折线图
for i, model in enumerate(models_to_use):
    ann_return = [all_results[model][j]['annual_return']*100 for j in range(len(window_labels))]
    ax2.plot(x, ann_return, color=line_colors[i], linestyle=line_styles[i],
             marker=markers[i], markersize=10, linewidth=2.5, markerfacecolor='white', markeredgewidth=2)

ax1.set_xlabel('时间窗口', fontsize=18)
ax1.set_xticks(x)
ax1.set_xticklabels(window_labels, fontsize=16)
ax1.set_ylabel('交易次数', fontsize=16)
ax2.set_ylabel('Ann.Return (%)', fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
ax1.set_axisbelow(True)

# 图例
legend_elements = []
for i, model in enumerate(models_to_use):
    legend_elements.append(Patch(facecolor='white', edgecolor=pattern_colors[i], hatch=patterns[i]*4,
                                label=f'{model}: 交易次数', linewidth=2))
for i, model in enumerate(models_to_use):
    legend_elements.append(Line2D([0],[0], color=line_colors[i], linestyle=line_styles[i], marker=markers[i],
                                  markersize=10, linewidth=2.5, markerfacecolor='white', markeredgewidth=2,
                                  label=f'{model}:Ann.Return'))
ax1.legend(handles=legend_elements, loc='upper left', fontsize=14)

plt.tight_layout()
plt.savefig("Ann_Return.png", dpi=600)
plt.show()