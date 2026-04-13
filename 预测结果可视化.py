import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.dates as mdates
import os

# -----------------------------
# 中文字体设置
# -----------------------------
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# -----------------------------
# 文件路径（修改后）
# -----------------------------
file_path = "prediction_results_all_windows.xlsx"
save_folder = os.path.dirname(os.path.abspath(file_path))

# 图1：四窗口总图
save_filename_1 = "预测结果对比图.png"
save_path_1 = os.path.join(save_folder, save_filename_1)

# 图2：30天窗口局部图
save_filename_2 = "预测结果细节_30天窗口.png"
save_path_2 = os.path.join(save_folder, save_filename_2)

os.makedirs(save_folder, exist_ok=True)

# -----------------------------
# 模型颜色
# -----------------------------
models = {
    "ARIMA": "orange",
    "LSTM": "green",
    "BiLSTM-Attention": "red",
    "Transformer": "blue"
}

sheet_names = ["Window_1", "Window_5", "Window_15", "Window_30"]
window_labels = [1, 5, 15, 30]

# =========================================================
# Part 1：四个窗口预测结果总对比图
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(30, 22))
fig.patch.set_facecolor('white')

for i, sheet_name in enumerate(sheet_names):
    ax = axes[i // 2, i % 2]
    window_label = window_labels[i]

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        # 自动识别时间列
        time_col = None
        for col in df.columns:
            if col.lower() in ['date', 'time', 'trade_date']:
                time_col = col
                break

        if not time_col:
            raise ValueError(f"{sheet_name} 中未找到时间列")

        df[time_col] = pd.to_datetime(df[time_col])

        # 真实值
        ax.plot(
            df[time_col],
            df["Actual"],
            color="black",
            linestyle="--",
            linewidth=2.2,
            label="Actual",
            zorder=10
        )

        # 模型预测
        for model_name, color in models.items():
            if model_name in df.columns:
                ax.plot(
                    df[time_col],
                    df[model_name],
                    color=color,
                    linewidth=2,
                    label=model_name
                )

        # 标题
        ax.set_title(
            f"时间窗口为 {window_label} 的预测结果",
            fontsize=32,
            fontweight='bold',
            pad=20
        )

        # 坐标轴标签
        ax.set_xlabel("时间", fontsize=26)
        ax.set_ylabel("收盘价（元/桶）", fontsize=26)

        # 坐标轴刻度
        ax.tick_params(axis='both', labelsize=22)

        # 时间格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        # 不显示网格
        ax.grid(False)

    except Exception as e:
        print(f"处理 Sheet '{sheet_name}' 时出错: {e}")

# 全局图例
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='upper center',
    ncol=5,
    fontsize=28,
    frameon=True,
    markerscale=3,
    handlelength=4,
    borderpad=1.5
)

plt.tight_layout(rect=[0, 0, 1, 0.92])

# 保存图1
plt.savefig(
    save_path_1,
    dpi=600,
    bbox_inches='tight',
    facecolor='white'
)

print("图1已保存:", save_path_1)
plt.show()

# =========================================================
# Part 2：30天窗口局部细节图
# =========================================================

# 时间段
start_date = '2025-03-01'
end_date = '2025-10-01'
target_sheet = "Window_30"

try:
    df = pd.read_excel(file_path, sheet_name=target_sheet)

    # 自动识别时间列
    time_col = None
    for col in df.columns:
        if col.lower() in ['date', 'time', 'trade_date']:
            time_col = col
            break

    if not time_col:
        raise ValueError(f"{target_sheet} 中未找到时间列")

    df[time_col] = pd.to_datetime(df[time_col])

    # 筛选局部时间段
    mask = (df[time_col] >= start_date) & (df[time_col] <= end_date)
    df_filtered = df.loc[mask].reset_index(drop=True)

    if df_filtered.empty:
        raise ValueError(f"{target_sheet} 在 {start_date} 到 {end_date} 之间无数据，请调整时间范围。")

    fig, ax = plt.subplots(figsize=(30, 12))
    fig.patch.set_facecolor('white')

    # 真实值
    ax.plot(
        df_filtered[time_col],
        df_filtered["Actual"],
        color="black",
        linestyle="--",
        linewidth=2.5,
        label="Actual（真实值）",
        marker='o',
        markersize=5,
        markevery=2
    )

    # 模型预测
    for model_name, color in models.items():
        if model_name in df_filtered.columns:
            ax.plot(
                df_filtered[time_col],
                df_filtered[model_name],
                color=color,
                linewidth=2,
                label=model_name,
                marker='s',
                markersize=4,
                markevery=3
            )

    # 坐标轴标签
    ax.set_xlabel("交易日期", fontsize=28)
    ax.set_ylabel("收盘价（元/桶）", fontsize=28)

    # 时间刻度
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

    plt.setp(
        ax.xaxis.get_majorticklabels(),
        rotation=0,
        ha='center',
        fontsize=22
    )

    # Y轴刻度字体
    ax.tick_params(axis='y', labelsize=22)

    # 去除网格
    ax.grid(False)

    # 图例
    ax.legend(
        loc='upper right',
        fontsize=24,
        frameon=True,
        markerscale=2.5,
        handlelength=3,
        borderpad=1.5
    )

    plt.tight_layout()

    # 保存图2
    plt.savefig(
        save_path_2,
        dpi=600,
        bbox_inches='tight',
        facecolor='white'
    )

    print("图2已保存:", save_path_2)
    plt.show()

except Exception as e:
    print(f"处理局部细节图时出错: {e}")