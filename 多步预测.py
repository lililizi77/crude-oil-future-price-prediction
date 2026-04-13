import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
import math
import copy
import sys

# ==============================
# 中文显示
# ==============================
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# ==============================
# 配置类
# ==============================
class Config:
    FILE_MAIN = '日度数据_SI.csv'
    TARGET_COL = 'close'
    SI_COL = 'SI'

    INPUT_WINDOW = 15

    # 多步预测期限
    HORIZONS = [1, 5, 15, 30]

    # 训练参数
    EPOCHS = 120
    BATCH_SIZE = 32
    LR = 0.001

    EARLY_STOP_PATIENCE = 10
    EARLY_STOP_MIN_DELTA = 0.0001

    # Transformer 参数
    TRANSFORMER_D_MODEL = 128
    TRANSFORMER_N_HEAD = 8
    TRANSFORMER_ENC_LAYERS = 2
    TRANSFORMER_DEC_LAYERS = 2
    TRANSFORMER_DIM_FF = 256
    TRANSFORMER_DROPOUT = 0.1

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Running on: {Config.DEVICE}")

# ==============================
# 早停机制
# ==============================
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        self.best_model_wts = copy.deepcopy(model.state_dict())

    def load_checkpoint(self, model):
        if self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)

# ==============================
# 数据加载
# ==============================
def load_data():
    def read_csv_safe(filepath):
        try:
            return pd.read_csv(filepath)
        except UnicodeDecodeError:
            return pd.read_csv(filepath, encoding='gbk')
        except Exception as e:
            print(f"读取 {filepath} 失败: {e}")
            return None

    df = read_csv_safe(Config.FILE_MAIN)
    if df is None:
        return None, None, None

    # 日期列识别
    if 'trade_date' not in df.columns:
        date_cols = [c for c in df.columns if 'date' in c.lower()]
        if date_cols:
            df.rename(columns={date_cols[0]: 'trade_date'}, inplace=True)
        else:
            print("未找到日期列")
            return None, None, None

    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)

    # 收盘价列识别
    if Config.TARGET_COL not in df.columns:
        close_candidates = [c for c in df.columns if 'close' in c.lower()]
        if close_candidates:
            df.rename(columns={close_candidates[0]: Config.TARGET_COL}, inplace=True)
        else:
            print("未找到目标列 close")
            return None, None, None

    # 数值列
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    all_features = [c for c in all_numeric if c != Config.TARGET_COL and 'Unnamed' not in c]

    # 填补缺失
    df[all_features] = df[all_features].ffill().bfill().fillna(0)
    df[Config.TARGET_COL] = df[Config.TARGET_COL].ffill().bfill().fillna(0)

    # SI 列
    if Config.SI_COL not in df.columns:
        print(f"警告：未找到情绪指标列 {Config.SI_COL}，将无法进行情绪对比实验。")
        si_exists = False
    else:
        si_exists = True

    return df, all_features, si_exists

# ==============================
# 构造多步预测数据集
# ==============================
class MultiStepDataset(Dataset):
    def __init__(self, X, y, input_window, horizon):
        self.X = []
        self.y = []

        for i in range(len(X) - input_window - horizon + 1):
            self.X.append(X[i:i + input_window])
            self.y.append(y[i + input_window + horizon - 1])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==============================
# Transformer 模型
# ==============================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, input_dim):
        super(EncoderDecoderTransformer, self).__init__()
        d_model = Config.TRANSFORMER_D_MODEL
        nhead = Config.TRANSFORMER_N_HEAD
        num_encoder_layers = Config.TRANSFORMER_ENC_LAYERS
        num_decoder_layers = Config.TRANSFORMER_DEC_LAYERS
        dim_feedforward = Config.TRANSFORMER_DIM_FF
        dropout = Config.TRANSFORMER_DROPOUT

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, src):
        batch_size, seq_len, _ = src.shape

        x_enc = self.input_proj(src) * math.sqrt(self.d_model)
        x_enc = self.pos_encoder(x_enc)
        memory = self.transformer_encoder(x_enc)

        tgt = torch.zeros(batch_size, seq_len, self.d_model).to(src.device)
        x_dec = self.pos_decoder(tgt)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=src.device), diagonal=1).bool()
        output = self.transformer_decoder(tgt=x_dec, memory=memory, tgt_mask=causal_mask)

        last_output = output[:, -1, :]
        return self.fc_out(last_output)

class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super(TransformerModel, self).__init__()
        self.model = EncoderDecoderTransformer(input_dim)

    def forward(self, x):
        return self.model(x)

# ==============================
# 单次训练与评估
# ==============================
def train_and_evaluate(df, feature_list, horizon):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    data_X = scaler_X.fit_transform(df[feature_list])
    data_y = scaler_y.fit_transform(df[[Config.TARGET_COL]])

    train_size = int(len(data_X) * 0.8)
    if train_size <= Config.INPUT_WINDOW + horizon or train_size >= len(data_X):
        return np.nan, np.nan

    train_X, test_X = data_X[:train_size], data_X[train_size:]
    train_y, test_y = data_y[:train_size], data_y[train_size:]

    train_ds = MultiStepDataset(train_X, train_y, Config.INPUT_WINDOW, horizon)
    test_ds = MultiStepDataset(test_X, test_y, Config.INPUT_WINDOW, horizon)

    if len(train_ds) == 0 or len(test_ds) == 0:
        return np.nan, np.nan

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = TransformerModel(len(feature_list)).to(Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    criterion = nn.MSELoss()

    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOP_PATIENCE,
        min_delta=Config.EARLY_STOP_MIN_DELTA
    )

    for epoch in range(Config.EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred.squeeze(), y_batch.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(Config.DEVICE), y_batch.to(Config.DEVICE)
                pred = model(X_batch)
                loss = criterion(pred.squeeze(), y_batch.squeeze())
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            break

    early_stopping.load_checkpoint(model)

    # 预测评估
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(Config.DEVICE)
            pred = model(X_batch).cpu().numpy().flatten()
            preds.extend(pred)
            actuals.extend(y_batch.numpy().flatten())

    if len(preds) == 0:
        return np.nan, np.nan

    final_pred = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    final_act = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    mape = mean_absolute_percentage_error(final_act, final_pred) * 100
    rmse = np.sqrt(mean_squared_error(final_act, final_pred))

    return mape, rmse

# ==============================
# 可视化
# ==============================
def plot_results(windows, no_si_mape, with_si_mape, no_si_rmse, with_si_rmse):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    marker_size = 9
    line_width = 2.2

    # -----------------------------
    # 左图：MAPE
    # -----------------------------
    ax1 = axes[0]

    ax1.plot(windows, with_si_mape,
             color='red', marker='o',
             markersize=marker_size,
             linewidth=line_width,
             label='包含情绪指标')

    ax1.plot(windows, no_si_mape,
             color='blue', marker='^',
             markersize=marker_size,
             linewidth=line_width,
             label='不包含情绪指标')

    ax1.set_xlabel("预测期限（天）", fontsize=16)
    ax1.set_ylabel("MAPE", fontsize=16)
    ax1.set_xticks(windows)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.legend(loc='upper left', fontsize=13, frameon=True)

    mape_all = [x for x in (no_si_mape + with_si_mape) if not np.isnan(x)]
    if len(mape_all) > 0:
        ymin1 = min(mape_all) - 0.1
        ymax1 = max(mape_all) + 0.1
        ax1.set_ylim(ymin1, ymax1)

    # -----------------------------
    # 右图：RMSE
    # -----------------------------
    ax2 = axes[1]

    ax2.plot(windows, with_si_rmse,
             color='red', marker='o',
             markersize=marker_size,
             linewidth=line_width,
             label='包含情绪指标')

    ax2.plot(windows, no_si_rmse,
             color='blue', marker='^',
             markersize=marker_size,
             linewidth=line_width,
             label='不包含情绪指标')

    ax2.set_xlabel("预测期限（天）", fontsize=16)
    ax2.set_ylabel("RMSE", fontsize=16)
    ax2.set_xticks(windows)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.legend(loc='upper left', fontsize=13, frameon=True)

    rmse_all = [x for x in (no_si_rmse + with_si_rmse) if not np.isnan(x)]
    if len(rmse_all) > 0:
        ymin2 = min(rmse_all) - 0.5
        ymax2 = max(rmse_all) + 0.5
        ax2.set_ylim(ymin2, ymax2)

    plt.tight_layout()
    plt.savefig("multistep_compare.png", dpi=600, bbox_inches='tight')
    plt.show()

# ==============================
# 主程序
# ==============================
if __name__ == "__main__":
    print("正在加载数据...")
    df, all_features, si_exists = load_data()

    if df is None:
        print("数据加载失败，请检查文件。")
        sys.exit(1)

    if not si_exists:
        print(f"文件中缺少 {Config.SI_COL} 列，无法进行有/无情绪指标对比。")
        sys.exit(1)

    # 过滤掉纯数字列名
    valid_features = []
    for feat in all_features:
        try:
            float(feat)
            continue
        except:
            valid_features.append(feat)

    # 无情绪指标特征
    features_no_si = [f for f in valid_features if f != Config.SI_COL]

    # 有情绪指标特征
    features_with_si = valid_features.copy()

    print("\n==============================")
    print("Transformer 多步预测实验开始")
    print("==============================")

    results = []

    no_si_mape_list = []
    with_si_mape_list = []
    no_si_rmse_list = []
    with_si_rmse_list = []

    for horizon in Config.HORIZONS:
        print(f"\n------------------------------")
        print(f"预测未来第 {horizon} 天")
        print(f"------------------------------")

        # 无情绪
        print("训练：Transformer（无情绪指标）...")
        mape_no_si, rmse_no_si = train_and_evaluate(df, features_no_si, horizon)

        # 有情绪
        print("训练：Transformer（有情绪指标）...")
        mape_with_si, rmse_with_si = train_and_evaluate(df, features_with_si, horizon)

        no_si_mape_list.append(mape_no_si)
        with_si_mape_list.append(mape_with_si)
        no_si_rmse_list.append(rmse_no_si)
        with_si_rmse_list.append(rmse_with_si)

        results.append({
            "预测期限": horizon,
            "无情绪_MAPE": mape_no_si,
            "有情绪_MAPE": mape_with_si,
            "无情绪_RMSE": rmse_no_si,
            "有情绪_RMSE": rmse_with_si
        })

        print(f"【MAPE】无情绪: {mape_no_si:.4f} | 有情绪: {mape_with_si:.4f}")
        print(f"【RMSE】无情绪: {rmse_no_si:.4f} | 有情绪: {rmse_with_si:.4f}")

    # ==============================
    # 控制台打印结果表
    # ==============================
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("Transformer 多步预测实验结果")
    print("=" * 90)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 90)

    # ==============================
    # 可视化
    # ==============================
    plot_results(
        Config.HORIZONS,
        no_si_mape_list,
        with_si_mape_list,
        no_si_rmse_list,
        with_si_rmse_list
    )