import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
import math
import sys
import copy

# ================= 配置类 =================
class Config:
    FILE_MAIN = '日度数据_SI.csv'
    FILE_TECH = '技术指标.csv'

    TARGET_COL = 'close'
    OUTPUT_EXCEL = 'ablation_study_results.xlsx'

    WINDOW_SIZES = [1, 5, 15, 30]

    EPOCHS = 120
    BATCH_SIZE = 32
    LR = 0.001

    EARLY_STOP_PATIENCE = 10
    EARLY_STOP_MIN_DELTA = 0.0001

    # Transformer 配置
    TRANSFORMER_D_MODEL = 128
    TRANSFORMER_N_HEAD = 8
    TRANSFORMER_ENC_LAYERS = 2
    TRANSFORMER_DEC_LAYERS = 2
    TRANSFORMER_DIM_FF = 256
    TRANSFORMER_DROPOUT = 0.1

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


warnings.filterwarnings('ignore')
print(f"Running on: {Config.DEVICE}")


# ================= 早停机制 =================
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


# ================= 数据加载 =================
def load_data():
    def read_csv_safe(filepath):
        try:
            return pd.read_csv(filepath)
        except UnicodeDecodeError:
            return pd.read_csv(filepath, encoding='gbk')
        except Exception as e:
            print(f"读取 {filepath} 失败: {e}")
            return None

    df_main = read_csv_safe(Config.FILE_MAIN)
    df_tech = read_csv_safe(Config.FILE_TECH)

    if df_main is None or df_tech is None:
        return None, None

    for df in [df_main, df_tech]:
        if 'trade_date' not in df.columns:
            date_cols = [c for c in df.columns if 'date' in c.lower()]
            if date_cols:
                df.rename(columns={date_cols[0]: 'trade_date'}, inplace=True)
            else:
                print("未找到 trade_date 列")
                return None, None

        df['trade_date'] = pd.to_datetime(
            df['trade_date'].astype(str),
            format='%Y%m%d',
            errors='coerce'
        )

    df_main = df_main.dropna(subset=['trade_date'])
    df_tech = df_tech.dropna(subset=['trade_date'])

    merge_keys = ['trade_date']
    if 'ts_code' in df_main.columns and 'ts_code' in df_tech.columns:
        merge_keys.append('ts_code')
    elif 'instrument' in df_main.columns and 'instrument' in df_tech.columns:
        merge_keys.append('instrument')

    df = pd.merge(df_main, df_tech, on=merge_keys, how='inner')

    if df.empty:
        print("合并后数据为空")
        return None, None

    if Config.TARGET_COL not in df.columns:
        close_candidates = [c for c in df.columns if 'close' in c.lower()]
        if close_candidates:
            df.rename(columns={close_candidates[0]: Config.TARGET_COL}, inplace=True)
        else:
            print("未找到 close 列")
            return None, None

    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in all_numeric if c != Config.TARGET_COL and 'Unnamed' not in c]

    df[features] = df[features].ffill().bfill().fillna(0)
    df[Config.TARGET_COL] = df[Config.TARGET_COL].ffill().bfill().fillna(0)

    final_df = df[['trade_date', Config.TARGET_COL] + features].copy()
    final_df = final_df.sort_values('trade_date').reset_index(drop=True)

    return final_df, features


# ================= 特征分类 =================
def get_feature_categories(all_features):
    """
    将特征分为：
    1. 期货市场因素
    2. 基本面因素
    3. 技术指标因素
    4. 宏观经济因素
    5. 情绪指标（SI）
    """

    futures_keywords = ['open', 'high', 'low', 'settle', 'vol', 'oi', 'WTI', '布伦特', 'Brent']
    fundamental_keywords = ['美元兑人民币', '现货价Brent', '现货价WTI', '现货价柴油', '现货价石油沥青', '原油产量']
    tech_keywords = ['MACD', 'MA', 'EMA', 'TRIX', 'BOLL', 'K', 'D', 'J', 'WilliamsR', 'RSI',
                     'ROC', 'ADX', 'OBV', 'MFI', 'DX', 'MOM', 'BIAS']
    macro_keywords = ['中国:经济政策不确定性', '美国:经济政策不确定性', '消费者指数:信心指数']

    cat_futures = []
    cat_fundamental = []
    cat_technical = []
    cat_macro = []
    cat_sentiment = []
    valid_features = []

    for feat in all_features:
        try:
            float(feat)
            continue
        except:
            pass

        valid_features.append(feat)

        # 单独识别 SI
        if feat == 'SI':
            cat_sentiment.append(feat)
            continue

        matched = False

        for kw in futures_keywords:
            if kw in feat:
                cat_futures.append(feat)
                matched = True
                break
        if matched:
            continue

        for kw in fundamental_keywords:
            if kw in feat:
                cat_fundamental.append(feat)
                matched = True
                break
        if matched:
            continue

        for kw in macro_keywords:
            if kw in feat:
                cat_macro.append(feat)
                matched = True
                break
        if matched:
            continue

        for kw in tech_keywords:
            if kw in feat:
                cat_technical.append(feat)
                matched = True
                break

        if not matched and feat != 'SI':
            print(f"Warning: 特征 '{feat}' 未分类，将默认保留在基准模型中。")

    categories = {
        "期货市场因素": cat_futures,
        "基本面因素": cat_fundamental,
        "技术指标因素": cat_technical,
        "宏观经济因素": cat_macro,
        "情绪指标": cat_sentiment
    }

    return valid_features, categories


# ================= 模型定义 =================
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
        if x.size(1) > self.pe.size(1):
            max_len = x.size(1)
            pe = torch.zeros(max_len, self.pe.size(2), device=x.device)
            position = torch.arange(0, max_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.pe.size(2), 2, device=x.device).float() *
                                 (-math.log(10000.0) / self.pe.size(2)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            x = x + pe
        else:
            x = x + self.pe[:, :x.size(1), :]
        return x


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


# ================= 数据集 =================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, window):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window = window

    def __len__(self):
        return max(0, len(self.X) - self.window)

    def __getitem__(self, i):
        return self.X[i:i+self.window], self.y[i+self.window]


# ================= 训练与评估 =================
def train_model_ablation(features_list, df, window):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    data_X = scaler_X.fit_transform(df[features_list])
    data_y = scaler_y.fit_transform(df[[Config.TARGET_COL]])

    train_size = int(len(data_X) * 0.8)
    if train_size == 0 or train_size >= len(data_X):
        return 0.0, 0.0

    train_X, test_X = data_X[:train_size], data_X[train_size:]
    train_y, test_y = data_y[:train_size], data_y[train_size:]

    train_ds = TimeSeriesDataset(train_X, train_y, window)
    test_ds = TimeSeriesDataset(test_X, test_y, window)

    if len(train_ds) == 0 or len(test_ds) == 0:
        return 0.0, 0.0

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    model = TransformerModel(len(features_list)).to(Config.DEVICE)
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
            early_stopping.load_checkpoint(model)
            break

    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(Config.DEVICE)
            p = model(X_batch).cpu().numpy().flatten()
            preds.extend(p)
            actuals.extend(y_batch.numpy().flatten())

    if len(preds) == 0:
        return 0.0, 0.0

    final_pred = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    final_act = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

    try:
        mape = mean_absolute_percentage_error(final_act, final_pred) * 100
        rmse = np.sqrt(mean_squared_error(final_act, final_pred))
    except:
        mape, rmse = 0.0, 0.0

    return mape, rmse


# ================= ER 计算 =================
def calc_er(value, baseline):
    if baseline == 0:
        return "-"
    return round((value - baseline) / baseline * 100, 1)


# ================= 主程序 =================
if __name__ == "__main__":
    print("正在加载数据...")
    df, all_features_raw = load_data()

    if df is None:
        print("数据加载失败，请检查文件路径。")
        sys.exit(1)

    valid_features, categories = get_feature_categories(all_features_raw)

    print(f"\n有效特征数: {len(valid_features)}")
    print("特征分类统计：")
    for cat_name, feats in categories.items():
        print(f"  - {cat_name}: {len(feats)} 个")

    experiment_rows = [
        ("无缺失因素", None),
        ("期货市场因素", "期货市场因素"),
        ("基本面因素", "基本面因素"),
        ("技术指标因素", "技术指标因素"),
        ("宏观经济因素", "宏观经济因素"),
        ("情绪指标", "情绪指标")
    ]

    windows = Config.WINDOW_SIZES
    raw_results = {}

    print("\n开始运行消融实验...")
    print("-" * 80)

    for row_name, missing_cat in experiment_rows:
        raw_results[row_name] = {}

        if missing_cat is None:
            current_features = valid_features
            print(f"\n运行基准模型：{row_name}")
        else:
            features_to_remove = set(categories[missing_cat])
            current_features = [f for f in valid_features if f not in features_to_remove]
            print(f"\n运行消融模型：删除 {missing_cat}（剩余特征数 {len(current_features)}）")

        if len(current_features) == 0:
            for w in windows:
                raw_results[row_name][w] = {"MAPE": np.nan, "RMSE": np.nan}
            continue

        for w in windows:
            print(f"  -> Window={w}", end=" | ")
            mape, rmse = train_model_ablation(current_features, df, w)
            raw_results[row_name][w] = {"MAPE": mape, "RMSE": rmse}
            print(f"MAPE={mape:.4f}, RMSE={rmse:.4f}")

    # ================= 构建结果表 =================
    baseline_name = "无缺失因素"

    mape_rows = []
    rmse_rows = []

    for row_name, _ in experiment_rows:
        mape_row = {"缺失因素": row_name}
        rmse_row = {"缺失因素": row_name}

        for w in windows:
            cur_mape = raw_results[row_name][w]["MAPE"]
            cur_rmse = raw_results[row_name][w]["RMSE"]

            base_mape = raw_results[baseline_name][w]["MAPE"]
            base_rmse = raw_results[baseline_name][w]["RMSE"]

            mape_row[f"Window={w}"] = round(cur_mape, 4) if not np.isnan(cur_mape) else np.nan
            rmse_row[f"Window={w}"] = round(cur_rmse, 4) if not np.isnan(cur_rmse) else np.nan

            if row_name == baseline_name:
                mape_row[f"ER_{w}"] = "-"
                rmse_row[f"ER_{w}"] = "-"
            else:
                mape_row[f"ER_{w}"] = calc_er(cur_mape, base_mape)
                rmse_row[f"ER_{w}"] = calc_er(cur_rmse, base_rmse)

        mape_rows.append(mape_row)
        rmse_rows.append(rmse_row)

    df_mape = pd.DataFrame(mape_rows)
    df_rmse = pd.DataFrame(rmse_rows)

    # 列顺序
    mape_cols = ["缺失因素"]
    rmse_cols = ["缺失因素"]

    for w in windows:
        mape_cols.extend([f"Window={w}", f"ER_{w}"])
        rmse_cols.extend([f"Window={w}", f"ER_{w}"])

    df_mape = df_mape[mape_cols]
    df_rmse = df_rmse[rmse_cols]

    # ================= 输出控制台 =================
    print("\n" + "=" * 100)
    print("不同类别特征缺失下的 MAPE 比较")
    print("=" * 100)
    print(df_mape.to_string(index=False))

    print("\n" + "=" * 100)
    print("不同类别特征缺失下的 RMSE 比较")
    print("=" * 100)
    print(df_rmse.to_string(index=False))

    # ================= 保存 Excel =================
    with pd.ExcelWriter(Config.OUTPUT_EXCEL, engine='openpyxl') as writer:
        df_mape.to_excel(writer, sheet_name='MAPE_Comparison', index=False)
        df_rmse.to_excel(writer, sheet_name='RMSE_Comparison', index=False)

    print(f"\n结果已保存到：{Config.OUTPUT_EXCEL}")