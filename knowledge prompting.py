import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


CONFIG = {
    "autodl_chatglm4": {
        "base_url": "https://u788855-cby6-d41ee786.westd.seetacloud.com:8443",
        "model_name": "ChatGLM4-9B",
        "api_key": "dummy-key"
    }
}


def build_endpoint():
    conf = CONFIG["autodl_chatglm4"]
    base_url = conf["base_url"].rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    return {
        "chat_completion_url": f"{base_url}/chat/completions",
        "model_name": conf["model_name"],
        "api_key": conf["api_key"]
    }

# 新增：读取外部知识库文件（支持txt格式，可根据实际文件类型调整读取方式）
def load_knowledge(knowledge_path="行业知识库.txt"):
    """
    读取外部原油期货知识库文件，返回知识库文本内容
    :param knowledge_path: 外部知识库文件路径（默认当前目录下crude_oil_knowledge.txt）
    :return: 知识库文本字符串（若读取失败返回空字符串）
    """
    try:
        with open(knowledge_path, "r", encoding="utf-8") as f:
            knowledge_content = f.read().strip()
        return knowledge_content
    except Exception as e:
        print(f"读取外部知识库失败：{str(e)}，将使用空知识库进行分类")
        return ""

# 加载外部知识库
knowledge = load_knowledge()


SYSTEM_PROMPT = (
    "你是⼀个上海原油期货新闻⽂本标题情感识别分类模型。"
    "请结合用户提供的原油期货价格变动相关知识库，判断新闻标题对上海原油期货价格的影响倾向，"
    "看涨标记为 1，看跌标记为 -1，中性标记为 0。"
    "只输出一个数字：1 或 -1 或 0，不要输出其他内容。"
)


def classify_title(session, endpoint, title, timeout=20):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {endpoint['api_key']}"
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"参考以下原油期货领域的知识:{knowledge}基于上述知识，请对以下原油期货相关新闻标题进行情感分类，只输出数字。标题:{title}"}
    ]
    payload = {
        "model": endpoint["model_name"],
        "messages": messages,
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.0
    }
    try:
        resp = session.post(endpoint["chat_completion_url"], headers=headers, json=payload, timeout=timeout)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if "choices" not in data or not data["choices"]:
            return None
        content = data["choices"][0]["message"]["content"].strip()
        normalized = content.replace(" ", "").replace("。", "").replace("\n", "")
        if normalized in {"1", "-1", "0"}:
            return int(normalized)
        mapping = {
            "看涨": 1, "看跌": -1, "中性": 0, "利好": 1, "利空": -1,
            "neutral": 0, "bullish": 1, "bearish": -1
        }
        for k, v in mapping.items():
            if k in normalized.lower():
                return v
        for token in ["-1", "1", "0"]:
            if token in normalized:
                return int(token)
        return None
    except requests.exceptions.RequestException:
        return None


def compute_metrics(df, pred_col="情感标签", true_col="情感"):
    valid_set = {-1, 0, 1}

    def to_int_or_none(x):
        try:
            v = int(x)
            return v if v in valid_set else None
        except Exception:
            return None

    y_true = df[true_col].apply(to_int_or_none)
    y_pred = df[pred_col].apply(to_int_or_none)
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return None, None

    yt = y_true[mask].astype(int).tolist()
    yp = y_pred[mask].astype(int).tolist()
    correct = sum(1 for a, b in zip(yt, yp) if a == b)
    acc = correct / len(yt)

    labels = [-1, 0, 1]
    f1s = []
    for lbl in labels:
        tp = sum(1 for a, b in zip(yt, yp) if b == lbl and a == lbl)
        fp = sum(1 for a, b in zip(yt, yp) if b == lbl and a != lbl)
        fn = sum(1 for a, b in zip(yt, yp) if b != lbl and a == lbl)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)
    return acc, macro_f1


def main():
    # 读取当前目录文件
    input_path = "sentiment2000.xlsx"
    df = pd.read_excel(input_path)

    expected_cols = ["标题", "时间"]
    rename_map = {}
    for col in df.columns:
        if col.lower() in ["title", "标题"]:
            rename_map[col] = "标题"
        elif col.lower() in ["time", "日期", "时间"]:
            rename_map[col] = "时间"
    if rename_map:
        df = df.rename(columns=rename_map)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    # 模型调用
    endpoint = build_endpoint()
    session = requests.Session()
    titles = df["标题"].astype(str).tolist()
    results = [None] * len(titles)
    max_workers = min(32, max(4, len(titles) // 50))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_title, session, endpoint, title): i for i, title in enumerate(titles)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                label = fut.result()
            except Exception:
                label = None
            results[idx] = label

    df["情感标签"] = results

    # 输出准确率和F1
    if "情感" in df.columns:
        acc, macro_f1 = compute_metrics(df)
        if acc is not None:
            print(f"准确率 = {acc:.4f}，宏平均F1 = {macro_f1:.4f}")
        else:
            print("评估失败：无有效对比样本")


if __name__ == "__main__":
    main()
