import requests
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 模型配置 =====================
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

# 加载知识库
def load_knowledge(knowledge_path="行业知识库.txt"):
    try:
        with open(knowledge_path, "r", encoding="utf-8") as f:
            knowledge_content = f.read().strip()
        return knowledge_content
    except Exception as e:
        print(f"读取知识库失败：{str(e)}")
        return ""

knowledge = load_knowledge()

SYSTEM_PROMPT = (
    "你是一个上海原油期货新闻文本标题情感识别分类模型。"
    "结合原油期货知识库，判断新闻标题对油价影响：看涨1，看跌-1，中性0。"
    "仅输出单个数字：1 / -1 / 0，不要多余内容。"
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
        normalized = content.replace(" ","").replace("。","").replace("\n","")
        if normalized in {"1","-1","0"}:
            return int(normalized)
        mapping = {"看涨":1,"看跌":-1,"中性":0,"利好":1,"利空":-1}
        for k,v in mapping.items():
            if k in normalized:
                return v
        for t in ["-1","1","0"]:
            if t in normalized:
                return int(t)
        return None
    except requests.exceptions.RequestException:
        return None

# ===================== 主流程：情感分析+SI计算 =====================
def main():
    # 读取新闻
    df = pd.read_excel("datas.xlsx")

    # 统一列名
    rename_map = {}
    for col in df.columns:
        cl = col.lower()
        if "标题" in cl or "title" in cl:
            rename_map[col] = "标题"
        if "时间" in cl or "日期" in cl or "time" in cl:
            rename_map[col] = "时间"
    df = df.rename(columns=rename_map)
    if "标题" not in df.columns:
        df["标题"] = ""
    if "时间" not in df.columns:
        df["时间"] = pd.NaT

    # 批量调用模型
    endpoint = build_endpoint()
    session = requests.Session()
    titles = df["标题"].astype(str).tolist()
    results = [None]*len(titles)
    max_workers = min(32, max(4, len(titles)//50))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_title, session, endpoint, t):i for i,t in enumerate(titles)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except:
                results[idx] = None

    df["情感标签"] = results

    # ========== SI指数计算 ==========
    news_df = df.dropna(subset=["时间"]).copy()
    news_df["trade_date"] = pd.to_datetime(news_df["时间"]).dt.strftime("%Y-%m-%d")

    pos = news_df[news_df["情感标签"]==1].groupby("trade_date").size()
    neg = news_df[news_df["情感标签"]==-1].groupby("trade_date").size()
    sent = pd.DataFrame({"NewsPos":pos,"NewsNeg":neg}).fillna(0).astype(int)
    sent["SI"] = np.log((1+sent["NewsPos"])/(1+sent["NewsNeg"]))

    # 读取原油数据并对齐
    oil = pd.read_csv("日度数据.csv")
    oil["trade_date"] = pd.to_datetime(oil["trade_date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")

    oil = oil.merge(sent[["SI"]], left_on="trade_date", right_index=True, how="left")
    oil["SI"] = oil["SI"].fillna(0).round(2)

    oil.to_csv("日度数据_SI.csv", index=False, encoding="utf-8-sig")
    print("已完成情感分析与SI计算，保存至：日度数据_SI.csv")

if __name__ == "__main__":
    main()