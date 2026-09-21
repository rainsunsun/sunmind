# -*- coding: utf-8 -*-
"""探测 SiliconFlow embedding / rerank API 连通性（不打印任何 key）"""
import os, json, requests
from dotenv import load_dotenv

load_dotenv(r"C:\Users\rainsunsun\Desktop\learn\echomind\EchoMind\.env")

EMB_URL = os.getenv("EMBEDDING_URL", "https://api.siliconflow.cn/v1")
EMB_MODEL = os.getenv("EMBEDDING_MODEL")
EMB_KEY = os.getenv("SILICONFLOW_API_KEY") or os.getenv("RERANK_API_KEY")

RERANK_URL = os.getenv("RERANK_URL", "https://api.siliconflow.cn/v1/rerank")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_KEY = os.getenv("SILICONFLOW_API_KEY") or os.getenv("RERANK_API_KEY")

print("EMBEDDING_URL =", EMB_URL)
print("EMBEDDING_MODEL =", EMB_MODEL)
print("RERANK_URL =", RERANK_URL)
print("RERANK_MODEL =", RERANK_MODEL)

# 1. embedding
try:
    r = requests.post(
        f"{EMB_URL.rstrip('/')}/embeddings",
        headers={"Authorization": f"Bearer {EMB_KEY}", "Content-Type": "application/json"},
        json={"model": EMB_MODEL, "input": ["数据库索引的作用是什么", "测试文本二"]},
        timeout=30,
    )
    print("\n[embedding] status =", r.status_code)
    d = r.json()
    if r.status_code == 200 and "data" in d:
        print("  维度 =", len(d["data"][0]["embedding"]))
        print("  返回条数 =", len(d["data"]))
    else:
        print("  响应 =", json.dumps(d, ensure_ascii=False)[:400])
except Exception as e:
    print("[embedding] 异常:", repr(e))

# 2. rerank
try:
    r2 = requests.post(
        RERANK_URL,
        headers={"Authorization": f"Bearer {RERANK_KEY}", "Content-Type": "application/json"},
        json={
            "model": RERANK_MODEL,
            "query": "什么是数据库索引",
            "documents": ["索引可以加速查询", "缓存用于加速读取", "事务保证一致性"],
            "top_n": 3,
            "return_documents": True,
        },
        timeout=30,
    )
    print("\n[rerank] status =", r2.status_code)
    d2 = r2.json()
    if r2.status_code == 200:
        print("  results 数 =", len(d2.get("results", [])))
        if d2.get("results"):
            print("  首个结果 keys =", list(d2["results"][0].keys()))
            print("  首个结果 =", json.dumps(d2["results"][0], ensure_ascii=False)[:300])
    else:
        print("  响应 =", json.dumps(d2, ensure_ascii=False)[:400])
except Exception as e:
    print("[rerank] 异常:", repr(e))
