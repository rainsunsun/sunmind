"""验证 EchoMind 各外部服务连通性（不改动任何代码，只读 .env 并做最小探测）。

用法: python verify_env.py
"""
import os
import httpx
from dotenv import load_dotenv

# .env 在上一级目录
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))


def check(name, ok, detail):
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}: {detail}")


# ========== 1. DeepSeek LLM ==========
print("=" * 60)
print("1) DeepSeek LLM (chat)")
key = os.getenv("DASHSCOPE_API_KEY")
model = os.getenv("AGENT_BASE_MODEL")
base = os.getenv("BASE_URL", "https://api.deepseek.com")
try:
    r = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5},
        timeout=25,
    )
    if r.status_code == 200:
        check("DeepSeek", True, f"model={model} -> {r.json()['choices'][0]['message']['content']!r}")
    else:
        check("DeepSeek", False, f"model={model} HTTP {r.status_code}: {r.text[:200]}")
except Exception as e:
    check("DeepSeek", False, f"异常: {e}")

# ========== 2. SiliconFlow embedding ==========
print("=" * 60)
print("2) SiliconFlow embedding (bge-m3)")
sf_key = os.getenv("RERANK_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
emb_model = os.getenv("EMBEDDING_MODEL")
try:
    r = httpx.post(
        "https://api.siliconflow.cn/v1/embeddings",
        headers={"Authorization": f"Bearer {sf_key}"},
        json={"model": emb_model, "input": "你好世界"},
        timeout=30,
    )
    if r.status_code == 200:
        dim = len(r.json()["data"][0]["embedding"])
        check("SiliconFlow embed", True, f"model={emb_model} dims={dim}")
    else:
        check("SiliconFlow embed", False, f"HTTP {r.status_code}: {r.text[:200]}")
except Exception as e:
    check("SiliconFlow embed", False, f"异常: {e}")

# ========== 3. SiliconFlow rerank ==========
print("=" * 60)
print("3) SiliconFlow rerank (bge-reranker-v2-m3)")
rerank_model = os.getenv("RERANK_MODEL")
rerank_url = os.getenv("RERANK_URL")
try:
    r = httpx.post(
        rerank_url,
        headers={"Authorization": f"Bearer {sf_key}"},
        json={"model": rerank_model, "query": "你好", "documents": ["你好世界", "今天天气不错"]},
        timeout=30,
    )
    if r.status_code == 200:
        res = r.json()["results"][0]
        check("SiliconFlow rerank", True,
              f"model={rerank_model} top={res.get('document', {}).get('text', '')!r} "
              f"score={res['relevance_score']:.3f}")
    else:
        check("SiliconFlow rerank", False, f"HTTP {r.status_code}: {r.text[:200]}")
except Exception as e:
    check("SiliconFlow rerank", False, f"异常: {e}")

# ========== 4. Zilliz Cloud ==========
print("=" * 60)
print("4) Zilliz Cloud (Milvus)")
try:
    from pymilvus import MilvusClient
    uri = os.getenv("Milvus_url")
    token = os.getenv("Token")
    client = MilvusClient(uri=uri, token=token)
    cols = client.list_collections()
    check("Zilliz", True, f"连接成功，已有集合: {cols}")
except Exception as e:
    check("Zilliz", False, f"{e}")

print("=" * 60)
print("完成。PostgreSQL/Redis 是本地服务，请另用: docker ps 或 telnet localhost 5432/6379 检查。")
