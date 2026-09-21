# -*- coding: utf-8 -*-
"""
检索评测：单一大块纯稠密  vs  父子块 + 稠密 + BM25 + RRF + rerank
指标：hit@1/3/5、MRR、recall@5；并按 query 类型分组统计。
结果写入 eval_report.md（UTF-8），控制台只打摘要。
"""
import os, re, math, time
import requests
import numpy as np
from dotenv import load_dotenv
from eval_corpus import SECTIONS, QUERIES

load_dotenv(r"C:\Users\rainsunsun\Desktop\learn\echomind\EchoMind\.env")

EMB_URL = os.getenv("EMBEDDING_URL", "https://api.siliconflow.cn/v1").rstrip("/")
EMB_MODEL = os.getenv("EMBEDDING_MODEL")
EMB_KEY = os.getenv("SILICONFLOW_API_KEY") or os.getenv("RERANK_API_KEY")
RERANK_URL = os.getenv("RERANK_URL", "https://api.siliconflow.cn/v1/rerank")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
REPORT = r"C:\Users\rainsunsun\Desktop\learn\echomind\EchoMind\backend\test\eval_report.md"

K_VALUES = [1, 3, 5]
RRF_K = 60
CHILD_SIZE, CHILD_OVERLAP = 200, 50


# ---------------- 分块 ----------------
def make_child_chunks(text, size=CHILD_SIZE, overlap=CHILD_OVERLAP):
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


# ---------------- 分词（中文 bigram + 英文数字整词） ----------------
def tokenize(text):
    text = text.lower()
    out = []
    for seg in re.findall(r'[\u4e00-\u9fff]+|[a-z0-9]+', text):
        if re.fullmatch(r'[\u4e00-\u9fff]+', seg):
            out.append(seg) if len(seg) == 1 else out.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        else:
            out.append(seg)
    return out


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.N = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        self.df, self.tf = {}, []
        for d in self.docs:
            tfd = {}
            for t in d:
                tfd[t] = tfd.get(t, 0) + 1
            self.tf.append(tfd)
            for t in tfd:
                self.df[t] = self.df.get(t, 0) + 1

    def score(self, query, idx):
        q = tokenize(query)
        tfd, dl = self.tf[idx], self.doc_len[idx]
        s = 0.0
        for t in q:
            if t not in tfd:
                continue
            idf = math.log((self.N - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1)
            f = tfd[t]
            s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


# ---------------- embedding ----------------
def embed_batch(texts):
    vecs = []
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        r = requests.post(
            f"{EMB_URL}/embeddings",
            headers={"Authorization": f"Bearer {EMB_KEY}", "Content-Type": "application/json"},
            json={"model": EMB_MODEL, "input": batch}, timeout=60,
        )
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda x: x["index"])
        vecs += [d["embedding"] for d in data]
        time.sleep(0.2)
    arr = np.array(vecs, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


# ---------------- rerank ----------------
def rerank(query, docs_text, top_n=5):
    try:
        r = requests.post(
            RERANK_URL,
            headers={"Authorization": f"Bearer {EMB_KEY}", "Content-Type": "application/json"},
            json={"model": RERANK_MODEL, "query": query, "documents": docs_text,
                  "top_n": top_n, "return_documents": True}, timeout=30,
        )
        r.raise_for_status()
        return [x["index"] for x in r.json().get("results", [])]
    except Exception:
        return None  # 降级：调用方用原顺序


# ---------------- 主流程 ----------------
parent_ids = [s["id"] for s in SECTIONS]
parent_texts = [s["text"] for s in SECTIONS]
parent_id_to_text = {s["id"]: s["text"] for s in SECTIONS}

child_ids, child_texts = [], []
for s in SECTIONS:
    for c in make_child_chunks(s["text"]):
        child_texts.append(c)
        child_ids.append(s["id"])

print(f"父块 {len(parent_ids)} 个, 子块 {len(child_texts)} 个, query {len(QUERIES)} 条", flush=True)
print("embedding 父块/子块/query ...", flush=True)
parent_vecs = embed_batch(parent_texts)
child_vecs = embed_batch(child_texts)
query_vecs = embed_batch([q["q"] for q in QUERIES])
bm25 = BM25(child_texts)
print("检索评测中 ...", flush=True)


def rank_A(qi, q):
    scores = parent_vecs @ query_vecs[qi]
    return [parent_ids[i] for i in np.argsort(-scores)]


def rank_B(qi, q):
    d_scores = child_vecs @ query_vecs[qi]
    d_order = np.argsort(-d_scores)
    s_scores = np.array([bm25.score(q, i) for i in range(len(child_texts))])
    s_order = np.argsort(-s_scores)
    rrf = {}
    for order in (d_order, s_order):
        for rank, idx in enumerate(order):
            rrf[idx] = rrf.get(idx, 0) + 1.0 / (RRF_K + rank + 1)
    child_ranked = sorted(rrf.keys(), key=lambda i: -rrf[i])
    seen, candidates = set(), []
    for ci in child_ranked:
        pid = child_ids[ci]
        if pid not in seen:
            seen.add(pid)
            candidates.append(pid)
    top = candidates[:12]
    idxs = rerank(q, [parent_id_to_text[p] for p in top], top_n=5)
    if idxs is None:
        return top[:5]
    return [top[i] for i in idxs[:5]]


def evaluate(name, rank_fn):
    hits = {k: 0 for k in K_VALUES}
    recall = {k: 0.0 for k in K_VALUES}
    mrr = 0.0
    cat_hit5 = {}
    per = []
    for qi, q in enumerate(QUERIES):
        ranked = rank_fn(qi, q["q"])
        gt = q["gt"]
        found = None
        for rank, pid in enumerate(ranked, 1):
            if pid in gt and found is None:
                found = rank
        for k in K_VALUES:
            top = ranked[:k]
            nh = sum(1 for p in top if p in gt)
            hits[k] += (1 if nh > 0 else 0)
            recall[k] += nh / len(gt)
        if found:
            mrr += 1.0 / found
        cat_hit5.setdefault(q["cat"], {"n": 0, "hit": 0})
        cat_hit5[q["cat"]]["n"] += 1
        cat_hit5[q["cat"]]["hit"] += (1 if any(p in gt for p in ranked[:5]) else 0)
        per.append((q["cat"], q["q"], gt, ranked[:5], found))
    N = len(QUERIES)
    return {"name": name, "hit": {k: hits[k] / N for k in K_VALUES},
            "recall": {k: recall[k] / N for k in K_VALUES}, "mrr": mrr / N,
            "cat_hit5": cat_hit5, "per": per}


A = evaluate("A 单一大块纯稠密", rank_A)
B = evaluate("B 父子块+稠密+BM25+RRF+rerank", rank_B)

# ---------------- 报告 ----------------
lines = []
lines.append("# 检索策略评测报告（sunmind）\n")
lines.append(f"- 语料：{len(parent_ids)} 个父块（约 {sum(len(t) for t in parent_texts)} 字），切出 {len(child_texts)} 个子块（200 字符 / 重叠 50）")
lines.append(f"- 评测集：{len(QUERIES)} 条 query（fact 事实 / paraphrase 改写 / multi 跨块 / exact 术语）")
lines.append(f"- embedding：{EMB_MODEL}（{parent_vecs.shape[1]} 维）；rerank：{RERANK_MODEL}")
lines.append(f"- 指标：hit@1/3/5、MRR、recall@5\n")

lines.append("## 总体指标\n")
lines.append("| 系统 | hit@1 | hit@3 | hit@5 | MRR | recall@5 |")
lines.append("|---|---|---|---|---|---|")
for R in (A, B):
    lines.append(f"| {R['name']} | {R['hit'][1]:.3f} | {R['hit'][3]:.3f} | {R['hit'][5]:.3f} | {R['mrr']:.3f} | {R['recall'][5]:.3f} |")

lines.append("\n## 分类型 hit@5\n")
lines.append("| 类型 | A 单一块 | B 父子+混合+rerank |")
lines.append("|---|---|---|")
cats = ["fact", "paraphrase", "multi", "exact"]
for c in cats:
    if c not in A["cat_hit5"]:
        continue
    a = A["cat_hit5"][c]["hit"] / A["cat_hit5"][c]["n"]
    b = B["cat_hit5"][c]["hit"] / B["cat_hit5"][c]["n"]
    lines.append(f"| {c}（n={A['cat_hit5'][c]['n']}） | {a:.3f} | {b:.3f} |")

lines.append("\n## 逐条明细（B 系统）\n")
for cat, q, gt, ranked, found in B["per"]:
    mark = "✓" if found else "✗"
    lines.append(f"- {mark} [{cat}] {q}  → 命中{'#'+str(found) if found else '未命中'}")

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# 控制台摘要（ASCII，避免乱码）
print("\n===== SUMMARY =====")
for R in (A, B):
    print(f"{R['name']}: hit@1={R['hit'][1]:.3f} hit@3={R['hit'][3]:.3f} hit@5={R['hit'][5]:.3f} MRR={R['mrr']:.3f} recall@5={R['recall'][5]:.3f}")
print("report ->", REPORT)
