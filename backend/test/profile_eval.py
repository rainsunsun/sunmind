"""画像拆分（正画像 + 负画像）的最小评测脚本。

验证三件事：
1. 提取分离：LLM 是否把「正向偏好」和「禁忌/反感」分到 user_profile / negative_profile 两个字段。
2. 负画像合并：NEGATIVE_PROFILE_MERGE_PROMPT 是否「只增不减」，旧禁忌不丢、新禁忌加入。
   （这就是「动态更新」的单元级代理——后台每轮都是走这个 merge 来保留旧负画像的）
3. 注入格式：system prompt 是否正确注入负画像，且负画像排在正画像前面（优先级更高）。

用法（在任意目录均可运行，路径用 __file__ 绝对定位）：
    cd backend && python test/profile_eval.py

需要 .env 里的 LLM 配置（DeepSeek），会发起真实 LLM 调用、消耗少量 token。
注意：LLM 输出非确定，本脚本是「启发式」断言，用于改 prompt 后快速回归，不是精确单测。
"""
import os
import sys
import asyncio
from pathlib import Path

# 把 backend/ 加入 sys.path，并加载项目根目录（EchoMind/）的 .env
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR.parent / ".env")

import config
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from auto_store_memory_from_psql import extract_memories


results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}: {detail}")


def make_model() -> ChatOpenAI:
    """与后台记忆提取任务同款模型（SUMMARIZATION_MODEL）。"""
    return ChatOpenAI(
        model=os.getenv("SUMMARIZATION_MODEL", "deepseek-flash"),
        openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
        openai_api_base=os.getenv("BASE_URL"),
        temperature=0.2,
    )


# ===== 测试 1：提取分离（正 / 负画像） =====
async def test_extract_separation(model: ChatOpenAI):
    messages = [
        {"id": "1", "role": "human",
         "content": "我叫小明，之前做 Java 后端，现在在学大模型应用开发。",
         "created_at": "2026-09-15 10:00:00"},
        {"id": "2", "role": "ai",
         "content": "好的小明，后端转大模型这个方向挺热门的。",
         "created_at": "2026-09-15 10:00:05"},
        {"id": "3", "role": "human",
         "content": "对，我比较喜欢简洁直白的回答，另外你千万别给我贴代码块，也别叫我老师。",
         "created_at": "2026-09-15 10:00:10"},
        {"id": "4", "role": "ai",
         "content": "记住了，我会简洁回答，不贴代码块，不叫老师。",
         "created_at": "2026-09-15 10:00:15"},
    ]
    try:
        result = await extract_memories(messages, model)
    except Exception as e:
        check("提取分离", False, f"extract_memories 抛异常: {e}")
        return

    pos = result.get("user_profile", "")
    neg = result.get("negative_profile", "")
    print(f"    user_profile     = {pos!r}")
    print(f"    negative_profile = {neg!r}")

    check("提取·正画像含姓名", "小明" in pos, f"user_profile={pos!r}")
    check("提取·负画像含禁忌（代码块）", "代码" in neg, f"negative_profile={neg!r}")
    check("提取·负画像含禁忌（称呼）", "老师" in neg, f"negative_profile={neg!r}")


# ===== 测试 2：负画像合并（只增不减） =====
async def test_negative_merge(model: ChatOpenAI):
    old_neg = "不要用代码展示；讨厌冗长回答"
    new_neg = "拒绝被叫老师"
    try:
        resp = await model.ainvoke([
            HumanMessage(content=config.NEGATIVE_PROFILE_MERGE_PROMPT.format(
                old_negative_profile=old_neg,
                new_negative_profile=new_neg,
            ))
        ])
        merged = resp.content.strip()
    except Exception as e:
        check("负画像合并", False, f"模型调用异常: {e}")
        return

    print(f"    合并结果 = {merged!r}")
    check("负画像合并·旧禁忌保留（代码）", "代码" in merged, f"merged={merged!r}")
    check("负画像合并·新禁忌加入（老师）", "老师" in merged, f"merged={merged!r}")


# ===== 测试 3：注入格式 =====
def test_injection():
    neg = "不要用代码展示"
    pos = "用户名为小明，后端开发"
    prompt = config.SYSTEM_PROMPT_DEFAULT.format(
        negative_profile=neg, user_profile=pos
    )
    neg_idx = prompt.find("用户负画像")
    pos_idx = prompt.find("用户画像：")
    neg_before_pos = neg_idx != -1 and pos_idx != -1 and neg_idx < pos_idx

    check("注入·负画像存在", neg in prompt)
    check("注入·负画像排在正画像前", neg_before_pos,
          f"neg_idx={neg_idx}, pos_idx={pos_idx}")


async def main():
    print("=" * 60)
    print("画像拆分评测（正画像 + 负画像）")
    print("=" * 60)
    model = make_model()

    print("\n--- 测试 1：提取分离 ---")
    await test_extract_separation(model)

    print("\n--- 测试 2：负画像合并（只增不减）---")
    await test_negative_merge(model)

    print("\n--- 测试 3：注入格式 ---")
    test_injection()

    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    print("\n" + "=" * 60)
    print(f"结果：{passed}/{total} 通过")
    print("=" * 60)
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
