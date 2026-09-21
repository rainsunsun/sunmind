"""查询 users 表的用户画像（正画像 + 负画像）。

用法（在任意目录均可）：
    cd backend && python test/check_profile.py

会直接连本地 PostgreSQL（读 .env 的 PG_* 配置），打印所有用户的画像。
同时会幂等补上 negative_profile 列（老库第一次跑会自动 ALTER）。
"""
import os
import sys
import asyncio
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR.parent / ".env")

import asyncpg


async def main():
    user = os.getenv("PG_USER")
    password = os.getenv("PG_PASSWORD")
    host = os.getenv("PG_HOST")
    port = int(os.getenv("PG_PORT", "5432"))
    db = os.getenv("PG_DB_NAME")

    try:
        conn = await asyncpg.connect(
            user=user, password=password, host=host, port=port, database=db
        )
    except Exception as e:
        print(f"❌ 连接 PostgreSQL 失败: {e}")
        print("   请确认后端已启动（或本地 PG 正在运行），且 .env 里 PG_* 配置正确。")
        return

    # 幂等补列，避免老库没有 negative_profile 列
    await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS negative_profile TEXT")

    rows = await conn.fetch(
        "SELECT user_id, user_profile, negative_profile, updated_at "
        "FROM users ORDER BY user_id"
    )
    await conn.close()

    if not rows:
        print("users 表为空：还没有任何用户画像被写入。")
        print("原因通常是：对话还没累计到 1000 token，后台压缩任务尚未触发提取。")
        return

    print(f"共 {len(rows)} 个用户：\n")
    for r in rows:
        print(f"user_id = {r['user_id']}")
        print(f"  正画像 user_profile     = {r['user_profile']!r}")
        print(f"  负画像 negative_profile = {r['negative_profile']!r}")
        print(f"  更新时间 updated_at      = {r['updated_at']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
