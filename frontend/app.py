# -*- coding: utf-8 -*-
"""sunmind 前端（浅蓝版）

结构：顶栏（Logo + 状态）→ Tab（对话 / 知识库 / 我的画像）→ 内容区。
数据加载 / 上传 / 删除 / 画像拉取均为同步封装 aiohttp（用独立 event loop 跑），
聊天为 requests 流式接收。
"""
import os
import json
import asyncio
from datetime import datetime
from urllib.parse import unquote
from html import escape

import streamlit as st
import aiohttp
import requests
import nest_asyncio

nest_asyncio.apply()

API_BASE_URL = "http://localhost:8000"
USER_ID = 1

st.set_page_config(
    page_title="sunmind · 个性化 AI 助手",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================ 样式 ============================================
def load_css() -> str:
    css = ""
    path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            css = f.read()
    return f"<style>{css}</style>"


st.markdown(load_css(), unsafe_allow_html=True)


# ============================================ 状态 ============================================
def init_state():
    defaults = {
        "chat_history": [],          # [{"role","content","time"}]
        "knowledge_bases": {},       # kb_id -> [file_name]
        "kb_files_info": {},         # kb_id -> {file_name: file_hash}
        "selected_kb": None,
        "active_tab": "对话",
        "streaming": False,
        "pending_query": None,
        "data_loaded": False,
        "load_error": None,
        "profile": {"user_profile": "", "negative_profile": ""},
        "profile_loaded": False,
        "profile_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def run(coro):
    """在独立事件循环中跑一个协程，返回结果（同步阻塞）。"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def now() -> str:
    return datetime.now().strftime("%H:%M")


# ============================================ 数据加载 ============================================
async def _fetch_files(session, kb_id):
    try:
        url = f"{API_BASE_URL}/knowledge-bases/{kb_id}/files"
        async with session.get(url, params={"user_id": USER_ID},
                               timeout=aiohttp.ClientTimeout(total=15)) as r:
            data = await r.json()
        files = data.get("files") or []
        names, info = [], {}
        for f in files:
            if isinstance(f, dict):
                name, h = f.get("file_name", ""), f.get("file_hash", "")
            else:
                name, h = str(f), ""
            if name:
                names.append(name)
                info[name] = h
        return kb_id, names, info
    except Exception:
        return kb_id, [], {}


async def _load_all():
    try:
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(f"{API_BASE_URL}/knowledge-bases",
                             params={"user_id": USER_ID}) as r:
                data = await r.json()
            kbs = data.get("knowledge_bases") or []
            if not kbs:
                kbs = ["默认知识库"]
            kbs = [k.get("knowledge_base_id") or k.get("name") or k.get("id") or str(k)
                   if isinstance(k, dict) else str(k) for k in kbs]
            kbs = list(dict.fromkeys(kbs))  # 去重保序
            if "默认知识库" not in kbs:
                kbs.append("默认知识库")
            results = await asyncio.gather(*[_fetch_files(s, kb) for kb in kbs])

        knowledge_bases, kb_files_info = {}, {}
        for kb_id, names, info in results:
            knowledge_bases[kb_id] = names
            kb_files_info[kb_id] = info
        return {
            "knowledge_bases": knowledge_bases,
            "kb_files_info": kb_files_info,
            "selected_kb": "默认知识库" if "默认知识库" in knowledge_bases else (kbs[0] if kbs else None),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================ API 操作 ============================================
async def _upload(kb_id, file_name, content, ctype):
    form = aiohttp.FormData()
    form.add_field("file", content, filename=file_name,
                   content_type=ctype or "application/octet-stream")
    form.add_field("knowledge_base_id", kb_id)
    form.add_field("user_id", str(USER_ID))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
            async with s.post(f"{API_BASE_URL}/document_upload", data=form) as r:
                return await r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def _delete_file(kb_id, file_hash):
    try:
        url = f"{API_BASE_URL}/knowledge-bases/{kb_id}/documents/{file_hash}"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.delete(url, params={"user_id": USER_ID}) as r:
                return await r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def _create_kb(kb_id):
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.post(f"{API_BASE_URL}/knowledge-bases",
                              params={"knowledge_base_id": kb_id, "user_id": USER_ID}) as r:
                return await r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def _delete_kb(kb_id):
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.delete(f"{API_BASE_URL}/knowledge-bases/{kb_id}",
                                params={"user_id": USER_ID}) as r:
                return await r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def _fetch_profile():
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.get(f"{API_BASE_URL}/user_profile",
                             params={"user_id": USER_ID}) as r:
                return await r.json()
    except Exception as e:
        return {"status": "error", "user_profile": "", "negative_profile": "",
                "message": str(e)}


# ============================================ 首次加载 ============================================
if not st.session_state.data_loaded:
    with st.spinner("正在连接后端并加载知识库..."):
        result = run(_load_all())
    if result.get("error"):
        st.session_state.load_error = result["error"]
        st.session_state.knowledge_bases = {"默认知识库": []}
        st.session_state.kb_files_info = {"默认知识库": {}}
        st.session_state.selected_kb = "默认知识库"
    else:
        st.session_state.knowledge_bases = result["knowledge_bases"]
        st.session_state.kb_files_info = result["kb_files_info"]
        st.session_state.selected_kb = result["selected_kb"]
    st.session_state.data_loaded = True


# ============================================ 渲染：顶栏 ============================================
st.markdown("""
<div class="topbar">
    <div class="brand">
        <span class="logo">◇ sunmind</span>
        <span class="status"><span class="status-dot"></span>在线</span>
    </div>
    <div class="tagline">personalized RAG · long-term memory</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.load_error:
    st.warning(f"⚠️ 后端连接异常：{st.session_state.load_error}（请确认后端已启动）")


# ============================================ 渲染：Tab 栏 ============================================
TABS = [("💬 对话", "对话"), ("📚 知识库", "知识库"), ("🧠 我的画像", "我的画像")]
cols = st.columns(len(TABS), gap="small")
for i, (label, key) in enumerate(TABS):
    active = st.session_state.active_tab == key
    with cols[i]:
        if st.button(label, key=f"tab_{key}", use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state.active_tab = key

st.markdown("")


# ============================================ 消息气泡 ============================================
def render_message(role: str, content: str, ts: str) -> str:
    from markdown import markdown
    try:
        body = markdown(content, extensions=["fenced_code", "tables"])
    except Exception:
        body = escape(content)
    who = "你" if role == "user" else "sunmind"
    row = "user" if role == "user" else "assistant"
    return f"""
    <div class="msg-row {row}">
        <div class="msg-bubble">
            <div class="msg-content">{body}</div>
            <div class="msg-meta"><span>{who}</span><span>•</span><span>{ts}</span></div>
        </div>
    </div>
    """


# ============================================ 渲染：对话 ============================================
def render_chat():
    s = st.session_state
    if not s.knowledge_bases:
        st.info("还没有知识库，请先到「知识库」标签创建。")
        return

    kb_list = list(s.knowledge_bases.keys())
    if s.selected_kb not in kb_list:
        s.selected_kb = kb_list[0]

    c1, c2 = st.columns([3, 1])
    with c1:
        s.selected_kb = st.selectbox("当前知识库", kb_list,
                                     index=kb_list.index(s.selected_kb),
                                     key="kb_select")
    with c2:
        cnt = len(s.knowledge_bases.get(s.selected_kb, []))
        st.caption(f"📄 {cnt} 个文件")

    # 历史消息
    if s.chat_history:
        for m in s.chat_history:
            st.markdown(render_message(m["role"], m["content"], m["time"]),
                        unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="welcome">
            <div class="glyph">❯_</div>
            <div>选择「默认知识库」可发散检索所有内容；选择指定知识库则严格只答库内内容。</div>
        </div>
        """, unsafe_allow_html=True)

    # 流式生成
    if s.streaming:
        query = s.pending_query
        s.pending_query = None
        status_ph = st.empty()
        answer_ph = st.empty()
        acc = ""
        try:
            url = f"{API_BASE_URL}/chat_with_agent/stream"
            params = {"query": query, "knowledge_base_id": s.selected_kb,
                      "user_id": USER_ID, "top_k": 5}
            with requests.get(url, params=params, stream=True, timeout=120) as r:
                if r.status_code != 200:
                    acc = f"❌ 后端错误：{r.status_code}"
                else:
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            d = json.loads(line.strip())
                            t = d.get("type")
                            c = d.get("content", "")
                            if t == "status":
                                status_ph.markdown(
                                    f"<div class='tool-status'>{escape(c)}</div>",
                                    unsafe_allow_html=True)
                            elif t == "answer":
                                acc += c
                                answer_ph.markdown(
                                    render_message("assistant", acc, now()),
                                    unsafe_allow_html=True)
                            elif t == "error":
                                acc = f"❌ {c}"
                                answer_ph.markdown(
                                    render_message("assistant", acc, now()),
                                    unsafe_allow_html=True)
                        except Exception:
                            acc += line
                            answer_ph.markdown(
                                render_message("assistant", acc, now()),
                                unsafe_allow_html=True)
        except Exception as e:
            acc = f"❌ 连接失败：{e}"
        status_ph.empty()
        s.chat_history.append({"role": "assistant", "content": acc, "time": now()})
        s.streaming = False
        st.rerun()

    # 输入框
    prompt = st.chat_input("给 sunmind 发消息...")
    if prompt:
        s.chat_history.append({"role": "user", "content": prompt.strip(), "time": now()})
        s.streaming = True
        s.pending_query = prompt.strip()
        st.rerun()


# ============================================ 渲染：知识库 ============================================
def render_knowledge_bases():
    s = st.session_state

    with st.expander("＋ 新建知识库", expanded=False):
        name = st.text_input("知识库名称", key="new_kb_name",
                             placeholder="例如：技术文档 / 产品手册",
                             label_visibility="collapsed")
        if st.button("创建", key="create_kb_btn", use_container_width=True):
            if not name or not name.strip():
                st.warning("请输入知识库名称")
            elif name.strip() == "默认知识库":
                st.warning("「默认知识库」是保留名称，请换一个")
            elif name.strip() in s.knowledge_bases:
                st.warning("该知识库已存在")
            else:
                with st.spinner("创建中..."):
                    run(_create_kb(name.strip()))
                s.knowledge_bases[name.strip()] = []
                s.kb_files_info[name.strip()] = {}
                s.selected_kb = name.strip()
                st.rerun()

    if not s.knowledge_bases:
        st.info("还没有知识库，先创建一个吧。")
        return

    for kb_name, files in s.knowledge_bases.items():
        with st.expander(f"📁 {kb_name} · {len(files)} 个文件",
                         expanded=(kb_name == s.selected_kb)):
            # 上传
            up = st.file_uploader("上传 PDF / Word",
                                  type=["pdf", "docx", "doc"],
                                  key=f"up_{kb_name}",
                                  label_visibility="collapsed")
            if up is not None:
                if st.button("⬆ 上传", key=f"upbtn_{kb_name}"):
                    if up.name in files:
                        st.warning("该文件已存在")
                    else:
                        with st.spinner("上传中..."):
                            res = run(_upload(kb_name, up.name, up.getvalue(), up.type))
                        if res.get("status") == "error":
                            st.error(f"上传失败：{res.get('message')}")
                        else:
                            s.knowledge_bases[kb_name].append(up.name)
                            s.kb_files_info[kb_name][up.name] = res.get("file_hash", "")
                            st.success("已上传，后台正在分块/向量化...")
                            st.rerun()

            # 文件列表
            if files:
                for i, fname in enumerate(files):
                    c1, c2 = st.columns([7, 1])
                    with c1:
                        st.markdown(
                            f"<div class='file-item'>📄 {escape(unquote(fname))}</div>",
                            unsafe_allow_html=True)
                    with c2:
                        h = s.kb_files_info.get(kb_name, {}).get(fname, "")
                        if st.button("🗑", key=f"del_{kb_name}_{i}"):
                            if h:
                                run(_delete_file(kb_name, h))
                            s.knowledge_bases[kb_name].remove(fname)
                            st.rerun()
            else:
                st.caption("📭 暂无文件")

            # 删除知识库
            if kb_name != "默认知识库":
                if st.button("🗑 删除该知识库", key=f"delkb_{kb_name}"):
                    run(_delete_kb(kb_name))
                    s.knowledge_bases.pop(kb_name, None)
                    s.kb_files_info.pop(kb_name, None)
                    if s.selected_kb == kb_name:
                        s.selected_kb = "默认知识库"
                    st.rerun()


# ============================================ 渲染：我的画像 ============================================
def render_profile():
    s = st.session_state

    if not s.profile_loaded:
        with st.spinner("拉取画像..."):
            res = run(_fetch_profile())
        if res.get("status") == "success":
            s.profile = {"user_profile": res.get("user_profile", ""),
                         "negative_profile": res.get("negative_profile", "")}
        else:
            s.profile_error = res.get("message", "后端未返回画像")
        s.profile_loaded = True

    if st.button("🔄 刷新画像", key="refresh_profile"):
        with st.spinner("刷新中..."):
            res = run(_fetch_profile())
        s.profile = {"user_profile": res.get("user_profile", ""),
                     "negative_profile": res.get("negative_profile", "")}
        s.profile_error = None if res.get("status") == "success" else res.get("message")

    if s.profile_error:
        st.warning(f"⚠️ {s.profile_error}（确认后端已重启并包含 /user_profile 接口）")

    pos = s.profile.get("user_profile", "").strip()
    neg = s.profile.get("negative_profile", "").strip()

    st.markdown("<div class='section-title'>User Profile · 用户画像</div>",
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="profile-card positive">
        <div class="profile-label">◆ 正画像 · 偏好 / 背景</div>
        <div class="profile-text{'' if pos else ' empty'}">{escape(pos) if pos else '（暂无）发一段超过 1000 token 的自我介绍，后台每 3 分钟自动提取后即可看到。'}</div>
    </div>
    <div class="profile-card negative">
        <div class="profile-label">◆ 负画像 · 禁忌 / 硬约束（只增不减）</div>
        <div class="profile-text{'' if neg else ' empty'}">{escape(neg) if neg else '（暂无）在对话中明确说出你的禁忌（如「不要贴代码」「别叫我老师」）即可被提取。'}</div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("提示：画像由后台任务每 3 分钟从对话中提取并合并；负画像作为硬约束注入每次回答。")


# ============================================ 主渲染 ============================================
active = st.session_state.active_tab
if active == "对话":
    render_chat()
elif active == "知识库":
    render_knowledge_bases()
else:
    render_profile()
