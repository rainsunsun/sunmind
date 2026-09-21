# SunMind — 个性化智能问答助手

**念念不忘，必有回响**

SunMind 是一款基于 **FastAPI** 和 **LangChain** 构建的个性化 AI 问答助手。系统采用 **RAG（检索增强生成）** 技术，集成 **Milvus 向量数据库**与 **PostgreSQL 关系数据库**，具备**长期记忆能力**和**自定义知识库管理**功能。SunMind 能够记住历史对话内容，自动提取并存储用户偏好与习惯，结合私有知识库检索，为用户提供**精准、连贯、个性化**的智能问答服务。

**核心价值**：让 AI 真正「记住」你，而非每次对话都是初次见面。

- 仓库地址：https://github.com/rainsunsun/sunmind
- 技术栈：FastAPI · LangChain · LangGraph · Milvus · PostgreSQL · Redis · Streamlit · DashScope

## 📋 目录

- [🎯 项目概述](#-项目概述)
- [✨ 核心特性](#-核心特性)
- [🏗️ 系统架构](#️-系统架构)
- [🚀 快速开始](#-快速开始)
- [📖 API 文档](#-api-文档)
- [🔧 配置说明](#-配置说明)
- [📁 项目结构](#-项目结构)
- [🤝 贡献指南](#-贡献指南)

## 🎯 项目概述

SunMind 是一个全功能的个性化 AI 问答系统，主要特点包括：

- **短期记忆**：支持多轮对话，Redis 缓存会话历史，保证对话连贯性
- **长期记忆系统**：自动提取和存储用户对话记忆，支持个性化回答
- **知识库管理**：支持多知识库文档上传、检索和管理
- **混合检索**：结合稠密向量检索和稀疏检索，提供准确的文档匹配
- **实时对话**：流式响应，提供打字机效果的交互体验
- **智能记忆**：自动识别和过滤重复记忆，保持记忆库的高效性

## ✨ 核心特性

### 🧠 记忆系统
- **多类型记忆**：支持摘要记忆、语义记忆、情节记忆、程序记忆和用户画像
- **自动提取**：后台任务每 3 分钟自动检查并提取未压缩的对话
- **智能过滤**：相似度高于 0.9 的记忆自动标记为重复并过滤
- **记忆优化**：按主题合并碎片化记忆，确保信息完整性

### 📚 知识库管理
- **多知识库支持**：用户可创建多个知识库，分别管理不同领域的文档
- **文档处理**：支持 PDF、Word 文档上传，自动分块和向量化
- **混合检索**：结合稠密检索（语义搜索）和稀疏检索（关键词匹配）
- **重排序机制**：使用外部 API 进行相关性重排序，提供最优结果

### 🔍 检索系统
- **两层降级策略**：
  - 混合检索失败时自动降级为稠密向量检索
  - 重排序失败时降级为 RRF 融合排序结果
- **动态提示词**：根据选择的知识库动态切换系统提示词
- **检索范围控制**：「默认知识库」检索所有知识，指定知识库限定检索范围

### 💬 对话系统
- **流式响应**：实时逐 token 输出，提供打字机效果
- **工具调用展示**：显示 AI 的思考过程和工具调用状态
- **会话管理**：Redis 缓存会话历史，保证对话连贯性
- **实时保存**：对话内容实时异步写入 PostgreSQL 数据库

## 🌟 技术亮点

| 特性 | 技术实现 | 优势 |
|------|----------|------|
| **父子块混合检索** | 父块存 PostgreSQL，子块存 Milvus | 兼顾检索精度与上下文完整性 |
| **双重降级策略** | 混合检索→稠密检索 / Rerank→RRF | 保证服务高可用 |
| **乐观 UI 更新** | 前端立即响应 + 后台异步确认 | 流畅的用户体验 |
| **动态提示词切换** | 默认/指定知识库不同策略 | 灵活控制回答发散度与检索范围 |
| **定时记忆压缩** | 每 3 分钟检查 Token 阈值 | 自动化长期记忆管理 |
| **记忆分层存储** | 语义/情景/程序/摘要自动分类 | 独立管理，模拟人脑 |
| **记忆并行检索** | 四种记忆并行查询融合排序 | 耗时降低 60% |
| **三维智能排序** | 相似度 + 时近性 + 重要性评分 | 优先返回最相关记忆 |
| **用户画像融合** | 新旧画像智能合并 | 持续演进不丢失 |
| **智能记忆去重** | 向量相似度 ≥ 0.9 自动过滤 | 避免记忆库膨胀 |

## 🏗️ 系统架构

系统由 **前端、后端服务、Agent、三大存储** 四层构成：

```
┌──────────────────────────────────────────────┐
│              前端（frontend/app.py）          │
│          Streamlit · SSE 流式接收 · 乐观更新  │
└──────────────────────┬───────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼───────────────────────┐
│              后端入口（backend/main.py）      │
│              FastAPI · 生命周期 · 定时任务    │
└──────────────────────┬───────────────────────┘
                       │
┌──────────────────────▼───────────────────────┐
│           Agent 核心（backend/agent.py）      │
│   LangChain Agent · 工具调用 · 动态提示词     │
└───────┬──────────────┬──────────────┬────────┘
        │              │              │
   ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
   │  Milvus │    │PostgreSQL│   │  Redis  │
   │ 向量检索 │    │ 关系存储 │   │  缓存   │
   └─────────┘    └─────────┘   └─────────┘
   知识库子块     父块文本       会话检查点
   四类记忆向量   对话记录       用户画像缓存
                文件元数据
```

### 核心模块

**1. Agent 系统（`agent.py`）**
- 基于 LangChain 构建的智能代理
- 支持工具调用和流式响应生成
- 集成 Redis 检查点机制，实现会话状态管理
- 动态提示词切换，根据知识库选择调整回答策略

**2. 工具框架（`tools.py`）**
- `search_knowledge_base`：知识库混合检索工具
- `get_memory`：多类型记忆检索工具，模型根据当前对话内容自主决策各类型记忆检索的数量
- `get_raw_conversation_by_summary_id`：原始对话获取工具，输入 summary_id 获取摘要对应的原始对话

**3. 记忆管理（`auto_store_memory_from_psql.py`）**
- 后台任务自动提取对话记忆
- 支持 4 种记忆类型：摘要、语义、情节、程序，多轮对话自动提取用户画像
- 记忆冲突检测和重复过滤
- 用户画像智能融合更新

**4. 知识库管理（`knowledeg_base_manager.py`）**
- 父子分块策略（父块 1000 字符，子块 200 字符）
- 向量化存储和混合检索，RRF 融合排序
- 文件整体去重（SHA-256 哈希检测）+ 子块内容去重，避免重复存储

**5. 数据处理管道（`documents_process.py`）**
- 支持 PDF、Word 文档处理
- 文本分块和嵌入生成
- 重排序文档相关性评分

## 🚀 快速开始

### 环境要求

- Python 3.12+
- PostgreSQL 14+
- Redis 7.2+（本项目使用 redis-stack-server 镜像）
- Milvus 2.4+（或 Zilliz Cloud 云服务）
- DashScope API Key（阿里云灵积平台）

### 安装步骤

1. **克隆项目**

```bash
git clone https://github.com/rainsunsun/sunmind.git
cd sunmind
```

2. **创建虚拟环境**

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# 或
.venv\Scripts\activate      # Windows
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **配置环境变量**

复制 `.env.example` 为 `.env` 并填写以下参数（完整说明见下方「配置说明」）：

```env
# 大模型
DASHSCOPE_API_KEY=your_dashscope_api_key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
AGENT_BASE_MODEL=qwen3.5-flash-2026-02-23
SUMMARIZATION_MODEL=qwen-turbo

# 嵌入模型
EMBEDDING_MODEL=text-embedding-v4
dense_dimension=1024

# Rerank 重排序
RERANK_MODEL=gte-rerank-v2
RERANK_URL=https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
RERANK_API_KEY=your_dashscope_api_key

# Milvus / Zilliz Cloud
Milvus_url=your_milvus_url
Token=your_milvus_token
knowledge_base_collection=knowledge_base_collection
memory_collection=memory_collection

# PostgreSQL
DATABASE_URL=postgresql://user:pass@localhost:5432/sunmind_db?sslmode=disable
user=your_db_user
password=your_db_password
host=localhost
port=5432
db_name=sunmind_db

# Redis
REDIS_URL=redis://localhost:6379/0

# 临时文件目录
TEMP_DIR=/tmp
```

5. **启动后端服务**

```bash
cd backend
python main.py
# 或
uvicorn main:app --reload
```

6. **启动前端界面**

```bash
cd frontend
streamlit run app.py
```

### 访问应用

- **API 文档**：http://localhost:8000/docs
- **Web 界面**：http://localhost:8501

## 📖 API 文档

### 核心端点

#### 对话接口

```http
GET /chat_with_agent/stream
Parameters:
- query: 用户消息
- knowledge_base_id: 知识库ID
- user_id: 用户ID
Returns: 流式文本响应
```

#### 知识库管理

```http
POST /knowledge-bases                    # 创建知识库
DELETE /knowledge-bases/{id}             # 删除知识库
GET /knowledge-bases                     # 获取用户知识库列表
GET /knowledge-bases/{id}/files          # 获取知识库文件列表
```

#### 文档管理

```http
POST /document_upload                          # 上传文档
DELETE /knowledge-bases/{id}/documents/{hash}  # 删除文档
```

## 🔧 配置说明

### 记忆提取配置

记忆提取通过 LLM 根据对话内容动态评估重要性，评分标准在提示词中定义：

| 记忆类型 | 评分范围 | 评估依据 |
|----------|----------|----------|
| 语义记忆 | 0.1-1.0 | 事实稳定性、知识通用性、用户关联度 |
| 情节记忆 | 0.1-1.0 | 事件重要性、时间敏感性、与用户目标关联 |
| 程序记忆 | 0.1-1.0 | 方法复用性、任务频率、执行效果 |
| 摘要记忆 | 0.1-1.0 | 对话信息密度、关键决策点、主题连贯性 |

**基础评分参考**：
- 0.8-0.9：个人身份、核心价值观、重大决定
- 0.7-0.8：职业信息、学习目标、重要偏好
- 0.6-0.7：工作相关知识、技能学习、待办任务
- 0.3-0.5：日常对话、临时信息
- 0.1-0.3：闲聊、一次性信息

**配置方式**：修改 `config.py` 中的 `MEMORY_EXTRACT_PROMPT` 提示词模板。

### 记忆检索配置

记忆检索采用综合评分算法，从三个维度评估记忆相关性：

| 评分维度 | 权重 | 说明 |
|----------|------|------|
| 语义相似度 | 45% | 向量相似度，匹配问题与记忆内容 |
| 访问时间 | 25% | 最近访问的记忆得分更高（每小时衰减 0.5%） |
| 重要性评分 | 30% | LLM 提取时赋予的重要性（0-1 分） |

**记忆类型权重系数**：

| 类型 | 权重 | 说明 |
|------|------|------|
| 语义记忆 | 1.3 | 事实知识，优先级最高 |
| 程序记忆 | 1.2 | 方法流程，优先级较高 |
| 情节记忆 | 1.0 | 历史事件，标准优先级 |
| 摘要记忆 | 0.7 | 对话摘要，优先级较低 |

**配置位置**：`backend/memory_manager.py` 中的 `get_the_top_k_memories` 方法。

### 文档分块配置

```python
# config.py
parent_chunk_size = 1000        # 父块大小
parent_chunk_overlap = 200      # 父块重叠
child_chunk_size = 200          # 子块大小
child_chunk_overlap = 50        # 子块重叠
```

### Redis TTL 配置

```python
TTL_CONFIG = {
    "default_ttl": 60,          # 60 分钟后过期
    "refresh_on_read": True     # 读取时刷新 TTL
}
```

## 📁 项目结构

```
sunmind/
├── backend/
│   ├── tests/                         # 测试与评测脚本
│   ├── agent.py                       # LangChain 智能代理
│   ├── api.py                         # FastAPI 路由定义
│   ├── main.py                        # 应用入口
│   ├── tools.py                       # 工具函数集合
│   ├── config.py                      # 系统配置与提示词
│   ├── schemas.py                     # Pydantic 数据模型
│   ├── documents_process.py           # 文档处理管道
│   ├── auto_store_memory_from_psql.py # 记忆提取后台任务
│   ├── knowledeg_base_manager.py      # 知识库管理（父子块 + 混合检索）
│   ├── memory_manager.py              # 记忆管理（检索 + 综合评分）
│   ├── milvus_client.py               # Milvus 客户端
│   ├── postgresql_client.py           # PostgreSQL 客户端
│   ├── redis_cache.py                 # Redis 缓存管理
│   ├── hash_storage.py                # 文件哈希去重
│   └── verify_env.py                  # 环境连通性检测
├── frontend/
│   ├── app.py                         # Streamlit 前端应用
│   └── style.css                      # 前端样式
├── test_docs/                         # 测试文档
├── requirements.txt                   # Python 依赖
├── .env.example                       # 环境变量模板
└── README.md                          # 项目文档
```

## 🔄 数据流

### 对话流程

```
用户输入 → FastAPI 接收 → Agent 处理 → 工具调用 →
  ├→ 记忆检索（Milvus + PostgreSQL）
  ├→ 知识库搜索（Milvus + PostgreSQL + 重排序）
  └→ LLM 响应生成 → 流式返回
```

### 记忆提取流程

```
对话存储 → PostgreSQL（summary_id=NULL）→ 后台任务检查 →
Token 数量超过阈值 → 记忆提取 → 冲突检测 →
用户画像融合 → 存储到 Milvus 和 PostgreSQL
```

### 文档处理流程

```
文档上传 → 文件去重检查 → 临时存储 → 后台处理 →
文档解析 → 文本分块 → 向量化 →
存储到 Milvus（子块）+ PostgreSQL（父块）
```

## 🚀 性能优化

### 已知性能考虑

1. **多数据库往返**：每次请求涉及 PostgreSQL、Redis、Milvus 多次查询
2. **记忆检索**：涉及 4 种不同类型记忆的并行搜索
3. **知识库搜索**：包含向量搜索、PostgreSQL 查询和重排序
4. **Redis 连接**：当前为每个请求创建新连接，建议实现连接池
5. **耗时分析**：当前数据检索约 2-3 秒，主要耗时在 API 请求上——问题嵌入约 1 秒，重排序约 0.8-1.2 秒。为降低重复查询延迟，已在提示词中规定相关性问题利用上下文回答

### 优化策略

- 优化文件上传处理方式，可考虑使用对象存储（如 S3、OSS、GCS 等）
- 实现 Redis 连接池
- 缓存用户画像减少 PostgreSQL 查询（已实现）
- 批量数据库操作（已实现）
- 延迟加载非关键记忆
- 监控生产环境中的时间日志

## 🤝 贡献指南


### 开发流程

1. Fork 项目
2. 创建功能分支（`git checkout -b feature/amazing-feature`）
3. 提交更改（`git commit -m 'Add amazing feature'`）
4. 推送到分支（`git push origin feature/amazing-feature`）
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 代码规范，类名采用驼峰命名法，函数名采用下划线命名法
- 为新功能添加适当的测试
- 更新相关文档
- 保持代码注释的清晰和完整

## 📞 联系方式

- **项目维护者**：rainsunsun
- **邮箱**：1069217859@qq.com
- **GitHub**：[@rainsunsun](https://github.com/rainsunsun)

## 🙏 致谢

感谢以下开源项目和框架的支持：

- [SuperMew](https://github.com/icey1287/SuperMew)
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Milvus](https://milvus.io/)
- [Streamlit](https://streamlit.io/)
- [DashScope](https://dashscope.aliyun.com/)

---

<div align="center">

**[🔝 返回顶部](#sunmind--个性化智能问答助手)**

Made with ❤️ by rainsunsun · SunMind

</div>
