# SmartEmailAgent RAG 模块说明

## 模块概览

`rag_modules/` 包含 SmartEmailAgent 的核心业务逻辑，四个模块各司其职，由 `VectorStorageAndMemorySystemModule` 统一编排。

---

## 1. `__init__.py`

包入口，统一导出四个模块类：

```python
EmailAccessAndSynchronizationModule   # 邮件访问
AiAnalysisCoreModule                  # AI 分析
MilvusConnectionModule                # 向量数据库连接
VectorStorageAndMemorySystemModule    # 存储与记忆系统
```

---

## 2. `email_access_and_synchronization_module.py` — 邮件访问与同步

**类比：系统的"眼睛和手"**，负责从 Gmail 获取数据。

| 方法 | 功能 |
|------|------|
| `authenticate()` | OAuth2 授权登录 Gmail，token 缓存在 `token.pickle` |
| `fetch_new_emails(n)` | 增量拉取收件箱邮件（基于 `last_sync_time`，有 1 天安全重叠） |
| `fetch_new_emails_sent(n)` | 增量拉取已发送邮件 |
| `parse_email(msg_id)` | 解析单封邮件：提取标题、发件人、日期、正文、附件 |
| `extract_attachments(msg_id, part)` | 下载附件到本地 `attachments/` 目录，文本文件自动解码 |

**依赖**：`google-auth-oauthlib`、`googleapiclient`

---

## 3. `ai_analysis_core_module.py` — AI 分析核心

**类比：系统的"大脑"**，调用 LLM 对邮件进行智能处理。使用 LangChain 的 `ChatOpenAI`（兼容 OpenAI API 格式）。

| Agent | 功能 | 输出格式 |
|-------|------|----------|
| `summarizer_agent` | 3-5 句邮件摘要 + 提取 Action Items | 纯文本 |
| `reply_agent` | 按指定语气/长度/语言起草回信 | JSON (subject + body) |
| `classifier_agent` | 判断邮件优先级（High/Medium/Low/Spam） | JSON (priority + reason) |
| `task_extractor_agent` | 提取待办事项和会议信息 | JSON (tasks + meetings) |
| `context_summary_agent` | 将冗长上下文压缩为结构化记忆文本 | 纯文本 |
| `orchestrator_agent` | **占位方法**，预留给未来的协调调度逻辑 | — |

所有 Agent 共享同一个 LLM 模型实例，通过 `temperature` 参数控制输出随机性（摘要/分类用 0.1，回信用 0.7）。

**依赖**：`langchain_openai`、`langchain_core`、`pydantic`

---

## 4. `milvus_connection_module.py` — 向量数据库连接

**类比：系统的"存储层入口"**，对 Milvus 的薄封装。

| 方法 | 功能 |
|------|------|
| `connection()` | 建立 Milvus 连接（URI 从 `SmartEmailAgentConfig.milvus_url` 读取） |
| `close()` | 关闭连接 |

**依赖**：`pymilvus`

---

## 5. `vector_storage_and_memory_system_module.py` — 向量存储与记忆系统

**类比：系统的"调度中心"**，把所有模块串联起来，实现完整的 RAG 流水线。

### 管理的三个 Milvus 集合

| 集合名 | 用途 | 关键字段 |
|--------|------|----------|
| `accept_email_collection` | 收件箱邮件 | id, subject, from, body, priority, reason, embedding(512维) |
| `send_email_collection` | 已发送邮件 | id, subject, to, body, embedding(512维) |
| `context_historical` | 长期记忆 | content, embedding(512维), timestamp |

### 核心方法

| 方法 | 流水线 |
|------|--------|
| `store_historical_emails()` | 授权 → 拉取收件箱 → 逐封解析 → LLM 摘要 → 向量化 → 优先级分类 → 写入 Milvus |
| `email_reply_tone()` | 授权 → 拉取发件箱 → 逐封解析 → LLM 摘要 → 向量化 → 写入 Milvus（用于回复风格学习） |
| `historical_chat()` | 统计收/发件箱中高频联系人 Top-N |
| `context_historical(text)` | 将文本 LLM 压缩 → 向量化 → 存入长期记忆集合 |

### 模块依赖关系

```
VectorStorageAndMemorySystemModule (调度中心)
├── EmailAccessAndSynchronizationModule  (取邮件)
├── MilvusConnectionModule               (连数据库)
└── AiAnalysisCoreModule                (AI 分析)
```

所有向量索引使用 **HNSW + COSINE 相似度**，维度默认 **512**。
