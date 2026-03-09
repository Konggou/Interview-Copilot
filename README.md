# Interview Copilot

`Interview Copilot` 是一个基于简历知识库的 AI 模拟面试系统。用户上传简历 PDF、填写岗位 JD 后，系统会以技术面试官视角发起多轮面试，并基于简历检索证据对回答进行评分、追问和报告生成。

当前项目采用前后端分离架构：

- 前端：[`client`](client)，基于 `Streamlit`
- 后端：[`server`](server)，基于 `FastAPI`

## 核心能力

- 简历入库：支持 PDF 上传、文本切块、向量化、Chroma 持久化
- 模拟面试：支持开始面试、逐轮回答、结束面试、生成报告
- 证据约束：评分与反馈基于简历检索证据，返回 `sources / page / score / snippet`
- 流式输出：聊天和面试接口支持 SSE 流式返回，前端可逐 token 渲染
- 语义缓存：基于 Redis 进行语义级响应缓存，返回 `X-Cache: HIT|MISS`
- 可观测性：提供 `GET /health` 与 `GET /metrics`，便于健康检查和指标采集
- 会话持久化：面试轮次与报告保存到本地 JSON 会话存储

## 技术栈

- 前端：Streamlit
- 后端：FastAPI
- 模型接入：DeepSeek API
- LLM 编排：LangChain
- 向量数据库：Chroma
- Embedding：sentence-transformers（`all-MiniLM-L12-v2`）
- 缓存：Redis
- 文档处理：PyPDF + TokenTextSplitter
- 日志与指标：structlog + prometheus-client
- 测试：pytest

## 快速开始

### 方式一：Docker Compose

```bash
cp .env.example .env
docker-compose up --build
```

默认端口：

- FastAPI：`http://127.0.0.1:8000`
- Streamlit：`http://127.0.0.1:8501`
- Redis：`redis://127.0.0.1:6379/0`

### 方式二：本地运行

```bash
git clone https://github.com/Konggou/Interview-Copilot.git
cd Interview-Copilot
python -m venv venv
```

Windows PowerShell 激活虚拟环境：

```powershell
.\venv\Scripts\Activate.ps1
```

安装依赖：

```bash
pip install -r server/requirements.txt
pip install -r client/requirements.txt
```

启动后端：

```bash
cd server
python -m uvicorn main:app --reload
```

启动前端：

```bash
cd client
python -m streamlit run app.py
```

## 环境变量

项目会优先读取仓库根目录 `.env`，也兼容 `server/.env`。建议从 [`.env.example`](.env.example) 复制一份后按需修改。

```env
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
REDIS_URL=redis://localhost:6379/0
SIMILARITY_THRESHOLD=1.4
SEMANTIC_CACHE_TTL_SECONDS=86400
SEMANTIC_CACHE_SIMILARITY_THRESHOLD=0.95
MAX_CONCURRENT_INTERVIEWS=5
DEEPSEEK_TIMEOUT_SECONDS=45
LOG_LEVEL=INFO
CLIENT_API_URL=http://127.0.0.1:8000
```

补充说明：

- 当前默认模型提供方为 `deepseek`
- Docker 环境下 `REDIS_URL` 通常为 `redis://redis:6379/0`
- `SIMILARITY_THRESHOLD` 用于简历检索阈值判断
- `SEMANTIC_CACHE_*` 控制 Redis 语义缓存时长与命中阈值

## 主要接口

后端核心接口定义在 [`server/api/routes.py`](server/api/routes.py)。

基础能力：

- `GET /health`
- `GET /metrics`
- `GET /llm`
- `GET /llm/{model_provider}`
- `POST /upload_and_process_pdfs`
- `POST /vector_store/search`
- `GET /vector_store/count/{model_provider}`

聊天能力：

- `POST /chat`
- `POST /chat/stream`

面试能力：

- `POST /interview/start`
- `POST /interview/start/stream`
- `POST /interview/answer`
- `POST /interview/answer/stream`
- `POST /interview/end`
- `GET /interview/report/{session_id}`

## 项目结构

```text
rag-bot-fastapi/
├── client/                         # Streamlit 前端
│   ├── app.py
│   ├── components/
│   │   ├── chat.py                # 面试对话流与统一输入框
│   │   ├── interview.py           # 面试报告展示
│   │   └── sidebar.py             # 模型、上传、JD、工具区
│   ├── state/
│   │   └── session.py
│   └── utils/
│       ├── api.py
│       └── helpers.py
├── server/                         # FastAPI 后端
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── config/
│   │   └── settings.py
│   ├── core/
│   │   ├── document_processor.py
│   │   ├── interview_session_store.py
│   │   ├── llm_chain_factory.py
│   │   ├── llm_service.py
│   │   ├── metrics.py
│   │   ├── semantic_cache.py
│   │   ├── sse.py
│   │   └── vector_database.py
│   ├── tests/
│   │   └── test_api.py
│   ├── utils/
│   │   └── logger.py
│   ├── Dockerfile
│   └── main.py
├── samples/                        # 手工测试样例文件
├── docker-compose.yml
└── .env.example
```

## 样例文件

[`samples`](samples) 目录存放非运行时样例数据，可用于手工联调：

- `samples/resumes/`：示例简历 PDF 与 HTML 导出文件
- `samples/documents/`：其他测试文档

## 测试

运行核心接口测试：

```bash
python -m pytest server/tests/test_api.py -q
```

如果只想运行单个文件：

```bash
pytest server/tests/test_api.py
```

## 典型流程

1. 上传简历 PDF 并写入向量库
2. 在侧边栏填写岗位 JD
3. 开始面试
4. 在主界面逐轮回答问题
5. 结束面试
6. 查看结构化面试报告

## 当前实现说明

- 面试问题由 LLM 按轮动态生成，不是一次性固定整套题
- 回答评分会结合当前问题、绑定的简历证据和最近几轮上下文
- 面试流式接口适合前端实时展示生成过程
- 当前会话默认存储在本地 JSON 文件中，适合单机演示与开发

## 后续可扩展方向

- 将会话存储从 JSON 升级为 SQLite 或数据库服务
- 增加更稳定的中文 Prompt 与评测集
- 增加岗位匹配分析和候选人综合评价
- 补充更完整的自动化测试与 CI
