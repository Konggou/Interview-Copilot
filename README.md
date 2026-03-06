# Interview Copilot

`Interview Copilot` 是一个基于简历知识库的 AI 模拟面试系统。用户上传简历 PDF、填写岗位 JD 后，系统会以技术面试官视角发起多轮面试，并基于简历证据对回答进行评分、追问和报告生成。

当前项目采用前后端分离架构：
- 前端：[client](/D:/Desktop/study/rag-bot-fastapi/client)，基于 `Streamlit`
- 后端：[server](/D:/Desktop/study/rag-bot-fastapi/server)，基于 `FastAPI`

## 核心能力

- 简历入库：支持 PDF 上传、文本切块、向量化、Chroma 持久化
- 模拟面试：支持显式开始面试、逐轮回答、显式结束面试
- 证据约束：评分与反馈基于简历检索证据，返回 `sources / page / score / snippet`
- JD 对齐：问题生成会结合岗位 JD 调整方向
- 会话持久化：面试轮次与报告保存到本地 JSON 会话存储
- 自动化测试：提供核心接口的 `pytest` 测试

## 技术栈

- 前端：Streamlit
- 后端：FastAPI
- 模型接入：DeepSeek API
- LLM 编排：LangChain
- 向量数据库：Chroma
- Embedding：sentence-transformers（`all-MiniLM-L12-v2`）
- 文档处理：PyPDF + TokenTextSplitter
- 测试：Pytest

## 当前产品形态

当前版本不再是通用 PDF Chatbot，而是单一目标的面试产品：

- 主界面只保留面试对话流和一个输入框
- 用户在侧边栏上传简历、填写 JD、开始/结束面试
- 后端按轮生成问题，并对每轮回答做结构化评分
- 面试结束后生成总结报告

## 项目结构

```text
rag-bot-fastapi/
├── client/                         # Streamlit 前端
│   ├── app.py
│   ├── components/
│   │   ├── chat.py                # 面试对话流与统一输入框
│   │   ├── interview.py           # 面试报告展示
│   │   ├── inspector.py           # 遗留调试组件
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
│   │   └── vector_database.py
│   ├── tests/
│   │   └── test_api.py
│   ├── utils/
│   │   └── logger.py
│   └── main.py
└── assets/
```

## 主要接口

后端核心接口定义在 [server/api/routes.py](/D:/Desktop/study/rag-bot-fastapi/server/api/routes.py)：

- `POST /upload_and_process_pdfs`
- `POST /interview/start`
- `POST /interview/answer`
- `POST /interview/end`
- `GET /interview/report/{session_id}`
- `POST /vector_store/search`
- `GET /vector_store/count/{model_provider}`
- `GET /llm`
- `GET /llm/{model_provider}`
- `GET /health`

## 环境变量

在 [server/config/settings.py](/D:/Desktop/study/rag-bot-fastapi/server/config/settings.py) 中读取：

```env
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
SIMILARITY_THRESHOLD=1.4
```

说明：
- 当前项目只保留 `deepseek` 模型提供商
- `SIMILARITY_THRESHOLD` 用于检索阈值判断

## 安装

```bash
git clone https://github.com/Konggou/Interview-Copilot.git
cd Interview-Copilot
```

创建虚拟环境：

```bash
python -m venv venv
```

Windows PowerShell 激活：

```bash
.\venv\Scripts\Activate.ps1
```

安装依赖：

```bash
pip install -r server/requirements.txt
pip install -r client/requirements.txt
```

## 启动方式

启动后端：

```bash
cd server
python -m uvicorn main:app
```

启动前端：

```bash
cd client
python -m streamlit run app.py
```

默认地址：
- 后端：`http://127.0.0.1:8000`
- 前端：`http://localhost:8501`

## 测试

运行核心接口测试：

```bash
python -m pytest server/tests/test_api.py -q
```

当前已覆盖的主流程包括：
- 健康检查
- `/chat`
- `/interview/start`
- `/interview/answer`
- `/interview/end`
- `/interview/report`

## 典型流程

1. 上传简历 PDF 并写入向量库
2. 在侧边栏填写岗位 JD
3. 点击开始面试
4. 在主界面逐轮回答问题
5. 点击结束面试
6. 生成并查看面试报告

## 当前实现说明

- 面试问题由 LLM 按轮生成，不是一次性固定整套题
- 回答评分会结合：
  - 当前问题
  - 当前轮绑定的简历证据
  - 最近几轮面试摘要
- 会话当前存储在本地 JSON 文件中，适合单机演示和项目展示

## 后续可扩展方向

- 将会话存储从 JSON 升级为 SQLite
- 增加更稳定的中文 Prompt 与评测集
- 增加岗位匹配分析和候选人总评
- 增加 Docker 化部署方式
