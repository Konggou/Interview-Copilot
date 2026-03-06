# Interview Copilot - Server

这是 `Interview Copilot` 的 FastAPI 后端，负责简历解析、向量检索、面试问题生成、回答评分、会话持久化和报告生成。

## 核心能力

- 上传并处理简历 PDF
- 构建 Chroma 向量库并执行证据检索
- 驱动面试生命周期：
  - 开始面试
  - 逐轮评分
  - 结束面试
  - 生成报告
- 基于简历证据返回结构化 `sources`
- 提供接口级自动化测试

## 目录结构

```text
server/
├── api/                        # FastAPI 路由与 Schema
├── config/                     # 配置与环境变量
├── core/                       # 文档处理、向量库、LLM、会话存储
├── tests/                      # pytest 测试
├── utils/                      # 日志与工具
└── main.py                     # 应用入口
```

## 安装

```bash
cd server
pip install -r requirements.txt
```

## 配置

在 [config/settings.py](/D:/Desktop/study/rag-bot-fastapi/server/config/settings.py) 中读取环境变量：

```env
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
SIMILARITY_THRESHOLD=1.4
```

当前后端仅保留 `deepseek` 模型提供商。

## 启动

```bash
cd server
python -m uvicorn main:app
```

服务默认地址：

```text
http://127.0.0.1:8000
```

## 主要接口

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

## 测试

运行接口测试：

```bash
python -m pytest tests/test_api.py -q
```

## 实现说明

- 简历会被切块后写入 Chroma 向量库
- 面试问题由 LLM 按轮生成
- 每轮评分基于当前问题、简历证据和最近几轮摘要
- 会话当前持久化到本地 JSON 文件，适合单机演示
