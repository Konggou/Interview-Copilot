# Interview Copilot - Client

这是 `Interview Copilot` 的 Streamlit 前端，负责简历上传、岗位 JD 输入、面试控制、对话流展示与报告查看。

## 核心能力

- 上传简历 PDF 并提交到后端处理
- 选择模型并填写岗位 JD
- 显式开始和结束模拟面试
- 使用单一输入框完成多轮问答
- 查看并下载面试报告

## 目录结构

```text
client/
├── app.py                      # Streamlit 入口
├── components/
│   ├── chat.py                 # 面试对话流与统一输入框
│   ├── interview.py            # 报告展示
│   ├── inspector.py            # 遗留调试组件
│   └── sidebar.py              # 模型、上传、JD、工具区
├── state/
│   └── session.py              # Session State 初始化
└── utils/
    ├── api.py                  # 后端接口调用
    └── helpers.py              # 前端编排逻辑
```

## 安装

```bash
cd client
pip install -r requirements.txt
```

## 启动

```bash
cd client
python -m streamlit run app.py
```

默认地址：

```text
http://localhost:8501
```

## 使用流程

1. 选择模型
2. 上传简历 PDF 并提交
3. 填写岗位 JD
4. 点击开始面试
5. 在主界面逐轮回答问题
6. 点击结束面试并查看报告

## 界面说明

- 主界面采用类聊天产品形态
- 面试消息、用户回答、评分反馈显示在同一条消息流中
- 侧边栏负责配置、上传、JD 输入和面试控制
