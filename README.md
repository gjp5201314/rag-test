# 本地知识库 Web 控制台 + 千问远程模型

这是一个可直接运行的本地知识库项目，支持：

- 扫描本地目录中的文本文件
- 自动切分文档并建立本地 TF-IDF 检索索引
- 调用千问兼容接口完成基于上下文的问答
- 提供命令行方式与浏览器交互页面

## 项目结构

- `app.py`：后端服务、CLI、索引与问答主逻辑
- `templates/index.html`：前端页面模板
- `static/style.css`：页面样式
- `static/app.js`：前端交互逻辑
- `requirements.txt`：Python 依赖
- `.env.example`：环境变量示例
- `.env`：你的本地密钥配置（不要提交到仓库）
- `.gitignore`：忽略敏感配置与索引产物

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境变量

复制 `.env.example` 为 `.env`，填写 DashScope Key：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 准备知识库文件

默认扫描目录为 `./data`，可放入如下文件：

```text
data/
  docs.txt
  notes.md
  manual.md
```

当前支持的文本扩展名包括：

- `.txt`
- `.md`
- `.py`
- `.js`
- `.ts`
- `.tsx`
- `.jsx`
- `.json`
- `.yaml`
- `.yml`
- `.toml`
- `.ini`
- `.csv`
- `.html`
- `.css`
- `.java`
- `.go`
- `.rs`
- `.c`
- `.cpp`
- `.h`
- `.hpp`
- `.sql`
- `.log`

## 启动 Web 服务

```bash
python app.py serve --host 127.0.0.1 --port 7860
```

启动后打开：

```text
http://127.0.0.1:7860
```

## 部署到 Render

仓库中已提供 [`render.yaml`](render.yaml)，可直接用于 Render Blueprint / Web Service 部署。

### 1. 连接仓库

将当前 GitHub 仓库导入 Render。

### 2. 构建与启动

[`render.yaml`](render.yaml) 中已配置：

- 构建命令：`pip install -r requirements.txt`
- 启动命令：`waitress-serve --host 0.0.0.0 --port $PORT app:app`

### 3. 配置环境变量

在 Render 控制台中设置：

```env
DASHSCOPE_API_KEY=你的千问密钥
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 4. 注意事项

- Render 上的 `data/` 是部署产物中的目录，不是你本机磁盘。
- 如果你在网页中重建索引，索引文件会写入运行实例本地文件系统；免费实例重启后可能丢失，需要重新构建。
- 如果希望知识库内容长期稳定，建议把初始文档直接提交到仓库中的 [`data/`](data/) 目录。

页面中可以：

1. 输入文档目录与索引路径
2. 一键建立索引
3. 输入问题并查看模型回答
4. 查看命中的文档片段与相似度分数

## 命令行用法

### 建立索引

```bash
python app.py build --source ./data --index ./data/index.json
```

可选参数：

- `--chunk-size`：分片长度，默认 `1000`
- `--chunk-overlap`：分片重叠长度，默认 `200`

### 提问

```bash
python app.py ask "这个项目如何检索本地文件？" --index ./data/index.json --top-k 5
```

可选参数：

- `--top-k`：返回分片数，默认 `5`

## 后端 API

### 查询状态

```http
GET /api/status?index=./data/index.json
```

### 建立索引

```http
POST /api/build
Content-Type: application/json

{
  "source": "./data",
  "index": "./data/index.json",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

### 发起提问

```http
POST /api/ask
Content-Type: application/json

{
  "question": "这个项目怎么检索本地文件？",
  "index": "./data/index.json",
  "top_k": 5
}
```

## 工作流程

1. 从本地目录读取文本文件
2. 按固定长度切分为文档分片
3. 使用 TF-IDF 向量化分片
4. 对用户问题进行相似度检索
5. 将命中上下文发送给千问模型生成答案

## 安全说明

- `.env` 已被写入 `.gitignore`，避免提交密钥
- 不要把真实 API Key 写入源码文件
- 如果真实 Key 已经暴露，建议立即在 DashScope 控制台轮换
- 当前实现使用本地 TF-IDF，适合快速搭建；后续可替换为 embedding + 向量数据库方案
