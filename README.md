# 本地知识库 Web 控制台 + 千问远程模型

一个开箱即用的本地知识库问答系统，支持 PDF/Office 文档解析、TF-IDF 检索与千问大模型回答生成。

## 功能特性

- **多格式支持**：支持 TXT、MD、PDF、DOCX、DOC、PPTX、PPT、XLSX、XLS 及代码文件
- **知识库管理**：支持创建、切换、删除多个独立知识库
- **文档索引**：自动分片、建立 TF-IDF 倒排索引
- **智能问答**：基于检索上下文的 RAG 问答，支持调整 Top-K 参数
- **进度可视化**：构建索引和问答请求的实时进度条
- **溯源追踪**：展示命中的文档片段、相似度分数与原文引用

## 项目结构

```
rag-test/
├── app.py                 # 后端服务、CLI 命令、索引与问教主逻辑
├── templates/
│   └── index.html        # 前端页面模板
├── static/
│   ├── style.css         # 页面样式（深色主题）
│   └── app.js            # 前端交互逻辑
├── data/                  # 知识库文件与索引存储目录
│   ├── uploads/          # 上传的临时文件
│   └── index.json        # 索引数据文件
├── requirements.txt       # Python 依赖
├── .env.example          # 环境变量示例
├── .env                  # 本地密钥配置（不提交）
├── .gitignore            # 忽略敏感文件与索引产物
└── render.yaml           # Render 部署配置
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填写 DashScope API Key：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

> API Key 可在阿里云 DashScope 控制台获取：https://dashscope.console.aliyun.com/

### 3. 启动服务

```bash
python app.py serve --host 127.0.0.1 --port 7860
```

启动后访问：http://127.0.0.1:7860

### 4. 使用知识库

1. **选择或创建知识库**：点击知识库标签切换，或点击「+ 新建」创建新知识库
2. **上传文件**：支持拖拽或点击上传按钮，支持格式包括：
   - 文档：`.txt`, `.md`, `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`
   - 代码：`.py`, `.js`, `.ts`, `.json`, `.yaml`, `.csv`, `.html`, `.css` 等
3. **构建索引**：设置分片长度和重叠长度，点击「构建索引」
4. **开始提问**：输入问题，选择知识库，调整 Top-K 值后发送

## 支持的文件格式

| 类型 | 扩展名 |
|------|--------|
| 纯文本 | `.txt`, `.md` |
| PDF | `.pdf` |
| Office Word | `.docx`, `.doc` |
| Office Excel | `.xlsx`, `.xls` |
| Office PowerPoint | `.pptx`, `.ppt` |
| 代码 | `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.json`, `.yaml`, `.yml`, `.toml`, `.ini`, `.sql`, `.log`, `.html`, `.css`, `.java`, `.go`, `.rs`, `.c`, `.cpp`, `.h`, `.hpp` |

## 命令行用法

### Web 服务模式

```bash
python app.py serve --host 127.0.0.1 --port 7860
```

### 建立索引

```bash
python app.py build --source ./data --index ./data/index.json
```

可选参数：
- `--chunk-size`：分片长度，默认 `1000`
- `--chunk-overlap`：分片重叠长度，默认 `200`

### 提问（CLI 模式）

```bash
python app.py ask "你的问题" --index ./data/index.json --top-k 5
```

可选参数：
- `--top-k`：返回的分片数量，默认 `5`

## API 接口

### 查询状态

```http
GET /api/status?kb_id=default
```

响应示例：
```json
{
  "ready": true,
  "kb_name": "我的知识库",
  "kb_loaded": true,
  "chunk_count": 100,
  "api_configured": true
}
```

### 知识库管理

**获取知识库列表**

```http
GET /api/kb
```

**创建知识库**

```http
POST /api/kb
Content-Type: application/json

{"name": "新知识库", "description": "描述"}
```

**删除知识库**

```http
DELETE /api/kb/{kb_id}
```

**获取文件列表**

```http
GET /api/kb/{kb_id}/files
```

**上传文件**

```http
POST /api/kb/{kb_id}/files
Content-Type: multipart/form-data

file: <文件>
```

**删除文件**

```http
DELETE /api/kb/{kb_id}/files?filename=<编码后的文件名>
```

### 索引与问答

**建立索引**

```http
POST /api/build
Content-Type: application/json

{
  "kb_id": "default",
  "chunk_size": 1000,
  "chunk_overlap": 200
}
```

**查询构建状态**

```http
GET /api/build-status?job_id=<job_id>
```

**发起提问**

```http
POST /api/ask
Content-Type: application/json

{
  "question": "问题内容",
  "kb_id": "default",
  "top_k": 5
}
```

响应示例：
```json
{
  "answer": "根据检索到的上下文...",
  "results": [
    {
      "file": "文档.txt",
      "chunk_id": 42,
      "score": 0.85,
      "content": "相关片段内容..."
    }
  ]
}
```

## 部署到 Render

### 方式一：Blueprint 部署（推荐）

1. 将代码推送到 GitHub 仓库
2. 在 Render 控制台点击「New」→ 「Blueprint」
3. 连接你的 GitHub 仓库
4. Render 会自动读取 `render.yaml` 并配置服务

### 方式二：手动部署

1. 创建 Web Service，连接 GitHub 仓库
2. 设置环境变量：
   - `DASHSCOPE_API_KEY`
   - `QWEN_MODEL=qwen-plus`
   - `QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
3. 设置构建命令：`pip install -r requirements.txt`
4. 设置启动命令：`waitress-serve --host 0.0.0.0 --port $PORT app:app`

### 部署注意事项

- Render 免费实例重启后本地文件系统会清空，建议将初始文档提交到仓库 `data/` 目录
- 或使用对象存储保存索引文件

## 工作原理

```
用户问题
   ↓
TF-IDF 向量化（用户问题）
   ↓
相似度检索（与索引中的分片匹配）
   ↓
获取 Top-K 相关片段
   ↓
构建 Prompt（问题 + 上下文片段）
   ↓
调用千问 API 生成回答
   ↓
返回回答 + 命中的分片列表
```

## 常见问题

**Q: 上传 PDF 文件后无法构建索引？**

确保已安装 PDF 解析库：
```bash
pip install PyPDF2 pdfplumber
```

**Q: 删除文件提示「文件不存在」？**

这是旧版本的一个 bug，已在最新代码中修复。请更新代码并重启服务。

**Q: 索引构建很久正常吗？**

对于大型 PDF 文件（>1MB），解析和分片需要较长时间。可以在构建时调整分片大小（减小可提高速度，但可能影响检索精度）。

**Q: 如何清空所有索引？**

删除 `data/index.json` 文件，然后重启服务。

## 安全建议

- `.env` 文件已加入 `.gitignore`，请勿提交真实 API Key
- 如果 Key 泄露，请在 DashScope 控制台立即轮换
- 生产环境建议使用环境变量而非 `.env` 文件

## 技术栈

- **后端**：Flask + Waitress
- **检索**：TF-IDF（scikit-learn/joblib）
- **PDF 解析**：PyPDF2 + pdfplumber
- **前端**：原生 HTML/CSS/JavaScript
- **大模型**：阿里云千问（DashScope API）
