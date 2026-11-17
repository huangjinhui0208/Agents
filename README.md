# 代码库理解 Agent

这是一个面向代码仓库的只读问答 Agent。它使用 `.env` 中的 Ollama / DeepSeek 配置，通过浏览器聊天界面选择仓库、搜索源码并回答问题。关键观察保留文件路径、行号和内容哈希；工具轨迹可在界面中展开检查。

## 启动

克隆仓库后创建虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env` 选择 Ollama、DeepSeek 或其他 OpenAI-compatible 服务。使用 Ollama 时，先确保服务已启动且模型已经安装：

```bash
ollama serve
python codebase_understanding_agent/run.py
```

默认界面是 <http://127.0.0.1:8765>。如果不希望自动打开浏览器：

```bash
python codebase_understanding_agent/run.py --no-browser
```

应用通过仓库根目录 `.env` 中的 `CODE_AGENT_PROVIDER` 选择模型配置组。默认使用 `ollama` 组；填写 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_API_KEY` 后，把提供方改为 `deepseek` 即可切换，无需改源码。原有 `LLM_*` 变量继续作为 Ollama 的兼容回退配置。

远程模型会接收 Agent 检索到的相关源码片段。分析私有仓库前，请确认所选模型服务符合你的代码保密要求。

## 使用

1. 点击“选择代码仓库”。可以在目录浏览器中逐级进入目录，也可以粘贴绝对路径。
2. 选择当前目录后，在右侧聊天框提问，例如：
   - “这个仓库的主要模块和入口是什么？”
   - “`FunctionCallAgent.run()` 如何执行工具并把结果交回模型？”
   - “谁调用了这个函数？调用前有哪些条件？”
3. 回答下方会显示本次实际使用的工具轨迹和源码证据范围。
4. “新建会话”会为同一仓库创建新的对话历史；重要的主题笔记仍可跨会话检索。

## 设计边界

- **只读源码**：没有编辑、删除、执行脚本、运行测试或 Git 提交工具。
- **仓库隔离**：模型工具只接受仓库相对路径；拒绝 `..`、绝对路径和越界符号链接。
- **敏感文件保护**：默认跳过 `.env`、私钥、证书及凭据类文件，并排除 `.git`、`.venv`、`node_modules`、构建目录等。
- **可回溯**：Glob、Find、Grep、Read、LSP 与 Note 工具返回统一结果包络；完整事件日志写入独立数据目录。
- **原生工具循环**：只接受模型返回的 OpenAI-compatible Function Call；没有 Tool Call 时不会由程序伪造或代替模型搜索。
- **上下文完整**：每次模型请求都重建完整原生工具消息历史，不设置应用级输入或输出 token 预算；单次工具结果仍通过分页、行数和文件大小限制保持有界。
- **长期知识**：SQLite NoteStore 保存带源码证据的主题知识，`note_search` 使用向量和余弦相似度检索，而不是词频排序。
- **受限终端后端**：Grep 默认由白名单 `rg` 子进程执行；使用参数数组、`shell=False`、仓库工作目录、超时和有界 stdout。模型没有任意 Terminal/Shell 工具。
- **本地绑定**：Web 服务只监听 `127.0.0.1`，不会把仓库上传到第三方服务器。

默认数据目录是 `~/.codebase_understanding_agent/`，与所分析仓库分离。可以通过 `CODE_AGENT_DATA_DIR` 改到其他目录。

## 原生 Function Calling

应用只运行原生工具循环：

1. ContextBuilder 将系统规则、相关向量笔记、完整会话历史和工具 Schema 交给模型；
2. 模型返回真实 `tool_calls`；
3. ToolRegistry 校验并执行调用，将 Observation 写入 Runtime Memory；
4. 下一轮模型根据 Observation 继续调用工具或完成回答。

当前 `deepseek-r1:7b` 是否返回原生 Tool Call 取决于 Ollama 模型模板。若模型只返回普通文字，应用会明确终止该轮并提示更换支持 Function Calling 的模型，不会执行 evidence 预取、文本 JSON 工具兼容或确定性 bootstrap。

## 模型提供方切换

```dotenv
CODE_AGENT_PROVIDER=ollama

OLLAMA_MODEL_ID=deepseek-r1:7b
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
OLLAMA_API_KEY=ollama

DEEPSEEK_MODEL_ID=deepseek-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=
```

填好远程配置后，只需改成：

```dotenv
CODE_AGENT_PROVIDER=deepseek
```

API Key 只保存在本地 `.env`，不要提交到版本控制。

## Note 向量检索

默认 `local_hash` 是离线、零模型下载的特征哈希向量，用于立即启用向量与余弦检索。若需要语义嵌入，可配置任意 OpenAI-compatible Embeddings API：

```dotenv
CODE_AGENT_EMBEDDING_PROVIDER=openai_compatible
CODE_AGENT_EMBEDDING_MODEL=<embedding-model>
CODE_AGENT_EMBEDDING_BASE_URL=<.../v1>
CODE_AGENT_EMBEDDING_API_KEY=<secret>
```

NoteStore 会保存向量签名和内容哈希；切换嵌入模型或笔记内容变化后，会自动重建对应向量。

## LSP

应用会按文件类型探测本机已有的 `clangd`、`basedpyright-langserver` / `pyright-langserver`、`typescript-language-server` 或 `rust-analyzer`。也可设置 `CODE_AGENT_LSP_COMMAND` 指定一个使用 stdio 的语言服务器命令。未找到服务器或能力不支持时，工具返回明确错误，Agent 应退回 Grep + Read。

对不可信第三方仓库使用 LSP 时仍建议在 OS 沙箱或容器中运行语言服务器；路径校验不是完整的进程隔离。

## 测试

测试只针对 Agent 自身的小型临时 fixture，不会执行所选仓库的代码：

```bash
python -m unittest discover -s codebase_understanding_agent/tests -v
```

## 可调配置

常用环境变量：

| 变量 | 默认值 | 用途 |
| --- | ---: | --- |
| `CODE_AGENT_PORT` | `8765` | 本地界面端口 |
| `CODE_AGENT_MAX_STEPS` | `10` | 单个问题最大工具轮次 |
| `CODE_AGENT_MAX_FILE_LINES` | `160` | 单次 Read 最大行数 |
| `CODE_AGENT_MAX_MATCHES` | `50` | 单页 Grep 结果上限 |
| `CODE_AGENT_PROVIDER` | `ollama` | 当前聊天模型配置组 |
| `CODE_AGENT_EMBEDDING_PROVIDER` | `local_hash` | Note 向量生成方式 |
| `CODE_AGENT_TERMINAL_ENABLED` | `1` | 是否启用受限终端后端 |
| `CODE_AGENT_TERMINAL_ALLOWED_PROGRAMS` | `rg` | 允许执行的只读程序白名单 |
| `CODE_AGENT_TERMINAL_TIMEOUT` | `15` | 子进程超时秒数 |
| `CODE_AGENT_TERMINAL_MAX_STDOUT_BYTES` | `8000000` | 子进程最大 stdout 收集量 |

项目不再设置输入 token、输出 token、上下文窗口或 evidence 模式相关环境变量。
