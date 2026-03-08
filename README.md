# Agent Router Platform

一个面向展示、实验和治理的 AI 路由平台。

它不是单纯的聊天后端，而是把一条请求从“进入系统”到“选模型”“执行结果”“治理分析”整条链路都显式展示出来。你既可以把它当成一个多模型对话平台来用，也可以把它当成一个 AI 路由治理样板来演示。

## 现在这套系统在做什么

当前系统由 4 层组成：

1. 输入理解层
   - 提取结构化特征值，如 `code_signal`、`reasoning_signal`、`retrieval_signal`、`summarization_signal`
   - 将请求归类为 `general_qa`、`summarization`、`coding`、`reasoning`、`rag`

2. 决策层
   - 当前实时决策模型使用 `DeepSeek`
   - 通过 `AI_DECISION_MODEL=deepseek/deepseek-chat` 做路由判断
   - 如果 AI 决策失败、返回非法结果、或置信度不足，则回退到本地 `RuleRouter`

3. 执行层
   - 根据最终路由结果调用实际执行模型
   - 统一通过 LiteLLM Gateway 访问不同 provider

4. 治理层
   - 记录 execution logs、policy recommendation、audit timeline
   - 提供统计、分析、推荐、模拟、rollout plan、apply readiness、dashboard、report、portfolio 等只读治理能力

一句话说明当前状态：

`DeepSeek 负责实时路由决策，RuleRouter 负责安全兜底，router.py 仍然是线上最终保护边界。`

## 页面说明

启动后你会看到 4 个主要页面：

- `/`
  - 首页
  - 适合演示系统入口、页面分工和当前 live 状态

- `/workspace`
  - 对话页
  - 用来直接发消息拿回答
  - 右侧会显示本轮任务类型、选中路由、决策模型、置信度、是否回退

- `/classification`
  - AI 决策层展示页
  - 用来解释系统为什么这样路由
  - 只做决策检查，不生成最终回答

- `/governance`
  - 治理中心
  - 用来查看 policy dashboard、portfolio、timeline、report 等治理信息

另外：

- `/docs`
  - Swagger/OpenAPI 文档

## 决策链路

当前实时请求的主要链路如下：

```text
用户输入
  -> FeatureExtractor
  -> TaskClassifier
  -> AIDecisionEngine (DeepSeek)
  -> RuleRouter fallback
  -> LiteLLM Gateway
  -> Validation / Logging / Governance
```

当前可用的主要路由目标：

- `fast_general`
- `strong_reasoning`
- `code_specialist`
- `long_context_rag`

当前规则路由不是简单关键词匹配，而是基于特征值和 route score 进行选择；AI 决策模型则在这之上给出实时路由建议和覆盖。

## 主要能力

- 多模型聊天入口
- 只读路由检查 `/api/v1/router/inspect`
- AI 决策可解释展示
- 模型注册表展示
- 执行日志与统计聚合
- AI 分析与路由建议
- recommendation review / approval / rejection
- policy snapshot / dashboard / audit / timeline
- dry-run simulation / rollout planning / apply readiness
- 单条 recommendation report
- portfolio 级治理视图

重要说明：

- 治理接口全部是只读或治理态操作
- 不会自动修改 `router.py`
- 不会自动应用 approved recommendation
- 不会自动执行 rollout

## 项目结构

```text
app/
├─ agent/                     # 编排层，串起分类、决策、调用、验证、落库
├─ api/                       # FastAPI 路由
├─ core/                      # 配置、数据库、日志
├─ evaluator/                 # 输出校验
├─ models/                    # LiteLLM Gateway 与模型注册表
├─ router/                    # 特征提取、分类、规则路由、AI 决策
├─ schemas/                   # 请求/响应 schema
├─ static/                    # 首页、对话页、决策页、治理页
└─ storage/                   # SQLAlchemy 实体与 repository

tests/                        # unittest + TestClient
agent_router.db               # 本地 SQLite 数据库
```

## 快速开始

### 1. 创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 复制环境变量

```powershell
Copy-Item .env.example .env
```

### 3. 启动服务

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

### 4. 打开页面

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/workspace`
- `http://127.0.0.1:8000/classification`
- `http://127.0.0.1:8000/governance`
- `http://127.0.0.1:8000/docs`

## 运行模式

### Mock 模式

默认配置：

```env
MOCK_LLM_RESPONSES=true
```

适合：

- UI 展示
- 离线开发
- 不依赖真实 provider key 的联调

在这个模式下：

- `/workspace` 可以完整演示
- `/classification` 可以展示路由判断
- `/governance` 可以展示治理界面
- 不必配置外部 API key

### 真实 Provider 模式

如果要切到真实调用：

```env
MOCK_LLM_RESPONSES=false
ENABLE_AI_DECISION_ENGINE=true
AI_DECISION_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=your_deepseek_key
```

同时配置执行模型，例如：

```env
FAST_GENERAL_MODEL=dashscope/qwen-plus
STRONG_REASONING_MODEL=deepseek/deepseek-reasoner
CODE_SPECIALIST_MODEL=deepseek/deepseek-chat
RAG_MODEL=volcengine/your-doubao-model-or-endpoint
LOCAL_FALLBACK_MODEL=dashscope/qwen-plus
```

说明：

- `AI_DECISION_MODEL` 只负责“选路由”
- `FAST_GENERAL_MODEL` / `STRONG_REASONING_MODEL` / `CODE_SPECIALIST_MODEL` / `RAG_MODEL` 负责“真正生成回答”
- 如果 `DeepSeek` 决策失败，系统会退回本地规则路由

## 核心环境变量

| 变量 | 说明 |
| --- | --- |
| `APP_NAME` | 页面和 API 使用的应用名 |
| `API_PREFIX` | API 前缀，默认 `/api/v1` |
| `DATABASE_URL` | 默认使用本地 SQLite |
| `MOCK_LLM_RESPONSES` | 是否启用 mock 响应 |
| `ENABLE_AI_DECISION_ENGINE` | 是否启用 AI 决策层 |
| `AI_DECISION_MODEL` | 当前路由决策模型，默认 DeepSeek |
| `AI_DECISION_MIN_CONFIDENCE` | AI 决策覆盖规则的最低置信度 |
| `FAST_GENERAL_MODEL` | 普通问答路由使用的模型 |
| `STRONG_REASONING_MODEL` | 强推理路由使用的模型 |
| `CODE_SPECIALIST_MODEL` | 代码路由使用的模型 |
| `RAG_MODEL` | 长上下文 / 检索路由使用的模型 |
| `LOCAL_FALLBACK_MODEL` | 执行回退模型 |

## 常用接口

### 健康检查

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
```

### 查看模型注册表

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/models
```

### 做一次路由检查

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/router/inspect `
  -ContentType "application/json" `
  -Body '{"message":"请比较规则路由和 AI 决策路由的优缺点。"}'
```

### 发起一次真实对话

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/chat `
  -ContentType "application/json" `
  -Body '{"message":"请解释为什么多模型架构比单模型入口更灵活。","metadata":{"source":"manual-test"}}'
```

### 查看治理看板

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/admin/policy/dashboard
```

## 推荐的演示顺序

如果你要对外展示，建议按这个路径：

1. 打开 `/`
   - 先讲系统是什么

2. 打开 `/workspace`
   - 直接发一条普通问答
   - 再发一条代码问题
   - 让别人看到“能用”

3. 打开 `/classification`
   - 输入同样的问题
   - 展示任务类型、决策模型、置信度、规则回退
   - 解释“系统为什么这样选”

4. 打开 `/governance`
   - 展示推荐、审查、模拟、rollout、audit、report
   - 说明这套系统不仅能跑，还能治理

## 测试

### 语法校验

```powershell
.\.venv\Scripts\python.exe -m compileall app tests
```

### 跑测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 常见问题

### 1. 为什么我会看到 DeepSeek / LiteLLM 的网络告警？

如果你在受限网络环境、未配置可用 key、或当前环境禁止外网访问，而又把：

```env
MOCK_LLM_RESPONSES=false
```

那么系统尝试访问真实 provider 时会出现 LiteLLM 告警。

这时你有两个选择：

- 回到 mock 模式：

```env
MOCK_LLM_RESPONSES=true
```

- 或者补齐真实 provider 配置和网络权限

### 2. 如果 SQLite schema mismatch 了怎么办？

删除本地数据库文件后重新启动：

```text
agent_router.db
```

项目会在启动时自动重新建表。

### 3. `/classification` 会不会真的调用模型？

会。

现在它调用的是“决策模型”，用于路由判断；但它不会生成最终回答。

### 4. 治理接口会不会自动改线上策略？

不会。

当前治理相关接口只做：

- 统计
- 分析
- 推荐
- 审核
- 模拟
- rollout 规划
- readiness 评估
- 报告导出

真实 live routing 仍然由当前 `router.py` 逻辑控制。

## 当前适合做什么

这套系统现在最适合：

- 做 AI 路由平台 Demo
- 做多模型编排展示
- 做治理闭环展示
- 做后续“自训练决策模型”的承载底座

如果你后面想把实时决策从 `DeepSeek` 切换成你自己训练的专用决策模型，最合适的接入位置就是：

```text
FeatureExtractor -> TaskClassifier -> AIDecisionEngine -> RuleRouter fallback
```

也就是替换或增强当前 `AIDecisionEngine`。
#   n e w - p r o j e c t  
 