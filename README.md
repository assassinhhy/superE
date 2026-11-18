# superE

> 面向 ETHUSDT 永续合约的本地行情采集、结构化存储、只读看盘与策略研究桌面工具。  
> **核心原则：仅消费公开行情数据，不读取交易所 API Key，永不下单。**

---

## 核心特性

- **进程隔离与只读安全**
  - 采集器（`ingestor`）是唯一写入主库 `superE.sqlite` 的进程，通过文件锁防双实例。
  - 只读 API（`127.0.0.1:8766`）、盘面分析（`127.0.0.1:8770`）、新闻服务（`127.0.0.1:8769`）与桌面窗口（`pywebview`）均以只读方式访问数据，互不阻塞，支持独立启停。
  - 前端构建产物由本机 API 直接托管，无外部 CDN 依赖，断网也可使用预览模式（`supere preview`）查看界面。
- **高可靠行情采集与归档回填**
  - 支持合约 15 个周期 K 线、标记价/指数价/溢价指数 K 线、资金费率及持仓量/多空比指标（metrics）。
  - 支持从 `data.binance.vision` 拉取月度/日度归档并做 SHA-256 校验，断线重连后自动触发 `GAP_FILL`。
  - 盘口、逐笔成交、强平数据按 UTC+8 日期分区写入 zstd 压缩 Parquet，支持 DuckDB 只读查询。
  - 每日 UTC+8 03:00 自动执行 SQLite 在线备份与 Parquet 增量复制，并提供完整性恢复演练（`restore-drill`）。
- **多模块看盘桌面端（Vue 3 + KLineChart）**
  - **行情页**：15 周期合约 K 线 + 现货 1m 基差参考，支持 MA20/MA50/BOLL、RSI/MACD/ATR 与本地持久化画线工具，统一 UTC+8 时间轴。
  - **盘面页**：基于固定进场价与失效时刻生成情景计划，记录于独立 `plan_journal.sqlite`，自动用 1m K 线回补成交/T1/T2/止损/MFE/MAE 与扣费后 R 倍数，支持 CSV 导出与历史回测。
  - **消息页**：定时抓取加密新闻源，支持基于内容哈希的评分复用与规则兜底，提供点击时刻无泄漏消息面聚合。
  - **研究与训练页**：提供数据覆盖校验、1m 聚合校验及多周期分位数研究/推理框架（`supere.train`）。

---

## 系统架构

```text
Binance Public WS / REST / data.binance.vision
                    │
                    ▼
        ┌───────────────────────┐
        │   ingestor (唯一写库)   │
        └─────┬───────────┬─────┘
              │           │ 本机 WS (:8767)
              ▼           ▼
    superE.sqlite + raw/*.parquet (zstd)
      │           │           │
      ▼           ▼           ▼
  API (:8766)  Board (:8770)  News (:8769)
      │           │           │
      └───────────┼───────────┘
                  ▼
     Vue 3 + KLineChart (pywebview 桌面窗口)
```

更多设计细节见 [`docs/架构.md`](docs/架构.md)。

---

## 环境要求

- **Python**：`3.13`（使用 `uv` 与 `uv.lock` 管理依赖）
- **Node.js**：`18+`（推荐使用 `pnpm`，也兼容 `npm`）
- **系统**：Windows 10/11（桌面端基于系统 WebView2）；后端服务与测试支持跨平台运行

---

## 快速开始

### 1. 安装后端依赖

```bash
uv sync --all-groups
# 若 uv 未加入 PATH，可使用：python -m uv sync --all-groups
```

### 2. 构建前端静态资源

```bash
cd frontend
pnpm install
pnpm build
cd ..
```

### 3. 可选：本机配置覆盖

复制默认配置为本机配置（`*.local.toml` 已被 `.gitignore` 忽略）：

```bash
copy config.toml config.local.toml
copy analysis.toml analysis.local.toml
```

- 网络代理可在 `config.local.toml` 的 `[proxy]` 下配置，或通过环境变量 `HTTPS_PROXY` / `HTTP_PROXY` 覆盖（程序会自动设置 `NO_PROXY=127.0.0.1,localhost`）。
- 消息面 LLM 摘要密钥仅从环境变量读取（`SUPERE_LLM_BASE_URL`、`SUPERE_LLM_API_KEY`、`SUPERE_LLM_MODEL`），不写入任何配置文件；未配置时自动使用规则兜底。

### 4. 启动与停止

**Windows 一键启停：**

```bat
启动superE.bat
停止superE.bat
```

**CLI 常用命令：**

```bash
# 查看各后台服务进程与端口状态
.venv\Scripts\python.exe -m supere status

# 停止全部后台服务（或指定 --role）
.venv\Scripts\python.exe -m supere stop

# 分别在前台运行各模块
.venv\Scripts\python.exe -m supere ingestor
.venv\Scripts\python.exe -m supere api
.venv\Scripts\python.exe -m supere board
.venv\Scripts\python.exe -m supere news
.venv\Scripts\python.exe -m supere desktop

# 无外网/无币安连接时，使用合成数据预览界面
.venv\Scripts\python.exe -m supere preview --no-window

# 历史全量回填（需先停止采集器释放文件锁）
.venv\Scripts\python.exe -m supere backfill

# 手动备份与恢复演练
.venv\Scripts\python.exe -m supere backup
.venv\Scripts\python.exe -m supere restore-drill
```

---

## 测试与验证

单元测试全部基于录制样例与 Mock，不访问外网：

```bash
# 后端全量测试
.venv\Scripts\pytest.exe

# 前端测试
cd frontend && pnpm test
```

实网连通性与验收脚本（需在可访问公开行情的代理环境下运行）：

```bash
# 实网连通冒烟
.venv\Scripts\python.exe scripts\smoke_live.py

# 数据层与采集链路验收
.venv\Scripts\python.exe scriptsccept_1a.py

# 盘面策略历史回测
.venv\Scripts\python.exe scriptsoard_backtest.py --compare
```

---

## 目录结构

```text
backend/supere/     核心后端（采集、存储、只读 API、盘面、新闻、训练研究、桌面壳）
frontend/           Vue 3 + KLineChart 前端工程
scripts/            实网冒烟、验收与回测脚本
tests/              单元测试与回放夹具
docs/               架构说明、阶段进度与工程维护记录
config.toml         默认基础配置
analysis.toml       默认盘面分析参数
CHANGELOG.md        版本更新记录
启动superE.bat       Windows 一键启动脚本
停止superE.bat       Windows 一键停止脚本
```

---

## 文档索引

- [`CHANGELOG.md`](CHANGELOG.md) — 更新日志
- [`docs/架构.md`](docs/架构.md) — 系统架构、数据流与模块边界说明
- [`docs/进度.md`](docs/进度.md) — 各阶段实现清单与里程碑记录
- [`docs/内核修复与工程优化记录.md`](docs/内核修复与工程优化记录.md) — 内核缺陷修复与工程优化记录
- [`frontend/NOTICE`](frontend/NOTICE) — 前端第三方图表库开源声明（KLineChart / TradingView Lightweight Charts）

---

## 免责声明

本项目仅用于公开行情数据采集、可视化看盘与量化研究回测，**不构成任何投资建议**，也不包含任何实盘下单功能。
