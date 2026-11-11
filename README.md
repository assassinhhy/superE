# superE

ETHUSDT 永续合约的行情采集、存储和只读查询。这一阶段是数据层：看行情、把数据落地，供以后做分析和预测。程序不下单，也不读取币安 API key。

界面（Vue3 + KLineChart、pywebview）属于下一阶段，这里还没有。

## 环境

- Python 3.13（本仓库用 `uv` 锁定依赖，见 `uv.lock`）
- Windows 上建议使用已有的 `D:\dev\python\3.13.15`
- 出网代理默认 `http://127.0.0.1:7890`，可在配置里改
- 本机回环不走代理：进程会设置 `NO_PROXY=127.0.0.1,localhost`

开发机如果访问不了 `fapi.binance.com`（常见是 HTTP 451），单元测试仍然可以跑，它们用录制样例和 mock。真实连通用下面的冒烟脚本，在能走代理的 Windows 上跑。

## 安装

在仓库根目录。这台开发机上的 `uv` 不在 PATH 里，用 Python 模块方式调用：

```bat
python -m uv sync --all-groups
```

已经把 `uv` 加进 PATH 的环境也可以直接写 `uv sync --all-groups`。没有 uv 时，用 Python 3.13 创建虚拟环境后执行 `pip install -e ".[dev]"` 也可以，但以 `uv.lock` 为准。

如果 `git clone` 或 `git fetch` 走到了 `ghfast.top` 而不是 `github.com`，那是本机全局 git 的 `insteadOf` 改写，不是这个仓库的配置。需要直连 GitHub 时在本机改 `git config --global --get-regexp url`，不要把这条改写写进仓库。

复制一份本机配置（可选，已被 git 忽略）：

```bat
copy config.toml config.local.toml
```

`config.local.toml` 里可以改数据目录，例如 `dir = "D:\\Tool\\cg\\superE\\data"`。不要把密钥写进任何 toml。以后用到的 DeepSeek 密钥只读环境变量 `DEEPSEEK_API_KEY`。

## 启动和停止

Windows 在仓库根目录双击，或在命令行执行：

```bat
启动superE.bat
停止superE.bat
```

启动脚本调用 `python -m supere detach`。这个命令用 `DETACHED_PROCESS` 拉起服务，并且只把 stdin/stdout/stderr 交给子进程（`close_fds`），所以从 PowerShell 执行 `cmd /c 启动superE.bat`，哪怕套在管道里，也会在服务还活着的时候返回。进程自己的 print 写到 `data\logs\ingestor.stdout.log` 和 `data\logs\api.stdout.log`。程序日志是按进程分开的 `data\logs\supere-ingestor-YYYY-MM-DD.log` 和 `supere-api-YYYY-MM-DD.log`，避免两个进程同时追加把汉字切断。

在本机检查启动会不会卡住（会真正启动再停止）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_launch_returns.ps1
```

停止脚本不再用 `timeout /t`（非交互时会立刻失败），而是最多等 30 秒让采集器先把暂存写成 Parquet 并做完 checkpoint，超时才按 PID 结束。关闭时日志里会有「采集器开始关闭」和「采集器已关闭」。

启动脚本会拉起两个进程：

- 采集器：唯一可以写 SQLite 的进程。第二个实例拿不到 `data\superE.lock` 会直接退出。
- 只读 API：`http://127.0.0.1:8766`（8765 留给别的项目）。

停止脚本先请求 `POST /api/shutdown`。请求失败时，按 `data\run\ingestor.pid` 和 `data\run\api.pid` 结束进程。

等价命令：

```bat
.venv\Scripts\python.exe -m supere ingestor
.venv\Scripts\python.exe -m supere api
```

API 和采集器之间用本机 WebSocket 传未收盘 K 线和盘口，默认 `127.0.0.1:8767`。训练和界面都不要直接写数据库。

## 历史数据

空库启动时，采集器只补每个周期最近一页，并在 `app_events` 里写 `BACKFILL_REQUIRED`。全量历史要单独跑，它会占用文件锁，所以先停掉采集器：

```bat
.venv\Scripts\python.exe -m supere backfill
```

回填内容：

- 合约 ETHUSDT 的 15 个周期 K 线。`1m` 到 `12h`、`1d`、`3d` 从 data.binance.vision 的月度和日度 zip 导入，并校验 SHA-256。`1w` 和 `1M` 数据量小，直接用 REST。
- 标记价、指数价、溢价指数 K 线，同样的规则。
- 资金费率月度存档，再加 REST 尾部。
- 2021-12-01 起的 metrics 日度存档（持仓量和多空比）。文件内行会先按时间排序。
- 现货只拉 1m，用来做基差。现货 REST 失败时改走 `data-api.binance.vision`。收到 HTTP 418 时不换域名，避免绕过封禁。

合约逐笔的全历史 zip 很大，默认不下载。实时流会自己录；超过 2 小时的缺口优先用已经发布的日度存档，当天的缺口用 REST。

## 测试

```bat
uv run pytest
```

这里不访问币安。覆盖限速、去重、缺口、UTC+8 显示、存档时间戳单位、K 线聚合、重叠切换、文件锁、Parquet、DuckDB 只读、只读 API，以及一条用假 WebSocket 跑通的采集器。

### 在 Windows 上做真实连通冒烟

代理开着，在仓库根目录：

```bat
.venv\Scripts\python.exe scripts\smoke_live.py
```

它会请求合约时间、两根合约 K 线、现货备用域名，并各听几秒 `/market` 和 `/public`。每行输出 `PASS` 或 `FAIL`。它不会发下单请求。

### 验收

```bat
.venv\Scripts\python.exe scripts\accept_1a.py
.venv\Scripts\python.exe scripts\accept_1a.py --data-dir data
```

不带 `--data-dir` 时，用合成数据检查缺口检测、聚合、重复写入和文件锁。带上数据目录后，再检查实库的连续性、500 根读取耗时、418 次数、自计权重、时钟事件和重叠切换。24 小时连续运行只能在本机做，脚本会把还没跑到的项标成 `SKIP`。

断网演练不要去关系统代理。先把 `config.toml` 里的 `proxy.url` 改成 `http://127.0.0.1:7891`，重启采集器，然后：

```bat
.venv\Scripts\python.exe scripts\accept_1a.py --live-outage --pause-seconds 300
```

这个命令在 7891 上开一个转发，把字节转到 `127.0.0.1:7890`，暂停 5 分钟再恢复。恢复后 60 秒内应看到重连和 `GAP_FILL`。然后再跑一次 `--data-dir`。

把 `ingest.reconnect_hours` 临时改成 `0.05` 可以在几分钟内演练 23 小时那种重叠切换，不用真的等 23 小时。演练完改回 `23`。

备份和恢复演练：

```bat
.venv\Scripts\python.exe -m supere backup
.venv\Scripts\python.exe -m supere restore-drill
```

采集器在每天 UTC+8 的 03:00 也会自己备份一次。恢复演练只复制最近的备份做 `integrity_check`，不会替换正在用的库。

## 目录

```
backend/supere/     采集器、API、存储、消息面骨架
config.toml         可提交的默认配置
data/               运行后生成，不进 git
docs/架构.md
docs/进度.md
scripts/            冒烟和验收
启动superE.bat
停止superE.bat
```

数据目录里：

- `superE.sqlite`：K 线、资金费率、metrics、覆盖表、事件
- `raw/`：按 UTC+8 日期分区的 zstd Parquet（盘口、逐笔、强平）
- `archive_cache/`：下载的 zip，可以删
- `backups/`：SQLite 备份和 Parquet 增量副本
- `logs/`：按进程、按 UTC+8 日期滚动（`supere-ingestor-*.log`、`supere-api-*.log`），密钥样式文本会打码

## 时间

数据库里的时间是毫秒时间戳。日志、API 里的 `*_utc8` 字段和分区目录名用 UTC+8。K 线仍按币安的 UTC 边界切，不会改成 UTC+8 的自然日。
