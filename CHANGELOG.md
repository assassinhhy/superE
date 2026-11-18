# Changelog

所有重要代码变更、内核修复与工程优化均记录在此文件。

## [Unreleased / 2026-11-18]

### Fixed
- **盘面回测（`board/journal.py`）**：消除成交当根 K 线同时触及 `entry` 与 `t1`/`t2` 导致的同柱乐观偏差（成交当根仅允许触发止损，目标位从下一根 K 线起检查），并清理不可达死代码。
- **预测服务（`train/predict.py`）**：为 `QuantilePredictService` 补齐模型目录 `mtime` 缓存失效检查，支持新模型热重载；将高周期 `higher` 查询合并到同一只读连接并限制时间窗口。
- **实时 K 线聚合（`ingest/kline_manager.py`）**：更新同一未收盘 K 线时保留 `last_agg_id` 水位，将 `on_agg_trade` 时间过滤改为严格小于 `kline_event_ms`，避免同毫秒成交被误丢。
- **新闻叠加排序（`news/overlay.py`）**：修正 `age_ms` 排序符号，确保同贡献度/紧急度下越新的消息排越前。
- **回放索引缓存（`board/replay.py`）**：移除全局 `_TIME_INDEX`，改为绑定 `fixture` 生命周期，避免内存地址复用命中脏索引。
- **盘面结论生成（`board/engine.py`）**：清理无意义的半成品空赋值。
- **存档补缺（`ingest/gapfill.py`）**：单日归档缺失时跳过当天继续补后续日期，不再直接中断整个补缺循环。
- **Windows 进程探测（`spawn.py`）**：修复中文 Windows 下 `netstat` 输出的 GBK 解码异常。

### Added
- **盘面绩效统计与复盘备注（`board/journal.py`、`board/serve.py`、`BoardPanel.vue`）**：新增 `GET /api/board/stats` 与 `POST /api/board/plans/{plan_id}/note`，支持按策略类型（顺势/破位/逆势）、状态（可进场）及跟单标记筛选查看成交率、胜率与累计/平均 R，并支持单条计划勾选跟单与记录复盘备注。
- **盘面点位上图表联动（`ChartPane.vue`、`boardOverlays.js`）**：选中/图看某条情景计划时，在 K 线主图上同步绘制带标签的固定进场（`entry`）、止损（`stop`）、`T1`、`T2` 水平参考线。
- 新增统一代理配置解析模块 `backend/supere/netproxy.py`（环境变量优先 > 显式参数 > 默认配置）。
- 新增 CLI 命令 `supere status` 与 `supere stop`，以及 Windows 一键停止脚本 `停止superE.bat`。
- 补充对应单元测试（`tests/test_cli_status_stop.py`、`tests/test_board_journal.py`、`frontend/src/board.test.mjs` 等）。

### Changed
- **盘面性能优化（`board/fetcher.py`、`board/serve.py`）**：消除 `_next_job` 每次轮询 3 次重复 `assemble` 读库；`/api/board` 轮询路径增加 60 秒结算节流；`board/serve.py` 接入 `resolve_proxy`。

### Docs
- 重构 `README.md` 为对外展示主页，脱敏文档中的本机绝对路径与内部笔记引用。
- 新增 `CHANGELOG.md`、`LICENSE`（MIT）与 `docs/内核修复与工程优化记录.md`。
