"""SQLite schema for the single market database.

``coverage.range_from`` / ``range_to`` are the plan's ``from`` / ``to`` columns.
Those names are reserved words, so they are stored under explicit names.
``ingested_local`` / ``ts_local`` keep the uncorrected machine clock beside the
calibrated timestamp (plan section 5.5).
"""

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS klines (
  market TEXT NOT NULL,
  symbol TEXT NOT NULL,
  interval TEXT NOT NULL,
  open_time INTEGER NOT NULL,
  close_time INTEGER NOT NULL,
  o TEXT NOT NULL,
  h TEXT NOT NULL,
  l TEXT NOT NULL,
  c TEXT NOT NULL,
  volume TEXT NOT NULL,
  quote_volume TEXT NOT NULL,
  trades INTEGER NOT NULL,
  taker_buy_base TEXT NOT NULL,
  taker_buy_quote TEXT NOT NULL,
  source TEXT NOT NULL,
  ingested_at INTEGER NOT NULL,
  ingested_local INTEGER NOT NULL,
  PRIMARY KEY (market, symbol, interval, open_time)
);

CREATE TABLE IF NOT EXISTS um_metrics (
  symbol TEXT NOT NULL,
  ts INTEGER NOT NULL,
  oi TEXT,
  oi_value TEXT,
  top_pos_lsr TEXT,
  top_acc_lsr TEXT,
  global_lsr TEXT,
  taker_lsr TEXT,
  source TEXT,
  PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS funding (
  symbol TEXT NOT NULL,
  funding_time INTEGER NOT NULL,
  rate TEXT NOT NULL,
  mark_price TEXT,
  source TEXT NOT NULL DEFAULT 'rest',
  PRIMARY KEY (symbol, funding_time)
);

CREATE TABLE IF NOT EXISTS funding_live (
  symbol TEXT PRIMARY KEY,
  rate TEXT,
  next_funding_time INTEGER,
  mark_price TEXT,
  index_price TEXT,
  updated_at INTEGER NOT NULL,
  updated_local INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS coverage (
  market TEXT NOT NULL,
  symbol TEXT NOT NULL,
  interval TEXT NOT NULL,
  range_from INTEGER,
  range_to INTEGER,
  gaps_json TEXT NOT NULL DEFAULT '[]',
  last_check INTEGER,
  PRIMARY KEY (market, symbol, interval)
);

CREATE TABLE IF NOT EXISTS raw_coverage (
  stream TEXT NOT NULL,
  date TEXT NOT NULL,
  first_ts INTEGER,
  last_ts INTEGER,
  msg_count INTEGER NOT NULL DEFAULT 0,
  gap_count INTEGER NOT NULL DEFAULT 0,
  gaps_json TEXT NOT NULL DEFAULT '[]',
  PRIMARY KEY (stream, date)
);

CREATE TABLE IF NOT EXISTS stream_state (
  stream TEXT PRIMARY KEY,
  last_id INTEGER,
  last_ts INTEGER,
  updated_at INTEGER
);

CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id TEXT NOT NULL,
  interval TEXT NOT NULL,
  bar_open_time INTEGER NOT NULL,
  p_up REAL,
  p_flat REAL,
  p_down REAL,
  decision TEXT,
  created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS outcomes (
  signal_id INTEGER PRIMARY KEY,
  label TEXT,
  ret REAL,
  filled_at INTEGER,
  detail_json TEXT
);

CREATE TABLE IF NOT EXISTS models (
  model_id TEXT PRIMARY KEY,
  interval TEXT NOT NULL,
  status TEXT NOT NULL,
  feature_hash TEXT,
  dataset_hash TEXT,
  metrics_json TEXT,
  created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  progress REAL,
  started_at INTEGER,
  finished_at INTEGER,
  log_path TEXT
);

CREATE TABLE IF NOT EXISTS app_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  ts_local INTEGER,
  level TEXT NOT NULL,
  component TEXT NOT NULL,
  code TEXT,
  msg TEXT NOT NULL,
  detail_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_app_events_ts ON app_events(ts);
CREATE INDEX IF NOT EXISTS idx_app_events_code ON app_events(code);

CREATE TABLE IF NOT EXISTS runtime_stats (
  ts INTEGER NOT NULL,
  venue TEXT NOT NULL,
  self_weight_minute INTEGER NOT NULL,
  self_weight_peak INTEGER NOT NULL,
  header_weight INTEGER,
  header_limit INTEGER,
  halted INTEGER NOT NULL,
  PRIMARY KEY (ts, venue)
);

CREATE TABLE IF NOT EXISTS news (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_published INTEGER,
  ts_ingested INTEGER NOT NULL,
  ts_local INTEGER NOT NULL,
  source TEXT NOT NULL,
  source_tier INTEGER NOT NULL,
  lang TEXT,
  title TEXT,
  body TEXT,
  url TEXT,
  url_hash TEXT,
  text_simhash TEXT,
  cluster_id INTEGER,
  category TEXT,
  direction INTEGER,
  impact INTEGER,
  eth_relevance REAL,
  credibility REAL,
  entities TEXT,
  scored_by TEXT,
  score_json TEXT,
  user_label INTEGER,
  status TEXT NOT NULL DEFAULT 'new'
);

CREATE INDEX IF NOT EXISTS idx_news_url_hash ON news(url_hash);
CREATE INDEX IF NOT EXISTS idx_news_ingested ON news(ts_ingested);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_event INTEGER NOT NULL,
  kind TEXT NOT NULL,
  title TEXT NOT NULL,
  importance INTEGER NOT NULL,
  source TEXT,
  url TEXT,
  pause_before_min INTEGER NOT NULL DEFAULT 30,
  pause_after_min INTEGER NOT NULL DEFAULT 60,
  actual TEXT,
  forecast TEXT,
  previous TEXT,
  confirmed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS news_reaction (
  news_id INTEGER NOT NULL,
  horizon TEXT NOT NULL,
  ts_ref INTEGER,
  px_ref TEXT,
  px_after TEXT,
  ret REAL,
  vol_ref TEXT,
  vol_after TEXT,
  vol_ratio REAL,
  max_up REAL,
  max_down REAL,
  PRIMARY KEY (news_id, horizon)
);

CREATE TABLE IF NOT EXISTS market_anomaly (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  kind TEXT NOT NULL,
  value REAL,
  threshold REAL,
  detail_json TEXT,
  linked_news_ids TEXT
);

CREATE INDEX IF NOT EXISTS idx_anomaly_ts ON market_anomaly(ts);

CREATE TABLE IF NOT EXISTS fear_greed (
  ts INTEGER PRIMARY KEY,
  value INTEGER NOT NULL,
  classification TEXT,
  ts_ingested INTEGER NOT NULL,
  ts_local INTEGER NOT NULL,
  known_at INTEGER NOT NULL
);
"""


def init_schema(conn) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.execute("PRAGMA user_version=1")
    conn.commit()
