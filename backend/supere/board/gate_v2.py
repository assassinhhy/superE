"""v2 backtest gate. Rewritten by scripts/board_backtest_v2.py."""

GATE = {
  "passed": False,
  "eligible": False,
  "failed": [
    "样本内或样本外期望不大于 0",
    "总体期望 -0.438R < 0.1R",
    "总样本 58 < 100",
    "命中率 37.9% < 45%",
    "平均盈亏比 0.70 < 1.3",
    "最大回撤 23.61R > 15R"
  ],
  "label": "未通过回测，仅供参考",
  "selected": {
    "b": 0.35,
    "t1_min_r": 1.2,
    "time_stop_bars": 8
  },
  "by_kind": {
    "顺势": {
      "expectancy": -0.11828857147927341,
      "counted": 15
    },
    "区间": {
      "expectancy": -0.3148812250760358,
      "counted": 2
    },
    "破位": {
      "expectancy": -0.5680770634454564,
      "counted": 39
    },
    "逆势": {
      "expectancy": -0.4260535167061084,
      "counted": 2
    }
  },
  "by_session": {
    "亚洲": {
      "expectancy": -0.14517877629381792,
      "counted": 19
    },
    "欧洲": {
      "expectancy": -0.6828369952734414,
      "counted": 18
    },
    "美国": {
      "expectancy": -0.4934162316962715,
      "counted": 21
    }
  },
  "disabled_sessions": [
    "亚洲",
    "欧洲",
    "美国"
  ],
  "in_sample": {
    "counted": 34,
    "hits": 14,
    "hit_rate": 0.4117647058823529,
    "expectancy": -0.37771035793943447,
    "payoff": 0.6762719217106326,
    "max_loss_streak": 8,
    "max_drawdown_r": 15.801710400047494,
    "avg_hold_min": 57.647042647058825,
    "time_stop_pct": 0.029411764705882353
  },
  "out_of_sample": {
    "counted": 24,
    "hits": 8,
    "hit_rate": 0.3333333333333333,
    "expectancy": -0.523710473341059,
    "payoff": 0.7530521450353737,
    "max_loss_streak": 6,
    "max_drawdown_r": 9.787093766032923,
    "avg_hold_min": 60.41665208333333,
    "time_stop_pct": 0.0
  },
  "overall": {
    "counted": 58,
    "hits": 22,
    "hit_rate": 0.3793103448275862,
    "expectancy": -0.43812419879527914,
    "payoff": 0.7029060458742366,
    "max_loss_streak": 8,
    "max_drawdown_r": 23.613553185391044,
    "avg_hold_min": 58.793087931034485,
    "time_stop_pct": 0.017241379310344827
  }
}
