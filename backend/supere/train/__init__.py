"""Offline training skeleton (phase 2 of plan section 10).

Reads ``superE.sqlite`` read-only and produces labels, features, a LightGBM
model, out-of-sample metrics and a simple backtest. It never places orders and
never touches credentials. Run with ``python -m supere.train``.
"""
