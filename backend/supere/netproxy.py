"""One place to resolve the outbound HTTP proxy.

Several clients used to hard-code ``http://127.0.0.1:7890``.  That ignored a
proxy configured by the desktop environment or a VPN client, and it made
"connect directly" impossible to express without editing code.  The order is:

1. an explicit empty string means "connect directly" (``None``);
2. the first non-empty ``HTTPS_PROXY`` / ``https_proxy`` / ``HTTP_PROXY`` /
   ``http_proxy``;
3. an explicitly supplied proxy (for example from ``config.toml``);
4. :data:`DEFAULT_PROXY`.

The environment wins over the explicit argument on purpose: callers normally
pass the value from ``config.toml``, while a shell override should still be
able to redirect every client at once.
"""

from __future__ import annotations

import os

DEFAULT_PROXY = "http://127.0.0.1:7890"
PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")


def resolve_proxy(explicit: str | None = None, *, default: str | None = DEFAULT_PROXY) -> str | None:
    """Return the proxy URL to use, or ``None`` for a direct connection."""
    if explicit == "":
        return None
    for name in PROXY_ENV:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    if explicit:
        return explicit
    return default
