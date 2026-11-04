"""Project exceptions. None of these represent a trading operation."""


class SuperEError(Exception):
    """Base error for the data layer."""


class RestHalted(SuperEError):
    """REST is stopped after an HTTP 418 ban response."""


class TradingDisabled(SuperEError):
    """Raised if code tries to call an order or account endpoint."""


class RateLimited(SuperEError):
    """Raised when retries after HTTP 429 are exhausted."""


class ChecksumError(SuperEError):
    """Archive zip does not match its SHA-256 checksum."""


class ConfigError(SuperEError):
    """Configuration is missing or invalid."""
