"""Custom exceptions returned by the FastAPI global error handler."""


class AppError(Exception):
    """Base application error with an API code and HTTP status."""

    # Khoi tao loi ung dung voi code, message va HTTP status.
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class ConfigurationError(AppError):
    """Raised when required runtime configuration is missing or invalid."""

    # Bao loi cau hinh thieu hoac sai.
    def __init__(self, message: str) -> None:
        super().__init__("configuration_error", message, status_code=500)


class RetrievalError(AppError):
    """Raised when retrieval cannot complete because of storage/search failures."""

    # Bao loi khi retrieval/storage khong hoan tat duoc.
    def __init__(self, message: str) -> None:
        super().__init__("retrieval_error", message, status_code=500)
