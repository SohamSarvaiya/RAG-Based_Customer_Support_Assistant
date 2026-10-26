class AppError(Exception):
    """Base error that maps cleanly to an HTTP response."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DocumentError(AppError):
    pass


class LLMError(AppError):
    def __init__(self, message: str = "The language model is unavailable. Try again."):
        super().__init__(message, 502)
