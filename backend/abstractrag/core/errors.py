"""Domain errors. The API layer maps these to HTTP status codes."""


class RagError(Exception):
    """Base class for every error raised by the engine."""


class SourceNotFoundError(RagError):
    """The requested arXiv ID or Wikipedia article does not exist."""


class FetchError(RagError):
    """The source exists but could not be downloaded."""


class ParseError(RagError):
    """The document was downloaded but could not be parsed into blocks."""


class InvalidInputError(RagError):
    """The ingest request is malformed (no source, or more than one source)."""


class LLMError(RagError):
    """The llama.cpp server is unreachable or returned an unusable response."""
