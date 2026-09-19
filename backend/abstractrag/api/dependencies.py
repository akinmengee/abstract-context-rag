"""FastAPI dependencies. The API only ever reaches the engine through this."""

from typing import Annotated

from fastapi import Depends

from abstractrag.core.container import get_engine
from abstractrag.rag.engine import RagEngine

EngineDep = Annotated[RagEngine, Depends(get_engine)]
