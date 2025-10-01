from pydantic import BaseModel
from typing import Dict, List

class ElementState(BaseModel):
    status: str # satisfied | missing | unclear
    summary: str = ""

class ExtractionResult(BaseModel):
    elements: Dict[str, ElementState]