from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Question(BaseModel):
    id: str
    text: str
    slot: Optional[str] = None    # 이 질문이 채우는 슬롯

class Slots(BaseModel):
    must: List[str] = Field(default_factory=list)
    nice_to_have: List[str] = Field(default_factory=list)

class Element(BaseModel):
    id: str
    label: str
    required: bool = False
    slots: Slots = Field(default_factory=Slots)
    questions: List[Question]

class Offense(BaseModel):
    offense: str
    title_ko: str
    statute_ref: str
    elements: List[Element]
    templates: Dict[str, Any] = Field(default_factory=dict)
    includes: List[str] = Field(default_factory=list)
    party_info: List[Question] = Field(default_factory=list)
