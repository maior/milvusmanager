from datetime import datetime
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field

class Reference(BaseModel):
    """참조 문서 모델"""
    text: str
    page: Optional[int] = None
    file_name: Optional[str] = None
    score: Optional[float] = None

class CitedReference(BaseModel):
    """인용된 참조 모델"""
    text: Optional[str] = None
    page: Optional[int] = None
    file_name: Optional[str] = None
    index: Optional[int] = None

class CacheEntry(BaseModel):
    query_id: str
    query_text: str
    query_vector: List[float]
    result_vector: List[float]
    email: str
    result_text: str
    references: Optional[List[Union[Dict, Reference]]] = Field(default_factory=list)
    pdf_names: Optional[List[str]] = Field(default_factory=list)
    cited_refs: Optional[List[Union[Dict, CitedReference, int]]] = Field(default_factory=list)  # int도 허용
    pdf_info: Optional[Dict] = Field(default_factory=dict)
    created_at: datetime
    last_accessed: datetime
    hit_count: int
    relevance_score: float
    project_id: str
    cache_management: Optional[Dict] = Field(default_factory=dict)
    
class CacheSearchResult(BaseModel):
    found: bool
    entry: Optional[CacheEntry] = None
    similarity_score: Optional[float] = None

class CacheStats(BaseModel):
    total_entries: int
    hit_rate: float
    avg_response_time: float
    memory_usage: float