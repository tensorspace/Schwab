"""
Pydantic models for the stock news analysis application.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class NewsItem(BaseModel):
    """Model for a single news item."""
    id: str = Field(..., description="Unique identifier for the news item")
    title: str = Field(..., description="Title of the news article")
    content: str = Field(..., description="Full content of the news article")
    url: Optional[str] = Field(None, description="URL of the original article")
    published_date: Optional[datetime] = Field(None, description="Publication date")
    source: Optional[str] = Field(None, description="News source")
    symbols: Optional[List[str]] = Field(default_factory=list, description="Related stock symbols")
    sentiment: Optional[float] = Field(None, description="Sentiment score (-1 to 1)")
    summary: Optional[str] = Field(None, description="Generated summary")
    relevance_score: Optional[float] = Field(None, description="Relevance score for query")

class QueryRequest(BaseModel):
    """Model for query requests."""
    query: str = Field(..., description="Search query", min_length=1)
    top_k: int = Field(default=10, description="Number of results to return", ge=1, le=100)
    include_summary: bool = Field(default=True, description="Whether to include summaries")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional filters")

class QueryResponse(BaseModel):
    """Model for query responses."""
    query: str = Field(..., description="Original query")
    results: List[NewsItem] = Field(..., description="List of matching news items")
    total_found: int = Field(..., description="Total number of results found")
    processing_time: Optional[float] = Field(None, description="Query processing time in seconds")

class IndexStats(BaseModel):
    """Model for index statistics."""
    total_documents: int = Field(..., description="Total number of documents in index")
    vocabulary_size: int = Field(..., description="Size of vocabulary")
    last_updated: datetime = Field(..., description="Last index update time")
    index_type: str = Field(..., description="Type of index (e.g., 'tfidf', 'bm25')")

class SummaryRequest(BaseModel):
    """Model for summary requests."""
    text: str = Field(..., description="Text to summarize", min_length=1)
    method: str = Field(default="lead3", description="Summarization method")
    max_sentences: int = Field(default=3, description="Maximum sentences in summary", ge=1, le=10)

class SummaryResponse(BaseModel):
    """Model for summary responses."""
    original_text: str = Field(..., description="Original text")
    summary: str = Field(..., description="Generated summary")
    method: str = Field(..., description="Summarization method used")
    compression_ratio: float = Field(..., description="Compression ratio (summary/original)")
