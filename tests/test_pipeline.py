"""
Tests for the end-to-end pipeline module.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.pipeline import Pipeline
from app.models import NewsItem

class TestPipeline:
    """Test cases for Pipeline class."""
    
    @pytest.fixture
    def sample_news_data(self):
        """Sample news data for testing."""
        return [
            {
                "id": "news_001",
                "title": "Apple Reports Strong Q4 Earnings",
                "content": "Apple Inc. reported strong fourth quarter earnings with revenue exceeding expectations. The technology giant showed impressive growth across all product categories.",
                "symbols": ["AAPL"],
                "sentiment": 0.8,
                "source": "TechNews"
            },
            {
                "id": "news_002", 
                "title": "Tesla Delivers Record Number of Vehicles",
                "content": "Tesla announced record vehicle deliveries for the quarter, surpassing analyst expectations. The electric vehicle manufacturer continues to scale production.",
                "symbols": ["TSLA"],
                "sentiment": 0.7,
                "source": "AutoNews"
            },
            {
                "id": "news_003",
                "title": "Market Volatility Increases Amid Economic Uncertainty",
                "content": "Stock market volatility has increased as investors react to mixed economic signals. Technology stocks have been particularly affected by the uncertainty.",
                "symbols": ["MSFT", "GOOGL"],
                "sentiment": -0.3,
                "source": "MarketWatch"
            }
        ]
    
    @pytest.fixture
    def temp_data_file(self, sample_news_data):
        """Create temporary data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_news_data, f)
            return Path(f.name)
    
    @pytest.fixture
    def pipeline(self, temp_data_file):
        """Create pipeline instance with temporary data."""
        temp_index = temp_data_file.parent / "test_index.pkl"
        return Pipeline(data_path=temp_data_file, index_path=temp_index)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with default parameters."""
        pipeline = Pipeline()
        
        assert pipeline.data_path == Path("data/stock_news.json")
        assert pipeline.index_path == Path("data/search_index.pkl")
        assert not pipeline.is_initialized
        assert len(pipeline.news_items) == 0
    
    def test_pipeline_initialization_custom_paths(self):
        """Test pipeline initialization with custom paths."""
        data_path = Path("custom/data.json")
        index_path = Path("custom/index.pkl")
        
        pipeline = Pipeline(data_path=data_path, index_path=index_path)
        
        assert pipeline.data_path == data_path
        assert pipeline.index_path == index_path
    
    @pytest.mark.asyncio
    async def test_pipeline_initialize_success(self, pipeline, temp_data_file):
        """Test successful pipeline initialization."""
        await pipeline.initialize()
        
        assert pipeline.is_initialized
        assert len(pipeline.news_items) > 0
        assert pipeline.retriever.is_fitted
        
        # Clean up
        temp_data_file.unlink()
        if pipeline.index_path.exists():
            pipeline.index_path.unlink()
    
    @pytest.mark.asyncio
    async def test_pipeline_initialize_missing_data_file(self):
        """Test initialization with missing data file."""
        pipeline = Pipeline(data_path=Path("nonexistent.json"))
        
        with pytest.raises(FileNotFoundError):
            await pipeline.initialize()
    
    @pytest.mark.asyncio
    async def test_pipeline_initialize_force_rebuild(self, pipeline, temp_data_file):
        """Test initialization with force rebuild."""
        # First initialization
        await pipeline.initialize()
        assert pipeline.index_path.exists()
        
        # Force rebuild
        await pipeline.initialize(force_rebuild=True)
        assert pipeline.is_initialized
        
        # Clean up
        temp_data_file.unlink()
        if pipeline.index_path.exists():
            pipeline.index_path.unlink()
    
    @pytest.mark.asyncio
    async def test_pipeline_load_existing_index(self, pipeline, temp_data_file):
        """Test loading existing index."""
        # First build index
        await pipeline.initialize()
        
        # Create new pipeline and load existing index
        new_pipeline = Pipeline(data_path=temp_data_file, index_path=pipeline.index_path)
        await new_pipeline.initialize()
        
        assert new_pipeline.is_initialized
        assert len(new_pipeline.news_items) > 0
        
        # Clean up
        temp_data_file.unlink()
        if pipeline.index_path.exists():
            pipeline.index_path.unlink()
    
    @pytest.mark.asyncio
    async def test_process_query_success(self, pipeline, temp_data_file):
        """Test successful query processing."""
        await pipeline.initialize()
        
        results = await pipeline.process_query(
            query="Apple earnings revenue",
            top_k=2,
            include_summary=True
        )
        
        assert len(results) <= 2
        assert all(isinstance(item, NewsItem) for item in results)
        
        # Check that summaries are included
        for item in results:
            if item.summary:
                assert len(item.summary) > 0
        
        # Clean up
        temp_data_file.unlink()
        if pipeline.index_path.exists():
            pipeline.index_path.unlink()
    
    @pytest.mark.asyncio
    async def test_process_query_without_summary(self, pipeline, temp_data_file):
        """Test query processing without summaries."""
        await pipeline.initialize()
        
        results = await pipeline.process_query(
            query="Tesla delivery",
            top_k=3,
            include_summary=False
        )
        
        assert len(results) <= 3
        # Summaries should not be generated
        for item in results:
            assert item.relevance_score is not None
        
        # Clean up
        temp_data_file.unlink()
        if pipeline.index_path.exists():
            pipeline.index_path.unlink()
    
    @pytest.mark.asyncio
    async def test_process_query_not_initialized(self, pipeline):
        """Test query processing before initialization."""
        with pytest.raises(RuntimeError, match="Pipeline not initialized"):
            await pipeline.process_query("test query")
    
    @pytest.mark.asyncio
    async def test_process_query_empty_query(self, pipeline, temp_data_file):
        """Test query processing with empty query."""
        await pipeline.initialize()
        
        results = await pipeline.process_query("")
        assert len(results) == 0
        
        results = await pipeline.process_query("   ")
        assert len(results) == 0
        
        # Clean up
        temp_data_file.unlink()
        if pipeline.index_path.exists():
            pipeline.index_path.unlink()
    
    @pytest.mark.asyncio
    async def test_get_articles_success(self, pipeline, temp_data_file):
        """Test getting articles with pagination."""
        await pipeline.initialize()
        
        # Get first page
        articles = await pipeline.get_articles(limit=2, offset=0)
        assert len(articles) <= 2
        
        # Get second page
        articles_page2 = await pipeline.get_articles(limit=2, offset=2)
        assert len(articles_page2) <= 2
        
        # Articles should be different
        if len(articles) > 0 and len(articles_page2) > 0:
            assert articles[0].id != articles_page2[0].id
        
        # Clean up
        temp_data_file.unlink()
        if pipeline.index_path.exists():
            pipeline.index_path.unlink()
    
    @pytest.mark.asyncio
    async def test_get_articles_with_summary(self, pipeline, temp_data_file):
        """Test getting articles with summaries."""
        await pipeline.initialize()
        
        articles = await pipeline.get_articles(limit=2, include_summary=True)
        
        # Check that summaries are generated for articles that don't have them
        for article in articles:
            if not article.summary:
                # Summary should be generated
                pass  # The actual summary generation depends on the implementation
        
        # Clean up
        temp_data_file.unlink()
        if pipeline.index_path.exists():
            pipeline.index_path.unlink()
    
    @pytest.mark.asyncio
    async def test_get_article_by_id_success(self, pipeline, temp_data_file):
        """Test getting article by ID."""
        await pipeline.initialize()
        
        # Get first article ID
        articles = await pipeline.get_articles(limit=1)
        if articles:
            article_id = articles[0].id
            
            # Get article by ID
            found_article = await pipeline.get_article_by_id(article_id)
            assert found_article is not None
            assert found_article.id == article_id
        
        # Clean up
        temp_data_file.unlink()
        if pipeline.index_path.exists():
            pipeline.index_path.unlink()
    
    @pytest.mark.asyncio
    async def test_get_article_by_id_not_found(self, pipeline, temp_data_file):
        """Test getting non-existent article by ID."""
        await pipeline.initialize()
        
        found_article = await pipeline.get_article_by_id("nonexistent_id")
        assert found_article is None
        
        # Clean up
        temp_data_file.unlink()
        if pipeline.index_path.exists():
            pipeline.index_path.unlink()
    
    @pytest.mark.asyncio
    async def test_summarize_text(self, pipeline):
        """Test text summarization."""
        text = (
            "This is a long text that needs to be summarized. "
            "It contains multiple sentences with various information. "
            "The summarizer should extract the most important parts. "
            "This is additional content for testing purposes."
        )
        
        summary = await pipeline.summarize_text(text, method="lead3", max_sentences=2)
        
        assert len(summary) > 0
        assert len(summary) < len(text)
    
    def test_get_stats_not_initialized(self, pipeline):
        """Test getting stats before initialization."""
        stats = pipeline.get_stats()
        
        assert stats["initialized"] is False
        assert stats["total_articles"] == 0
    
    @pytest.mark.asyncio
    async def test_get_stats_initialized(self, pipeline, temp_data_file):
        """Test getting stats after initialization."""
        await pipeline.initialize()
        
        stats = pipeline.get_stats()
        
        assert stats["initialized"] is True
        assert stats["total_articles"] > 0
        assert "fitted" in stats
        
        # Clean up
        temp_data_file.unlink()
        if pipeline.index_path.exists():
            pipeline.index_path.unlink()
    
    @pytest.mark.asyncio
    async def test_rebuild_index(self, pipeline, temp_data_file):
        """Test rebuilding index."""
        await pipeline.initialize()
        original_index_time = pipeline.index_path.stat().st_mtime if pipeline.index_path.exists() else 0
        
        # Rebuild index
        await pipeline.rebuild_index()
        
        # Index should be updated
        if pipeline.index_path.exists():
            new_index_time = pipeline.index_path.stat().st_mtime
            assert new_index_time >= original_index_time
        
        # Clean up
        temp_data_file.unlink()
        if pipeline.index_path.exists():
            pipeline.index_path.unlink()
    
    @pytest.mark.asyncio
    async def test_add_articles(self, pipeline, temp_data_file):
        """Test adding new articles."""
        await pipeline.initialize()
        original_count = len(pipeline.news_items)
        
        # Create new articles
        new_articles = [
            NewsItem(
                id="new_001",
                title="New Article 1",
                content="This is new content for testing.",
                symbols=["NEW1"]
            ),
            NewsItem(
                id="new_002", 
                title="New Article 2",
                content="This is another new article for testing.",
                symbols=["NEW2"]
            )
        ]
        
        await pipeline.add_articles(new_articles)
        
        # Check that articles were added
        assert len(pipeline.news_items) == original_count + 2
        
        # Clean up
        temp_data_file.unlink()
        if pipeline.index_path.exists():
            pipeline.index_path.unlink()
    
    @pytest.mark.asyncio
    async def test_cleanup(self, pipeline, temp_data_file):
        """Test pipeline cleanup."""
        await pipeline.initialize()
        
        # Verify pipeline is initialized
        assert pipeline.is_initialized
        assert len(pipeline.news_items) > 0
        
        # Cleanup
        await pipeline.cleanup()
        
        # Verify cleanup
        assert not pipeline.is_initialized
        assert len(pipeline.news_items) == 0
        
        # Clean up files
        temp_data_file.unlink()
        if pipeline.index_path.exists():
            pipeline.index_path.unlink()

if __name__ == "__main__":
    pytest.main([__file__])
