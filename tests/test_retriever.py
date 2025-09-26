"""
Tests for the retrieval module.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.retriever import HybridRetriever, BM25
from app.models import NewsItem

class TestBM25:
    """Test cases for BM25 class."""
    
    @pytest.fixture
    def sample_corpus(self):
        """Sample corpus for testing."""
        return [
            "Apple reports strong quarterly earnings with revenue growth",
            "Tesla stock price increases after delivery numbers announcement",
            "Market volatility continues as investors watch economic indicators",
            "Technology stocks show mixed performance in trading session",
            "Apple and Tesla lead technology sector gains today"
        ]
    
    @pytest.fixture
    def bm25(self, sample_corpus):
        """Create fitted BM25 instance."""
        bm25 = BM25()
        bm25.fit(sample_corpus)
        return bm25
    
    def test_bm25_initialization(self):
        """Test BM25 initialization with default parameters."""
        bm25 = BM25()
        assert bm25.k1 == 1.2
        assert bm25.b == 0.75
        assert bm25.corpus_size == 0
    
    def test_bm25_initialization_custom_params(self):
        """Test BM25 initialization with custom parameters."""
        bm25 = BM25(k1=1.5, b=0.8)
        assert bm25.k1 == 1.5
        assert bm25.b == 0.8
    
    def test_bm25_fit(self, sample_corpus):
        """Test BM25 fitting on corpus."""
        bm25 = BM25()
        bm25.fit(sample_corpus)
        
        assert bm25.corpus_size == len(sample_corpus)
        assert len(bm25.doc_len) == len(sample_corpus)
        assert bm25.avgdl > 0
        assert len(bm25.idf) > 0
    
    def test_bm25_get_scores(self, bm25):
        """Test BM25 score calculation."""
        query = "Apple earnings revenue"
        scores = bm25.get_scores(query)
        
        assert len(scores) == bm25.corpus_size
        assert isinstance(scores, np.ndarray)
        assert scores[0] > 0  # First document should have positive score
    
    def test_bm25_empty_query(self, bm25):
        """Test BM25 with empty query."""
        scores = bm25.get_scores("")
        assert len(scores) == bm25.corpus_size
        assert all(score == 0 for score in scores)
    
    def test_bm25_unknown_terms(self, bm25):
        """Test BM25 with query containing unknown terms."""
        scores = bm25.get_scores("unknown terms not in corpus")
        assert len(scores) == bm25.corpus_size
        assert all(score == 0 for score in scores)

class TestHybridRetriever:
    """Test cases for HybridRetriever class."""
    
    @pytest.fixture
    def sample_news_items(self):
        """Sample news items for testing."""
        return [
            NewsItem(
                id="news_001",
                title="Apple Reports Strong Q4 Earnings",
                content="Apple Inc. reported strong fourth quarter earnings with revenue exceeding expectations. The company showed significant growth in iPhone sales.",
                symbols=["AAPL"],
                sentiment=0.8
            ),
            NewsItem(
                id="news_002",
                title="Tesla Stock Surges on Delivery Numbers",
                content="Tesla's stock price increased significantly following better than expected delivery numbers. Electric vehicle sales continue to grow.",
                symbols=["TSLA"],
                sentiment=0.6
            ),
            NewsItem(
                id="news_003",
                title="Market Volatility Continues",
                content="Stock market volatility persisted as investors weigh economic indicators. Technology stocks showed mixed performance.",
                symbols=["MSFT", "GOOGL"],
                sentiment=-0.2
            ),
            NewsItem(
                id="news_004",
                title="Apple and Tesla Lead Tech Gains",
                content="Apple and Tesla stocks led technology sector gains today. Both companies reported positive developments.",
                symbols=["AAPL", "TSLA"],
                sentiment=0.7
            )
        ]
    
    @pytest.fixture
    def retriever(self, sample_news_items):
        """Create fitted HybridRetriever instance."""
        retriever = HybridRetriever()
        retriever.fit(sample_news_items)
        return retriever
    
    def test_retriever_initialization(self):
        """Test retriever initialization with default parameters."""
        retriever = HybridRetriever()
        assert retriever.tfidf_weight == 0.5
        assert retriever.bm25_weight == 0.5
        assert not retriever.is_fitted
    
    def test_retriever_initialization_custom_weights(self):
        """Test retriever initialization with custom weights."""
        retriever = HybridRetriever(tfidf_weight=0.7, bm25_weight=0.3)
        assert retriever.tfidf_weight == 0.7
        assert retriever.bm25_weight == 0.3
    
    def test_retriever_fit(self, sample_news_items):
        """Test retriever fitting on news items."""
        retriever = HybridRetriever()
        retriever.fit(sample_news_items)
        
        assert retriever.is_fitted
        assert len(retriever.news_items) == len(sample_news_items)
        assert len(retriever.documents) == len(sample_news_items)
        assert retriever.tfidf_matrix is not None
    
    def test_retriever_fit_empty_items(self):
        """Test retriever fitting with empty news items."""
        retriever = HybridRetriever()
        retriever.fit([])
        
        # Should handle empty input gracefully
        assert len(retriever.news_items) == 0
        assert len(retriever.documents) == 0
    
    def test_retriever_search_before_fit(self):
        """Test search before fitting should raise error."""
        retriever = HybridRetriever()
        
        with pytest.raises(ValueError, match="Retriever must be fitted"):
            retriever.search("test query")
    
    def test_retriever_search_success(self, retriever):
        """Test successful search operation."""
        results = retriever.search("Apple earnings revenue", top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(item, tuple) for item in results)
        assert all(len(item) == 2 for item in results)
        
        # Check that results are sorted by score (descending)
        if len(results) > 1:
            assert results[0][1] >= results[1][1]
    
    def test_retriever_search_empty_query(self, retriever):
        """Test search with empty query."""
        results = retriever.search("", top_k=5)
        assert len(results) == 0
    
    def test_retriever_search_no_matches(self, retriever):
        """Test search with query that has no matches."""
        results = retriever.search("completely unrelated query xyz", top_k=5)
        # Should return empty list or items with very low scores
        assert len(results) >= 0
    
    def test_retriever_search_top_k_limit(self, retriever):
        """Test that search respects top_k limit."""
        results = retriever.search("stock", top_k=2)
        assert len(results) <= 2
    
    def test_retriever_relevance_scores(self, retriever):
        """Test that relevance scores are added to results."""
        results = retriever.search("Apple", top_k=3)
        
        for news_item, score in results:
            assert hasattr(news_item, 'relevance_score')
            assert news_item.relevance_score is not None
            assert news_item.relevance_score >= 0
    
    def test_retriever_get_stats_unfitted(self):
        """Test get_stats on unfitted retriever."""
        retriever = HybridRetriever()
        stats = retriever.get_stats()
        
        assert stats["fitted"] is False
    
    def test_retriever_get_stats_fitted(self, retriever):
        """Test get_stats on fitted retriever."""
        stats = retriever.get_stats()
        
        assert stats["fitted"] is True
        assert stats["num_documents"] > 0
        assert stats["tfidf_features"] > 0
        assert "tfidf_weight" in stats
        assert "bm25_weight" in stats
    
    def test_retriever_save_load_index(self, retriever):
        """Test saving and loading retriever index."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = Path(f.name)
        
        # Save index
        retriever.save_index(temp_path)
        assert temp_path.exists()
        
        # Create new retriever and load index
        new_retriever = HybridRetriever()
        new_retriever.load_index(temp_path)
        
        assert new_retriever.is_fitted
        assert len(new_retriever.news_items) == len(retriever.news_items)
        assert new_retriever.tfidf_weight == retriever.tfidf_weight
        
        # Test that loaded retriever works
        results = new_retriever.search("Apple", top_k=2)
        assert len(results) > 0
        
        # Clean up
        temp_path.unlink()
    
    def test_retriever_save_unfitted_error(self):
        """Test saving unfitted retriever should raise error."""
        retriever = HybridRetriever()
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = Path(f.name)
        
        with pytest.raises(ValueError, match="Cannot save unfitted retriever"):
            retriever.save_index(temp_path)
        
        temp_path.unlink()
    
    def test_tfidf_scores_calculation(self, retriever):
        """Test TF-IDF scores calculation."""
        scores = retriever._get_tfidf_scores("Apple earnings")
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(retriever.documents)
        assert all(0 <= score <= 1 for score in scores)
    
    def test_bm25_scores_calculation(self, retriever):
        """Test BM25 scores calculation."""
        scores = retriever._get_bm25_scores("Apple earnings")
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(retriever.documents)
        assert all(score >= 0 for score in scores)
    
    def test_combined_scoring(self, retriever):
        """Test that combined scoring works correctly."""
        # Test with different weight combinations
        retriever.tfidf_weight = 0.8
        retriever.bm25_weight = 0.2
        
        results = retriever.search("Apple", top_k=2)
        assert len(results) > 0
        
        # Scores should be influenced by both methods
        for news_item, score in results:
            assert score > 0

if __name__ == "__main__":
    pytest.main([__file__])
