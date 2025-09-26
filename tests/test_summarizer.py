"""
Tests for the summarization module.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.summarizer import Lead3Summarizer, TextRankSummarizer, HybridSummarizer

class TestLead3Summarizer:
    """Test cases for Lead3Summarizer class."""
    
    @pytest.fixture
    def summarizer(self):
        """Create Lead3Summarizer instance."""
        return Lead3Summarizer(max_sentences=3)
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return (
            "Apple Inc. reported strong fourth quarter earnings yesterday. "
            "The company's revenue exceeded analyst expectations by 5%. "
            "iPhone sales were particularly strong in international markets. "
            "The stock price increased by 3% in after-hours trading. "
            "Analysts are optimistic about the company's future prospects. "
            "The next earnings call is scheduled for next quarter."
        )
    
    def test_lead3_initialization(self):
        """Test Lead3Summarizer initialization."""
        summarizer = Lead3Summarizer(max_sentences=5)
        assert summarizer.max_sentences == 5
    
    def test_lead3_summarize_short_text(self, summarizer):
        """Test summarization of text shorter than max_sentences."""
        short_text = "This is a short text. It has only two sentences."
        result = summarizer.summarize(short_text)
        assert result == short_text.strip()
    
    def test_lead3_summarize_long_text(self, summarizer, sample_text):
        """Test summarization of text longer than max_sentences."""
        result = summarizer.summarize(sample_text)
        sentences = result.split('. ')
        
        # Should return first 3 sentences
        assert len(sentences) <= 3
        assert "Apple Inc. reported strong fourth quarter earnings yesterday" in result
        assert "The company's revenue exceeded analyst expectations by 5%" in result
    
    def test_lead3_summarize_empty_text(self, summarizer):
        """Test summarization of empty text."""
        result = summarizer.summarize("")
        assert result == ""
    
    def test_lead3_summarize_custom_max_sentences(self, summarizer, sample_text):
        """Test summarization with custom max_sentences parameter."""
        result = summarizer.summarize(sample_text, max_sentences=2)
        sentences = result.split('. ')
        assert len(sentences) <= 2

class TestTextRankSummarizer:
    """Test cases for TextRankSummarizer class."""
    
    @pytest.fixture
    def summarizer(self):
        """Create TextRankSummarizer instance."""
        return TextRankSummarizer(max_sentences=3)
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return (
            "Apple Inc. is a technology company based in Cupertino, California. "
            "The company designs and manufactures consumer electronics and software. "
            "Apple's most popular products include the iPhone, iPad, and Mac computers. "
            "The iPhone is Apple's flagship product and generates most of its revenue. "
            "Apple also provides services like the App Store and iCloud. "
            "The company has a strong focus on innovation and design. "
            "Apple's stock is traded on the NASDAQ under the symbol AAPL. "
            "The company has a market capitalization of over one trillion dollars."
        )
    
    def test_textrank_initialization(self):
        """Test TextRankSummarizer initialization."""
        summarizer = TextRankSummarizer(
            max_sentences=5,
            similarity_threshold=0.2,
            damping=0.9
        )
        assert summarizer.max_sentences == 5
        assert summarizer.similarity_threshold == 0.2
        assert summarizer.damping == 0.9
    
    def test_textrank_summarize_short_text(self, summarizer):
        """Test summarization of text shorter than max_sentences."""
        short_text = "This is short. Only two sentences here."
        result = summarizer.summarize(short_text)
        assert result == short_text.strip()
    
    def test_textrank_summarize_long_text(self, summarizer, sample_text):
        """Test summarization of text longer than max_sentences."""
        result = summarizer.summarize(sample_text)
        
        # Should return a summary
        assert len(result) > 0
        assert len(result) < len(sample_text)
        
        # Should contain some key information
        sentences = result.split('. ')
        assert len(sentences) <= 3
    
    def test_textrank_summarize_empty_text(self, summarizer):
        """Test summarization of empty text."""
        result = summarizer.summarize("")
        assert result == ""
    
    def test_textrank_calculate_similarity_matrix(self, summarizer):
        """Test similarity matrix calculation."""
        sentences = [
            "Apple is a technology company.",
            "The company makes iPhones and iPads.",
            "Technology companies are innovative.",
            "Apple's products are popular worldwide."
        ]
        
        matrix = summarizer._calculate_similarity_matrix(sentences)
        
        assert matrix.shape == (4, 4)
        assert matrix[0][0] == 0  # Diagonal should be 0 (no self-similarity)
        assert matrix[0][1] >= 0  # Similarities should be non-negative
    
    def test_textrank_algorithm(self, summarizer):
        """Test TextRank algorithm execution."""
        import numpy as np
        
        # Create a simple similarity matrix
        similarity_matrix = np.array([
            [0.0, 0.5, 0.3, 0.2],
            [0.5, 0.0, 0.4, 0.6],
            [0.3, 0.4, 0.0, 0.1],
            [0.2, 0.6, 0.1, 0.0]
        ])
        
        scores = summarizer._textrank(similarity_matrix)
        
        assert len(scores) == 4
        assert all(score >= 0 for score in scores)
        assert abs(sum(scores) - 1.0) < 0.1  # Scores should roughly sum to 1
    
    def test_textrank_select_top_sentences(self, summarizer):
        """Test top sentence selection."""
        import numpy as np
        
        sentences = [
            "First sentence.",
            "Second sentence.",
            "Third sentence.",
            "Fourth sentence."
        ]
        scores = np.array([0.1, 0.4, 0.2, 0.3])
        
        selected = summarizer._select_top_sentences(sentences, scores, 2)
        
        assert len(selected) == 2
        # Should preserve original order
        assert selected[0] == "Second sentence."  # Highest score
        assert selected[1] == "Fourth sentence."  # Second highest score

class TestHybridSummarizer:
    """Test cases for HybridSummarizer class."""
    
    @pytest.fixture
    def summarizer(self):
        """Create HybridSummarizer instance."""
        return HybridSummarizer(default_method="textrank", max_sentences=3)
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return (
            "Tesla Inc. is an electric vehicle and clean energy company. "
            "The company was founded by Elon Musk and others in 2003. "
            "Tesla's main products are electric cars, energy storage systems, and solar panels. "
            "The Model S was Tesla's first mass-produced electric vehicle. "
            "Tesla has revolutionized the automotive industry with its innovations. "
            "The company's stock price has been highly volatile in recent years. "
            "Tesla continues to expand its manufacturing capabilities globally. "
            "The company aims to accelerate the world's transition to sustainable energy."
        )
    
    def test_hybrid_initialization(self):
        """Test HybridSummarizer initialization."""
        summarizer = HybridSummarizer(default_method="lead3", max_sentences=5)
        assert summarizer.default_method == "lead3"
        assert summarizer.max_sentences == 5
    
    def test_hybrid_summarize_lead3(self, summarizer, sample_text):
        """Test summarization using Lead-3 method."""
        result = summarizer.summarize(sample_text, method="lead3")
        
        assert len(result) > 0
        assert "Tesla Inc. is an electric vehicle" in result
        # Should start with first sentence for Lead-3
    
    def test_hybrid_summarize_textrank(self, summarizer, sample_text):
        """Test summarization using TextRank method."""
        result = summarizer.summarize(sample_text, method="textrank")
        
        assert len(result) > 0
        assert len(result) < len(sample_text)
    
    def test_hybrid_summarize_hybrid_method(self, summarizer, sample_text):
        """Test summarization using hybrid method."""
        result = summarizer.summarize(sample_text, method="hybrid")
        
        assert len(result) > 0
        assert len(result) < len(sample_text)
    
    def test_hybrid_summarize_unknown_method(self, summarizer, sample_text):
        """Test summarization with unknown method falls back to default."""
        result = summarizer.summarize(sample_text, method="unknown_method")
        
        # Should fall back to default method (textrank)
        assert len(result) > 0
    
    def test_hybrid_summarize_empty_text(self, summarizer):
        """Test summarization of empty text."""
        result = summarizer.summarize("")
        assert result == ""
    
    def test_hybrid_summarize_custom_max_sentences(self, summarizer, sample_text):
        """Test summarization with custom max_sentences."""
        result = summarizer.summarize(sample_text, max_sentences=2)
        sentences = result.split('. ')
        assert len(sentences) <= 2
    
    def test_hybrid_method_short_text(self, summarizer):
        """Test hybrid method with short text (should use Lead-3)."""
        short_text = (
            "First sentence here. "
            "Second sentence follows. "
            "Third sentence continues. "
            "Fourth sentence ends. "
            "Fifth sentence concludes. "
            "Sixth and final sentence."
        )
        
        result = summarizer._hybrid_summarize(short_text, 3)
        
        # For short texts (<=6 sentences), should use Lead-3
        assert "First sentence here" in result
    
    def test_hybrid_method_long_text(self, summarizer, sample_text):
        """Test hybrid method with long text (should use TextRank with first sentence)."""
        result = summarizer._hybrid_summarize(sample_text, 3)
        
        # Should include first sentence for hybrid method
        sentences = result.split('. ')
        assert len(sentences) <= 3
    
    def test_get_compression_ratio(self, summarizer):
        """Test compression ratio calculation."""
        original = "This is a long original text with many words and sentences."
        summary = "This is a summary."
        
        ratio = summarizer.get_compression_ratio(original, summary)
        
        assert 0 < ratio < 1
        assert ratio == len(summary) / len(original)
    
    def test_get_compression_ratio_empty_original(self, summarizer):
        """Test compression ratio with empty original text."""
        ratio = summarizer.get_compression_ratio("", "summary")
        assert ratio == 0.0

if __name__ == "__main__":
    pytest.main([__file__])
