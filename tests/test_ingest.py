"""
Tests for the data ingestion module.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.ingest import DataIngestor
from app.models import NewsItem

class TestDataIngestor:
    """Test cases for DataIngestor class."""
    
    @pytest.fixture
    def ingestor(self):
        """Create a DataIngestor instance."""
        return DataIngestor()
    
    @pytest.fixture
    def sample_json_data(self):
        """Sample JSON data for testing."""
        return [
            {
                "id": "news_001",
                "title": "Apple Reports Strong Q4 Earnings",
                "content": "Apple Inc. reported strong fourth quarter earnings with revenue exceeding expectations.",
                "url": "https://example.com/apple-earnings",
                "published_date": "2024-01-15 10:30:00",
                "source": "TechNews",
                "symbols": ["AAPL"],
                "sentiment": 0.8
            },
            {
                "title": "Tesla Stock Surges on Delivery Numbers",
                "content": "Tesla's stock price increased significantly following better than expected delivery numbers.",
                "date": "2024-01-16",
                "source": "AutoNews",
                "symbol": "TSLA",
                "sentiment": 0.6
            },
            {
                "title": "Market Volatility Continues",
                "content": "Stock market volatility persisted as investors weigh economic indicators.",
                "published_date": "invalid_date",
                "symbols": "MSFT,GOOGL",
                "sentiment": -0.2
            }
        ]
    
    @pytest.fixture
    def temp_json_file(self, sample_json_data):
        """Create a temporary JSON file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_json_data, f)
            return Path(f.name)
    
    def test_load_json_file_success(self, ingestor, temp_json_file, sample_json_data):
        """Test successful JSON file loading."""
        data = ingestor.load_json_file(temp_json_file)
        assert len(data) == len(sample_json_data)
        assert data[0]["title"] == "Apple Reports Strong Q4 Earnings"
        
        # Clean up
        temp_json_file.unlink()
    
    def test_load_json_file_not_found(self, ingestor):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            ingestor.load_json_file(Path("nonexistent.json"))
    
    def test_load_json_file_invalid_json(self, ingestor):
        """Test loading invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = Path(f.name)
        
        with pytest.raises(json.JSONDecodeError):
            ingestor.load_json_file(temp_path)
        
        temp_path.unlink()
    
    def test_normalize_item_complete(self, ingestor):
        """Test normalizing item with all fields."""
        raw_item = {
            "id": "test_001",
            "title": "Test Article",
            "content": "This is test content.",
            "url": "https://example.com",
            "published_date": "2024-01-15 10:30:00",
            "source": "TestSource",
            "symbols": ["AAPL", "MSFT"],
            "sentiment": 0.5
        }
        
        news_item = ingestor.normalize_item(raw_item)
        
        assert news_item.id == "test_001"
        assert news_item.title == "Test Article"
        assert news_item.content == "This is test content."
        assert news_item.url == "https://example.com"
        assert news_item.source == "TestSource"
        assert news_item.symbols == ["AAPL", "MSFT"]
        assert news_item.sentiment == 0.5
        assert isinstance(news_item.published_date, datetime)
    
    def test_normalize_item_minimal(self, ingestor):
        """Test normalizing item with minimal fields."""
        raw_item = {
            "title": "Minimal Article",
            "content": "Minimal content."
        }
        
        news_item = ingestor.normalize_item(raw_item)
        
        assert news_item.title == "Minimal Article"
        assert news_item.content == "Minimal content."
        assert news_item.id.startswith("news_")
        assert news_item.symbols == []
        assert news_item.published_date is None
    
    def test_normalize_item_symbols_string(self, ingestor):
        """Test normalizing symbols from string format."""
        raw_item = {
            "title": "Test",
            "content": "Content",
            "symbols": "AAPL,MSFT,GOOGL"
        }
        
        news_item = ingestor.normalize_item(raw_item)
        assert news_item.symbols == ["AAPL", "MSFT", "GOOGL"]
    
    def test_normalize_item_single_symbol(self, ingestor):
        """Test normalizing single symbol field."""
        raw_item = {
            "title": "Test",
            "content": "Content",
            "symbol": "tsla"
        }
        
        news_item = ingestor.normalize_item(raw_item)
        assert news_item.symbols == ["TSLA"]
    
    def test_parse_date_valid_formats(self, ingestor):
        """Test parsing various valid date formats."""
        valid_dates = [
            "2024-01-15 10:30:00",
            "2024-01-15",
            "2024/01/15 10:30:00",
            "01/15/2024",
            "2024-01-15T10:30:00",
            "2024-01-15T10:30:00Z"
        ]
        
        for date_str in valid_dates:
            result = ingestor._parse_date(date_str)
            assert isinstance(result, datetime)
    
    def test_parse_date_invalid(self, ingestor):
        """Test parsing invalid date formats."""
        invalid_dates = [
            "invalid_date",
            "2024-13-45",
            "",
            None
        ]
        
        for date_str in invalid_dates:
            result = ingestor._parse_date(date_str)
            assert result is None
    
    def test_ingest_file_success(self, ingestor, temp_json_file):
        """Test successful file ingestion."""
        news_items = ingestor.ingest_file(temp_json_file)
        
        assert len(news_items) == 3
        assert all(isinstance(item, NewsItem) for item in news_items)
        assert news_items[0].title == "Apple Reports Strong Q4 Earnings"
        assert news_items[1].symbols == ["TSLA"]
        
        # Check cache
        cached_items = ingestor.get_cached_items()
        assert len(cached_items) == 3
        
        # Clean up
        temp_json_file.unlink()
    
    def test_ingest_file_with_errors(self, ingestor):
        """Test ingestion with some invalid items."""
        data_with_errors = [
            {
                "title": "Valid Article",
                "content": "Valid content"
            },
            {
                # Missing required fields - should be handled gracefully
            },
            {
                "title": "Another Valid Article",
                "content": "More valid content"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data_with_errors, f)
            temp_path = Path(f.name)
        
        news_items = ingestor.ingest_file(temp_path)
        
        # Should get valid items even if some fail
        assert len(news_items) >= 2
        
        temp_path.unlink()
    
    def test_clear_cache(self, ingestor, temp_json_file):
        """Test cache clearing functionality."""
        # Ingest some data
        ingestor.ingest_file(temp_json_file)
        assert len(ingestor.get_cached_items()) > 0
        
        # Clear cache
        ingestor.clear_cache()
        assert len(ingestor.get_cached_items()) == 0
        
        temp_json_file.unlink()
    
    def test_load_single_object_json(self, ingestor):
        """Test loading JSON file with single object instead of array."""
        single_item = {
            "title": "Single Article",
            "content": "Single content"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(single_item, f)
            temp_path = Path(f.name)
        
        data = ingestor.load_json_file(temp_path)
        assert len(data) == 1
        assert data[0]["title"] == "Single Article"
        
        temp_path.unlink()

if __name__ == "__main__":
    pytest.main([__file__])
