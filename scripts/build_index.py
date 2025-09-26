"""
Script to precompute and build search indexes for the stock news analysis system.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('build_index.log')
    ]
)

logger = logging.getLogger(__name__)

async def build_index(data_path: Path, index_path: Path, force_rebuild: bool = False):
    """
    Build search index from data file.
    
    Args:
        data_path: Path to the JSON data file
        index_path: Path where to save the index
        force_rebuild: Whether to force rebuild even if index exists
    """
    logger.info(f"Building index from {data_path}")
    logger.info(f"Index will be saved to {index_path}")
    
    # Check if data file exists
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return False
    
    # Check if index already exists
    if index_path.exists() and not force_rebuild:
        logger.warning(f"Index already exists at {index_path}")
        logger.warning("Use --force to rebuild existing index")
        return False
    
    try:
        # Initialize pipeline
        pipeline = Pipeline(data_path=data_path, index_path=index_path)
        
        # Build index
        start_time = time.time()
        await pipeline.initialize(force_rebuild=True)
        build_time = time.time() - start_time
        
        # Get statistics
        stats = pipeline.get_stats()
        
        logger.info("Index built successfully!")
        logger.info(f"Build time: {build_time:.2f} seconds")
        logger.info(f"Total documents: {stats.get('total_articles', 0)}")
        logger.info(f"TF-IDF features: {stats.get('tfidf_features', 0)}")
        logger.info(f"Average document length: {stats.get('avg_doc_length', 0):.1f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        return False

async def validate_index(index_path: Path, data_path: Path):
    """
    Validate that the index works correctly.
    
    Args:
        index_path: Path to the index file
        data_path: Path to the data file
    """
    logger.info("Validating index...")
    
    try:
        # Load pipeline with existing index
        pipeline = Pipeline(data_path=data_path, index_path=index_path)
        await pipeline.initialize()
        
        # Test search functionality
        test_queries = [
            "Apple earnings revenue",
            "Tesla delivery numbers",
            "market volatility",
            "technology stocks"
        ]
        
        for query in test_queries:
            results = await pipeline.process_query(query, top_k=3, include_summary=False)
            logger.info(f"Query '{query}': {len(results)} results")
        
        logger.info("Index validation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Index validation failed: {e}")
        return False

async def get_index_info(index_path: Path, data_path: Path):
    """
    Get information about an existing index.
    
    Args:
        index_path: Path to the index file
        data_path: Path to the data file
    """
    if not index_path.exists():
        logger.error(f"Index file not found: {index_path}")
        return
    
    try:
        pipeline = Pipeline(data_path=data_path, index_path=index_path)
        await pipeline.initialize()
        
        stats = pipeline.get_stats()
        
        print("\n" + "="*50)
        print("INDEX INFORMATION")
        print("="*50)
        print(f"Data file: {data_path}")
        print(f"Index file: {index_path}")
        print(f"Index size: {index_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"Total documents: {stats.get('total_articles', 0)}")
        print(f"TF-IDF features: {stats.get('tfidf_features', 0)}")
        print(f"Average document length: {stats.get('avg_doc_length', 0):.1f}")
        print(f"TF-IDF weight: {stats.get('tfidf_weight', 0)}")
        print(f"BM25 weight: {stats.get('bm25_weight', 0)}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Failed to get index info: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build search index for stock news analysis")
    
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/stock_news.json"),
        help="Path to the JSON data file (default: data/stock_news.json)"
    )
    
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/search_index.pkl"),
        help="Path to save the index file (default: data/search_index.pkl)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index exists"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing index"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show information about existing index"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Ensure paths are absolute
    data_path = args.data.resolve()
    index_path = args.index.resolve()
    
    # Create directories if they don't exist
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def run():
        if args.info:
            await get_index_info(index_path, data_path)
        elif args.validate:
            success = await validate_index(index_path, data_path)
            sys.exit(0 if success else 1)
        else:
            success = await build_index(data_path, index_path, args.force)
            
            if success and not args.validate:
                # Automatically validate after building
                logger.info("Running validation...")
                await validate_index(index_path, data_path)
            
            sys.exit(0 if success else 1)
    
    # Run the async function
    asyncio.run(run())

if __name__ == "__main__":
    main()
