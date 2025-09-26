"""
Streamlit web interface for the Stock News Analysis application.
"""

import streamlit as st
import asyncio
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.pipeline import Pipeline
from app.models import NewsItem

# Page configuration
st.set_page_config(
    page_title="Stock News Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .news-item {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .relevance-score {
        background-color: #e8f4fd;
        color: #1f77b4;
        padding: 0.25rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    """Initialize and cache the pipeline."""
    return Pipeline()

@st.cache_data
def load_pipeline_data(_pipeline):
    """Load pipeline data with caching."""
    try:
        asyncio.run(_pipeline.initialize())
        return True, "Pipeline initialized successfully"
    except Exception as e:
        return False, f"Error initializing pipeline: {str(e)}"

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock News Analysis</h1>', unsafe_allow_html=True)
    
    # Initialize pipeline
    pipeline = get_pipeline()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Initialize pipeline
        if st.button("🔄 Initialize/Reload Pipeline"):
            st.cache_data.clear()
            with st.spinner("Initializing pipeline..."):
                success, message = load_pipeline_data(pipeline)
                if success:
                    st.success(message)
                else:
                    st.error(message)
                    return
        
        # Check if pipeline is initialized
        if not pipeline.is_initialized:
            with st.spinner("Loading pipeline..."):
                success, message = load_pipeline_data(pipeline)
                if not success:
                    st.error(message)
                    st.stop()
        
        # Search parameters
        st.subheader("Search Parameters")
        top_k = st.slider("Number of results", min_value=1, max_value=50, value=10)
        include_summary = st.checkbox("Include summaries", value=True)
        
        if include_summary:
            summary_method = st.selectbox(
                "Summary method",
                ["textrank", "lead3", "hybrid"],
                index=0
            )
            max_sentences = st.slider("Max summary sentences", min_value=1, max_value=10, value=3)
        else:
            summary_method = "textrank"
            max_sentences = 3
        
        # Pipeline stats
        st.subheader("📊 Pipeline Stats")
        stats = pipeline.get_stats()
        st.metric("Total Articles", stats.get("total_articles", 0))
        if stats.get("fitted", False):
            st.metric("TF-IDF Features", stats.get("tfidf_features", 0))
            st.metric("Avg Doc Length", f"{stats.get('avg_doc_length', 0):.1f}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "📰 Browse Articles", "📈 Analytics"])
    
    with tab1:
        search_tab(pipeline, top_k, include_summary, summary_method, max_sentences)
    
    with tab2:
        browse_tab(pipeline)
    
    with tab3:
        analytics_tab(pipeline)

def search_tab(pipeline, top_k, include_summary, summary_method, max_sentences):
    """Search functionality tab."""
    
    st.header("🔍 Search Stock News")
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., Tesla earnings, Apple stock price, market volatility",
            key="search_query"
        )
    with col2:
        search_button = st.button("🔍 Search", type="primary")
    
    # Perform search
    if search_button and query:
        with st.spinner("Searching..."):
            try:
                results = asyncio.run(pipeline.process_query(
                    query=query,
                    top_k=top_k,
                    include_summary=include_summary,
                    summary_method=summary_method,
                    max_summary_sentences=max_sentences
                ))
                
                if results:
                    st.success(f"Found {len(results)} relevant articles")
                    
                    # Display results
                    for i, item in enumerate(results, 1):
                        display_news_item(item, i)
                else:
                    st.warning("No results found for your query.")
                    
            except Exception as e:
                st.error(f"Search error: {str(e)}")
    
    elif query and not search_button:
        st.info("Click the Search button to find relevant articles.")

def browse_tab(pipeline):
    """Browse articles tab."""
    
    st.header("📰 Browse All Articles")
    
    # Pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        page_size = st.selectbox("Articles per page", [10, 25, 50, 100], index=1)
    with col2:
        page_number = st.number_input("Page", min_value=1, value=1)
    with col3:
        include_summaries = st.checkbox("Include summaries", key="browse_summaries")
    
    # Calculate offset
    offset = (page_number - 1) * page_size
    
    # Get articles
    try:
        articles = asyncio.run(pipeline.get_articles(
            limit=page_size,
            offset=offset,
            include_summary=include_summaries
        ))
        
        if articles:
            st.info(f"Showing articles {offset + 1} to {offset + len(articles)}")
            
            for i, article in enumerate(articles, offset + 1):
                display_news_item(article, i, show_relevance=False)
        else:
            st.warning("No articles found.")
            
    except Exception as e:
        st.error(f"Error loading articles: {str(e)}")

def analytics_tab(pipeline):
    """Analytics and statistics tab."""
    
    st.header("📈 Analytics Dashboard")
    
    try:
        # Get all articles for analysis
        all_articles = asyncio.run(pipeline.get_articles(limit=1000))
        
        if not all_articles:
            st.warning("No articles available for analysis.")
            return
        
        # Create DataFrame for analysis
        df_data = []
        for article in all_articles:
            df_data.append({
                'id': article.id,
                'title': article.title,
                'content_length': len(article.content),
                'title_length': len(article.title),
                'published_date': article.published_date,
                'source': article.source or 'Unknown',
                'symbols': len(article.symbols) if article.symbols else 0,
                'sentiment': article.sentiment
            })
        
        df = pd.DataFrame(df_data)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Articles", len(df))
        with col2:
            avg_length = df['content_length'].mean()
            st.metric("Avg Content Length", f"{avg_length:.0f} chars")
        with col3:
            sources_count = df['source'].nunique()
            st.metric("Unique Sources", sources_count)
        with col4:
            with_symbols = df[df['symbols'] > 0].shape[0]
            st.metric("Articles with Symbols", with_symbols)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Content length distribution
            fig_length = px.histogram(
                df, 
                x='content_length',
                title="Content Length Distribution",
                nbins=30
            )
            fig_length.update_layout(height=400)
            st.plotly_chart(fig_length, use_container_width=True)
        
        with col2:
            # Source distribution
            source_counts = df['source'].value_counts().head(10)
            fig_sources = px.bar(
                x=source_counts.values,
                y=source_counts.index,
                orientation='h',
                title="Top 10 Sources"
            )
            fig_sources.update_layout(height=400)
            st.plotly_chart(fig_sources, use_container_width=True)
        
        # Timeline if dates available
        if df['published_date'].notna().any():
            st.subheader("📅 Publication Timeline")
            df_with_dates = df.dropna(subset=['published_date'])
            df_with_dates['date'] = pd.to_datetime(df_with_dates['published_date']).dt.date
            
            timeline_data = df_with_dates.groupby('date').size().reset_index(name='count')
            
            fig_timeline = px.line(
                timeline_data,
                x='date',
                y='count',
                title="Articles Published Over Time"
            )
            fig_timeline.update_layout(height=400)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Sentiment analysis if available
        if df['sentiment'].notna().any():
            st.subheader("😊 Sentiment Analysis")
            sentiment_df = df.dropna(subset=['sentiment'])
            
            fig_sentiment = px.histogram(
                sentiment_df,
                x='sentiment',
                title="Sentiment Score Distribution",
                nbins=20
            )
            fig_sentiment.update_layout(height=400)
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error generating analytics: {str(e)}")

def display_news_item(item: NewsItem, index: int, show_relevance: bool = True):
    """Display a news item in a formatted card."""
    
    with st.container():
        st.markdown(f'<div class="news-item">', unsafe_allow_html=True)
        
        # Header with title and relevance score
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{index}. {item.title}**")
        with col2:
            if show_relevance and item.relevance_score is not None:
                score_pct = item.relevance_score * 100
                st.markdown(
                    f'<span class="relevance-score">Relevance: {score_pct:.1f}%</span>',
                    unsafe_allow_html=True
                )
        
        # Metadata
        metadata_parts = []
        if item.source:
            metadata_parts.append(f"**Source:** {item.source}")
        if item.published_date:
            metadata_parts.append(f"**Date:** {item.published_date.strftime('%Y-%m-%d %H:%M')}")
        if item.symbols:
            metadata_parts.append(f"**Symbols:** {', '.join(item.symbols)}")
        if item.sentiment is not None:
            sentiment_emoji = "😊" if item.sentiment > 0.1 else "😐" if item.sentiment > -0.1 else "😞"
            metadata_parts.append(f"**Sentiment:** {sentiment_emoji} {item.sentiment:.2f}")
        
        if metadata_parts:
            st.markdown(" | ".join(metadata_parts))
        
        # Summary or content
        if item.summary:
            st.markdown(f"**Summary:** {item.summary}")
            
            # Expandable full content
            with st.expander("📄 View full article"):
                st.markdown(item.content)
        else:
            # Show truncated content
            content_preview = item.content[:500] + "..." if len(item.content) > 500 else item.content
            st.markdown(content_preview)
        
        # URL if available
        if item.url:
            st.markdown(f"[🔗 Read original article]({item.url})")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

if __name__ == "__main__":
    main()
