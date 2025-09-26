# Multi-stage Docker build for Stock News Analysis application

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements files
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install .

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory and set permissions
WORKDIR /app
RUN chown -R appuser:appuser /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser ui/ ./ui/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser pyproject.toml ./

# Create data directory
RUN mkdir -p data && chown -R appuser:appuser data

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create startup script
COPY --chown=appuser:appuser <<EOF /app/start.sh
#!/bin/bash
set -e

# Function to start FastAPI
start_fastapi() {
    echo "Starting FastAPI server..."
    uvicorn app.main:app --host 0.0.0.0 --port 8000 &
    FASTAPI_PID=\$!
    echo "FastAPI started with PID \$FASTAPI_PID"
}

# Function to start Streamlit
start_streamlit() {
    echo "Starting Streamlit server..."
    streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
    STREAMLIT_PID=\$!
    echo "Streamlit started with PID \$STREAMLIT_PID"
}

# Function to handle shutdown
shutdown() {
    echo "Shutting down services..."
    if [ ! -z "\$FASTAPI_PID" ]; then
        kill \$FASTAPI_PID 2>/dev/null || true
    fi
    if [ ! -z "\$STREAMLIT_PID" ]; then
        kill \$STREAMLIT_PID 2>/dev/null || true
    fi
    exit 0
}

# Set up signal handlers
trap shutdown SIGTERM SIGINT

# Check if data file exists
if [ ! -f "data/stock_news.json" ]; then
    echo "Warning: data/stock_news.json not found. Please mount your data file."
    echo "Example: docker run -v /path/to/your/data:/app/data stock-news-analysis"
fi

# Build index if it doesn't exist and data file is present
if [ -f "data/stock_news.json" ] && [ ! -f "data/search_index.pkl" ]; then
    echo "Building search index..."
    python scripts/build_index.py --data data/stock_news.json --index data/search_index.pkl
fi

# Start services based on environment variable
case "\${SERVICE:-both}" in
    "fastapi")
        echo "Starting FastAPI only..."
        start_fastapi
        wait \$FASTAPI_PID
        ;;
    "streamlit")
        echo "Starting Streamlit only..."
        start_streamlit
        wait \$STREAMLIT_PID
        ;;
    "both"|*)
        echo "Starting both FastAPI and Streamlit..."
        start_fastapi
        start_streamlit
        
        # Wait for both processes
        wait \$FASTAPI_PID \$STREAMLIT_PID
        ;;
esac
EOF

# Make startup script executable
RUN chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]

# Labels for metadata
LABEL maintainer="your.email@example.com" \
      version="0.1.0" \
      description="Stock News Analysis - FastAPI + Streamlit application" \
      org.opencontainers.image.title="Stock News Analysis" \
      org.opencontainers.image.description="A comprehensive application for stock news analysis with hybrid retrieval and summarization" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.vendor="Your Organization" \
      org.opencontainers.image.licenses="MIT"
