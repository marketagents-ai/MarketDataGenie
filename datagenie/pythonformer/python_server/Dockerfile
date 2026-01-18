# Pythonformer REPL Server
# Isolated sandbox with pre-installed scientific packages

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including git for cloning minference)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for math/code tasks
RUN pip install --no-cache-dir \
    flask \
    numpy \
    pandas \
    sympy \
    scipy \
    matplotlib \
    scikit-learn \
    networkx \
    requests \
    beautifulsoup4 \
    lxml \
    litellm \
    openai \
    pydantic \
    tenacity \
    python-dotenv

# Clone and install minference from MarketAgents repo (has lite module)
RUN git clone --depth 1 https://github.com/marketagents-ai/MarketAgents.git /tmp/marketagents && \
    pip install --no-cache-dir /tmp/marketagents/minference && \
    rm -rf /tmp/marketagents

# Copy sandbox code
COPY sandbox.py server.py __init__.py ./

# Create workspace directory
RUN mkdir -p /app/workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 5003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5003/health')" || exit 1

# Run server
CMD ["python", "server.py", "--port", "5003", "--host", "0.0.0.0"]
