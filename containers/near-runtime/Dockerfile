# NeAR Runtime Container for ProRL-Agent-Server
# Provides enhanced tools for deep research tasks

ARG BASE_IMAGE=nikolaik/python-nodejs:python3.12-nodejs22
FROM ${BASE_IMAGE}

# Install system packages (core + browser automation)
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        # Core utilities
        git curl wget vim nano tree jq \
        ca-certificates sudo build-essential \
        # Process management
        tmux procps \
        # Browser automation (Playwright dependencies)
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
        libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
        libxdamage1 libxfixes3 libxrandr2 libgbm1 \
        libasound2 libpango-1.0-0 libcairo2 \
        libdbus-1-3 libexpat1 libgdk-pixbuf2.0-0 \
        libgtk-3-0 libx11-xcb1 libxcb-dri3-0 \
        # Additional tools
        zip unzip rsync htop netcat-openbsd; \
    # Cleanup
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Install UV package manager
COPY --from=ghcr.io/astral-sh/uv /uv /uvx /bin/

# Install Python packages
RUN uv pip install --system --no-cache-dir \
    # Scientific computing
    numpy pandas matplotlib scipy scikit-learn \
    # Web & data
    requests beautifulsoup4 lxml html2text markdownify \
    # Browser automation
    playwright \
    # IPython
    ipython \
    # Tavily search (via langchain-community)
    langchain-community tavily-python \
    # Image processing
    pillow

# Install Playwright browsers (Chromium only for efficiency)
RUN playwright install chromium && \
    playwright install-deps chromium

# Create workspace and skills directories
RUN mkdir -p /workspace/assets /workspace/mounted /workspace/outputs /skills

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Set working directory
WORKDIR /workspace

# Keep container running
CMD ["tail", "-f", "/dev/null"]
