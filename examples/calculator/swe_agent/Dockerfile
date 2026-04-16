FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install SWE-agent from git (PyPI version has missing dependency)
RUN pip install --no-cache-dir "git+https://github.com/SWE-agent/SWE-agent.git" \
    && SITE=$(python -c "import site; print(site.getsitepackages()[0])") \
    && git clone --depth 1 https://github.com/SWE-agent/SWE-agent.git /tmp/swe-agent-src \
    && cp -r /tmp/swe-agent-src/config "$SITE/config" \
    && cp -r /tmp/swe-agent-src/tools "$SITE/tools" \
    && mkdir -p "$SITE/trajectories" \
    && rm -rf /tmp/swe-agent-src

# Pre-install tool dependencies so install.sh is a no-op at runtime
# (compute nodes may lack internet access).
RUN pip install --no-cache-dir 'tree-sitter==0.21.3' 'tree-sitter-languages'

WORKDIR /polar/session
