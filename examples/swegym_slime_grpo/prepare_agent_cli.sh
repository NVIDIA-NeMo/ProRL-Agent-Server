#!/usr/bin/env bash
# Prepare the shared agent CLI directory (Node 22 + coding agent CLIs)
# without Docker. Downloads Node.js and installs agent CLIs directly.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

AGENT_CLI_DIR="${AGENT_CLI_DIR:-${PROJECT_ROOT}/tmp/swegym_agent_cli/opt_node}"
NODE_VERSION="${NODE_VERSION:-22.11.0}"

required_bins=(node codex claude)

_all_present() {
    for bin in "${required_bins[@]}"; do
        [[ ! -x "${AGENT_CLI_DIR}/bin/${bin}" ]] && return 1
    done
    return 0
}

if _all_present; then
    echo "Agent CLI directory already prepared: ${AGENT_CLI_DIR}"
    exit 0
fi

echo "Preparing agent CLI directory: ${AGENT_CLI_DIR}"
rm -rf "${AGENT_CLI_DIR}"
mkdir -p "${AGENT_CLI_DIR}"

echo "Downloading Node.js v${NODE_VERSION}..."
curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
    | tar -xJ --strip-components=1 -C "${AGENT_CLI_DIR}"

export PATH="${AGENT_CLI_DIR}/bin:${PATH}"
echo "Node version: $(node --version)"

echo "Installing agent CLIs..."
npm install -g --no-audit --no-fund --prefix="${AGENT_CLI_DIR}" \
    @openai/codex@latest \
    @anthropic-ai/claude-code@latest \
    @qwen-code/qwen-code@latest \
    opencode-ai@latest \
    @mariozechner/pi-coding-agent@latest

echo "Installed binaries:"
ls -la "${AGENT_CLI_DIR}/bin/"

echo "Done: ${AGENT_CLI_DIR}"
