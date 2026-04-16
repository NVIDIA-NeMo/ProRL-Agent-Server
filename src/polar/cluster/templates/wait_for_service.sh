#!/bin/bash
# wait_for_service.sh — Poll an HTTP endpoint until it responds 2xx.
#
# Usage:
#   source wait_for_service.sh
#   wait_for_service "http://host:8000/health" "vLLM" 120 5

wait_for_service() {
    local url="$1"
    local name="${2:-service}"
    local max_attempts="${3:-60}"
    local interval="${4:-5}"

    echo "[wait] Waiting for ${name} at ${url} (max ${max_attempts} attempts, ${interval}s interval)..."
    for ((i=1; i<=max_attempts; i++)); do
        if curl -sf --max-time 5 "${url}" > /dev/null 2>&1; then
            echo "[wait] ${name} is ready at ${url} (attempt ${i}/${max_attempts})"
            return 0
        fi
        if (( i % 10 == 0 )); then
            echo "[wait] Still waiting for ${name}... (${i}/${max_attempts})"
        fi
        sleep "${interval}"
    done
    echo "[wait] FATAL: ${name} did not become ready at ${url} after ${max_attempts} attempts"
    return 1
}

# Allow direct invocation
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    wait_for_service "$@"
fi
