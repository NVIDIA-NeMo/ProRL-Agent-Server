#!/usr/bin/env bash
# Submit a relay chain that keeps profiling/report jobs under observation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MONITOR_SCRIPT="${SCRIPT_DIR}/monitor_profile.py"
SBATCH_BIN="${SBATCH_BIN:-/cm/shared/apps/slurm/current/bin/sbatch}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage: submit_monitor.sh STATUS_JSON LABEL=JOB_ID [LABEL=JOB_ID ...]

Environment:
  MONITOR_SEGMENTS=12          number of sequential cpu_short relays
  MONITOR_PARTITION=cpu_short  Slurm partition
  MONITOR_WALL_TIME=01:00:00   wall time per relay
  MONITOR_MAX_SECONDS=3300     runtime per relay before handing off
  MONITOR_POLL_SECONDS=30
  MONITOR_HEARTBEAT_SECONDS=300
EOF
}

if [ "$#" -lt 2 ]; then
    usage >&2
    exit 2
fi

status_json="$1"
shift
if [[ "${status_json}" != /* ]]; then
    echo "ERROR: STATUS_JSON must be absolute" >&2
    exit 2
fi

jobs=("$@")
for item in "${jobs[@]}"; do
    if ! [[ "${item}" =~ ^[A-Za-z0-9._-]+=[1-9][0-9]*$ ]]; then
        echo "ERROR: invalid monitor target ${item@Q}; expected LABEL=JOB_ID" >&2
        exit 2
    fi
done

segments="${MONITOR_SEGMENTS:-12}"
if ! [[ "${segments}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MONITOR_SEGMENTS must be a positive integer" >&2
    exit 2
fi

mkdir -p "$(dirname -- "${status_json}")"
job_args=()
for item in "${jobs[@]}"; do
    job_args+=(--job "${item}")
done

command=(
    "${PYTHON_BIN}" "${MONITOR_SCRIPT}"
    "${job_args[@]}"
    --status-json "${status_json}"
    --poll-seconds "${MONITOR_POLL_SECONDS:-30}"
    --heartbeat-seconds "${MONITOR_HEARTBEAT_SECONDS:-300}"
    --max-seconds "${MONITOR_MAX_SECONDS:-3300}"
)
printf -v wrapped_command '%q ' "${command[@]}"

previous_job_id=""
first_job_id=""
for segment in $(seq 1 "${segments}"); do
    dependency_args=()
    if [ -n "${previous_job_id}" ]; then
        dependency_args=(--dependency="afterany:${previous_job_id}")
    fi
    job_id="$(
        "${SBATCH_BIN}" \
            --parsable \
            --partition="${MONITOR_PARTITION:-cpu_short}" \
            --time="${MONITOR_WALL_TIME:-01:00:00}" \
            --nodes=1 \
            --ntasks=1 \
            --cpus-per-task="${MONITOR_CPUS:-2}" \
            --job-name="profile-monitor-${segment}" \
            --output="$(dirname -- "${status_json}")/monitor-${segment}-%j.out" \
            --error="$(dirname -- "${status_json}")/monitor-${segment}-%j.err" \
            "${dependency_args[@]}" \
            --wrap="${wrapped_command}"
    )"
    job_id="${job_id%%;*}"
    if ! [[ "${job_id}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: sbatch returned invalid job id ${job_id@Q}" >&2
        exit 1
    fi
    if [ -z "${first_job_id}" ]; then
        first_job_id="${job_id}"
    fi
    previous_job_id="${job_id}"
done

printf 'monitor_first_job_id=%s\n' "${first_job_id}"
printf 'monitor_last_job_id=%s\n' "${previous_job_id}"
printf 'monitor_status_json=%s\n' "${status_json}"
