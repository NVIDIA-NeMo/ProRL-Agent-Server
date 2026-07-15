#!/usr/bin/env bash
# Schedule the final SPilot + TMax report after the last profiling arm.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
SBATCH_BIN="${SBATCH_BIN:-/cm/shared/apps/slurm/current/bin/sbatch}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
POLAR_DATA_ROOT="${POLAR_DATA_ROOT:-${WORKSPACE_ROOT}/data}"
REPORT_ACCOUNT="${REPORT_ACCOUNT:-${ACCOUNT:-${SBATCH_ACCOUNT:-nvr_lpr_llm}}}"

if [ "$#" -ne 4 ]; then
    echo "Usage: submit_report.sh AFTER_JOB_ID SPILOT_MANIFEST TMAX_MANIFEST OUTPUT_DIR" >&2
    exit 2
fi

after_job_id="$1"
spilot_manifest="$2"
tmax_manifest="$3"
output_dir="$4"
if ! [[ "${after_job_id}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: AFTER_JOB_ID must be numeric" >&2
    exit 2
fi
for path in "${spilot_manifest}" "${tmax_manifest}"; do
    if [[ "${path}" != /* ]] || [ ! -s "${path}" ]; then
        echo "ERROR: manifest must be an existing non-empty absolute file: ${path}" >&2
        exit 2
    fi
done
if [[ "${output_dir}" != /* ]]; then
    echo "ERROR: OUTPUT_DIR must be absolute" >&2
    exit 2
fi

mkdir -p "${output_dir}"
command=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_profile_report.py"
    --data-root "${POLAR_DATA_ROOT}"
    --log-root "${POLAR_DATA_ROOT}/logs/slurm"
    --spilot-manifest "${spilot_manifest}"
    --tmax-manifest "${tmax_manifest}"
    --output-dir "${output_dir}"
    --expected-steps "${PROFILE_EXPECTED_STEPS:-3}"
    --warmup-steps "${PROFILE_WARMUP_STEPS:-1}"
    --log-timezone "${PROFILE_LOG_TIMEZONE:-UTC}"
)
printf -v wrapped_command '%q ' "${command[@]}"

job_id="$(
    "${SBATCH_BIN}" \
        --parsable \
        --account="${REPORT_ACCOUNT}" \
        --partition="${REPORT_PARTITION:-cpu_short}" \
        --time="${REPORT_WALL_TIME:-00:30:00}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="${REPORT_CPUS:-4}" \
        --mem="${REPORT_MEMORY:-16G}" \
        --dependency="afterany:${after_job_id}" \
        --job-name="profile-final-report" \
        --output="${output_dir}/report-%j.out" \
        --error="${output_dir}/report-%j.err" \
        --wrap="${wrapped_command}"
)"
job_id="${job_id%%;*}"
if ! [[ "${job_id}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: sbatch returned invalid job id ${job_id@Q}" >&2
    exit 1
fi
printf 'report_job_id=%s\n' "${job_id}"
printf 'report_html=%s\n' "${output_dir}/gpu-allocation-report.html"
