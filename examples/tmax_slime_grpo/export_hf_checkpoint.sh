#!/usr/bin/env bash
# One-click export of a numbered TMax torch_dist checkpoint to a complete
# HuggingFace safetensors directory while preserving checkpoint dtypes. The
# public mode submits one Slurm job; the hidden --worker mode performs the
# conversion inside that allocation.
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SPILOT_ROOT="$(cd -- "${PROJECT_ROOT}/../.." && pwd)"
USER_ROOT="$(dirname "${SPILOT_ROOT}")"
SELF="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"

RUN_ID="${TMAX_HF_EXPORT_RUN_ID:-${RUN_ID:-}}"
DATA_ROOT="${POLAR_DATA_ROOT:-${SPILOT_ROOT}/data}"
CKPT_ROOT="${TMAX_HF_EXPORT_CKPT_ROOT:-}"
OUTPUT_ROOT="${TMAX_HF_EXPORT_OUTPUT_ROOT:-}"
SLIME_DIR="${SLIME_DIR:-${SPILOT_ROOT}/src/slime}"
PYTHON_BIN="${TMAX_HF_EXPORT_PYTHON:-${USER_ROOT}/.python/polar/bin/python3}"
MEGATRON_DIR="${MEGATRON_DIR:-${USER_ROOT}/spilot-router/data/Megatron-LM-slime-v0.3.0}"
ORIGIN_HF_DIR="${TMAX_HF_EXPORT_ORIGIN:-}"
CHUNK_SIZE="${TMAX_HF_EXPORT_CHUNK_SIZE:-5368709120}"

die() { echo "ERROR: $*" >&2; exit 1; }

require_absolute_path() {
    local name="$1" value="$2"
    case "${value}" in
        /*) ;;
        *) die "${name} must be an absolute path: ${value:-<empty>}" ;;
    esac
}

configure_paths() {
    [ -n "${ORIGIN_HF_DIR}" ] || die "set TMAX_HF_EXPORT_ORIGIN to the source Hugging Face model directory"
    require_absolute_path POLAR_DATA_ROOT "${DATA_ROOT}"
    require_absolute_path SLIME_DIR "${SLIME_DIR}"
    require_absolute_path TMAX_HF_EXPORT_PYTHON "${PYTHON_BIN}"
    require_absolute_path MEGATRON_DIR "${MEGATRON_DIR}"
    require_absolute_path TMAX_HF_EXPORT_ORIGIN "${ORIGIN_HF_DIR}"
    if [ -z "${CKPT_ROOT}" ]; then
        [ -n "${RUN_ID}" ] || die "set TMAX_HF_EXPORT_RUN_ID or TMAX_HF_EXPORT_CKPT_ROOT"
        CKPT_ROOT="${DATA_ROOT}/ckpt/${RUN_ID}"
    elif [ -z "${RUN_ID}" ]; then
        RUN_ID="$(basename -- "${CKPT_ROOT}")"
    fi
    if [ -z "${OUTPUT_ROOT}" ]; then
        OUTPUT_ROOT="${DATA_ROOT}/hf_ckpt/${RUN_ID}"
    fi
    require_absolute_path TMAX_HF_EXPORT_CKPT_ROOT "${CKPT_ROOT}"
    require_absolute_path TMAX_HF_EXPORT_OUTPUT_ROOT "${OUTPUT_ROOT}"
}

normalize_iteration() {
    local value="${1#iter_}"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "invalid iteration: $1"
    while [ "${#value}" -gt 1 ] && [[ "${value}" == 0* ]]; do value="${value#0}"; done
    printf '%s\n' "${value}"
}

validate_input() {
    local input_dir="$1" validation_error
    [ -s "${input_dir}/common.pt" ] || die "missing ${input_dir}/common.pt"
    [ -s "${input_dir}/.metadata" ] || die "missing ${input_dir}/.metadata"
    [ -x "${PYTHON_BIN}" ] || die "Python is not executable: ${PYTHON_BIN}"
    if ! validation_error="$("${PYTHON_BIN}" "${SCRIPT_DIR}/validate_torch_dist_checkpoint.py" "${input_dir}" 2>&1)"; then
        die "invalid torch distributed checkpoint ${input_dir}: ${validation_error}"
    fi
    [ -f "${SLIME_DIR}/tools/convert_torch_dist_to_hf.py" ] || die "Slime converter not found"
    [ -s "${ORIGIN_HF_DIR}/config.json" ] || die "origin HF snapshot not found: ${ORIGIN_HF_DIR}"
    [ -s "${ORIGIN_HF_DIR}/model.safetensors.index.json" ] || die "origin HF index not found"
}

run_worker() {
    [ "$#" -eq 3 ] || die "internal worker expected INPUT OUTPUT ITERATION"
    local input_dir="$1" output_dir="$2" iteration="$3"
    local staging_dir="${output_dir}.tmp-${SLURM_JOB_ID:-$$}"
    local ready=0 origin_model_type origin_vocab_size

    validate_input "${input_dir}"
    read -r origin_model_type origin_vocab_size < <(
        "${PYTHON_BIN}" - "${ORIGIN_HF_DIR}/config.json" <<'PY'
import json
import sys

def config_value(config, name):
    value = config.get(name)
    text_config = config.get("text_config")
    if value is None and isinstance(text_config, dict):
        value = text_config.get(name)
    return value


config = json.load(open(sys.argv[1], encoding="utf-8"))
model_type = config_value(config, "model_type")
vocab_size = config_value(config, "vocab_size")
if not isinstance(model_type, str) or not model_type:
    raise SystemExit("origin config has no model_type")
if not isinstance(vocab_size, int) or vocab_size <= 0:
    raise SystemExit("origin config has no positive vocab_size")
print(model_type, vocab_size)
PY
    )
    echo "[tmax hf export] origin model=${origin_model_type} vocab_size=${origin_vocab_size}"
    mkdir -p "$(dirname -- "${output_dir}")"
    command -v flock >/dev/null || die "flock is required"
    exec 9>"${output_dir}.lock"
    flock -n 9 || die "another export owns ${output_dir}.lock"

    if [ -s "${output_dir}/.export_complete.json" ]; then
        echo "[tmax hf export] already ready: ${output_dir}"
        return
    fi
    [ ! -e "${output_dir}" ] || die "refusing to overwrite incomplete output: ${output_dir}"
    case "${staging_dir}" in "${output_dir}.tmp-"*) ;; *) die "unsafe staging path" ;; esac
    rm -rf -- "${staging_dir}"
    cleanup() {
        local rc=$?
        trap - EXIT
        if [ "${ready}" != 1 ] && [ -e "${staging_dir}" ]; then rm -rf -- "${staging_dir}"; fi
        exit "${rc}"
    }
    trap cleanup EXIT

    echo "[tmax hf export] input=${input_dir}"
    echo "[tmax hf export] staging=${staging_dir}"
    cd "${SLIME_DIR}"
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" tools/convert_torch_dist_to_hf.py \
        --input-dir "${input_dir}" \
        --output-dir "${staging_dir}" \
        --origin-hf-dir "${ORIGIN_HF_DIR}" \
        --add-missing-from-origin-hf \
        --vocab-size "${origin_vocab_size}" \
        --chunk-size "${CHUNK_SIZE}"

    # Compare the complete HF key set and inspect headers without loading tensors.
    "${PYTHON_BIN}" - "${staging_dir}" "${ORIGIN_HF_DIR}" "${input_dir}" "${RUN_ID}" "${iteration}" <<'PY'
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import sys

out, origin, source = map(Path, sys.argv[1:4])
run_id, iteration = sys.argv[4], int(sys.argv[5])
load = lambda p: json.loads(p.read_text(encoding="utf-8"))


def config_value(config, name):
    value = config.get(name)
    text_config = config.get("text_config")
    if value is None and isinstance(text_config, dict):
        value = text_config.get(name)
    return value


out_map = load(out / "model.safetensors.index.json")["weight_map"]
origin_map = load(origin / "model.safetensors.index.json")["weight_map"]

missing = sorted(set(origin_map) - set(out_map))
extra = sorted(set(out_map) - set(origin_map))
if missing or extra:
    raise RuntimeError(f"weight-key mismatch: missing={missing[:20]} extra={extra[:20]}")

seen, tensor_dtypes, dtypes, weight_bytes = {}, {}, Counter(), 0
for name in sorted(set(out_map.values())):
    path = out / name
    if not path.is_file():
        raise RuntimeError(f"missing shard: {path}")
    weight_bytes += path.stat().st_size
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    for key, metadata in header.items():
        if key == "__metadata__":
            continue
        if key in seen:
            raise RuntimeError(f"duplicate tensor: {key}")
        seen[key] = name
        tensor_dtypes[key] = metadata["dtype"]
        dtypes[metadata["dtype"]] += 1

if seen != out_map:
    raise RuntimeError("safetensors index does not match shard contents")
unexpected_dtypes = {
    key: dtype
    for key, dtype in tensor_dtypes.items()
    if dtype != "BF16" and not (key == "lm_head.weight" and dtype == "F32")
}
if unexpected_dtypes:
    raise RuntimeError(
        f"unexpected tensor dtypes: {unexpected_dtypes}; counts={dict(dtypes)}"
    )
config = load(out / "config.json")
origin_config = load(origin / "config.json")
for field in ("model_type", "vocab_size"):
    output_value = config_value(config, field)
    origin_value = config_value(origin_config, field)
    if output_value != origin_value:
        raise RuntimeError(
            f"unexpected {field}: {output_value!r} != {origin_value!r}"
        )
for asset in ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
    if not (out / asset).is_file():
        raise RuntimeError(f"missing HF asset: {asset}")

manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "iteration": iteration,
    "source_checkpoint": str(source),
    "origin_hf_dir": str(origin),
    "model_type": config_value(config, "model_type"),
    "vocab_size": config_value(config, "vocab_size"),
    "weight_key_count": len(out_map),
    "weight_shard_count": len(set(out_map.values())),
    "weight_bytes": weight_bytes,
    "dtype_counts": dict(dtypes),
}
(out / ".export_complete.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"validated keys={len(out_map)} shards={manifest['weight_shard_count']} dtypes={dict(dtypes)}")
PY

    mv -- "${staging_dir}" "${output_dir}"
    ready=1
    trap - EXIT
    echo "[tmax hf export] ready: ${output_dir}"
}

if [ "${1:-}" = "--worker" ]; then
    shift
    configure_paths
    run_worker "$@"
    exit 0
fi

dry_run=0
case "${1:-}" in
    -h|--help)
        echo "Usage: TMAX_HF_EXPORT_RUN_ID=RUN_ID $0 [latest|ITERATION]"
        echo "       TMAX_HF_EXPORT_RUN_ID=RUN_ID $0 --dry-run [latest|ITERATION]"
        echo "       TMAX_HF_EXPORT_CKPT_ROOT=DIR $0 [latest|ITERATION]"
        exit 0
        ;;
    --dry-run)
        dry_run=1
        shift
        ;;
esac
configure_paths
[ "$#" -le 1 ] || die "expected at most one iteration argument"
requested="${1:-latest}"
if [ "${requested}" = latest ]; then
    [ -s "${CKPT_ROOT}/latest_checkpointed_iteration.txt" ] || die "latest checkpoint tracker not found"
    requested="$(tr -d '[:space:]' <"${CKPT_ROOT}/latest_checkpointed_iteration.txt")"
fi
iteration="$(normalize_iteration "${requested}")"
printf -v iteration_dir 'iter_%07d' "${iteration}"
input_dir="${CKPT_ROOT}/${iteration_dir}"
output_dir="${OUTPUT_ROOT}/${iteration_dir}-bf16"
validate_input "${input_dir}"

if [ -s "${output_dir}/.export_complete.json" ]; then
    echo "[tmax hf export] already ready: ${output_dir}"
    exit 0
fi
[ ! -e "${output_dir}" ] || die "refusing to overwrite incomplete output: ${output_dir}"

log_dir="${TMAX_HF_EXPORT_LOG_DIR:-${DATA_ROOT}/logs/slurm}"
require_absolute_path TMAX_HF_EXPORT_LOG_DIR "${log_dir}"
mkdir -p "${log_dir}"
slurm_export_values=(
    "TMAX_HF_EXPORT_RUN_ID=${RUN_ID}"
    "POLAR_DATA_ROOT=${DATA_ROOT}"
    "TMAX_HF_EXPORT_CKPT_ROOT=${CKPT_ROOT}"
    "TMAX_HF_EXPORT_OUTPUT_ROOT=${OUTPUT_ROOT}"
    "SLIME_DIR=${SLIME_DIR}"
    "TMAX_HF_EXPORT_PYTHON=${PYTHON_BIN}"
    "MEGATRON_DIR=${MEGATRON_DIR}"
    "TMAX_HF_EXPORT_ORIGIN=${ORIGIN_HF_DIR}"
    "TMAX_HF_EXPORT_CHUNK_SIZE=${CHUNK_SIZE}"
)
for export_value in "${slurm_export_values[@]}"; do
    case "${export_value}" in
        *','*|*$'\n'*) die "Slurm export values cannot contain commas or newlines" ;;
    esac
done
slurm_export_spec="$(IFS=,; printf '%s' "${slurm_export_values[*]}")"
sbatch_cmd=(
    sbatch --nodes=1 --ntasks=1 --ntasks-per-node=1
    --account="${TMAX_HF_EXPORT_ACCOUNT:-nvr_lpr_llm}"
    --job-name="tmax-hf-${iteration}"
    --partition="${TMAX_HF_EXPORT_PARTITION:-interactive,batch_short,backfill,batch}"
    --time="${TMAX_HF_EXPORT_WALL_TIME:-00:30:00}"
    --cpus-per-task="${TMAX_HF_EXPORT_CPUS:-32}"
    --mem=0 --gres="gpu:${TMAX_HF_EXPORT_GPUS:-1}" --constraint="${TMAX_HF_EXPORT_CONSTRAINT:-H100}"
    --chdir="${PROJECT_ROOT}" --export="${slurm_export_spec}" --parsable
    --output="${log_dir}/%x-%j.out" --error="${log_dir}/%x-%j.err"
)
worker_args=("${SELF}" --worker "${input_dir}" "${output_dir}" "${iteration}")

echo "[tmax hf export] ${input_dir} -> ${output_dir}"
if [ "${dry_run}" = 1 ]; then
    printf '[tmax hf export] dry-run:'; printf ' %q' "${sbatch_cmd[@]}" "${worker_args[@]}"; printf '\n'
    exit 0
fi

job_id="$("${sbatch_cmd[@]}" "${worker_args[@]}")"
job_id="${job_id%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || die "invalid sbatch response: ${job_id}"
echo "[tmax hf export] submitted job ${job_id}"
echo "[tmax hf export] logs: ${log_dir}/tmax-hf-${iteration}-${job_id}.{out,err}"
