#!/usr/bin/env python3
"""Single-GPU HF/Megatron log-prob parity smoke test for Qwen3.5.

The test intentionally loads the two implementations sequentially so that a
9B BF16 model fits on one 80GB GPU.  The Megatron half is constructed with the
same ``model_args.sh`` and loaded from the same torch_dist checkpoint as the
trainer, so this catches both conversion and runtime-model configuration bugs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
SPILOT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_ARGS_FILE = HERE / "model_args.sh"
DEFAULT_HF_CHECKPOINT = SPILOT_ROOT / "data/checkpoints/Qwen3.5-9B"
DEFAULT_TORCH_DIST = SPILOT_ROOT / "data/checkpoints/Qwen3.5-9B_torch_dist"


def _argv_value(name: str, default: str) -> str:
    for index, value in enumerate(sys.argv[1:]):
        if value == name:
            try:
                return sys.argv[index + 2]
            except IndexError as exc:
                raise SystemExit(f"{name} requires a value") from exc
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return default


def _source_model_args(path: str) -> list[str]:
    command = r'''
source "$1"
declare -p MODEL_ARGS >/dev/null 2>&1 || {
    echo "$1 did not define MODEL_ARGS" >&2
    exit 2
}
printf '%s\0' "${MODEL_ARGS[@]}"
'''
    result = subprocess.run(
        ["bash", "-c", command, "qwen35-parity", path],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [item.decode() for item in result.stdout.split(b"\0") if item]


# Put the shared model flags before explicit CLI flags so a diagnostic caller
# can still override one deliberately.  Megatron's parser sees one coherent
# argv and therefore constructs exactly the same model as conversion/training.
_MODEL_ARGS_FILE = _argv_value("--model-args-file", str(DEFAULT_MODEL_ARGS_FILE))
sys.argv[1:1] = _source_model_args(_MODEL_ARGS_FILE)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from megatron.core.enums import ModelType  # noqa: E402
from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402
from megatron.training.arguments import parse_args, validate_args  # noqa: E402
from megatron.training.training import get_model  # noqa: E402

import slime_plugins.mbridge  # noqa: E402,F401
from slime.backends.megatron_utils.arguments import set_default_megatron_args  # noqa: E402
from slime.backends.megatron_utils.checkpoint import load_checkpoint  # noqa: E402
from slime.backends.megatron_utils.initialize import init  # noqa: E402
from slime.backends.megatron_utils.model_provider import get_model_provider_func  # noqa: E402


def _add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hf-checkpoint", type=str, default=str(DEFAULT_HF_CHECKPOINT))
    # ``model_provider`` selects its construction path from this Slime
    # conversion argument.  The normal trainer parser supplies it, but this
    # standalone smoke test uses Megatron's base parser directly.
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=("raw", "bridge"),
        default="raw",
    )
    parser.add_argument("--model-args-file", type=str, default=str(DEFAULT_MODEL_ARGS_FILE))
    parser.add_argument(
        "--token-ids",
        type=str,
        default="151644,8948,198,151645,198,151644,872,198,9707,11,1879,30,151645,198",
        help="Fixed comma-separated token stream; positions 1..N-1 are scored.",
    )
    # Eager avoids Transformers' first-run download of a ~300 MB remote
    # FlashAttention kernel inside the job-local container HOME.  This is a
    # correctness smoke over a 14-token sequence, so the slower kernel is both
    # sufficient and more reproducible.
    parser.add_argument("--hf-attn-implementation", default="eager")
    parser.add_argument("--parity-max-mean-abs-diff", type=float, default=0.02)
    parser.add_argument("--parity-max-max-abs-diff", type=float, default=0.10)
    parser.add_argument("--parity-json", type=str, default=None)
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except argparse.ArgumentError:
        pass
    return parser


def _get_args():
    args = parse_args(_add_args)
    args = set_default_megatron_args(args)
    args.save_interval = 1
    args.micro_batch_size = 1
    args.global_batch_size = 1
    args.no_load_optim = True
    args.no_load_rng = True
    args.finetune = True
    args.exit_on_missing_checkpoint = True
    validate_args(args)
    if args.tensor_model_parallel_size != 1 or args.pipeline_model_parallel_size != 1:
        raise ValueError("parity smoke requires TP=1 and PP=1")
    if not args.load:
        args.load = str(DEFAULT_TORCH_DIST)
    return args


def _token_ids(raw: str, vocab_size: int) -> list[int]:
    ids = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if len(ids) < 2:
        raise ValueError("--token-ids needs at least two tokens")
    bad = [value for value in ids if value < 0 or value >= vocab_size]
    if bad:
        raise ValueError(f"token ids outside [0, {vocab_size}): {bad}")
    return ids


@torch.inference_mode()
def _hf_logprobs(args, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers import Qwen3_5ForConditionalGeneration

    print(f"[parity] loading Hugging Face model: {args.hf_checkpoint}", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.hf_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=args.hf_attn_implementation,
    ).cuda()
    model.eval()
    tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
    logits = model(input_ids=tokens, use_cache=False, logits_to_keep=0).logits.float()
    next_logits = logits[:, :-1]
    targets = tokens[:, 1:]
    logprobs = F.log_softmax(next_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    top1 = next_logits.argmax(dim=-1)
    result = logprobs.squeeze(0).cpu(), top1.squeeze(0).cpu()
    del next_logits, logits, tokens, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return result


@torch.inference_mode()
def _megatron_logprobs(args, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"[parity] building Megatron model and loading torch_dist: {args.load}", flush=True)
    model = get_model(
        get_model_provider_func(args),
        ModelType.encoder_or_decoder,
        wrap_with_ddp=False,
    )
    load_checkpoint(
        model,
        optimizer=None,
        opt_param_scheduler=None,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    model[0].eval()
    unpadded_tokens = torch.tensor([ids], dtype=torch.long, device="cuda")
    sequence_length = unpadded_tokens.size(1)
    # Match get_batch(): the packed token stream is padded to 128 and the pad
    # tail is represented as a separate sequence, so it cannot alter the real
    # sequence's recurrent GDN state.
    padded_length = ((sequence_length + 127) // 128) * 128
    tokens = F.pad(unpadded_tokens, (0, padded_length - sequence_length), value=0)
    boundaries = [0, sequence_length]
    if padded_length != sequence_length:
        boundaries.append(padded_length)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    packed = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(sequence_length, padded_length - sequence_length),
        max_seqlen_kv=max(sequence_length, padded_length - sequence_length),
        qkv_format="thd",
    )
    logits = model[0](
        input_ids=tokens,
        position_ids=None,
        attention_mask=None,
        labels=None,
        packed_seq_params=packed,
        loss_mask=torch.ones_like(tokens, dtype=torch.float32),
    ).float()
    next_logits = logits[:, : sequence_length - 1]
    targets = unpadded_tokens[:, 1:]
    logprobs = F.log_softmax(next_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    top1 = next_logits.argmax(dim=-1)
    return logprobs.squeeze(0).cpu(), top1.squeeze(0).cpu()


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5 parity smoke requires one CUDA GPU")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("launch this smoke with exactly one process/GPU")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    dist.init_process_group(
        backend="nccl",
        rank=0,
        world_size=1,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    try:
        args = _get_args()
        init(args)
        ids = _token_ids(args.token_ids, args.padded_vocab_size)
        hf_logprobs, hf_top1 = _hf_logprobs(args, ids)
        megatron_logprobs, megatron_top1 = _megatron_logprobs(args, ids)

        abs_diff = (hf_logprobs - megatron_logprobs).abs()
        mean_abs_diff = abs_diff.mean().item()
        max_abs_diff = abs_diff.max().item()
        top1_agreement = (hf_top1 == megatron_top1).float().mean().item()
        passed = (
            mean_abs_diff <= args.parity_max_mean_abs_diff
            and max_abs_diff <= args.parity_max_max_abs_diff
        )
        report = {
            "passed": passed,
            "token_ids": ids,
            "hf_logprobs": hf_logprobs.tolist(),
            "megatron_logprobs": megatron_logprobs.tolist(),
            "abs_diff": abs_diff.tolist(),
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff": max_abs_diff,
            "top1_agreement": top1_agreement,
            "thresholds": {
                "mean_abs_diff": args.parity_max_mean_abs_diff,
                "max_abs_diff": args.parity_max_max_abs_diff,
            },
        }
        print(json.dumps(report, indent=2), flush=True)
        if args.parity_json:
            output = Path(args.parity_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("[parity] PASS" if passed else "[parity] FAIL", flush=True)
        return 0 if passed else 1
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
