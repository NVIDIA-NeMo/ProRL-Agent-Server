# shellcheck shell=bash
# Qwen3.5-9B Megatron model args used by both checkpoint conversion and
# training. Keep this in sync with slime/scripts/models/qwen3.5-9B.sh.
#
# The Hugging Face checkpoint is a VLM container, while Polar trains its text
# backbone. slime_plugins.mbridge.qwen3_5 reads the nested text_config. The 9B
# checkpoint has tie_word_embeddings=false, so the untied flag is required.
# Slime's current model snippet targets a newer Megatron/TE combination.  The
# pinned training image needs the compatibility substitutions below: an
# explicit linear-attention frequency, unfused RoPE, and non-persistent
# RMSNorm kernels. Qwen3.5 stores its RMSNorm gamma as a zero-centred delta
# (the Hugging Face forward multiplies by ``1 + weight``), so
# --apply-layernorm-1p is required even when persistent layernorm kernels are
# disabled. Omitting it makes the Megatron policy numerically unrelated to
# SGLang while all checkpoint keys still appear to load successfully.
#
# This file is deliberately shared by conversion and training; the old 4B path
# used different flags for the two phases and loaded missing checkpoint keys as
# a result.
MODEL_ARGS=(
    --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
    --disable-bias-linear
    --qk-layernorm
    --group-query-attention
    --num-attention-heads 16
    --num-query-groups 4
    --kv-channels 256
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 12288
    --linear-attention-freq 4
    --normalization RMSNorm
    --apply-layernorm-1p
    --position-embedding-type rope
    --no-rope-fusion
    --no-persist-layer-norm
    --norm-epsilon 1e-6
    --rotary-percent 0.25
    --swiglu
    --untie-embeddings-and-output-weights
    --vocab-size 248320
    --rotary-base 10000000
    --attention-output-gate
)
