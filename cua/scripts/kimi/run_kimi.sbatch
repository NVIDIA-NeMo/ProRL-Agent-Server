#!/bin/bash
#SBATCH --job-name=gpu_server
#SBATCH --account=llmservice_fm_vision
#SBATCH --partition=batch_block1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm-%A-server.out
#SBATCH --error=logs/slurm-%A-server.out
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper": {"exemptIdleTimeMins": "240", "reason": "model_loading", "description": "Required model loading will take 90 minutes."}}'

# parameters
TOTAL_TP=$(( SLURM_JOB_NUM_NODES * 8 ))
TOTAL_PP=1
#TOTAL_TP=8
#TOTAL_PP=$SLURM_JOB_NUM_NODES
# CONTAINER_IMAGE="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/images/cua-vllm-0.16.0.sqsh"
CONTAINER_IMAGE="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/bcui/images/cua-vllm-0.16.0.sqsh"
CONTAINER_MOUNTS="/lustre:/lustre,/tmp:/tmp"


nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=( $nodes )
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
ray_port=6379
api_port=8000

echo "[INFO] VLLM Cluster: $SLURM_JOB_NUM_NODES nodes | Total TP: $TOTAL_TP | Total PP: $TOTAL_PP"
echo "[INFO] VLLM Head Node: $head_node ($head_node_ip)"

# Write head node to a file so launcher scripts can discover it reliably
HEAD_NODE_FILE="${LOG_DIR:-./logs}/head_node_${SLURM_JOB_ID}"
echo "$head_node" > "$HEAD_NODE_FILE"
echo "[INFO] Wrote head node to $HEAD_NODE_FILE"


# start ray head node
echo "[INFO] Starting VLLM Head on $head_node ($head_node_ip)"
srun --nodes=1 --ntasks=1 -w "$head_node" \
  --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
  bash -c \
  "
  export VLLM_HOST_IP=$head_node_ip;
  export RAY_CGRAPH_get_timeout=600;
  export NCCL_IB_HCA=mlx5; export NCCL_IB_DISABLE=0;
  ray start --head --node-ip-address=$head_node_ip --port=$ray_port --runtime-env-agent-port 9999 \
  --metrics-export-port 9998 --dashboard-agent-grpc-port 9997 --block
  " &

sleep 30

# start ray worker nodes
if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
  for worker_node in "${nodes_array[@]:1}"; do
    echo "[INFO] Starting VLLM Worker on $worker_node"
    srun --nodes=1 --ntasks=1 -w "$worker_node" \
    --container-image="$CONTAINER_IMAGE" --container-mounts="$CONTAINER_MOUNTS" \
    bash -c \
    "
    MY_IP=\$(hostname --ip-address);
    export VLLM_HOST_IP=\$MY_IP;
    export RAY_CGRAPH_get_timeout=600;
    echo \"[Worker] IP: \$MY_IP connected to Head: $head_node_ip\";
    ray start --address=$head_node_ip:$ray_port --node-ip-address=\$MY_IP --runtime-env-agent-port 9999 \
    --metrics-export-port 9998 --dashboard-agent-grpc-port 9997 --block
    " &
  done
fi

sleep 60

# === Launching vLLM server === #
echo "[INFO] Launching VLLM Server on $head_node with TP=$TOTAL_TP, PP=$TOTAL_PP..."
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  --container-image="$CONTAINER_IMAGE"  --container-mounts="$CONTAINER_MOUNTS" \
  bash -c \
  "
  export VLLM_HOST_IP=$head_node_ip;
  vllm serve /lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/Kimi-K2.5 \
    -tp $TOTAL_TP \
    -pp $TOTAL_PP \
    --host 0.0.0.0 \
    --port $api_port \
    --mm-encoder-tp-mode data \
    --compilation_config.pass_config.fuse_allreduce_rms true \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --enable-auto-tool-choice \
    --limit-mm-per-prompt.video 0 \
    --limit-mm-per-prompt.image 3 \
    --max-model-len 65536 \
    --trust-remote-code \
    --enable-expert-parallel \
    --distributed-executor-backend ray \
    --api-key gen
  "






