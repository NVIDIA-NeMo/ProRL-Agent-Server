#!/bin/bash
# ============================================================================
# 2-Node Batch Trajectory Collection
# ============================================================================
# Node 0 (Planner): Runs Qwen3-VL-235B vLLM server (tp=8, all 8 GPUs)
# Node 1 (Actor):   2x UI-TARS-1.5-7B vLLM servers (tp=4 each, GPUs 0-3 + 4-7)
#                    + round-robin load balancer on port 8000
#                    + trajectory collection
#
# Usage: sbatch collect_trajectories_sbatch.sh
# ============================================================================

#SBATCH --job-name=traj_collect
#SBATCH --account=llmservice_fm_vision
#SBATCH --reservation=sla_res_osworld_agent_vlm
#SBATCH --partition=batch_block1
#SBATCH --gpus-per-node=8
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/bcui/ProRL-Agent-Server/cua/scripts/logs-refactor/traj_collect_%j.out
#SBATCH --error=/lustre/fsw/portfolios/nvr/users/bcui/ProRL-Agent-Server/cua/scripts/logs-refactor/traj_collect_%j.err

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
IMAGE="/lustre/fsw/portfolios/nvr/users/bcui/images/cua-vllm-0.13.0.sqsh"
PROJECT_ROOT="/lustre/fsw/portfolios/nvr/users/bcui/ProRL-Agent-Server"
PROJECT_DIR="$PROJECT_ROOT/cua"

PLANNER_MODEL="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/Qwen3-VL-235B-A22B-Thinking"
ACTOR_MODEL="/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/bcui/huggingface_models/UI-TARS-1.5-7B"

PLANNER_PORT=8000
ACTOR_PORT=8000          # Load balancer port (what the code talks to)
ACTOR_PORT_1=8001        # Actor replica 1 (GPUs 0-3)
ACTOR_PORT_2=8002        # Actor replica 2 (GPUs 4-7)

# Trajectory collection settings
MAX_PARALLEL=16
MAX_TRAJECTORIES=1024

# Log file name (timestamped)
TIMESTAMP=$(date +%m-%d-%H%M)
LOG_FILE="$PROJECT_DIR/${TIMESTAMP}-logs.log"

# ============================================================================
# Resolve node assignments
# ============================================================================
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
PLANNER_NODE=$(echo "$ALL_NODES" | head -n 1)
ACTOR_NODE=$(echo "$ALL_NODES" | tail -n 1)

echo "[sbatch] Job ID: $SLURM_JOB_ID"
echo "[sbatch] Planner Node: $PLANNER_NODE"
echo "[sbatch] Actor Node:   $ACTOR_NODE"
echo "[sbatch] Log file: $LOG_FILE"

mkdir -p "$PROJECT_DIR/scripts/logs"

# ============================================================================
# Launch Planner vLLM server on Node 0
# ============================================================================
echo "[sbatch] Starting Planner vLLM server on $PLANNER_NODE..."
srun --nodes=1 --ntasks=1 --nodelist="$PLANNER_NODE" \
    --container-image="$IMAGE" \
    --container-mounts=/lustre:/lustre \
    --container-writable \
    bash -c "
        vllm serve $PLANNER_MODEL \
            --api-key gen \
            --tensor-parallel-size 8 \
            --enable-expert-parallel \
            --limit-mm-per-prompt.video 0 \
            --limit-mm-per-prompt.image 3 \
            --async-scheduling \
            --max-model-len 65536 \
            --gpu-memory-utilization 0.9 \
            > $PROJECT_DIR/scripts/logs/planner_${SLURM_JOB_ID}.log 2>&1
    " &
PLANNER_SRUN_PID=$!

# ============================================================================
# Launch Actor vLLM server + trajectory collection on Node 1
# ============================================================================
echo "[sbatch] Starting 2x Actor vLLM servers + collection on $ACTOR_NODE..."
srun --nodes=1 --ntasks=1 --nodelist="$ACTOR_NODE" \
    --container-image="$IMAGE" \
    --container-mounts=/lustre:/lustre \
    --container-writable \
    bash -c "
        # --- Start Actor vLLM replica 1 (GPUs 0-3) ---
        CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $ACTOR_MODEL \
            --served-model-name ByteDance-Seed/UI-TARS-1.5-7B \
            --api-key gen \
            --port $ACTOR_PORT_1 \
            --tensor-parallel-size 4 \
            --limit-mm-per-prompt.image 5 \
            --limit-mm-per-prompt.video 0 \
            --max-model-len 65536 \
            --disable-log-requests \
            --disable-log-stats \
            > $PROJECT_DIR/scripts/logs/actor1_${SLURM_JOB_ID}.log 2>&1 &
        ACTOR1_PID=\$!

        # --- Start Actor vLLM replica 2 (GPUs 4-7) ---
        CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve $ACTOR_MODEL \
            --served-model-name ByteDance-Seed/UI-TARS-1.5-7B \
            --api-key gen \
            --port $ACTOR_PORT_2 \
            --tensor-parallel-size 4 \
            --limit-mm-per-prompt.image 5 \
            --limit-mm-per-prompt.video 0 \
            --max-model-len 65536 \
            --disable-log-requests \
            --disable-log-stats \
            > $PROJECT_DIR/scripts/logs/actor2_${SLURM_JOB_ID}.log 2>&1 &
        ACTOR2_PID=\$!

        # --- Start round-robin load balancer on port $ACTOR_PORT ---
        python3 -c '
import http.server, http.client, threading, sys, io

backends = [(\"localhost\", $ACTOR_PORT_1), (\"localhost\", $ACTOR_PORT_2)]
counter = 0
lock = threading.Lock()

class LBHandler(http.server.BaseHTTPRequestHandler):
    def do_ANY(self, method):
        global counter
        with lock:
            host, port = backends[counter % len(backends)]
            counter += 1

        content_length = int(self.headers.get(\"Content-Length\", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        try:
            conn = http.client.HTTPConnection(host, port, timeout=300)
            conn.request(method, self.path, body=body, headers=dict(self.headers))
            resp = conn.getresponse()
            resp_body = resp.read()

            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in (\"transfer-encoding\",):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(resp_body)
            conn.close()
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f\"LB error: {e}\".encode())

    def do_GET(self): self.do_ANY(\"GET\")
    def do_POST(self): self.do_ANY(\"POST\")
    def do_PUT(self): self.do_ANY(\"PUT\")
    def do_DELETE(self): self.do_ANY(\"DELETE\")
    def log_message(self, format, *args): pass  # silence logs

server = http.server.ThreadingHTTPServer((\"0.0.0.0\", $ACTOR_PORT), LBHandler)
print(f\"[LB] Round-robin load balancer on port $ACTOR_PORT -> {backends}\", flush=True)
server.serve_forever()
' > $PROJECT_DIR/scripts/logs/actor_lb_${SLURM_JOB_ID}.log 2>&1 &
        LB_PID=\$!

        # --- Wait for all servers to be healthy ---
        echo '[actor-node] Waiting for vLLM servers to become healthy...'

        wait_for_server() {
            local host=\$1
            local port=\$2
            local name=\$3
            local max_wait=600
            local elapsed=0

            while [ \$elapsed -lt \$max_wait ]; do
                if curl -sf http://\${host}:\${port}/health > /dev/null 2>&1; then
                    echo \"[actor-node] \$name server healthy (\${elapsed}s)\"
                    return 0
                fi
                sleep 10
                elapsed=\$((elapsed + 10))
                if [ \$((elapsed % 60)) -eq 0 ]; then
                    echo \"[actor-node] Still waiting for \$name (\${elapsed}s)...\"
                fi
            done
            echo \"[actor-node] ERROR: \$name server did not start within \${max_wait}s\"
            return 1
        }

        # Wait for both actor replicas and planner
        wait_for_server localhost $ACTOR_PORT_1 'Actor-1 (GPU 0-3)'
        ACTOR1_OK=\$?

        wait_for_server localhost $ACTOR_PORT_2 'Actor-2 (GPU 4-7)'
        ACTOR2_OK=\$?

        wait_for_server $PLANNER_NODE $PLANNER_PORT Planner
        PLANNER_OK=\$?

        if [ \$ACTOR1_OK -ne 0 ] || [ \$ACTOR2_OK -ne 0 ] || [ \$PLANNER_OK -ne 0 ]; then
            echo '[actor-node] ERROR: One or more servers failed to start.'
            echo 'Planner log:' && tail -20 $PROJECT_DIR/scripts/logs/planner_${SLURM_JOB_ID}.log 2>/dev/null
            echo 'Actor-1 log:' && tail -20 $PROJECT_DIR/scripts/logs/actor1_${SLURM_JOB_ID}.log 2>/dev/null
            echo 'Actor-2 log:' && tail -20 $PROJECT_DIR/scripts/logs/actor2_${SLURM_JOB_ID}.log 2>/dev/null
            kill \$ACTOR1_PID \$ACTOR2_PID \$LB_PID 2>/dev/null
            exit 1
        fi

        echo '[actor-node] All servers healthy. Starting trajectory collection...'

        # --- Run trajectory collection ---
        cd $PROJECT_DIR
        source cua_env_reqs/bin/activate
        export PYTHONPATH=$PROJECT_ROOT:\$PYTHONPATH
        export OSWORLD_SETUP_CACHE_DIR=/tmp/osworld_cache

        python parallel_collect_trajectories.py \
            --planner_node $PLANNER_NODE \
            --actor_node $ACTOR_NODE \
            --runtime nvcf \
            --max_parallel $MAX_PARALLEL \
            --max_trajectories $MAX_TRAJECTORIES \
            2>&1 | tee $LOG_FILE

        COLLECT_EXIT=\$?

        # Cleanup
        kill \$ACTOR1_PID \$ACTOR2_PID \$LB_PID 2>/dev/null
        echo \"[actor-node] Collection finished (exit code: \$COLLECT_EXIT)\"
        exit \$COLLECT_EXIT
    " &
ACTOR_SRUN_PID=$!

# ============================================================================
# Wait for completion
# ============================================================================
# Wait for the actor srun (which runs collection). When it finishes, kill planner.
wait $ACTOR_SRUN_PID
COLLECT_EXIT=$?

echo "[sbatch] Actor node finished (exit: $COLLECT_EXIT). Stopping planner..."
kill $PLANNER_SRUN_PID 2>/dev/null
wait $PLANNER_SRUN_PID 2>/dev/null

echo "[sbatch] Done. Log: $LOG_FILE"
exit $COLLECT_EXIT
