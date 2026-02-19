#!/bin/bash

#SBATCH --array=0%1
#SBATCH --partition=interactive
#SBATCH --account=llmservice_fm_vision
#SBATCH --reservation=sla_res_osworld_agent_vlm
#SBATCH --gpus-per-node=8
#SBATCH --job-name=cua-check_kvm_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm-kvm-gpu-%A.out
#SBATCH --error=logs/slurm-kvm-gpu-%A.out

# ============================================================================
# Test KVM availability on GPU partition nodes
# This validates that /dev/kvm is accessible inside a container on GPU nodes,
# which is required for the consolidated 2-node setup.
# ============================================================================

CONTAINER_IMAGE=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/images/cua_vllm.sqsh

# 1. Identify the Compute Node
TARGET_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
echo "[Script] Job assigned to GPU node: $TARGET_NODE"

# 1.5. Check if /dev/kvm exists on the node (before container)
echo "[Script] Checking if /dev/kvm exists on $TARGET_NODE (host level)..."
ssh -q -o StrictHostKeyChecking=no "$TARGET_NODE" "
    echo '--- Host-level KVM check ---'
    if [ -e /dev/kvm ]; then
        echo '/dev/kvm EXISTS on host'
        ls -la /dev/kvm
        stat /dev/kvm
        getent group kvm 2>/dev/null || echo 'kvm group not found'
    else
        echo 'WARNING: /dev/kvm does NOT exist on this GPU node!'
        echo 'KVM may not be enabled on GPU partition nodes.'
        echo 'This is a HARD BLOCKER for the consolidated setup.'
    fi
    echo '--- End host-level check ---'
"

# 2. Launch Container on that Node (with /dev/kvm mount)
echo "[Script] Launching container (sleep infinity) on $TARGET_NODE..."
srun --nodelist="$TARGET_NODE" \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts=/lustre:/lustre,/dev/kvm:/dev/kvm \
     --container-writable \
     sleep infinity &

SRUN_PID=$!
trap "kill $SRUN_PID" EXIT

# 3. Wait/Poll for Enroot PID (via SSH)
echo "[Script] SSH-ing into $TARGET_NODE to find PID..."
CONTAINER_PID=""
while [ -z "$CONTAINER_PID" ]; do
    sleep 2

    CONTAINER_PID=$(ssh -q -o StrictHostKeyChecking=no "$TARGET_NODE" \
        "enroot list -f | grep 'pyxis' | grep 'sleep' | awk '{print \$2}' | head -n 1")

    if [ -z "$CONTAINER_PID" ]; then
        echo "[Script] Container not yet visible. Retrying..."
    fi
done

echo "[Script] Found PID: $CONTAINER_PID on $TARGET_NODE"

# 4. Execute KVM checks inside container
echo "[Script] Checking KVM inside container on GPU node..."
ssh -o StrictHostKeyChecking=no "$TARGET_NODE" "enroot exec $CONTAINER_PID bash -c '
    echo \"--- Container-level KVM check (GPU node) ---\"
    echo \"Running as user: \$(whoami) (uid=\$(id -u), gid=\$(id -g))\"
    echo \"User groups: \$(groups)\"
    echo \"\"

    if [ -e /dev/kvm ]; then
        echo \"/dev/kvm EXISTS in container\"
        ls -la /dev/kvm

        if [ -r /dev/kvm ]; then
            echo \"READ permission: YES\"
        else
            echo \"READ permission: NO\"
        fi

        if [ -w /dev/kvm ]; then
            echo \"WRITE permission: YES\"
            echo \"\"
            echo \"SUCCESS: /dev/kvm is writable on GPU node!\"
            echo \"The consolidated 2-node setup should work.\"
        else
            echo \"WRITE permission: NO\"
            echo \"\"
            echo \"FAILURE: Cannot write to /dev/kvm on GPU node\"
            echo \"Admin intervention may be needed.\"
            exit 1
        fi
    else
        echo \"FAILURE: /dev/kvm does NOT exist in container\"
        echo \"The /dev/kvm mount may have failed or KVM is not available on this GPU node.\"
        exit 1
    fi

    echo \"\"
    echo \"--- GPU check ---\"
    nvidia-smi -L 2>/dev/null || echo \"nvidia-smi not available (expected in non-CUDA container)\"
    echo \"--- End checks ---\"
'"
