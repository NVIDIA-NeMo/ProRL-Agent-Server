echo "Setting up environment..."
export PYTHONPATH=/lustre/fsw/portfolios/nvr/users/mingjiel/OpenHands_internal
pip install git+https://github.com/SWE-Gym/SWE-Bench-Package.git

echo "Launching servers..."
export VLLM_LOGGING_LEVEL=CRITICAL
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1 --host 127.0.0.1 --port 8000 --api-key mykey  --disable-log-requests --tensor-parallel-size 2 &
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1 --host 127.0.0.1 --port 8001 --api-key mykey  --disable-log-requests --tensor-parallel-size 2 &
CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1 --host 127.0.0.1 --port 8002 --api-key mykey  --disable-log-requests --tensor-parallel-size 2 &
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-14B --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1 --host 127.0.0.1 --port 8003 --api-key mykey  --disable-log-requests --tensor-parallel-size 2 &

echo "Waiting 60 seconds before checking if servers are ready..."
sleep 60
# Wait for all servers to start up
echo "Waiting for all servers to start on ports 8000, 8001, 8002, 8003..."
while true; do
    all_ready=true

    # Check each server
    for port in 8000 8001 8002 8003; do
        if curl -s -f http://127.0.0.1:$port/health > /dev/null 2>&1; then
            echo "Server on port $port is ready!"
        else
            echo "Server on port $port not ready yet..."
            all_ready=false
        fi
    done

    if [ "$all_ready" = true ]; then
        echo "All servers are ready!"
        break
    else
        echo "Waiting 10 seconds before checking again..."
        sleep 10
    fi
done

echo "Setup complete. Servers are running..."
