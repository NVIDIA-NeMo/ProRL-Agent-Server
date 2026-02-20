# OSWorld Evaluation Pipeline

This directory contains a three-stage pipeline for running OSWorld evaluations, processing results, and generating visualizations.

## Pipeline Overview

```
osworld.py → unpack.py → generate_gifs.py
   ↓              ↓              ↓
JSON files   Extracted    MP4 videos
             dirs with
             PNGs & JSON
```

## Stage 1: osworld.py - Run OSWorld Evaluation

The first stage runs OSWorld benchmark tasks using an OpenHands agent server with an LLM backend.

### Purpose
- Submits OSWorld test instances to an agent server
- Processes tasks in parallel with configurable concurrency
- Saves complete interaction logs as JSON files
- Supports resumption (skips already-completed tasks)

### Usage

```bash
python osworld.py \
    --data-file /path/to/osworld_test.json \
    --output-dir ./osworld_results \
    --llm-server-address http://localhost:8000/v1 \
    --model "hosted_vllm/Qwen/Qwen3-VL-235B-A22B-Instruct" \
    --max-parallel-jobs 2 \
    --max-iterations 15 \
    --timeout 6000
```

### Key Arguments
- `--data-file`: Path to OSWorld test data (JSONL format)
- `--output-dir`: Directory to save result JSON files (default: `./osworld_results`)
- `--llm-server-address`: LLM server endpoint (default: `http://localhost:8000/v1`)
- `--model`: Model identifier for the LLM
- `--max-parallel-jobs`: Number of concurrent tasks (default: 2)
- `--max-iterations`: Max agent iterations per task (default: 15)
- `--timeout`: Timeout per task in seconds (default: 6000)
- `--enable-vision/--no-vision`: Enable/disable vision capabilities (default: enabled)
- `--enable-a11y-tree`: Enable accessibility tree
- `--max-image-history`: Number of images to keep in context (default: 4)
- `--temperature`: Sampling temperature (default: 0.0)
- `--total-jobs`: Limit number of jobs to run (default: all)

### Output
Creates one JSON file per task in `--output-dir`:
- `<instance_id>.json`: Complete interaction log including messages, tool calls, and images
- `error_<n>.json`: Error logs for failed tasks

Each JSON contains:
- `instance_id`: Task identifier
- `messages`: Complete conversation history with the agent
- `resolved`: Boolean indicating task success
- Embedded base64-encoded screenshots

## Stage 2: unpack.py - Extract Images and Conversations

The second stage unpacks the JSON files into directories with extracted images and structured conversation data.

### Purpose
- Extracts base64-encoded images from JSON files
- Creates organized directory structure for each task
- Converts message history to readable conversation format
- Computes overall accuracy statistics

### Usage

```bash
python unpack.py \
    --dir_path ./osworld_results \
    --workers 64
```

### Key Arguments
- `--dir_path`: Directory containing JSON files from Stage 1 (default: current directory)
- `--workers`: Number of parallel worker processes (default: 64)
- `--skip_save`: Skip saving result.txt file (useful for dry runs)

### Output Structure
For each `<instance_id>.json`, creates:
```
<instance_id>/
├── conversation.json     # Structured conversation with roles and content
├── result.txt           # Boolean result (True/False)
├── 0.png               # First screenshot
├── 1.png               # Second screenshot
└── ...                 # Additional screenshots
```

The `conversation.json` format:
```json
[
  {
    "role": "user",
    "content": "Task instruction...",
    "img": "0.png"
  },
  {
    "role": "assistant",
    "content": "I'll help with that...",
    "tool_calls": [
      {
        "name": "mouse_click",
        "arguments": "{\"x\": 0.5, \"y\": 0.3}"
      }
    ]
  },
  ...
]
```

### Statistics
Prints summary statistics:
- Total correct tasks
- Total tasks processed
- Overall accuracy

## Stage 3: generate_gifs.py - Create Video Visualizations

The final stage generates MP4 videos showing the agent's actions with annotations.

### Purpose
- Creates videos from conversation and screenshots
- Overlays user instructions and assistant responses
- Visualizes click actions with red markers
- Generates one video per task

### Usage

```bash
python generate_gifs.py ./osworld_results \
    --fps 0.2 \
    --output-name output.mp4
```

### Key Arguments
- `work_dir`: Directory containing subdirectories from Stage 2 (positional)
- `--fps`: Frames per second for video (default: 0.2 = 5 seconds per frame)
- `--output-name`: Output filename (default: `output.mp4`)

### Output
Creates `output.mp4` in each task subdirectory with:
- **Top overlay** (first frame only): User instruction
- **Bottom overlay** (all frames): Assistant response
- **Red markers**: Click locations with tool names
- **Duration**: Configurable via FPS (default 5s per frame)

### Video Features
- Semi-transparent text backgrounds for readability
- Centered text with automatic wrapping
- Red circles mark click coordinates (normalized 0-1)
- Tool names labeled above click markers
- MP4 format with mp4v codec

## Complete Pipeline Example

```bash
# Step 1: Run evaluation
python osworld.py \
    --data-file /data/osworld_test.json \
    --output-dir ./osworld_results \
    --max-parallel-jobs 4 \
    --max-iterations 15

# Step 2: Extract images and conversations
python unpack.py \
    --dir_path ./osworld_results \
    --workers 64

# Step 3: Generate videos
python generate_gifs.py ./osworld_results --fps 0.2
```

### Expected Directory Structure

After running the complete pipeline:

```
osworld_results/
├── <instance_id_1>.json
├── <instance_id_2>.json
├── <instance_id_1>/
│   ├── conversation.json
│   ├── result.txt
│   ├── 0.png
│   ├── 1.png
│   └── output.mp4
├── <instance_id_2>/
│   ├── conversation.json
│   ├── result.txt
│   ├── 0.png
│   └── output.mp4
└── ...
```

## Requirements

### Python Dependencies
- `PIL` (Pillow): Image processing
- `cv2` (opencv-python): Video generation
- `numpy`: Array operations
- `openhands.nvidia.async_server_osworld`: Agent server interface

### System Dependencies
- Fonts for text overlay (DejaVu or Liberation Sans)
- Sufficient disk space for videos

## Tips and Best Practices

1. **Resumption**: osworld.py automatically skips completed tasks, so you can safely re-run it after interruptions

2. **Parallel Processing**: 
   - osworld.py: Set `--max-parallel-jobs` based on available GPU/CPU resources
   - unpack.py: Uses multiprocessing; adjust `--workers` for your CPU count

3. **Video Customization**: Adjust `--fps` in generate_gifs.py:
   - Lower FPS (e.g., 0.2) = More time per frame = Easier to read
   - Higher FPS (e.g., 1.0) = Faster playback = Shorter videos

4. **Storage**: Each task with screenshots and video can be several MB; plan storage accordingly
