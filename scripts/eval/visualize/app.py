#!/usr/bin/env python3
"""
Trajectory Viewer - A web app to visualize agent trajectories with screenshots.
"""

import argparse
import json
from pathlib import Path
from flask import Flask, render_template, send_from_directory, jsonify

app = Flask(__name__)


def get_results_dir():
    """Get the results directory from app config."""
    return app.config.get("RESULTS_DIR", Path(__file__).parent / "results")


def get_trajectories():
    """Get list of all trajectory folders."""
    results_dir = get_results_dir()
    if not results_dir.exists():
        return []
    
    trajectories = []
    for folder in sorted(results_dir.iterdir()):
        if folder.is_dir():
            result_file = folder / "result.txt"
            result = None
            if result_file.exists():
                result = result_file.read_text().strip().lower() == "true"
            
            trajectories.append({
                "id": folder.name,
                "result": result
            })
    
    return trajectories


def parse_coordinates(arguments_str):
    """Extract normalized coordinates (0-1) from tool call arguments."""
    try:
        args = json.loads(arguments_str)
        coords = []
        
        # Handle x, y coordinates (normalized 0-1)
        if "x" in args and "y" in args:
            x = float(args["x"])
            y = float(args["y"])
            coords.append({"x": x, "y": y})
        
        return coords
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def load_conversation(traj_id):
    """Load conversation.json for a trajectory."""
    conv_file = get_results_dir() / traj_id / "conversation.json"
    if not conv_file.exists():
        return [], ""
    
    with open(conv_file, "r") as f:
        conversation = json.load(f)
    
    # Extract task instruction from first user message
    task_instruction = ""
    for msg in conversation:
        if msg.get("role") == "user":
            task_instruction = msg.get("content", "")
            break
    
    # Build list of steps: each step pairs a screenshot with the next assistant action
    steps = []
    
    for i, msg in enumerate(conversation):
        role = msg.get("role", "unknown")
        img = msg.get("img")
        
        # Only process user/tool turns that have screenshots
        if role in ("user", "tool") and img:
            # Find the next assistant turn to get tool calls
            next_assistant = None
            for j in range(i + 1, len(conversation)):
                if conversation[j].get("role") == "assistant":
                    next_assistant = conversation[j]
                    break
            
            # Extract tool calls from next assistant turn
            tool_calls = []
            assistant_content = ""
            if next_assistant:
                assistant_content = next_assistant.get("content", "")
                if "tool_calls" in next_assistant and next_assistant["tool_calls"]:
                    for tc in next_assistant["tool_calls"]:
                        tool_call = {
                            "name": tc.get("name", "unknown"),
                            "arguments": tc.get("arguments", "{}"),
                            "coordinates": []
                        }
                        coords = parse_coordinates(tc.get("arguments", "{}"))
                        tool_call["coordinates"] = coords
                        tool_calls.append(tool_call)
            
            steps.append({
                "step_num": len(steps),
                "img": img,
                "tool_calls": tool_calls,
                "assistant_content": assistant_content
            })
    
    return steps, task_instruction


def get_screenshot_list(traj_id):
    """Get list of screenshot files for a trajectory."""
    traj_dir = get_results_dir() / traj_id
    if not traj_dir.exists():
        return []
    
    screenshots = []
    for f in traj_dir.iterdir():
        if f.suffix.lower() == ".png" and f.stem.isdigit():
            screenshots.append(f.name)
    
    # Sort by numeric value
    screenshots.sort(key=lambda x: int(Path(x).stem))
    return screenshots


@app.route("/")
def index():
    """Home page listing all trajectories."""
    trajectories = get_trajectories()
    return render_template("index.html", trajectories=trajectories)


@app.route("/trajectory/<traj_id>")
def view_trajectory(traj_id):
    """View a specific trajectory."""
    steps, task_instruction = load_conversation(traj_id)
    
    # Get result
    result_file = get_results_dir() / traj_id / "result.txt"
    result = None
    if result_file.exists():
        result = result_file.read_text().strip().lower() == "true"
    
    return render_template(
        "trajectory.html",
        traj_id=traj_id,
        steps=steps,
        task_instruction=task_instruction,
        result=result
    )


@app.route("/api/trajectory/<traj_id>")
def api_trajectory(traj_id):
    """API endpoint to get trajectory data as JSON."""
    steps, task_instruction = load_conversation(traj_id)
    
    result_file = get_results_dir() / traj_id / "result.txt"
    result = None
    if result_file.exists():
        result = result_file.read_text().strip().lower() == "true"
    
    return jsonify({
        "id": traj_id,
        "result": result,
        "task_instruction": task_instruction,
        "steps": steps
    })


@app.route("/results/<traj_id>/<filename>")
def serve_screenshot(traj_id, filename):
    """Serve screenshot files."""
    return send_from_directory(get_results_dir() / traj_id, filename)


def parse_args():
    parser = argparse.ArgumentParser(description="Trajectory Viewer - Visualize agent trajectories")
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="./results",
        help="Path to the results folder containing trajectory directories (default: ./results)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results_dir = Path(args.results_dir).resolve()
    
    # Store in app config so it's accessible in routes
    app.config["RESULTS_DIR"] = results_dir
    
    if not results_dir.exists():
        print(f"Warning: Results directory '{results_dir}' does not exist.")
    else:
        print(f"Serving trajectories from: {results_dir}")
    
    app.run(debug=True, host=args.host, port=args.port)
