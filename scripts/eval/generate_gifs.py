#!/usr/bin/env python3
"""
Generate videos from conversation data and screenshots.

This script takes a work directory containing conversation.json and PNG files,
matches each assistant turn with corresponding PNG images, overlays user turn
content as text on the images, draws click markers for tool calls with coordinates,
and creates a video with configurable FPS (default 0.2 FPS = 5 seconds per frame).
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
import textwrap
import cv2
import numpy as np


def parse_conversation(conversation_path: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse conversation.json to extract user and assistant turns.
    
    Args:
        conversation_path: Path to conversation.json file
        
    Returns:
        Tuple of (user_turns, assistant_turns) where:
        - user_turns is a list of dicts with 'content' key
        - assistant_turns is a list of dicts with 'content' and 'tool_calls' keys
    """
    with open(conversation_path, 'r', encoding='utf-8') as f:
        conversation = json.load(f)
    
    user_turns = []
    assistant_turns = []
    
    for msg in conversation:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role in ['user', 'tool']:
            user_turns.append({
                'content': content
            })
        elif role == 'assistant':
            assistant_turns.append({
                'content': content,
                'tool_calls': msg.get('tool_calls', [])
            })
    
    return user_turns, assistant_turns


def add_text_to_image(image: Image.Image, top_text: str = None, bottom_text: str = None, font_size: int = 15) -> Image.Image:
    """
    Add text overlay to an image at the top and/or bottom.
    
    Args:
        image: PIL Image object
        top_text: Text to overlay at the top of the image
        bottom_text: Text to overlay at the bottom of the image
        font_size: Font size for the text
        
    Returns:
        Modified PIL Image with text overlay
    """
    # Create a copy to avoid modifying the original
    img_with_text = image.copy()
    draw = ImageDraw.Draw(img_with_text)
    
    # Try to use a better font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
    
    # Get image dimensions
    img_width, img_height = img_with_text.size
    
    # Helper function to add text at a specific position
    def add_text_overlay(text: str, position: str):
        """Add text overlay at 'top' or 'bottom' position"""
        # Wrap text to fit image width (with some padding)
        max_chars_per_line = int(img_width / (font_size * 0.6))  # Rough estimate
        wrapped_text = textwrap.fill(text, width=max_chars_per_line)
        
        # Calculate text bounding box
        # For multi-line text, we need to calculate height manually
        lines = wrapped_text.split('\n')
        line_height = font_size + 5  # Add some spacing between lines
        text_height = len(lines) * line_height
        
        # Create a semi-transparent black background for the text
        padding = 10
        text_bg_height = text_height + 2 * padding
        text_bg = Image.new('RGBA', (img_width, text_bg_height), (0, 0, 0, 180))
        
        # Composite the background onto the image at the appropriate position
        if position == 'top':
            y_position = 0
            img_with_text.paste(text_bg, (0, y_position), text_bg)
            y_offset = padding
        else:  # bottom
            y_position = img_height - text_bg_height
            img_with_text.paste(text_bg, (0, y_position), text_bg)
            y_offset = y_position + padding
        
        # Draw each line of text
        for line in lines:
            # Center each line horizontally
            line_width = draw.textlength(line, font=font)
            x_position = (img_width - line_width) / 2
            draw.text((x_position, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += line_height
    
    # Add top text if provided
    if top_text:
        add_text_overlay(top_text, 'top')
    
    # Add bottom text if provided
    if bottom_text:
        add_text_overlay(bottom_text, 'bottom')
    
    return img_with_text


def add_click_marker(image: Image.Image, x: float, y: float, tool_name: str = None, radius: int = 10, font_size: int = 14) -> Image.Image:
    """
    Add a filled red circle at the specified coordinates with optional tool name label.
    
    Args:
        image: PIL Image object
        x: X coordinate (normalized 0-1, will be scaled to image width)
        y: Y coordinate (normalized 0-1, will be scaled to image height)
        tool_name: Name of the tool call to display above the circle
        radius: Radius of the circle in pixels
        font_size: Font size for the tool name text
        
    Returns:
        Modified PIL Image with circle marker and optional label
    """
    # Create a copy to avoid modifying the original
    img_with_marker = image.copy()
    draw = ImageDraw.Draw(img_with_marker)
    
    # Get image dimensions
    img_width, img_height = img_with_marker.size
    
    # Convert normalized coordinates to pixel coordinates
    pixel_x = x * img_width
    pixel_y = y * img_height
    
    # Draw filled red circle
    left_up = (pixel_x - radius, pixel_y - radius)
    right_down = (pixel_x + radius, pixel_y + radius)
    draw.ellipse([left_up, right_down], fill=(255, 0, 0, 255), outline=(255, 0, 0, 255))
    
    # Draw tool name above the circle if provided
    if tool_name:
        # Try to use a better font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
            except:
                # Fall back to default font
                font = ImageFont.load_default()
        
        # Calculate text position (centered above the circle)
        text_width = draw.textlength(tool_name, font=font)
        text_x = pixel_x - (text_width / 2)
        text_y = pixel_y - radius - font_size - 5  # 5 pixels padding above circle
        
        # Draw text in red
        draw.text((text_x, text_y), tool_name, fill=(255, 0, 0, 255), font=font)
    
    return img_with_marker


def create_video(work_dir: Path, output_path: Path = None, fps: float = 0.2) -> None:
    """
    Create a video from conversation and PNG files.
    
    Args:
        work_dir: Directory containing conversation.json and PNG files
        output_path: Path for the output video (default: work_dir/output.mp4)
        fps: Frames per second (default: 0.2 = 5 seconds per frame)
    """
    work_dir = Path(work_dir)
    conversation_path = work_dir / 'conversation.json'
    
    if not conversation_path.exists():
        raise FileNotFoundError(f"conversation.json not found in {work_dir}")
    
    # Parse conversation
    print(f"Parsing conversation from {conversation_path}")
    user_turns, assistant_turns = parse_conversation(conversation_path)
    
    print(f"Found {len(user_turns)} user turns and {len(assistant_turns)} assistant turns")
    
    # Find all PNG files, sorted numerically
    png_files = sorted(work_dir.glob('*.png'), key=lambda x: int(x.stem) if x.stem.isdigit() else float('inf'))
    
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {work_dir}")
    
    print(f"Found {len(png_files)} PNG files")
    
    # Process images and match with turns
    processed_images = []
    
    for i, png_path in enumerate(png_files):
        # Match with assistant turn (0.png -> first assistant turn, etc.)
        if i < len(assistant_turns):
            # Get the corresponding user turn and assistant turn
            user_turn = user_turns[i] if i < len(user_turns) else {'content': "No user instruction available"}
            assistant_turn = assistant_turns[i]
            
            user_text = user_turn['content']
            assistant_text = assistant_turn['content']
            
            ## Truncate long text for better display
            #if len(user_text) > 500:
            #    user_text = user_text[:500] + "..."
            #if len(assistant_text) > 500:
            #    assistant_text = assistant_text[:500] + "..."
            
            print(f"Processing {png_path.name} with user turn {i} and assistant turn {i}")
            
            # Load image
            img = Image.open(png_path)
            
            # Check if there are tool calls with x, y coordinates in the assistant turn
            tool_calls = assistant_turn.get('tool_calls', [])
            if tool_calls and len(tool_calls) > 0:
                first_tool_call = tool_calls[0]
                tool_name = first_tool_call.get('name', '')  # Default to '' if no name
                arguments = first_tool_call.get('arguments', {})
                
                # Parse arguments if it's a string (JSON)
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        print(f"  Warning: Could not parse tool_call arguments for {png_path.name}")
                        arguments = {}
                
                # Check if x and y coordinates exist
                if 'x' in arguments and 'y' in arguments:
                    try:
                        x = float(arguments['x'])
                        y = float(arguments['y'])
                        print(f"  Drawing click marker at ({x}, {y}) for tool '{tool_name}'")
                        img = add_click_marker(img, x, y, tool_name=tool_name)
                    except (ValueError, TypeError) as e:
                        print(f"  Warning: Invalid x/y coordinates in {png_path.name}: {e}")
            
            # Add text (user at top, assistant at bottom)
            if i == 0:
                img_with_text = add_text_to_image(img, top_text=user_text, bottom_text=assistant_text)
            else:
                img_with_text = add_text_to_image(img, bottom_text=assistant_text)
            
            # Convert to RGB if necessary (OpenCV uses BGR, but we'll convert later)
            if img_with_text.mode != 'RGB':
                img_with_text = img_with_text.convert('RGB')
            
            processed_images.append(img_with_text)
        else:
            print(f"Warning: No assistant turn for {png_path.name}, skipping")
    
    if not processed_images:
        raise ValueError("No images were processed")
    
    # Set output path
    if output_path is None:
        output_path = work_dir / 'output.mp4'
    else:
        output_path = Path(output_path)
    
    # Get dimensions from first image
    first_img = processed_images[0]
    width, height = first_img.size
    
    # Create video writer
    print(f"Creating video at {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    
    # Write frames to video
    for img_pil in processed_images:
        # Convert PIL Image to numpy array
        img_array = np.array(img_pil)
        # Convert RGB to BGR (OpenCV uses BGR)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        # Write frame
        video_writer.write(img_bgr)
    
    video_writer.release()
    
    duration_per_frame = 1.0 / fps
    total_duration = len(processed_images) * duration_per_frame
    
    print(f"✓ Video created successfully: {output_path}")
    print(f"  - {len(processed_images)} frames")
    print(f"  - {fps} FPS ({duration_per_frame:.1f}s per frame)")
    print(f"  - Total duration: {total_duration:.1f}s")
    print(f"  - Resolution: {width}x{height}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate videos from conversation data and screenshots'
    )
    parser.add_argument(
        'work_dir',
        type=str,
        help='Directory containing subdirectories with conversation.json and PNG files'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=0.2,
        help='Frames per second (default: 0.2, which is 5 seconds per frame)'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='output.mp4',
        help='Output video filename (default: output.mp4)'
    )
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    
    if not work_dir.exists():
        print(f"Error: Directory {work_dir} does not exist")
        return 1
    
    if not work_dir.is_dir():
        print(f"Error: {work_dir} is not a directory")
        return 1
    
    # Find all subdirectories that contain conversation.json
    subdirs_to_process = []
    for subdir in work_dir.iterdir():
        if subdir.is_dir():
            conversation_file = subdir / 'conversation.json'
            if conversation_file.exists():
                subdirs_to_process.append(subdir)
    
    if not subdirs_to_process:
        print(f"No subdirectories with conversation.json found in {work_dir}")
        return 1
    
    print(f"Found {len(subdirs_to_process)} subdirectories to process")
    print("=" * 80)
    
    # Process each subdirectory
    successful = 0
    failed = 0
    errors = []
    
    for i, subdir in enumerate(subdirs_to_process, 1):
        print(f"\n[{i}/{len(subdirs_to_process)}] Processing: {subdir.name}")
        print("-" * 80)
        
        try:
            output_path = subdir / args.output_name
            create_video(
                work_dir=subdir,
                output_path=output_path,
                fps=args.fps
            )
            successful += 1
        except Exception as e:
            print(f"✗ Error processing {subdir.name}: {e}")
            failed += 1
            errors.append((subdir.name, str(e)))
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total subdirectories: {len(subdirs_to_process)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if errors:
        print("\nFailed subdirectories:")
        for subdir_name, error in errors:
            print(f"  - {subdir_name}: {error}")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())

