#!/usr/bin/env python3
"""
Pre-process video files into numpy arrays for fast broadcasting.
This eliminates real-time video decoding/resizing CPU overhead.

Usage:
    python preprocess_videos.py           # Process all videos (1-4)
    python preprocess_videos.py --video 1 # Process only video 1
    python preprocess_videos.py --fps 15  # Custom FPS
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import time

# Default settings (should match video_broadcaster.py)
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 854
DEFAULT_FPS = 20


def preprocess_video(
    video_index: int,
    target_width: int = DEFAULT_WIDTH,
    target_height: int = DEFAULT_HEIGHT,
    target_fps: int = DEFAULT_FPS,
) -> Path:
    """Convert a video file to preprocessed numpy frames.
    
    Args:
        video_index: Video number (1-4)
        target_width: Output width
        target_height: Output height  
        target_fps: Output FPS (will sample frames from source)
    
    Returns:
        Path to the saved .npy file
    """
    video_dir = Path(__file__).parent.parent / "video"
    video_path = video_dir / f"bot{video_index}.mp4"
    output_path = video_dir / f"bot{video_index}_{target_width}x{target_height}_{target_fps}fps.npy"
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    print(f"[VIDEO {video_index}] Processing {video_path.name}...")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = src_frame_count / src_fps if src_fps > 0 else 0
    
    print(f"  Source: {src_width}x{src_height} @ {src_fps:.1f}fps, {src_frame_count} frames, {duration:.1f}s")
    print(f"  Target: {target_width}x{target_height} @ {target_fps}fps (I420)")
    
    # Calculate frame sampling
    frame_interval = src_fps / target_fps  # Sample every N source frames
    target_frame_count = int(src_frame_count / frame_interval)
    
    # I420 frame size
    frame_size = int(target_width * target_height * 1.5)
    
    # Pre-allocate array for all frames
    all_frames = np.zeros((target_frame_count, frame_size), dtype=np.uint8)
    
    target_aspect = target_width / target_height
    frame_idx = 0
    src_frame_idx = 0
    next_sample_frame = 0
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames at target FPS
        if src_frame_idx >= next_sample_frame and frame_idx < target_frame_count:
            # Crop to target aspect ratio
            h, w = frame.shape[:2]
            src_aspect = w / h
            
            if src_aspect > target_aspect:
                new_w = int(h * target_aspect)
                start_x = (w - new_w) // 2
                cropped = frame[:, start_x:start_x + new_w]
            else:
                new_h = int(w / target_aspect)
                start_y = (h - new_h) // 2
                cropped = frame[start_y:start_y + new_h, :]
            
            # Resize
            resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            # Convert to I420
            frame_i420 = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV_I420)
            
            # Store flattened frame
            all_frames[frame_idx] = frame_i420.flatten()
            frame_idx += 1
            next_sample_frame += frame_interval
            
            # Progress
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{target_frame_count} frames...", end='\r')
        
        src_frame_idx += 1
    
    cap.release()
    
    # Trim to actual frame count (in case video was shorter)
    all_frames = all_frames[:frame_idx]
    
    # Save as numpy file
    np.save(output_path, all_frames)
    
    elapsed = time.time() - start_time
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"  Saved: {output_path.name}")
    print(f"  Frames: {frame_idx}, Size: {file_size_mb:.1f}MB, Time: {elapsed:.1f}s")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Pre-process videos to numpy arrays")
    parser.add_argument("--video", "-v", type=int, help="Process only this video index (1-4)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Target width (default: {DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Target height (default: {DEFAULT_HEIGHT})")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"Target FPS (default: {DEFAULT_FPS})")
    args = parser.parse_args()
    
    video_indices = [args.video] if args.video else [1, 2, 3, 4]
    
    print(f"Pre-processing videos to {args.width}x{args.height} @ {args.fps}fps\n")
    
    total_size = 0
    for idx in video_indices:
        try:
            output_path = preprocess_video(idx, args.width, args.height, args.fps)
            total_size += output_path.stat().st_size
            print()
        except FileNotFoundError as e:
            print(f"[VIDEO {idx}] Skipped: {e}\n")
    
    print(f"Total size: {total_size / (1024*1024):.1f}MB")
    print("\nDone! Videos are ready for fast broadcasting.")


if __name__ == "__main__":
    main()
