"""
Extract a single frame from each bot's video/NPY to use as 'listening' mode image.
The frame can be used when the bot is not speaking.
"""
import numpy as np
from pathlib import Path
import cv2

# Constants from video_broadcaster.py
TARGET_WIDTH = 360
TARGET_HEIGHT = 640
FPS = 20

# I420 format sizes
Y_SIZE = TARGET_WIDTH * TARGET_HEIGHT
U_SIZE = (TARGET_WIDTH // 2) * (TARGET_HEIGHT // 2)
V_SIZE = U_SIZE
FRAME_SIZE = Y_SIZE + U_SIZE + V_SIZE

def i420_to_bgr(i420_frame: np.ndarray) -> np.ndarray:
    """Convert I420 format to BGR for saving as image."""
    # Reshape to I420 format
    i420 = i420_frame.reshape((TARGET_HEIGHT * 3 // 2, TARGET_WIDTH))
    bgr = cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)
    return bgr

def extract_frame_from_npy(video_index: int, frame_number: int = 0, output_dir: Path = None) -> tuple:
    """
    Extract a specific frame from bot's NPY file.
    
    Args:
        video_index: Bot number (1-4)
        frame_number: Which frame to extract (default: 0 = first frame)
        output_dir: Where to save the PNG (default: video directory)
    
    Returns:
        Tuple of (npy_path, png_path, total_frames) or (None, None, 0) if failed
    """
    video_dir = Path(__file__).parent.parent / "video"
    if output_dir is None:
        output_dir = video_dir
    
    npy_path = video_dir / f"bot{video_index}_{TARGET_WIDTH}x{TARGET_HEIGHT}_{FPS}fps.npy"
    
    if not npy_path.exists():
        print(f"[ERROR] NPY file not found: {npy_path}")
        return None, None, 0
    
    # Load frames
    frames = np.load(npy_path, mmap_mode='r')
    total_frames = len(frames)
    
    print(f"\n=== Bot {video_index} ===")
    print(f"NPY file: {npy_path.name}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames / FPS:.1f} seconds @ {FPS}fps")
    
    # Ensure frame_number is valid
    if frame_number >= total_frames:
        frame_number = 0
        print(f"[WARN] Requested frame {frame_number} exceeds total, using frame 0")
    
    # Extract the frame
    i420_frame = frames[frame_number]
    
    # Convert to BGR for saving
    bgr_frame = i420_to_bgr(i420_frame)
    
    # Save as PNG
    png_path = output_dir / f"bot{video_index}_listening.png"
    cv2.imwrite(str(png_path), bgr_frame)
    print(f"Saved: {png_path.name} (frame {frame_number})")
    
    # Also save the raw I420 data as NPY for direct use
    npy_listen_path = output_dir / f"bot{video_index}_listening.npy"
    np.save(npy_listen_path, i420_frame)
    print(f"Saved: {npy_listen_path.name} (I420 format, {FRAME_SIZE} bytes)")
    
    return npy_path, png_path, total_frames


def extract_frame_from_video(video_index: int, second: float = 0.0, output_dir: Path = None) -> tuple:
    """
    Extract a specific frame from bot's MP4 video at a given second.
    
    Args:
        video_index: Bot number (1-4)
        second: Time in seconds to extract frame from
        output_dir: Where to save the PNG (default: video directory)
    
    Returns:
        Tuple of (video_path, png_path, total_frames) or (None, None, 0) if failed
    """
    video_dir = Path(__file__).parent.parent / "video"
    if output_dir is None:
        output_dir = video_dir
    
    video_path = video_dir / f"bot{video_index}.mp4"
    
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return None, None, 0
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return None, None, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"\n=== Bot {video_index} (from MP4) ===")
    print(f"Video: {video_path.name}")
    print(f"Total frames: {total_frames}")
    print(f"Original FPS: {fps:.1f}")
    print(f"Duration: {duration:.1f} seconds")
    
    # Calculate frame number from second
    frame_number = int(second * fps)
    if frame_number >= total_frames:
        frame_number = 0
        print(f"[WARN] Requested second {second} exceeds duration, using second 0")
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"[ERROR] Could not read frame {frame_number}")
        return None, None, 0
    
    # Crop and resize to target resolution
    frame_processed = _crop_and_resize(frame)
    
    # Save as PNG
    png_path = output_dir / f"bot{video_index}_listening.png"
    cv2.imwrite(str(png_path), frame_processed)
    print(f"Saved: {png_path.name} (at {second:.1f}s, frame {frame_number})")
    
    # Convert to I420 and save as NPY
    i420 = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2YUV_I420)
    npy_listen_path = output_dir / f"bot{video_index}_listening.npy"
    np.save(npy_listen_path, i420.flatten().astype(np.uint8))
    print(f"Saved: {npy_listen_path.name} (I420 format, {FRAME_SIZE} bytes)")
    
    return video_path, png_path, total_frames


def _crop_and_resize(frame: np.ndarray) -> np.ndarray:
    """Crop center and resize frame to target aspect ratio."""
    TARGET_ASPECT = TARGET_WIDTH / TARGET_HEIGHT
    h, w = frame.shape[:2]
    src_aspect = w / h

    if src_aspect > TARGET_ASPECT:
        new_w = int(h * TARGET_ASPECT)
        start_x = (w - new_w) // 2
        cropped = frame[:, start_x:start_x + new_w]
    else:
        new_h = int(w / TARGET_ASPECT)
        start_y = (h - new_h) // 2
        cropped = frame[start_y:start_y + new_h, :]

    resized = cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    return resized


def main():
    print("=== Extracting Listening Mode Frames for Bots 1-4 ===")
    print(f"Target resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    
    video_dir = Path(__file__).parent.parent / "video"
    
    # Check which files exist
    print("\n--- Available Files ---")
    for bot_num in range(1, 5):
        npy_path = video_dir / f"bot{bot_num}_{TARGET_WIDTH}x{TARGET_HEIGHT}_{FPS}fps.npy"
        mp4_path = video_dir / f"bot{bot_num}.mp4"
        print(f"Bot {bot_num}: NPY={'✓' if npy_path.exists() else '✗'}, MP4={'✓' if mp4_path.exists() else '✗'}")
    
    # Extract first frame from each bot's NPY file
    print("\n--- Extracting First Frame (frame 0) ---")
    for bot_num in range(1, 5):
        extract_frame_from_npy(bot_num, frame_number=0)
    
    print("\n=== Done ===")
    print("Generated files:")
    print("  - bot{1-4}_listening.png : Viewable images")
    print("  - bot{1-4}_listening.npy : I420 format data for direct use")


if __name__ == "__main__":
    main()
