"""
Shared video frame broadcaster using shared memory.
One broadcaster per video file, multiple bots can read the current frame.

Uses I420 (YUV) format for efficiency - 1.5 bytes/pixel vs 3 bytes/pixel for RGB.

Supports two modes:
1. Preprocessed numpy files (fast, low CPU) - use preprocess_videos.py first
2. Real-time video decoding (fallback if numpy not found)
"""
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional
import numpy as np
import time

# Target resolution - 360p @ 20fps for stress testing
# For production, consider 720x1280 @ 24fps
TARGET_WIDTH = 360
TARGET_HEIGHT = 640
TARGET_ASPECT = TARGET_WIDTH / TARGET_HEIGHT
FPS = 20  # 20fps for smoother video

# I420 format: Y plane (full res) + U plane (half res) + V plane (half res)
# Total size = width * height * 1.5
Y_SIZE = TARGET_WIDTH * TARGET_HEIGHT
U_SIZE = (TARGET_WIDTH // 2) * (TARGET_HEIGHT // 2)
V_SIZE = U_SIZE
FRAME_SIZE = Y_SIZE + U_SIZE + V_SIZE  # 1.5x instead of 3x for RGB


class VideoBroadcaster:
    """Broadcasts video frames to shared memory for multiple consumers."""

    def __init__(self, video_index: int):
        self.video_index = video_index
        self.shm_name = f"video_frame_{video_index}"
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.running = False

    def _load_preprocessed(self) -> Optional[np.ndarray]:
        """Try to load preprocessed numpy frames."""
        video_dir = Path(__file__).parent.parent / "video"
        npy_path = video_dir / f"bot{self.video_index}_{TARGET_WIDTH}x{TARGET_HEIGHT}_{FPS}fps.npy"
        
        if npy_path.exists():
            print(f"[BROADCASTER {self.video_index}] Loading preprocessed: {npy_path.name}")
            frames = np.load(npy_path, mmap_mode='r')  # Memory-map for efficiency
            print(f"[BROADCASTER {self.video_index}] Loaded {len(frames)} frames")
            return frames
        return None

    def start(self):
        """Start broadcasting video frames to shared memory."""
        # Try preprocessed numpy first
        preprocessed_frames = self._load_preprocessed()
        
        if preprocessed_frames is not None:
            self._start_from_numpy(preprocessed_frames)
        else:
            self._start_from_video()

    def _start_from_numpy(self, frames: np.ndarray):
        """Broadcast from preprocessed numpy array (fast path)."""
        # Create shared memory for the frame
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=FRAME_SIZE)
        except FileExistsError:
            existing = shared_memory.SharedMemory(name=self.shm_name)
            existing.close()
            existing.unlink()
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=FRAME_SIZE)

        print(f"[BROADCASTER {self.video_index}] Started (numpy): {TARGET_WIDTH}x{TARGET_HEIGHT} @ {FPS}fps")

        self.running = True
        frame_interval = 1 / FPS
        num_frames = len(frames)
        frame_idx = 0
        shm_array = np.ndarray(FRAME_SIZE, dtype=np.uint8, buffer=self.shm.buf)

        while self.running:
            start_time = time.monotonic()

            # Copy frame to shared memory (very fast - just memcpy)
            np.copyto(shm_array, frames[frame_idx])
            
            frame_idx = (frame_idx + 1) % num_frames
            
            # Log loop point
            if frame_idx == 0:
                print(f"[BROADCASTER {self.video_index}] Video looped")

            # Maintain frame rate
            elapsed = time.monotonic() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.cleanup()

    def _start_from_video(self):
        """Broadcast from video file with real-time decoding (fallback)."""
        import cv2
        
        video_dir = Path(__file__).parent.parent / "video"
        video_path = video_dir / f"bot{self.video_index}.mp4"

        if not video_path.exists():
            print(f"[BROADCASTER {self.video_index}] ERROR: Video not found: {video_path}")
            print(f"[BROADCASTER {self.video_index}] Run: python preprocess_videos.py")
            return

        # Create shared memory
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=FRAME_SIZE)
        except FileExistsError:
            existing = shared_memory.SharedMemory(name=self.shm_name)
            existing.close()
            existing.unlink()
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=FRAME_SIZE)

        print(f"[BROADCASTER {self.video_index}] Started (video decode): {video_path.name} -> {TARGET_WIDTH}x{TARGET_HEIGHT} @ {FPS}fps")
        print(f"[BROADCASTER {self.video_index}] TIP: Run 'python preprocess_videos.py' for lower CPU usage")

        self.running = True
        frame_interval = 1 / FPS

        while self.running:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[BROADCASTER {self.video_index}] ERROR: Could not open video")
                break

            while self.running:
                start_time = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    break  # End of video, loop

                # Process frame: crop, resize, convert to I420
                frame_portrait = self._crop_and_resize(frame, cv2)
                frame_i420 = cv2.cvtColor(frame_portrait, cv2.COLOR_BGR2YUV_I420)

                np.copyto(
                    np.ndarray(FRAME_SIZE, dtype=np.uint8, buffer=self.shm.buf),
                    frame_i420.flatten()
                )

                elapsed = time.monotonic() - start_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            cap.release()
            print(f"[BROADCASTER {self.video_index}] Video looped")

        self.cleanup()

    def _crop_and_resize(self, frame: np.ndarray, cv2) -> np.ndarray:
        """Crop center and resize frame to target aspect ratio."""
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

        self.cleanup()

    def cleanup(self):
        """Clean up shared memory."""
        if self.shm:
            self.shm.close()
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass

    def stop(self):
        """Stop the broadcaster."""
        self.running = False


class SharedFrameReader:
    """Reads frames from shared memory written by VideoBroadcaster."""

    def __init__(self, video_index: int):
        self.video_index = video_index
        self.shm_name = f"video_frame_{video_index}"
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.connected = False

    def connect(self, max_retries: int = 30, retry_delay: float = 1.0) -> bool:
        """Connect to shared memory. Retries until broadcaster is ready."""
        for attempt in range(max_retries):
            try:
                self.shm = shared_memory.SharedMemory(name=self.shm_name)
                self.connected = True
                print(f"[FRAME_READER] Connected to shared memory: {self.shm_name}")
                return True
            except FileNotFoundError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        print(f"[FRAME_READER] Failed to connect to {self.shm_name} after {max_retries} attempts")
        return False

    def get_frame_bytes(self) -> Optional[bytearray]:
        """Get current I420 frame as bytes for LiveKit.
        
        Returns a bytearray to avoid creating new bytes objects each call.
        """
        if not self.connected or not self.shm:
            return None

        # Read I420 frame directly from shared memory
        # Use bytearray for better memory efficiency (reusable buffer)
        if not hasattr(self, '_frame_buffer'):
            self._frame_buffer = bytearray(FRAME_SIZE)
        self._frame_buffer[:] = self.shm.buf[:FRAME_SIZE]
        return self._frame_buffer

    def close(self):
        """Close the shared memory connection."""
        if self.shm:
            self.shm.close()
            self.connected = False


# Global broadcaster processes
_broadcasters: dict[int, mp.Process] = {}


def start_broadcaster(video_index: int) -> None:
    """Start a broadcaster process for a video index."""
    if video_index in _broadcasters and _broadcasters[video_index].is_alive():
        return  # Already running

    def run_broadcaster():
        broadcaster = VideoBroadcaster(video_index)
        try:
            broadcaster.start()
        except KeyboardInterrupt:
            broadcaster.stop()

    process = mp.Process(target=run_broadcaster, daemon=True)
    process.start()
    _broadcasters[video_index] = process
    print(f"[MAIN] Started broadcaster process for video {video_index} (PID: {process.pid})")


def stop_all_broadcasters():
    """Stop all broadcaster processes."""
    for video_index, process in _broadcasters.items():
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
    _broadcasters.clear()


if __name__ == "__main__":
    # Test: run broadcaster for video 1
    import signal

    broadcaster = VideoBroadcaster(1)

    def signal_handler(sig, frame):
        broadcaster.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    broadcaster.start()
