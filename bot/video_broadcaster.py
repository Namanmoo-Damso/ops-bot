"""
Shared video frame broadcaster using shared memory.
One broadcaster per video file, multiple bots can read the current frame.

Uses I420 (YUV) format for efficiency - 1.5 bytes/pixel vs 3 bytes/pixel for RGB.
"""
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
import time

# Target resolution - HD quality for iPhone
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
TARGET_ASPECT = TARGET_WIDTH / TARGET_HEIGHT
FPS = 30  # 30fps for CPU efficiency (still smooth on iPhone)

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

    def _crop_and_resize(self, frame: np.ndarray) -> np.ndarray:
        """Crop center and resize frame to 9:16 portrait aspect ratio."""
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

        # Use INTER_NEAREST for faster resizing (less CPU)
        resized = cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)
        return resized

    def start(self):
        """Start broadcasting video frames to shared memory."""
        video_dir = Path(__file__).parent.parent / "video"
        video_path = video_dir / f"bot{self.video_index}.mp4"

        if not video_path.exists():
            print(f"[BROADCASTER {self.video_index}] ERROR: Video not found: {video_path}")
            return

        # Create shared memory for the frame (I420 format)
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=FRAME_SIZE)
        except FileExistsError:
            # Already exists, attach to it and recreate
            existing = shared_memory.SharedMemory(name=self.shm_name)
            existing.close()
            existing.unlink()
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=FRAME_SIZE)

        print(f"[BROADCASTER {self.video_index}] Started: {video_path.name} -> {TARGET_WIDTH}x{TARGET_HEIGHT} @ {FPS}fps (I420)")

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
                frame_portrait = self._crop_and_resize(frame)
                # BGR -> YUV I420 (this is what video codecs use internally)
                frame_i420 = cv2.cvtColor(frame_portrait, cv2.COLOR_BGR2YUV_I420)

                # Write I420 frame to shared memory (it's already a flat buffer)
                np.copyto(
                    np.ndarray(FRAME_SIZE, dtype=np.uint8, buffer=self.shm.buf),
                    frame_i420.flatten()
                )

                # Maintain frame rate
                elapsed = time.monotonic() - start_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            cap.release()
            print(f"[BROADCASTER {self.video_index}] Video looped")

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

    def get_frame_bytes(self) -> Optional[bytes]:
        """Get current I420 frame as bytes for LiveKit."""
        if not self.connected or not self.shm:
            return None

        # Read I420 frame directly from shared memory
        # Use memoryview to avoid extra copy
        return bytes(self.shm.buf[:FRAME_SIZE])

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
