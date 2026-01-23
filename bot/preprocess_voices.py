"""Preprocess voice files and save PCM data with video frames.

This script pre-renders the waveform visualization for all voice files
and saves them as .npz files for fast loading at runtime.

Run this script once before starting the bot to speed up startup time.
"""

import io
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for rendering
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Video settings (must match bot_client.py)
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 300
VIDEO_FPS = 60
NUM_BARS = 96  # Fewer bars for smaller width

# Directories
SCRIPT_DIR = Path(__file__).parent
VOICE_DIR = SCRIPT_DIR.parent / "voice"
PREPROCESSED_DIR = SCRIPT_DIR.parent / "voice_preprocessed"


def load_voice_file(file_path: Path) -> tuple[bytes, int]:
    """Load an MP3 file and convert to raw PCM audio data."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-i", str(file_path),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", "24000",
        "-ac", "1",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout, 24000


def pre_render_visualization(pcm_data: bytes, sample_rate: int = 24000) -> list[np.ndarray]:
    """Pre-render visualization frames for audio data using matplotlib."""
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    total_samples = len(samples)
    audio_duration = total_samples / sample_rate
    num_video_frames = int(audio_duration * VIDEO_FPS)

    frames = []
    bar_values = np.zeros(NUM_BARS, dtype=np.float32)

    dpi = 100
    fig_width = VIDEO_WIDTH / dpi
    fig_height = VIDEO_HEIGHT / dpi

    print(f"  Rendering {num_video_frames} video frames...", flush=True)

    for frame_idx in range(num_video_frames):
        if frame_idx % 30 == 0:
            print(f"    Frame {frame_idx}/{num_video_frames}", flush=True)
            
        frame_progress = frame_idx / max(num_video_frames - 1, 1)
        center_sample = int(frame_progress * (total_samples - 1))
        
        window_size = sample_rate // VIDEO_FPS * 2
        start_sample = max(0, center_sample - window_size // 2)
        end_sample = min(total_samples, center_sample + window_size // 2)
        chunk = samples[start_sample:end_sample]

        if len(chunk) > 0:
            chunk_per_bar = max(1, len(chunk) // NUM_BARS)
            bar_targets = np.zeros(NUM_BARS, dtype=np.float32)

            for i in range(NUM_BARS):
                start_idx = i * chunk_per_bar
                end_idx = min(start_idx + chunk_per_bar, len(chunk))
                if start_idx < len(chunk):
                    bar_chunk = chunk[start_idx:end_idx]
                    rms = np.sqrt(np.mean(bar_chunk ** 2))
                    bar_targets[i] = min(1.0, rms * 5.0)

            for i in range(NUM_BARS):
                if bar_targets[i] > bar_values[i]:
                    bar_values[i] = bar_values[i] * 0.3 + bar_targets[i] * 0.7
                else:
                    bar_values[i] = bar_values[i] * 0.9 + bar_targets[i] * 0.1

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        ax.set_xlim(0, NUM_BARS)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        x_positions = np.arange(NUM_BARS) + 0.5
        bar_heights = bar_values.copy() * 0.6  # Reduce max height (60%)
        bar_heights = np.maximum(bar_heights, 0.02)
        
        ax.bar(x_positions, bar_heights, bottom=0.2, width=0.8,  # narrower bars
               color='white', edgecolor='white', linewidth=0)
        
        fig.canvas.draw()
        frame_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        # Resize if needed
        if frame_array.shape[0] != VIDEO_HEIGHT or frame_array.shape[1] != VIDEO_WIDTH:
            img = Image.fromarray(frame_array)
            img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
            frame_array = np.array(img)
        
        frames.append(frame_array)

    return frames


def preprocess_voice_file(voice_file: Path, output_dir: Path) -> None:
    """Preprocess a single voice file and save to .npz."""
    print(f"Processing {voice_file.name}...", flush=True)
    
    # Load audio
    pcm_data, sample_rate = load_voice_file(voice_file)
    print(f"  Loaded audio: {len(pcm_data)} bytes, {len(pcm_data) / (sample_rate * 2):.2f} seconds", flush=True)
    
    # Pre-render video frames
    video_frames = pre_render_visualization(pcm_data, sample_rate)
    print(f"  Rendered {len(video_frames)} video frames", flush=True)
    
    # Stack frames into single array for efficient storage
    frames_array = np.stack(video_frames, axis=0)
    
    # Save as .npz (compressed numpy archive)
    output_file = output_dir / f"{voice_file.stem}.npz"
    np.savez_compressed(
        output_file,
        pcm_data=np.frombuffer(pcm_data, dtype=np.uint8),
        video_frames=frames_array,
        sample_rate=sample_rate,
    )
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  Saved to {output_file.name} ({file_size_mb:.2f} MB)", flush=True)


def main():
    # Create output directory
    PREPROCESSED_DIR.mkdir(exist_ok=True)
    
    # Find all voice files
    voice_files = sorted(VOICE_DIR.glob("voice_*.mp3"))
    if not voice_files:
        print(f"No voice files found in {VOICE_DIR}")
        sys.exit(1)
    
    print(f"Found {len(voice_files)} voice files to preprocess")
    print(f"Output directory: {PREPROCESSED_DIR}")
    print()
    
    for voice_file in voice_files:
        preprocess_voice_file(voice_file, PREPROCESSED_DIR)
        print()
    
    print("Done! All voice files have been preprocessed.")
    print(f"The bot will now load from: {PREPROCESSED_DIR}")


if __name__ == "__main__":
    main()
