"""Bot client that creates a bot room via the ops-api and joins it as a
LiveKit participant. This bot LISTENS (VAD) and SPEAKS (pre-recorded audio)
to have two-way conversations with the simple_agent.

Includes video stream with audio waveform visualization.
Uses pre-recorded voice files and VAD for speech detection (no STT).
"""

import asyncio
import io
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import httpx
import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for rendering
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents.vad import VADEventType
from livekit.plugins import silero

# Video settings for waveform visualization
VIDEO_WIDTH = 960
VIDEO_HEIGHT = 300
VIDEO_FPS = 60
NUM_BARS = 96  # Fewer bars for smaller width
BACKGROUND_COLOR = (0, 0, 0, 255)  # Black


# Add parent directory (bot/) to path to import config
BOT_DIR = Path(__file__).parent  # ops-bot/
sys.path.insert(0, str(BOT_DIR))
from config import ConfigError, get_optional_config, validate_env_vars


# Global shared VAD model - loaded once, reused by all bots in this process
_shared_vad: silero.VAD | None = None


def get_shared_vad() -> silero.VAD:
    """Get or create shared VAD model (saves ~50MB per bot)."""
    global _shared_vad
    if _shared_vad is None:
        _shared_vad = silero.VAD.load(
            min_speech_duration=0.3,
            min_silence_duration=1.0,
            activation_threshold=0.7,
        )
    return _shared_vad


# Load environment variables from bot/.env
env_path = BOT_DIR / ".env"
if not env_path.exists():
    # Fallback: check in ops-bot root
    env_path = BOT_DIR.parent / ".env"
print(f"[DEBUG] Loading .env from: {env_path}", flush=True)
print(f"[DEBUG] .env exists: {env_path.exists()}", flush=True)
load_dotenv(dotenv_path=env_path)


try:
    _env_config = validate_env_vars()
    _optional = get_optional_config()
except ConfigError as e:
    print(f"Configuration Error in bot_client: {e}")
    raise SystemExit(1)

log_level = getattr(_optional["LOG_LEVEL"].upper(), "INFO", "INFO")
logging.basicConfig(
    level=getattr(logging, _optional["LOG_LEVEL"].upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy LiveKit SDK logs (e.g., "ignoring text stream" messages)
logging.getLogger("livekit").setLevel(logging.WARNING)
logging.getLogger("root").setLevel(logging.WARNING)

# Voice files directory
VOICE_DIR = Path(__file__).parent.parent / "voice"


def pre_render_visualization(pcm_data: bytes, sample_rate: int = 24000) -> list[np.ndarray]:
    """Pre-render visualization frames for audio data using matplotlib for smooth rendering.

    Args:
        pcm_data: Raw PCM audio bytes (16-bit signed, mono)
        sample_rate: Audio sample rate

    Returns:
        List of RGBA frame arrays (one per video frame)
    """
    samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
    total_samples = len(samples)
    audio_duration = total_samples / sample_rate
    num_video_frames = int(audio_duration * VIDEO_FPS)

    frames = []

    # State for smooth bar animation
    bar_values = np.zeros(NUM_BARS, dtype=np.float32)

    # Pre-compute figure settings
    dpi = 100
    fig_width = VIDEO_WIDTH / dpi
    fig_height = VIDEO_HEIGHT / dpi

    for frame_idx in range(num_video_frames):
        # Use floating point for precise sample positioning across full audio
        frame_progress = frame_idx / max(num_video_frames - 1, 1)  # 0.0 to 1.0
        center_sample = int(frame_progress * (total_samples - 1))
        
        # Get audio chunk around this position (window size for analysis)
        window_size = sample_rate // VIDEO_FPS * 2  # 2 frames worth of samples
        start_sample = max(0, center_sample - window_size // 2)
        end_sample = min(total_samples, center_sample + window_size // 2)
        chunk = samples[start_sample:end_sample]

        # Get amplitude for each bar with amplification
        if len(chunk) > 0:
            chunk_per_bar = max(1, len(chunk) // NUM_BARS)
            bar_targets = np.zeros(NUM_BARS, dtype=np.float32)

            for i in range(NUM_BARS):
                start_idx = i * chunk_per_bar
                end_idx = min(start_idx + chunk_per_bar, len(chunk))
                if start_idx < len(chunk):
                    bar_chunk = chunk[start_idx:end_idx]
                    # Use RMS with amplification
                    rms = np.sqrt(np.mean(bar_chunk ** 2))
                    bar_targets[i] = min(1.0, rms * 5.0)  # 5x amplification, capped at 1.0

            # Smooth transitions - fast attack, slower decay
            for i in range(NUM_BARS):
                if bar_targets[i] > bar_values[i]:
                    bar_values[i] = bar_values[i] * 0.3 + bar_targets[i] * 0.7  # Smooth attack
                else:
                    bar_values[i] = bar_values[i] * 0.9 + bar_targets[i] * 0.1  # Slower decay

        # Create matplotlib figure with black background
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Remove axes - bars grow upward from bottom (baseline at 0.2)
        ax.set_xlim(0, NUM_BARS)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Draw white bars growing upward from baseline (0.2)
        x_positions = np.arange(NUM_BARS) + 0.5  # Center bars
        bar_heights = bar_values.copy() * 0.6  # Reduce height (60%)
        bar_heights = np.maximum(bar_heights, 0.02)  # Minimum height
        
        # Draw bars from baseline (y=0.2) growing upward
        ax.bar(x_positions, bar_heights, bottom=0.2, width=1.0,  # width=1.0 for no gaps
               color='white', edgecolor='white', linewidth=0)
        
        # Render to numpy array
        fig.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format='rgba', dpi=dpi, facecolor='black', 
                    edgecolor='none', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        # Read RGBA data
        frame_data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        # Calculate expected size and reshape
        frame_rgba = frame_data.reshape(-1, 4)
        
        # Get actual rendered size from figure
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        
        plt.close(fig)
        buf.close()
        
        # Re-render with exact dimensions
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_xlim(0, NUM_BARS)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.bar(x_positions, bar_heights, bottom=0.2, width=1.0,  # width=1.0 for no gaps
               color='white', edgecolor='white', linewidth=0)
        
        fig.canvas.draw()
        frame_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        # Resize to exact VIDEO_WIDTH x VIDEO_HEIGHT if needed
        if frame_array.shape[0] != VIDEO_HEIGHT or frame_array.shape[1] != VIDEO_WIDTH:
            from PIL import Image
            img = Image.fromarray(frame_array)
            img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
            frame_array = np.array(img)
        
        frames.append(frame_array)

    return frames


def generate_idle_frame() -> np.ndarray:
    """Generate a static idle frame (no audio playing) using matplotlib."""
    dpi = 100
    fig_width = VIDEO_WIDTH / dpi
    fig_height = VIDEO_HEIGHT / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_xlim(0, NUM_BARS)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Draw small idle bars at baseline (0.2)
    x_positions = np.arange(NUM_BARS) + 0.5
    bar_heights = np.full(NUM_BARS, 0.02)  # Small constant height
    
    ax.bar(x_positions, bar_heights, bottom=0.2, width=1.0,  # width=1.0 for no gaps
           color='white', edgecolor='white', linewidth=0)
    
    fig.canvas.draw()
    frame_array = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    
    # Resize to exact VIDEO_WIDTH x VIDEO_HEIGHT if needed
    if frame_array.shape[0] != VIDEO_HEIGHT or frame_array.shape[1] != VIDEO_WIDTH:
        from PIL import Image
        img = Image.fromarray(frame_array)
        img = img.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)
        frame_array = np.array(img)
    
    return frame_array


def load_voice_file(file_path: Path) -> tuple[bytes, int]:
    """Load an MP3 file and convert to raw PCM audio data.

    Returns:
        Tuple of (raw PCM bytes, sample rate)
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    # Convert MP3 to raw PCM: mono, 24kHz, 16-bit signed little-endian
    cmd = [
        ffmpeg_exe,
        "-i",
        str(file_path),
        "-f",
        "s16le",  # 16-bit signed little-endian PCM
        "-acodec",
        "pcm_s16le",
        "-ar",
        "24000",  # 24kHz sample rate
        "-ac",
        "1",  # mono
        "-",  # output to stdout
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return result.stdout, 24000


async def create_bot_session(
    bot_number: int | None = None,
    user_id: str | None = None,
    bot_id: str | None = None,
) -> dict:
    """Call ops-api /bot/create to create a bot room and token.

    Args:
        bot_number: If provided, sets preferredIdentity to "bot-{bot_number}".
        user_id: If provided, uses the real user's identity (e.g., "kakao_123456").
        bot_id: If provided, joins a specific bot room (e.g., "0" -> room="bot-0").
    """
    api_base = os.getenv("OPS_API_URL", "http://localhost:8080")
    admin_token = os.getenv("ADMIN_ACCESS_TOKEN")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if admin_token:
        headers["Authorization"] = f"Bearer {admin_token}"

    # Build request body
    body = {}
    if bot_number is not None:
        body["preferredIdentity"] = f"bot-{bot_number}"
        if bot_id is None:
            body["botId"] = str(bot_number)
    if user_id is not None:
        body["userId"] = user_id
    if bot_id is not None:
        body["botId"] = bot_id

    url = f"{api_base}/v1/bot/create"
    print(f"[DEBUG] POST {url}", flush=True)
    print(f"[DEBUG] Body: {body}", flush=True)

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=body, headers=headers)
        print(f"[DEBUG] Response status: {resp.status_code}", flush=True)
        resp.raise_for_status()
        return resp.json()


class SimpleBotAgent:
    """Simple bot agent that responds to agent's speech using pre-recorded voice files."""

    def __init__(self):
        self.conversation_count = 0
        # Load preprocessed voice files (fast) or fall back to live processing (slow)
        self.voice_data: list[tuple[bytes, list[np.ndarray]]] = []
        
        # Try to load from preprocessed directory first
        preprocessed_dir = Path(__file__).parent.parent / "voice_preprocessed"
        
        for i in range(1, 6):  # voice_1 to voice_5
            npz_file = preprocessed_dir / f"voice_{i}.npz"
            mp3_file = VOICE_DIR / f"voice_{i}.mp3"
            
            if npz_file.exists():
                # Fast path: load preprocessed data
                print(f"[BOT] Loading preprocessed {npz_file.name}...", flush=True)
                data = np.load(npz_file, allow_pickle=True)
                pcm_data = data["pcm_data"].tobytes()
                video_frames = [frame for frame in data["video_frames"]]
                self.voice_data.append((pcm_data, video_frames))
                print(f"[BOT] Loaded {npz_file.name}: {len(video_frames)} video frames", flush=True)
            elif mp3_file.exists():
                # Slow path: process on the fly
                print(f"[BOT] Preprocessed file not found, processing {mp3_file.name}...", flush=True)
                pcm_data, sample_rate = load_voice_file(mp3_file)
                print(f"[BOT] Pre-rendering visualization for {mp3_file.name}...", flush=True)
                video_frames = pre_render_visualization(pcm_data, sample_rate)
                self.voice_data.append((pcm_data, video_frames))
                print(f"[BOT] Loaded {mp3_file.name}: {len(video_frames)} video frames", flush=True)
            else:
                print(f"[BOT] Warning: Voice file not found: voice_{i}", flush=True)
        
        if not self.voice_data:
            print("[BOT] Warning: No voice files loaded!", flush=True)

    def get_voice_response(self) -> tuple[bytes, list[np.ndarray]] | None:
        """Get a random pre-recorded voice response with video frames."""
        if not self.voice_data:
            return None
        self.conversation_count += 1
        voice_index = random.randint(0, len(self.voice_data) - 1)
        print(f"[BOT] Playing voice_{voice_index + 1}.mp3", flush=True)
        return self.voice_data[voice_index]


async def run_bot() -> None:
    # Get bot number from environment (set by stress_test_bots.py)
    bot_number = int(os.getenv("BOT_NUMBER", "1"))
    user_id = os.getenv("USER_ID")  # Optional: use real user's identity
    api_base = os.getenv("OPS_API_URL", "http://localhost:8080")

    # Step 1: ask ops-api to create a bot session and dispatch the voice agent
    bot_session = await create_bot_session(bot_number, user_id=user_id)

    # Handle queue response - poll until ready
    if bot_session.get("status") == "queued":
        identity = bot_session["identity"]
        room_name = bot_session["roomName"]
        position = bot_session.get("position", "?")
        print(f"[BOT] Queued at position {position}, polling...", flush=True)

        async with httpx.AsyncClient() as client:
            while True:
                retry_after = bot_session.get("retryAfter", 5)
                await asyncio.sleep(retry_after)

                queue_resp = await client.get(
                    f"{api_base}/v1/rtc/queue/{identity}",
                    params={"roomName": room_name},
                )
                bot_session = queue_resp.json()

                if bot_session.get("status") == "ready":
                    print(f"[BOT] Queue cleared, got token!", flush=True)
                    break

                new_position = bot_session.get("position", "?")
                print(f"[BOT] Still queued at position {new_position}", flush=True)

    livekit_url = bot_session["livekitUrl"]
    token = bot_session["token"]
    room_name = bot_session.get("roomName") or bot_session.get("room_name")
    identity = bot_session.get("identity")

    # Load voice files BEFORE connecting (so we don't miss agent's first words)
    bot_agent = SimpleBotAgent()

    # Initialize VAD (shared globally to save memory) - no STT needed
    vad = get_shared_vad()

    print(f"[BOT] Joining room={room_name} identity={identity}", flush=True)
    print(f"[BOT] LiveKit URL: {livekit_url}", flush=True)

    # Step 2: connect to LiveKit as the bot-* participant
    room = rtc.Room()
    await room.connect(livekit_url, token)

    # Create audio source and track for publishing bot's voice
    audio_source = rtc.AudioSource(24000, 1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("bot_voice", audio_source)

    # Create video source and track for waveform visualization
    video_source = rtc.VideoSource(VIDEO_WIDTH, VIDEO_HEIGHT)
    video_track = rtc.LocalVideoTrack.create_video_track("bot_waveform", video_source)

    # Lock to prevent concurrent speaking
    speak_lock = asyncio.Lock()
    # Flag to indicate bot is speaking (pause STT processing)
    is_speaking = False
    # Flag to track if initial greeting has been sent
    initial_greeting_sent = False
    # Current video frame to display (None = idle frame)
    current_video_frame: np.ndarray | None = None
    # Pre-generate idle frame
    idle_frame = generate_idle_frame()
    # Flag to control video loop
    video_running = True

    # Background task to continuously send video frames
    async def video_loop():
        """Continuously send video frames (idle or from pre-rendered queue)."""
        frame_interval = 1.0 / VIDEO_FPS
        while video_running:
            frame = current_video_frame if current_video_frame is not None else idle_frame
            video_frame = rtc.VideoFrame(
                width=VIDEO_WIDTH,
                height=VIDEO_HEIGHT,
                type=rtc.VideoBufferType.RGBA,
                data=frame.tobytes(),
            )
            video_source.capture_frame(video_frame)
            await asyncio.sleep(frame_interval)

    # Helper function to play pre-recorded audio with pre-rendered video
    async def speak(pcm_data: bytes, video_frames: list[np.ndarray]):
        """Play pre-recorded PCM audio with pre-rendered visualization."""
        nonlocal is_speaking, current_video_frame

        async with speak_lock:
            is_speaking = True

            # Audio frame settings
            frame_size = 480  # 20ms at 24kHz
            bytes_per_frame = frame_size * 2  # 16-bit = 2 bytes per sample

            audio_duration = len(pcm_data) / (24000 * 2)  # seconds
            
            # Video sync task - updates video frames based on elapsed time
            video_sync_running = True
            video_start_time = None
            
            async def video_sync_task():
                nonlocal current_video_frame, video_start_time
                while video_sync_running:
                    if video_start_time is not None:
                        elapsed = time.time() - video_start_time
                        # Add ~0.5s delay to account for audio buffering/network latency
                        adjusted_elapsed = max(0, elapsed - 0.5)
                        video_progress = min(adjusted_elapsed / audio_duration, 1.0)
                        video_idx = min(int(video_progress * len(video_frames)), len(video_frames) - 1)
                        current_video_frame = video_frames[video_idx]
                    await asyncio.sleep(1.0 / VIDEO_FPS)  # Update at video FPS
            
            # Start video sync task
            sync_task = asyncio.create_task(video_sync_task())
            
            # Mark start time when audio begins
            video_start_time = time.time()

            # Push all audio frames as fast as possible
            for i in range(0, len(pcm_data), bytes_per_frame):
                chunk = pcm_data[i : i + bytes_per_frame]
                if len(chunk) < bytes_per_frame:
                    chunk = chunk + b"\x00" * (bytes_per_frame - len(chunk))

                frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=24000,
                    num_channels=1,
                    samples_per_channel=frame_size,
                )
                await audio_source.capture_frame(frame)

            # Wait for audio to finish playing (including network buffer time)
            elapsed = time.time() - video_start_time
            remaining = (audio_duration + 0.5) - elapsed  # Add 0.5s for buffer
            if remaining > 0:
                await asyncio.sleep(remaining + 0.3)
            else:
                await asyncio.sleep(0.3)

            # Stop video sync and return to idle frame
            video_sync_running = False
            sync_task.cancel()
            try:
                await sync_task
            except asyncio.CancelledError:
                pass
            
            current_video_frame = None
            is_speaking = False

    async def process_audio_track(track: rtc.Track, participant: rtc.RemoteParticipant):
        """Process incoming audio track with VAD to detect when agent stops speaking."""
        audio_stream = rtc.AudioStream(track)
        vad_stream = vad.stream()
        
        # Track speech state
        agent_is_speaking = False
        response_task = None
        
        async def delayed_response():
            """Wait for confirmed silence, then respond."""
            nonlocal initial_greeting_sent
            
            await asyncio.sleep(1.5)  # Wait 1.5 seconds of silence
            
            if not is_speaking:
                print(f"[BOT] Agent finished speaking (VAD detected silence)", flush=True)
                
                if not initial_greeting_sent:
                    initial_greeting_sent = True
                    print(f"[BOT] bot {bot_number} 시작", flush=True)
                
                # Small delay before responding
                await asyncio.sleep(0.5)
                
                voice_response = bot_agent.get_voice_response()
                if voice_response:
                    pcm_data, video_frames = voice_response
                    await speak(pcm_data, video_frames)
        
        async def process_vad_events():
            nonlocal agent_is_speaking, response_task
            
            async for event in vad_stream:
                if event.type == VADEventType.START_OF_SPEECH:
                    agent_is_speaking = True
                    # Cancel any pending response if agent starts speaking again
                    if response_task and not response_task.done():
                        response_task.cancel()
                    print(f"[BOT] Agent started speaking", flush=True)
                    
                elif event.type == VADEventType.END_OF_SPEECH:
                    agent_is_speaking = False
                    # Only respond if bot is not currently speaking
                    if not is_speaking:
                        response_task = asyncio.create_task(delayed_response())
        
        vad_task = asyncio.create_task(process_vad_events())
        
        # Feed audio to VAD
        try:
            async for frame_event in audio_stream:
                if not is_speaking:
                    vad_stream.push_frame(frame_event.frame)
        except Exception as e:
            logger.error(f"Error reading audio stream: {e}")
        
        vad_stream.end_input()
        await vad_task

    # Track active audio processing tasks
    active_audio_tasks: dict[str, asyncio.Task] = {}

    # Track subscription handler - listen to agent's audio
    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            # Only process agent tracks, not our own echoes
            if participant.identity.startswith("agent-"):
                task_key = f"{participant.identity}_{publication.sid}"
                if task_key not in active_audio_tasks:
                    task = asyncio.create_task(process_audio_track(track, participant))
                    active_audio_tasks[task_key] = task

    @room.on("track_published")
    def on_track_published(
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        pass  # Silenced track published log

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        pass  # Silenced participant connected log

    # Publish audio track
    audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await room.local_participant.publish_track(audio_track, audio_options)

    # Publish video track for waveform visualization
    video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await room.local_participant.publish_track(video_track, video_options)

    # Start video loop for continuous waveform rendering
    video_task = asyncio.create_task(video_loop())

    # Wait for agent to join
    await asyncio.sleep(2)

    agent_identity = None
    for p in room.remote_participants.values():
        if p.identity.startswith("agent-"):
            agent_identity = p.identity
        # Check for existing audio tracks from this participant
        for sid, track_pub in p.track_publications.items():
            if track_pub.kind == rtc.TrackKind.KIND_AUDIO and p.identity.startswith(
                "agent-"
            ):
                if track_pub.track:
                    task_key = f"{p.identity}_{sid}"
                    if task_key not in active_audio_tasks:
                        task = asyncio.create_task(
                            process_audio_track(track_pub.track, p)
                        )
                        active_audio_tasks[task_key] = task

    print(
        f"[BOT] Ready and listening for agent speech (using pre-recorded voice files)...",
        flush=True,
    )

    # Keep the bot connected until interrupted
    try:
        await asyncio.Event().wait()
    finally:
        video_running = False
        video_task.cancel()
        await room.disconnect()


def main() -> None:
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        print("Bot client interrupted, exiting...")


if __name__ == "__main__":
    main()
