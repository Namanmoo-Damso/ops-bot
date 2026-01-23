"""
Spawns bot instances concurrently with intervals between each start.
All bots run until Ctrl+C is pressed, then they all exit cleanly.

Uses multiprocessing to share voice data - loads once, used by all bots.

Usage:
    python stress_test_bots.py              # Spawns 30 bots (default)
    python stress_test_bots.py -n 5         # Spawns 5 bots
    python stress_test_bots.py --num-bots 10  # Spawns 10 bots
"""

import argparse
import multiprocessing as mp
import time
import sys
import os
from pathlib import Path

import numpy as np


def get_user_id_for_bot(bot_number: int) -> str:
    """Generate dummy user ID for a bot.

    Format: 10000000-0000-0000-0000-000000000001 to 000000000100
    """
    return f"10000000-0000-0000-0000-{bot_number:012d}"


def load_voice_data() -> list[tuple[bytes, list[np.ndarray]]]:
    """Load all preprocessed voice files once."""
    voice_data = []
    bot_dir = Path(__file__).parent
    preprocessed_dir = bot_dir.parent / "voice_preprocessed"
    voice_dir = bot_dir.parent / "voice"

    print("[MAIN] Loading voice files (once for all bots)...", flush=True)

    for i in range(1, 6):
        npz_file = preprocessed_dir / f"voice_{i}.npz"
        mp3_file = voice_dir / f"voice_{i}.mp3"

        if npz_file.exists():
            print(f"[MAIN] Loading {npz_file.name}...", flush=True)
            data = np.load(npz_file, allow_pickle=True)
            pcm_data = data["pcm_data"].tobytes()
            video_frames = [frame for frame in data["video_frames"]]
            voice_data.append((pcm_data, video_frames))
            print(f"[MAIN] Loaded {npz_file.name}: {len(video_frames)} frames", flush=True)
        elif mp3_file.exists():
            # Import here to avoid loading unnecessary modules in main process
            from bot_client import load_voice_file, pre_render_visualization

            print(f"[MAIN] Processing {mp3_file.name}...", flush=True)
            pcm_data, sample_rate = load_voice_file(mp3_file)
            video_frames = pre_render_visualization(pcm_data, sample_rate)
            voice_data.append((pcm_data, video_frames))
            print(f"[MAIN] Processed {mp3_file.name}: {len(video_frames)} frames", flush=True)
        else:
            print(f"[MAIN] Warning: voice_{i} not found", flush=True)

    print(f"[MAIN] Loaded {len(voice_data)} voice files total.\n", flush=True)
    return voice_data


def run_bot_process(bot_number: int, user_id: str, voice_data: list):
    """Run a single bot in a subprocess with pre-loaded voice data."""
    import asyncio
    import logging
    import random

    from dotenv import load_dotenv
    from livekit import rtc
    from livekit.agents.vad import VADEventType

    # Load environment
    bot_dir = Path(__file__).parent
    env_path = bot_dir / ".env"
    if not env_path.exists():
        env_path = bot_dir.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    # Suppress noisy logs
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("livekit").setLevel(logging.WARNING)

    from bot_client import (
        create_bot_session,
        get_shared_vad,
        generate_idle_frame,
        VIDEO_WIDTH,
        VIDEO_HEIGHT,
        VIDEO_FPS,
    )

    async def run():
        # Create bot session
        bot_session = await create_bot_session(bot_number, user_id=user_id)

        # Handle queue
        if bot_session.get("status") == "queued":
            import httpx

            api_base = os.getenv("OPS_API_URL", "http://localhost:8080")
            identity = bot_session["identity"]
            room_name = bot_session["roomName"]
            print(f"[BOT-{bot_number}] Queued, polling...", flush=True)

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
                        break

        livekit_url = bot_session["livekitUrl"]
        token = bot_session["token"]
        room_name = bot_session.get("roomName") or bot_session.get("room_name")
        identity = bot_session.get("identity")

        print(f"[BOT-{bot_number}] Joining room={room_name}", flush=True)

        # Initialize VAD
        vad = get_shared_vad()

        # Connect to room
        room = rtc.Room()
        await room.connect(livekit_url, token)

        # Create audio/video sources
        audio_source = rtc.AudioSource(24000, 1)
        audio_track = rtc.LocalAudioTrack.create_audio_track("bot_voice", audio_source)
        video_source = rtc.VideoSource(VIDEO_WIDTH, VIDEO_HEIGHT)
        video_track = rtc.LocalVideoTrack.create_video_track("bot_waveform", video_source)

        # State
        speak_lock = asyncio.Lock()
        is_speaking = False
        initial_greeting_sent = False
        current_video_frame = None
        idle_frame = generate_idle_frame()
        video_running = True
        conversation_count = 0

        async def video_loop():
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

        async def speak(pcm_data: bytes, video_frames: list):
            nonlocal is_speaking, current_video_frame

            async with speak_lock:
                is_speaking = True
                frame_size = 480
                bytes_per_frame = frame_size * 2
                audio_duration = len(pcm_data) / (24000 * 2)

                video_sync_running = True
                video_start_time = None

                async def video_sync_task():
                    nonlocal current_video_frame, video_start_time
                    while video_sync_running:
                        if video_start_time is not None:
                            elapsed = time.time() - video_start_time
                            adjusted_elapsed = max(0, elapsed - 0.5)
                            video_progress = min(adjusted_elapsed / audio_duration, 1.0)
                            video_idx = min(int(video_progress * len(video_frames)), len(video_frames) - 1)
                            current_video_frame = video_frames[video_idx]
                        await asyncio.sleep(1.0 / VIDEO_FPS)

                sync_task = asyncio.create_task(video_sync_task())
                video_start_time = time.time()

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

                elapsed = time.time() - video_start_time
                remaining = (audio_duration + 0.5) - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining + 0.3)
                else:
                    await asyncio.sleep(0.3)

                video_sync_running = False
                sync_task.cancel()
                try:
                    await sync_task
                except asyncio.CancelledError:
                    pass

                current_video_frame = None
                is_speaking = False

        def get_voice_response():
            nonlocal conversation_count
            if not voice_data:
                return None
            conversation_count += 1
            voice_index = random.randint(0, len(voice_data) - 1)
            print(f"[BOT-{bot_number}] Playing voice_{voice_index + 1}.mp3", flush=True)
            return voice_data[voice_index]

        async def process_audio_track(track, participant):
            nonlocal initial_greeting_sent

            audio_stream = rtc.AudioStream(track)
            vad_stream = vad.stream()
            agent_is_speaking = False
            response_task = None

            async def delayed_response():
                nonlocal initial_greeting_sent
                await asyncio.sleep(1.5)
                if not is_speaking:
                    if not initial_greeting_sent:
                        initial_greeting_sent = True
                        print(f"[BOT-{bot_number}] Started", flush=True)
                    await asyncio.sleep(0.5)
                    voice_response = get_voice_response()
                    if voice_response:
                        pcm_data, video_frames = voice_response
                        await speak(pcm_data, video_frames)

            async def process_vad_events():
                nonlocal agent_is_speaking, response_task
                async for event in vad_stream:
                    if event.type == VADEventType.START_OF_SPEECH:
                        agent_is_speaking = True
                        if response_task and not response_task.done():
                            response_task.cancel()
                    elif event.type == VADEventType.END_OF_SPEECH:
                        agent_is_speaking = False
                        if not is_speaking:
                            response_task = asyncio.create_task(delayed_response())

            vad_task = asyncio.create_task(process_vad_events())

            try:
                async for frame_event in audio_stream:
                    if not is_speaking:
                        vad_stream.push_frame(frame_event.frame)
            except Exception as e:
                print(f"[BOT-{bot_number}] Audio error: {e}", flush=True)

            vad_stream.end_input()
            await vad_task

        active_audio_tasks = {}

        @room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                if participant.identity.startswith("agent-"):
                    task_key = f"{participant.identity}_{publication.sid}"
                    if task_key not in active_audio_tasks:
                        task = asyncio.create_task(process_audio_track(track, participant))
                        active_audio_tasks[task_key] = task

        # Publish tracks
        audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await room.local_participant.publish_track(audio_track, audio_options)
        video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        await room.local_participant.publish_track(video_track, video_options)

        video_task = asyncio.create_task(video_loop())

        await asyncio.sleep(2)

        # Check existing participants
        for p in room.remote_participants.values():
            for sid, track_pub in p.track_publications.items():
                if track_pub.kind == rtc.TrackKind.KIND_AUDIO and p.identity.startswith("agent-"):
                    if track_pub.track:
                        task_key = f"{p.identity}_{sid}"
                        if task_key not in active_audio_tasks:
                            task = asyncio.create_task(process_audio_track(track_pub.track, p))
                            active_audio_tasks[task_key] = task

        print(f"[BOT-{bot_number}] Ready and listening...", flush=True)

        try:
            await asyncio.Event().wait()
        finally:
            video_running = False
            video_task.cancel()
            await room.disconnect()

    asyncio.run(run())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spawn multiple bot instances for stress testing (shared voice data)"
    )
    parser.add_argument(
        "-n", "--num-bots", type=int, default=30,
        help="Number of bots to spawn (default: 30)",
    )
    parser.add_argument(
        "-i", "--interval", type=float, default=1,
        help="Seconds between spawning each bot (default: 1)",
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Enable resource monitoring",
    )
    parser.add_argument(
        "-s", "--start", type=int, default=1,
        help="Starting bot number (default: 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total_bots = args.num_bots
    start_bot = args.start
    spawn_interval = args.interval
    enable_monitor = args.monitor

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load voice data ONCE
    voice_data = load_voice_data()

    if not voice_data:
        print("[MAIN] ERROR: No voice files loaded!", flush=True)
        sys.exit(1)

    print(f"[MAIN] Spawning {total_bots} bots (starting from {start_bot}) with {spawn_interval}s intervals.", flush=True)
    print("[MAIN] Voice data shared across all bots.", flush=True)
    print("[MAIN] Press Ctrl+C to stop all bots.\n", flush=True)

    processes = []

    try:
        for i in range(start_bot, start_bot + total_bots):
            user_id = get_user_id_for_bot(i)
            print(f"[{time.strftime('%H:%M:%S')}] Starting bot {i} ({i - start_bot + 1}/{total_bots})...", flush=True)

            proc = mp.Process(
                target=run_bot_process,
                args=(i, user_id, voice_data),
            )
            proc.start()
            processes.append((i, proc))
            print(f"[{time.strftime('%H:%M:%S')}] Bot {i} started (PID: {proc.pid})", flush=True)

            if i < start_bot + total_bots - 1:
                time.sleep(spawn_interval)

        print(f"\n[{time.strftime('%H:%M:%S')}] All {total_bots} bots running.", flush=True)
        print("[MAIN] Press Ctrl+C to stop.\n", flush=True)

        if enable_monitor:
            try:
                import psutil
                while True:
                    alive = sum(1 for _, p in processes if p.is_alive())
                    total_mem = sum(
                        psutil.Process(p.pid).memory_info().rss / (1024 * 1024)
                        for _, p in processes if p.is_alive()
                    )
                    print(f"[MONITOR] {alive} bots | RAM: {total_mem:.0f}MB", flush=True)
                    time.sleep(10)
            except ImportError:
                print("[MAIN] psutil not installed", flush=True)
                while True:
                    time.sleep(1)
        else:
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%H:%M:%S')}] Shutting down...", flush=True)

    finally:
        for i, proc in processes:
            if proc.is_alive():
                print(f"[{time.strftime('%H:%M:%S')}] Terminating bot {i}...", flush=True)
                proc.terminate()

        time.sleep(2)

        for i, proc in processes:
            if proc.is_alive():
                print(f"[{time.strftime('%H:%M:%S')}] Force killing bot {i}...", flush=True)
                proc.kill()

        for _, proc in processes:
            proc.join()

        print(f"[{time.strftime('%H:%M:%S')}] All bots stopped.", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
