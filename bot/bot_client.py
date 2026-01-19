"""Bot client that creates a bot room via the ops-api and joins it as a
LiveKit participant. This bot can both SPEAK (TTS) and LISTEN (STT) to
have two-way conversations with the simple_agent.

Run this in the agents container or on a machine that can reach the
ops-api and LiveKit server.
"""

import asyncio
import logging
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import AgentSession, RoomInputOptions, RoomOutputOptions
from livekit.agents.stt import SpeechEventType
from livekit.agents.vad import VADEventType
from livekit.plugins import aws, silero

from config import ConfigError, get_optional_config, validate_env_vars


# Load environment variables from ../.env if present
# Load environment variables
env_path = Path(__file__).parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).parent.parent / ".env"
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
    """Simple bot agent that responds to agent's speech."""

    def __init__(self):
        self.conversation_count = 0
        # Elderly Korean speech patterns with informal/반말 style
        self.responses = [
            "응, 그래 그래. 요즘 허리가 좀 아프긴 해도 잘 지내고 있어.",
            "아이고, 그러게 말이야. 나이 먹으니까 여기저기 안 아픈 데가 없어.",
            "뭐라고? 아, 그래그래. 요즘 귀가 좀 어두워져서 말이야.",
            "그래, 고맙다 고마워. 젊은 사람이 이렇게 챙겨주니 좋구만.",
            "옛날에는 말이야, 이런 거 없었어. 세상 참 좋아졌어.",
            "아이고, 맞아 맞아. 요즘 젊은 것들은 바빠서 연락도 잘 안 해.",
            "밥은 먹었어? 끼니는 잘 챙겨 먹어야 해.",
            "그래, 알았어 알았어. 뭐 더 할 말 있어?",
        ]

    def get_response(self, agent_said: str) -> str:
        """Get a simple response based on conversation count."""
        response = self.responses[self.conversation_count % len(self.responses)]
        self.conversation_count += 1
        print(f"[BOT] Response: {response}", flush=True)
        return response


async def publish_video_from_file(room: rtc.Room, bot_number: int):
    """Publish video from a file, looping continuously in 9:16 portrait format."""
    import cv2

    # Video file path - look for video-1.mp4 in the video folder
    video_dir = Path(__file__).parent.parent / "video"
    video_path = video_dir / "video-1.mp4"

    if not video_path.exists():
        print(f"[BOT] ERROR: Video file not found: {video_path}", flush=True)
        print(f"[BOT] Please add a video file at: {video_path}", flush=True)
        return

    # Open video to get properties
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[BOT] ERROR: Could not open video: {video_path}", flush=True)
        return

    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # Target: 9:16 portrait aspect ratio (matching the UI)
    target_width = 720
    target_height = 1280
    target_aspect = target_width / target_height  # 0.5625

    print(f"[BOT] Source video: {src_width}x{src_height}, Target: {target_width}x{target_height} (9:16 portrait)", flush=True)

    # Create video source and track with portrait dimensions
    video_source = rtc.VideoSource(width=target_width, height=target_height)
    video_track = rtc.LocalVideoTrack.create_video_track("bot_camera", video_source)

    # Publish the video track
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await room.local_participant.publish_track(video_track, options)
    print(f"[BOT] Video track published: {video_path.name} @ {fps:.1f}fps -> {target_width}x{target_height}", flush=True)

    frame_interval = 1 / fps

    def crop_and_resize_to_portrait(frame):
        """Crop center and resize frame to 9:16 portrait aspect ratio."""
        h, w = frame.shape[:2]
        src_aspect = w / h

        if src_aspect > target_aspect:
            # Source is wider - crop sides
            new_w = int(h * target_aspect)
            start_x = (w - new_w) // 2
            cropped = frame[:, start_x : start_x + new_w]
        else:
            # Source is taller - crop top/bottom
            new_h = int(w / target_aspect)
            start_y = (h - new_h) // 2
            cropped = frame[start_y : start_y + new_h, :]

        # Resize to target dimensions
        resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        return resized

    # Loop video continuously
    while True:
        cap = cv2.VideoCapture(str(video_path))
        while True:
            ret, frame = cap.read()
            if not ret:
                # End of video, restart from beginning
                break

            # Crop and resize to 9:16 portrait
            frame_portrait = crop_and_resize_to_portrait(frame)

            # Convert BGR (OpenCV) to RGB (LiveKit)
            frame_rgb = cv2.cvtColor(frame_portrait, cv2.COLOR_BGR2RGB)

            video_frame = rtc.VideoFrame(
                width=target_width,
                height=target_height,
                type=rtc.VideoBufferType.RGB24,
                data=frame_rgb.tobytes(),
            )
            video_source.capture_frame(video_frame)
            await asyncio.sleep(frame_interval)

        cap.release()
        print(f"[BOT] Video looped, restarting...", flush=True)



async def run_bot() -> None:
    # Get bot number from environment (set by stress_test_bots.py)
    bot_number = int(os.getenv("BOT_NUMBER", "1"))
    user_id = os.getenv("USER_ID")  # Optional: use real user's identity

    # Step 1: ask ops-api to create a bot session and dispatch the voice agent
    bot_session = await create_bot_session(bot_number, user_id=user_id)

    livekit_url = bot_session["livekitUrl"]
    token = bot_session["token"]
    room_name = bot_session["roomName"]
    identity = bot_session["identity"]

    print(f"[BOT] Joined room={room_name} identity={identity}", flush=True)

    # Step 2: connect to LiveKit as the bot-* participant
    room = rtc.Room()
    await room.connect(livekit_url, token)

    # Initialize STT, TTS, and VAD for two-way communication
    stt = aws.STT(language="ko-KR")
    tts = aws.TTS(voice="Seoyeon", sample_rate=24000)
    vad = silero.VAD.load(
        min_speech_duration=0.3,
        min_silence_duration=1.0,
        activation_threshold=0.7,
    )

    bot_agent = SimpleBotAgent()

    # Create audio source and track for publishing bot's voice
    audio_source = rtc.AudioSource(24000, 1)
    audio_track = rtc.LocalAudioTrack.create_audio_track("bot_voice", audio_source)

    # Lock to prevent concurrent speaking
    speak_lock = asyncio.Lock()
    # Flag to indicate bot is speaking (pause STT processing)
    is_speaking = False
    # Flag to track if initial greeting has been sent
    initial_greeting_sent = False

    # Helper function to speak via TTS
    async def speak(text: str):
        """Synthesize and play TTS audio."""
        nonlocal is_speaking
        async with speak_lock:
            is_speaking = True
            # Silenced speaking log
            audio_stream = tts.synthesize(text)
            async for synthesized_audio in audio_stream:
                await audio_source.capture_frame(synthesized_audio.frame)
            # Silenced finished speaking log
            # Wait a bit for audio to finish playing before resuming STT
            await asyncio.sleep(0.5)
            is_speaking = False

    async def process_audio_track(track: rtc.Track, participant: rtc.RemoteParticipant):
        """Process incoming audio track with STT."""
        # Silenced verbose audio processing log

        audio_stream = rtc.AudioStream(track)
        stt_stream = stt.stream()
        frame_count = 0
        
        # Accumulated text and response task for debounce
        accumulated_text = []
        response_task = None

        # Task to read transcripts with debounce
        async def read_transcripts():
            nonlocal initial_greeting_sent, response_task
            
            async def delayed_response():
                """Wait for silence, then wait for TTS to finish playing, then respond."""
                await asyncio.sleep(2.0)  # Wait 2 seconds of silence (no new transcripts)
                if accumulated_text and not is_speaking:
                    full_text = " ".join(accumulated_text)
                    accumulated_text.clear()
                    
                    # Estimate TTS playback duration: ~100ms per character for Korean
                    estimated_tts_duration = len(full_text) * 0.1
                    print(f"[BOT] Agent finished generating: {full_text}", flush=True)
                    print(f"[BOT] Waiting {estimated_tts_duration:.1f}s for TTS playback to finish...", flush=True)
                    await asyncio.sleep(estimated_tts_duration)
                    
                    nonlocal initial_greeting_sent
                    if not initial_greeting_sent:
                        initial_greeting_sent = True
                        print(f"[BOT] bot {bot_number} 시작", flush=True)
                    
                    # Wait 3 seconds to allow data packets to reach web UI
                    await asyncio.sleep(3)
                    
                    response = bot_agent.get_response(full_text)
                    await speak(response)
            
            try:
                async for event in stt_stream:
                    if event.type == SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                        transcript = event.alternatives[0].text
                        if transcript and transcript.strip():
                            print(f"[BOT] Agent said: {transcript}", flush=True)
                            accumulated_text.append(transcript.strip())
                            
                            # Cancel previous timer and start new one
                            if response_task and not response_task.done():
                                response_task.cancel()
                            response_task = asyncio.create_task(delayed_response())
            except Exception as e:
                logger.error(f"Error in transcript reader: {e}", exc_info=True)

        transcript_task = asyncio.create_task(read_transcripts())

        # Feed audio to STT
        try:
            async for frame_event in audio_stream:
                frame_count += 1
                if not is_speaking:
                    stt_stream.push_frame(frame_event.frame)
        except Exception as e:
            logger.error(f"Error reading audio stream: {e}")

        await stt_stream.aclose()
        await transcript_task

    # Track active audio processing tasks
    active_audio_tasks: dict[str, asyncio.Task] = {}

    # Track subscription handler - listen to agent's audio
    # IMPORTANT: Register BEFORE publishing our track so we catch all subscriptions
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

    # Publish the track immediately so agent can hear us
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    publication = await room.local_participant.publish_track(audio_track, options)

    # Wait for agent to join
    await asyncio.sleep(2)

    agent_identity = None
    for p in room.remote_participants.values():
        if p.identity.startswith("agent-"):
            agent_identity = p.identity
        # Check for existing audio tracks from this participant
        for sid, track_pub in p.track_publications.items():
            if track_pub.kind == rtc.TrackKind.KIND_AUDIO and p.identity.startswith("agent-"):
                if track_pub.track:
                    task_key = f"{p.identity}_{sid}"
                    if task_key not in active_audio_tasks:
                        task = asyncio.create_task(process_audio_track(track_pub.track, p))
                        active_audio_tasks[task_key] = task

    print(f"[BOT] Ready and listening for agent speech...", flush=True)

    # Start video publishing in background
    video_task = asyncio.create_task(publish_video_from_file(room, bot_number))

    # Keep the bot connected until interrupted
    try:
        await asyncio.Event().wait()
    finally:
        await room.disconnect()


def main() -> None:
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        print("Bot client interrupted, exiting...")


if __name__ == "__main__":
    main()
