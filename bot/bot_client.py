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


# Female video indices (1, 2, 4) and male video index (3)
FEMALE_VIDEOS = [1, 2, 4]
MALE_VIDEO = 3


def get_video_for_bot(bot_number: int) -> int:
    """Get video index for a bot based on its number.
    
    Bot 1-25: Female (randomly picks from videos 1, 2, 4)
    Bot 26-50: Male (video 3)
    Cycles every 50 bots.
    
    Returns:
        video_index (1, 2, 3, or 4)
    """
    import random
    # Use bot_number as seed for reproducibility
    rng = random.Random(bot_number)
    
    # Cycle every 50 bots: 1-25 female, 26-50 male
    position = ((bot_number - 1) % 50) + 1
    
    if position <= 25:
        # Female - randomly pick from videos 1, 2, 4
        return rng.choice(FEMALE_VIDEOS)
    else:
        # Male - video 3
        return MALE_VIDEO


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
        # Covers: greetings, health, daily life, emotions, family, weather, food, memories
        self.responses = [
            # 일상 인사/안부 응답
            "응, 그래 그래. 요즘 허리가 좀 아프긴 해도 잘 지내고 있어.",
            "아이고, 덕분에 잘 지내지. 고마워.",
            "뭐, 그럭저럭 살고 있어. 하루하루가 다 비슷하지 뭐.",
            "응, 오늘은 좀 괜찮아. 어제보다 낫네.",
            
            # 건강 관련 응답
            "아이고, 그러게 말이야. 나이 먹으니까 여기저기 안 아픈 데가 없어.",
            "무릎이 좀 시큰시큰해. 비가 오려나 봐.",
            "오늘은 혈압이 좀 높은 것 같아. 조심해야겠어.",
            "허리가 아파서 오래 못 앉아 있겠어.",
            "눈이 침침해서 글씨가 잘 안 보여.",
            "아이고, 다리에 힘이 없어서 오래 못 걸어.",
            "요즘 잠을 잘 못 자. 새벽에 자꾸 깨.",
            
            # 식사/음식 관련 응답
            "밥은 먹었어? 끼니는 잘 챙겨 먹어야 해.",
            "오늘 아침에 죽 끓여 먹었어. 속이 편하더라고.",
            "점심은 된장찌개 끓여 먹었어. 맛있었어.",
            "배가 좀 고프네... 뭐 먹을까.",
            "요즘은 입맛이 없어서 많이 못 먹어.",
            "옛날에는 밥을 세 공기씩 먹었는데, 이제는 반 공기도 힘들어.",
            
            # 귀가 어두움/다시 말해달라
            "뭐라고? 아, 그래그래. 요즘 귀가 좀 어두워져서 말이야.",
            "뭐라고 했어? 좀 크게 말해줘.",
            "아, 그래? 뭐라고? 다시 한번 말해봐.",
            "귀가 어두워서... 천천히 다시 말해줄래?",
            
            # 감사/긍정 응답
            "그래, 고맙다 고마워. 젊은 사람이 이렇게 챙겨주니 좋구만.",
            "아이고, 고마워라. 네가 말 걸어주니까 덜 외롭네.",
            "그래그래, 좋아. 고맙다.",
            "참 착하구나. 이렇게 신경 써줘서 고마워.",
            
            # 옛날 이야기/추억
            "옛날에는 말이야, 이런 거 없었어. 세상 참 좋아졌어.",
            "우리 젊었을 때는 말이야, 많이 힘들었어.",
            "내가 젊었을 때는 산에 나무하러 많이 다녔지.",
            "옛날 생각이 나네... 그때가 좋았어.",
            "우리 신랑이 살아있을 때는 같이 산책 많이 다녔는데...",
            
            # 자녀/손주 이야기
            "아이고, 맞아 맞아. 요즘 젊은 것들은 바빠서 연락도 잘 안 해.",
            "우리 아들이 요즘 바쁜가 봐. 전화가 안 오네.",
            "손자가 어제 전화했어. 반갑더라고.",
            "우리 며느리가 잘해줘. 고맙지.",
            "손녀가 대학 갔어. 기특하지?",
            "자식들이 다 커서 제 앞가림 하니까 다행이야.",
            
            # 날씨/계절 관련
            "오늘 날씨가 좋네. 산책이라도 나가볼까.",
            "비가 올 것 같아. 빨래 걷어야겠다.",
            "요즘 날이 추워서 바깥에 못 나가.",
            "봄이 오나 봐. 꽃이 피기 시작했어.",
            
            # 일상 활동 응답
            "오늘 아침에 산책 좀 했어. 기분이 좋더라.",
            "텔레비전 보고 있었어. 드라마가 재밌어.",
            "아까 화분에 물 줬어. 꽃이 예쁘게 피었어.",
            "이웃집 아주머니랑 얘기 좀 하고 왔어.",
            
            # 감정 표현
            "아이고, 심심해. 말동무가 없으니까.",
            "오늘은 기분이 좋아. 날씨도 좋고.",
            "좀 우울해... 아무것도 하기 싫어.",
            "외로울 때가 있어. 혼자 있으니까.",
        ]

    def get_response(self, agent_said: str) -> str:
        """Get a simple response based on conversation count."""
        response = self.responses[self.conversation_count % len(self.responses)]
        self.conversation_count += 1
        print(f"[BOT] Response: {response}", flush=True)
        return response


async def publish_video_from_file(room: rtc.Room, bot_number: int):
    """Publish video frames from shared memory broadcaster using I420 format."""
    from video_broadcaster import SharedFrameReader, TARGET_WIDTH, TARGET_HEIGHT, FPS

    video_index = get_video_for_bot(bot_number)

    # Connect to shared frame broadcaster
    frame_reader = SharedFrameReader(video_index)
    if not frame_reader.connect(max_retries=30, retry_delay=1.0):
        print(f"[BOT] ERROR: Could not connect to video broadcaster for video {video_index}", flush=True)
        return

    # Create video source and track with portrait dimensions
    video_source = rtc.VideoSource(width=TARGET_WIDTH, height=TARGET_HEIGHT)
    video_track = rtc.LocalVideoTrack.create_video_track("bot_camera", video_source)

    # Publish the video track
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await room.local_participant.publish_track(video_track, options)
    print(f"[BOT] Video track published from shared broadcaster: video{video_index} @ {FPS}fps -> {TARGET_WIDTH}x{TARGET_HEIGHT} (I420)", flush=True)

    frame_interval = 1 / FPS

    try:
        while True:
            frame_bytes = frame_reader.get_frame_bytes()
            if frame_bytes:
                # Use I420 format - more efficient, less CPU for encoding
                video_frame = rtc.VideoFrame(
                    width=TARGET_WIDTH,
                    height=TARGET_HEIGHT,
                    type=rtc.VideoBufferType.I420,
                    data=frame_bytes,
                )
                video_source.capture_frame(video_frame)
            await asyncio.sleep(frame_interval)
    finally:
        frame_reader.close()



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
                    params={"roomName": room_name}
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
