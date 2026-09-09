"""
v-tutor voice entrypoint: microphone/LiveKit -> Silero VAD -> Whisper ->
LangGraph tutor -> Rime -> speakers/LiveKit.

Two ways to run (see docs/VOICE_PIPELINE.md):

  Local, no LiveKit -- laptop mic and speakers (wear headphones):
      v-tutor/Scripts/python main.py local [--lang en] [--pdf notes.pdf]
                                           [--session demo] [--no-evidence]

  LiveKit room -- the agent joins any room created on LIVEKIT_URL:
      v-tutor/Scripts/python main.py dev            # console logs, auto-reload
      v-tutor/Scripts/python main.py start          # production mode
  then join the room from a browser (scripts/room_token.py prints a link).
  Env for this mode: TUTOR_LANG=en (skip the language question),
  TUTOR_PDF=path.pdf, TUTOR_EVIDENCE_DIR=evidence.

What you see: every tutor line and every transcript on the console, all events
in evidence/<session>.csv, and in LiveKit mode the same lines as data messages
on topic "tutor" (JSON {"role": "tutor"|"learner", "text": ...}).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import config  # noqa: E402  (loads .env)

LOG_DIR = Path(os.getenv("TUTOR_LOG_DIR", "logs"))
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "voice.log", encoding="utf-8")],
)
for noisy in ("httpx", "httpcore", "openai", "urllib3", "livekit"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("v-tutor.main")

SHOW_EVENTS = {"vad_start", "transcript", "intent", "retrieve", "answer_mode", "web_search", "fence_drop",
               "discard", "tts_drop_stale", "tts_fallback", "filler", "playback_confirmed", "ingest",
               "localized_bg", "navigate", "interrupt", "graph_turn_done", "onboarding_reask",
               "echo_drop", "half_duplex_on", "switch_topic"}


# Events the web page reacts to (topic "events"). Everything else stays in
# the console and the evidence CSV.
UI_EVENTS = {"vad_start", "barge_in_deferred", "transcript", "echo_drop", "intent", "filler", "web_search", "web_aborted", "fragment_hold", "fragment_release",
             "tts_drop_stale", "fence_drop", "discard", "tts_fallback", "tts", "half_duplex_on",
             "playback_confirmed", "graph_turn_done", "second_learner_ignored"}


def _ui_state(bridge) -> dict:
    """What the page needs to draw the lesson header and progress bar."""
    s = bridge.runner.state or {}
    plan = s.get("lesson_plan") or []
    titles: list[str] = []
    for b in plan:
        t = b.get("section_title") if isinstance(b, dict) else getattr(b, "section_title", None)
        if t and (not titles or titles[-1] != t):
            titles.append(t)
    bi = int(s.get("beat_index") or 0)
    cur = plan[bi] if 0 <= bi < len(plan) else None
    cur_title = (cur.get("section_title") if isinstance(cur, dict) else getattr(cur, "section_title", None)) if cur else None
    return {
        "onboarding": s.get("onboarding_step"),
        "topic": s.get("topic"), "grade": s.get("grade"), "source_title": s.get("source_title"),
        "source_url": s.get("source_url"), "lang": s.get("active_lang"),
        "sections": titles, "section_index": titles.index(cur_title) if cur_title in titles else -1,
        "beat_index": bi, "beat_spoken": bool(s.get("beat_spoken")), "beats_total": len(plan),
        "paused": bool(s.get("paused")), "finished": bridge.finished,
        "stale_drops": s.get("stale_drops", 0),
        "provider": bridge.speaker.provider, "model": config.RIME_MODEL_ID,
        "speaker": config.LANG_SPEAKER.get(s.get("active_lang") or "en"),
        "stress_ms": config.STRESS_DELAY_MS,
    }


def _console_event(name: str, p: dict) -> None:
    if name == "transcript":
        # The learner's own words, printed like the tutor's so a session reads
        # as a conversation in the terminal, not as a stream of events.
        print(f"\n  [learner | {p.get('lang')} | whisper {p.get('whisper_ms')} ms]\n  {p.get('text') or '(nothing heard)'}\n")
    if name in SHOW_EVENTS:
        print(f"     . {name} {p}")


def _console_text(text: str, meta: dict) -> None:
    print(f"\n  [tutor | turn {meta.get('turn')} | {meta.get('lang')} | {meta.get('speaker')}]\n  {text}\n")


def _evidence_path(session: str, enabled: bool) -> str | None:
    if not enabled:
        return None
    d = Path(os.getenv("TUTOR_EVIDENCE_DIR", config.EVIDENCE_DIR))
    return str(d / f"{session}.csv")


# --------------------------------------------------------------------------
# local: laptop mic + speakers
# --------------------------------------------------------------------------

async def run_local(args: argparse.Namespace) -> None:
    from stt import settings as stt_settings
    from voice.audio_io import SpeechInput, mic_frames
    from voice.bridge import make_bridge
    from voice.player import LocalPlayer

    session = args.session or f"local-{int(time.time())}"
    player = LocalPlayer(device=args.output_device)
    bridge = make_bridge(
        player, session_id=session, pdf_paths=args.pdf, preset_lang=args.lang,
        evidence_path=_evidence_path(session, not args.no_evidence),
        checkpoint_db=None if args.no_persist else config.CHECKPOINT_DB,
        on_text=_console_text, on_event=_console_event if not args.quiet else None,
        stress_delay_ms=args.stress_ms,
        half_duplex=True if args.no_headphones else None,
    )
    print(f"\nsession={session}  tts={bridge.speaker.provider}/{config.RIME_MODEL_ID}  "
          f"llm={bridge.runner.deps.llm_fast.provider}/{bridge.runner.deps.llm_strong.model}  "
          f"whisper={stt_settings.WHISPER_MODEL_SIZE} (beam {stt_settings.WHISPER_BEAM_SIZE}, "
          f"{stt_settings.WHISPER_CPU_THREADS} threads)  stress={args.stress_ms}ms")
    print(f"transcripts: {stt_settings.TRANSCRIPTS_CSV}")
    # Which speaker the tutor's voice is actually going to. Windows routes to
    # the headphone jack by default on this hardware, so with no earphones
    # plugged in the whole pipeline looks healthy and you hear nothing.
    try:
        import sounddevice as sd
        out = args.output_device if args.output_device is not None else sd.default.device[1]
        in_ = args.input_device if args.input_device is not None else sd.default.device[0]
        print(f"audio out:   [{out}] {sd.query_devices(out)['name']}   "
              f"in: [{in_}] {sd.query_devices(in_)['name']}")
        print("             (list them: python -c \"import sounddevice; print(sounddevice.query_devices())\";"
              " pick with --output-device N / --input-device N)")
    except Exception as exc:  # noqa: BLE001
        print(f"audio devices unavailable: {exc}")
    if args.no_headphones:
        print("Speakers mode: the mic is ignored while the tutor talks, so you cannot interrupt it."
              "\nSpeak after it stops. Ctrl+C to quit.\n")
    else:
        print("Wear headphones (the mic must not hear the tutor), then interrupt whenever you like."
              "\nOn speakers, run with --no-headphones. Ctrl+C to quit.\n")

    speech = SpeechInput()
    speech.load()                       # models load before the tutor says anything
    bridge.start()
    stop = asyncio.Event()

    def _sig(*_: object) -> None:
        stop.set()
    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, _sig)
    except (NotImplementedError, RuntimeError):
        signal.signal(signal.SIGINT, lambda *_: stop.set())

    mic_task = asyncio.create_task(speech.run_frames(mic_frames(args.input_device), bridge, label="mic"))
    try:
        while not stop.is_set() and not bridge.finished:
            await asyncio.sleep(0.2)
    finally:
        mic_task.cancel()
        try:
            await mic_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        bridge.wait_idle(5)
        bridge.close()
        speech.close()
        ev = getattr(bridge, "evidence", None)
        if ev:
            ev.close()
            print(f"\nevidence: {ev.path} ({ev.rows} rows)")
        print(f"transcripts: {stt_settings.TRANSCRIPTS_CSV}")
        print(f"turns: {len(bridge.turns)}  stale drops: {bridge.runner.state.get('stale_drops')}  "
              f"rime calls: {bridge.speaker.synth.calls} (cache hits {bridge.speaker.synth.cache_hits})")
        print("session over.")


# --------------------------------------------------------------------------
# LiveKit worker: joins rooms, publishes the tutor's audio track
# --------------------------------------------------------------------------

def run_livekit() -> None:
    from livekit import agents, rtc
    # LiveKit plugins register themselves on import and refuse to do so off the
    # main thread. On Windows the job runs in a thread, so import Silero here,
    # before the worker starts; stt/vad.py then gets the cached module.
    from livekit.plugins import silero  # noqa: F401

    from voice.audio_io import SpeechInput
    from voice.bridge import make_bridge
    from voice.player import LiveKitPlayer

    speech = SpeechInput()

    def prewarm(proc: agents.JobProcess) -> None:
        speech.load()
        proc.userdata["speech"] = speech

    async def entrypoint(ctx: agents.JobContext) -> None:
        await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)
        room = ctx.room
        loop = asyncio.get_running_loop()
        logger.info("joined room %s", room.name)

        source = rtc.AudioSource(16000, 1, queue_size_ms=200)
        track = rtc.LocalAudioTrack.create_audio_track("tutor-voice", source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))

        def publish_json(topic: str, obj: dict) -> None:
            payload = json.dumps(obj, ensure_ascii=False, default=str)
            asyncio.run_coroutine_threadsafe(
                room.local_participant.publish_data(payload, reliable=True, topic=topic), loop)

        def publish(role: str, text: str, meta: dict | None = None) -> None:
            publish_json("tutor", {"role": role, "text": text, **(meta or {})})

        bridge = None   # assigned below; the callbacks run after it exists

        def publish_state() -> None:
            """A compact snapshot for the UI: lesson, progress, provider. Sent
            after every graph turn, so the page never has to infer state."""
            if bridge is None:
                return
            try:
                publish_json("state", _ui_state(bridge))
            except Exception:  # noqa: BLE001
                logger.exception("state snapshot failed")

        def on_text(text: str, meta: dict) -> None:
            _console_text(text, meta)

        def on_event(name: str, p: dict) -> None:
            _console_event(name, p)
            if name == "speak":
                publish("tutor", p.get("text") or "", {"turn": p.get("turn"), "kind": p.get("kind"), "lang": p.get("lang")})
            elif name == "transcript":
                publish("learner", p.get("text") or "", {"lang": p.get("lang")})
            if name in UI_EVENTS:
                publish_json("events", {"name": name, "payload": p})
            if name in ("graph_turn_done", "ingest", "playback_confirmed", "navigate", "switch_topic"):
                publish_state()

        session = f"{room.name}-{int(time.time())}"
        pdf = [p for p in (os.getenv("TUTOR_PDF") or "").split(";") if p.strip()] or None
        player = LiveKitPlayer(source, loop, queue_size_ms=200)
        bridge = make_bridge(
            player, session_id=session, pdf_paths=pdf, preset_lang=os.getenv("TUTOR_LANG") or None,
            evidence_path=_evidence_path(session, True), checkpoint_db=config.CHECKPOINT_DB,
            on_text=on_text, on_event=on_event,
        )
        sp = ctx.proc.userdata.get("speech") or speech
        sp.load()

        @room.on("data_received")
        def on_data(pkt: rtc.DataPacket) -> None:
            # Buttons on the web page. A press is delivered exactly like a
            # spoken utterance, so it goes through the same intent rules,
            # the same fence, and shows up in the same evidence CSV.
            if pkt.topic != "control":
                return
            try:
                msg = json.loads(pkt.data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return
            phrase = str(msg.get("say") or "").strip()[:200]
            if phrase:
                who = pkt.participant.identity if pkt.participant else "ui"
                logger.info("button from %s: %r", who, phrase)
                bridge.on_transcript(phrase, lang=None, prob=1.0)
            pdf_paths = msg.get("load_pdf")
            if pdf_paths and isinstance(pdf_paths, list):
                who = pkt.participant.identity if pkt.participant else "ui"
                logger.info("pdf upload from %s: %s", who, pdf_paths)
                bridge.load_pdf(pdf_paths)

        tasks: set[asyncio.Task] = set()
        # One learner per room. A second device joining the same room used to
        # get its microphone processed too, so every utterance arrived twice
        # (two VAD starts, two transcripts, two graph turns). The first audio
        # track wins; later participants are told and ignored.
        learner: dict[str, object] = {"identity": None, "task": None}

        @room.on("track_subscribed")
        def on_track(track: rtc.Track, pub: rtc.TrackPublication, participant: rtc.RemoteParticipant) -> None:
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            if learner["identity"] not in (None, participant.identity):
                logger.warning("ignoring a second microphone from %s (learner is %s)",
                               participant.identity, learner["identity"])
                publish_json("events", {"name": "second_learner_ignored",
                                        "payload": {"identity": participant.identity, "learner": learner["identity"]}})
                return
            old = learner["task"]
            if old is not None and not old.done():
                old.cancel()                     # same learner re-published (reconnect): one listener only
            logger.info("listening to %s", participant.identity)
            t = asyncio.create_task(sp.run_track(track, bridge, label=participant.identity))
            learner["identity"], learner["task"] = participant.identity, t
            tasks.add(t)
            t.add_done_callback(tasks.discard)

        @room.on("participant_disconnected")
        def on_leave(participant: rtc.RemoteParticipant) -> None:
            if participant.identity == learner["identity"]:
                logger.info("learner %s left; the next microphone to join is the learner", participant.identity)
                learner["identity"] = None

        # A participant may already be in the room when we join.
        for participant in room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track is not None and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                    on_track(pub.track, pub, participant)

        async def shutdown() -> None:
            for t in list(tasks):
                t.cancel()
            bridge.wait_idle(5)
            bridge.close()
            sp.close()
            ev = getattr(bridge, "evidence", None)
            if ev:
                ev.close()
                logger.info("evidence: %s (%d rows)", ev.path, ev.rows)
        ctx.add_shutdown_callback(shutdown)

        bridge.start()
        bridge.wait_idle(60)            # runner.start() has run: state exists
        publish_state()
        while not bridge.finished and room.connection_state != rtc.ConnectionState.CONN_DISCONNECTED:
            await asyncio.sleep(0.5)
        publish_state()
        logger.info("lesson finished or room gone; leaving")

    # Named agent = explicit dispatch. The join token from scripts/room_token.py
    # carries a RoomAgentDispatch for this name, so the tutor is dispatched on
    # EVERY join, even into a room that already exists. (Auto-dispatch only fires
    # when a room is created, which is how a stale room once left us waiting.)
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm,
                                            agent_name=os.getenv("TUTOR_AGENT_NAME", "v-tutor")))


# --------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "local":
        ap = argparse.ArgumentParser(prog="main.py local")
        ap.add_argument("--lang", default=None, help="en to skip the spoken language question")
        ap.add_argument("--pdf", nargs="*", default=None)
        ap.add_argument("--session", default=None)
        ap.add_argument("--stress-ms", type=int, default=config.STRESS_DELAY_MS)
        ap.add_argument("--input-device", default=None)
        ap.add_argument("--output-device", default=None)
        ap.add_argument("--no-headphones", action="store_true",
                        help="speakers: ignore the mic while the tutor talks (no barge-in, no echo)")
        ap.add_argument("--no-evidence", action="store_true")
        ap.add_argument("--no-persist", action="store_true", help="in-memory checkpoints (fresh session)")
        ap.add_argument("--quiet", action="store_true", help="hide internal events")
        args = ap.parse_args(sys.argv[2:])
        for k in ("input_device", "output_device"):
            v = getattr(args, k)
            if v is not None and str(v).isdigit():
                setattr(args, k, int(v))
        asyncio.run(run_local(args))
        return
    run_livekit()                       # dev | start | download-files ... handled by livekit cli


if __name__ == "__main__":
    main()
