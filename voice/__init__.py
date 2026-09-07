"""
voice/ -- the seam between the audio layer (stt/, Rime) and the agent layer
(agents/). Nothing in agents/ imports this package; nothing here changes how
the graph works. It only turns audio into the two events the graph already
understands (user_barge_in, playback_confirmed) and turns the graph's text
into audio.

  bridge.py    VoiceBridge: VAD start -> stop now; transcript -> barge-in;
               drained player + parked graph -> playback_confirmed.
  tts.py       RimeSpeaker: text -> PCM (Rime HTTP, cached) -> player, fenced.
  player.py    Players: local speakers (sounddevice) or a LiveKit audio track.
  audio_io.py  Microphone / LiveKit track -> stt.vad -> stt.transcriber.
"""
