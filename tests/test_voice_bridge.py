"""The audio <-> graph seam, with no audio: a ScriptedPlayer stands in for the
sound card and Rime is replaced by silence of the right length."""
from __future__ import annotations

import pytest

from agents.session import TurnClock
from conftest import make_deps
from agents.graph import TutorRunner
from voice.bridge import VoiceBridge
from voice.player import ScriptedPlayer
from voice.tts import RimeSpeaker, SilentSynth


def build(auto_play: bool = True, synth=None, **deps_overrides):
    clock = TurnClock()
    player = ScriptedPlayer(auto=auto_play)
    spoken: list[str] = []
    events: list[tuple[str, dict]] = []
    speaker = RimeSpeaker(player, clock, synth=synth or SilentSynth(),
                          on_text=lambda t, m: spoken.append(t), on_event=lambda n, p: events.append((n, p)))
    deps = make_deps(clock=clock, speaker=speaker, on_event=lambda n, p: events.append((n, p)), **deps_overrides)
    runner = TutorRunner(deps, "voice")
    bridge = VoiceBridge(runner, clock, speaker, player, on_event=lambda n, p: events.append((n, p)))
    return bridge, player, spoken, events


def say(bridge, text: str, words_heard: int | None = None, player: ScriptedPlayer | None = None) -> None:
    """What the audio layer does: VAD start (stop now), then the transcript."""
    if words_heard is not None and player is not None:
        player.play_words(words_heard)
    bridge.on_speech_start()
    bridge.on_transcript(text, lang="en", prob=0.9, duration_s=1.0, whisper_ms=300)
    assert bridge.wait_idle(20)


@pytest.fixture
def voice():
    bridge, player, spoken, events = build()
    bridge.start()
    assert bridge.wait_idle(20)
    yield bridge, player, spoken, events
    bridge.close()


def test_onboarding_by_voice(voice):
    bridge, player, spoken, events = voice
    assert "Which language" in spoken[0]
    say(bridge, "English")
    assert bridge.runner.state["active_lang"] == "en"
    say(bridge, "the heart for class six")
    st = bridge.runner.state
    assert st["onboarding_step"] == "done" and st["topic"] == "the heart"
    assert any("muscular organ" in s for s in spoken)
    assert all(kind for kind in (e for e, _ in events))


def onboard_manual():
    """Onboarding with hand-driven playback; returns with the first beat queued, unplayed."""
    bridge, player, spoken, events = build(auto_play=False)
    bridge.start()
    assert bridge.wait_idle(20)
    player.play_all()
    say(bridge, "English")
    player.play_all()
    say(bridge, "the heart for class six")
    # queue: "Give me a moment ...", then the first beat. Step past the wait line.
    cur = player.start_next()
    assert cur is not None and cur.text.startswith("Give me a moment")
    player.finish_current()
    return bridge, player, spoken, events


def test_barge_in_mid_beat_answers_then_resumes_from_heard_cursor():
    bridge, player, spoken, events = onboard_manual()
    cur = player.start_next()                             # the first beat starts "playing"
    assert cur is not None and "muscular organ" in cur.text
    n = len(spoken)
    say(bridge, "How many chambers does the heart have?", words_heard=3, player=player)
    new = spoken[n:]
    assert any("four chambers" in s.lower() for s in new)
    assert any("back to where we were" in s for s in new)
    assert "muscular organ" in new[-1]                    # interrupted early -> the beat is replayed
    vad = [p for e, p in events if e == "vad_start"]
    assert vad and vad[-1]["cursor"]["words_heard"] == 3
    tr = [p for e, p in events if e == "transcript"][-1]
    assert tr["cursor"] == {"word_index": 3}
    bridge.close()


def test_interrupt_while_earlier_item_plays_means_beat_unheard():
    """Answer + bridge + beat are queued; the learner speaks during the answer.
    The cursor must say 0 words of the beat were heard, not 'N words of the answer'."""
    bridge, player, spoken, events = onboard_manual()
    player.start_next()
    say(bridge, "How many chambers does the heart have?", words_heard=3, player=player)
    # queue now holds: answer, bridge, replayed beat. Start the answer, hear 4 words, interrupt.
    cur = player.start_next()
    assert "four chambers" in cur.text.lower()
    say(bridge, "slower", words_heard=4, player=player)
    tr = [p for e, p in events if e == "transcript"][-1]
    assert tr["cursor"] == {"word_index": 0}
    assert bridge.runner.state["speed_alpha"] != 1.0
    bridge.close()


def test_empty_transcript_is_a_backchannel_not_a_clarify():
    bridge, player, spoken, events = onboard_manual()
    player.start_next()
    n = len(spoken)
    say(bridge, "", words_heard=2, player=player)         # cough: VAD fired, Whisper heard nothing
    new = spoken[n:]
    assert new and not any("didn't catch" in s for s in new)
    assert bridge.runner.state["intent"] == "backchannel"
    assert "muscular organ" in new[-1]                    # lesson picked back up from the cursor
    bridge.close()


def test_playback_confirmed_only_when_drained_and_parked():
    bridge, player, spoken, events = build(auto_play=False)
    bridge.start()
    assert bridge.wait_idle(20)
    player.play_all()                                     # "Which language" finished playing
    say(bridge, "English")
    player.play_all()
    say(bridge, "the heart for class six")
    assert bridge.wait_idle(20)
    b0 = bridge.runner.state["beat_index"]
    assert bridge.runner.state["beat_spoken"] is True
    # Graph is parked but the beat is still "playing": no confirm may be sent.
    assert bridge.wait_idle(2)
    assert bridge.runner.state["beat_index"] == b0
    confirms = [e for e, _ in events if e == "playback_confirmed"]
    n_conf = len(confirms)
    player.play_all()                                     # drained -> exactly one confirm
    assert bridge.wait_idle(20)
    assert bridge.runner.state["beat_index"] == b0 + 1
    assert len([e for e, _ in events if e == "playback_confirmed"]) == n_conf + 1
    bridge.close()


def test_barge_in_during_playback_never_confirms_that_audio():
    bridge, player, spoken, events = build(auto_play=False)
    bridge.start()
    assert bridge.wait_idle(20)
    player.play_all()
    say(bridge, "English")
    player.play_all()
    say(bridge, "the heart for class six")
    b0 = bridge.runner.state["beat_index"]
    n_conf = len([e for e, _ in events if e == "playback_confirmed"])
    say(bridge, "pause", words_heard=2, player=player)    # interrupt the first beat
    assert bridge.runner.state["paused"] is True
    player.play_all()                                     # the "Okay, pausing" line finishes
    assert bridge.wait_idle(5)
    # confirm may have been sent for the pause line, but paused => graph stays parked, beat unchanged
    assert bridge.runner.state["beat_index"] == b0
    bridge.close()


def test_tts_rendered_after_a_barge_in_is_dropped():
    clock_ref: dict = {}

    class BumpingSynth(SilentSynth):
        def __init__(self):
            super().__init__()
            self.n = 0

        def synth(self, text, **kw):
            self.n += 1
            if "four chambers" in text.lower():
                clock_ref["clock"].bump()                 # learner speaks while Rime renders the answer
            return None                                   # silence fallback

    bridge, player, spoken, events = build(synth=BumpingSynth())
    clock_ref["clock"] = bridge.clock
    bridge.start()
    assert bridge.wait_idle(20)
    say(bridge, "English")
    say(bridge, "the heart for class six")
    n_items = len(player.enqueued)
    say(bridge, "How many chambers does the heart have?")
    assert not any("four chambers" in i.text.lower() for i in player.enqueued[n_items:])
    assert any(e == "tts_drop_stale" for e, _ in events)
    bridge.close()
