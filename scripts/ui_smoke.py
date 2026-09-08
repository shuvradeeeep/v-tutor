"""
Headless stand-in for the web page: joins a room with the same token the page
gets, listens on the same data topics ("tutor", "events", "state"), and presses
the same buttons (topic "control"). Proves the worker <-> UI contract without a
browser or a microphone.

    terminal 1:  v-tutor/Scripts/python main.py dev
    terminal 2:  v-tutor/Scripts/python scripts/ui_smoke.py [--room smoke] [--timeout 90]

Exit code 0 when: the tutor joined and published audio, a state snapshot
arrived, the language button produced the topic question, and the topic
button produced a lesson state with sections.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402,F401

from livekit import api, rtc  # noqa: E402


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default=f"smoke-{int(time.time())}")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--topic", default="the heart for class six")
    args = ap.parse_args()
    url, key, secret = os.getenv("LIVEKIT_URL"), os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")
    tok = (api.AccessToken(key, secret).with_identity("smoke").with_name("smoke").with_ttl(timedelta(minutes=30))
           .with_grants(api.VideoGrants(room_join=True, room=args.room, can_publish=True, can_subscribe=True,
                                        can_publish_data=True))
           .with_room_config(api.RoomConfiguration(agents=[api.RoomAgentDispatch(
               agent_name=os.getenv("TUTOR_AGENT_NAME", "v-tutor"))])))

    room = rtc.Room()
    got: dict[str, list] = {"tutor": [], "events": [], "state": []}
    audio = asyncio.Event()
    fresh = asyncio.Condition()

    @room.on("track_subscribed")
    def _sub(track, pub, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"  audio track from {participant.identity}")
            audio.set()

    @room.on("data_received")
    def _data(pkt: rtc.DataPacket):
        if pkt.topic not in got:
            return
        try:
            m = json.loads(pkt.data.decode("utf-8"))
        except Exception:  # noqa: BLE001
            return
        got[pkt.topic].append(m)
        if pkt.topic == "tutor":
            print(f"  [{m.get('role')}|{m.get('kind') or ''}] {m.get('text')}")
        elif pkt.topic == "events":
            print(f"    . {m['name']} {json.dumps(m['payload'], ensure_ascii=False)[:120]}")
        else:
            print(f"    = state onboarding={m.get('onboarding')} topic={m.get('topic')} sections={len(m.get('sections') or [])} "
                  f"provider={m.get('provider')}/{m.get('model')}/{m.get('speaker')}")
        loop.call_soon(lambda: asyncio.ensure_future(_notify()))

    async def _notify():
        async with fresh:
            fresh.notify_all()

    async def wait_for(pred, what: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return True
            async with fresh:
                try:
                    await asyncio.wait_for(fresh.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
        print(f"TIMEOUT waiting for {what}")
        return False

    async def press(phrase: str) -> None:
        print(f">>> button: {phrase!r}")
        await room.local_participant.publish_data(json.dumps({"say": phrase}), reliable=True, topic="control")

    loop = asyncio.get_running_loop()
    print(f"joining {args.room} on {url}")
    await room.connect(url, tok.to_jwt())
    ok = True
    t0 = time.monotonic()
    ok &= await wait_for(lambda: audio.is_set(), "tutor audio track", args.timeout)
    print(f"  tutor joined in {time.monotonic()-t0:.1f}s")
    ok &= await wait_for(lambda: got["state"], "first state snapshot", 60)
    ok &= await wait_for(lambda: any(m.get("role") == "tutor" for m in got["tutor"]), "first tutor line", 60)
    if ok and got["state"][-1].get("onboarding") == "language":
        n = len(got["tutor"])
        await press("English")
        ok &= await wait_for(lambda: len(got["tutor"]) > n, "reply to language button", 60)
    n = len(got["tutor"])
    await press(args.topic)
    ok &= await wait_for(lambda: any(s.get("sections") for s in got["state"]), "lesson state with sections", args.timeout)
    ok &= await wait_for(lambda: any(m.get("kind") == "lesson" for m in got["tutor"]), "a lesson beat", 60)
    await press("stop for today")
    await wait_for(lambda: any(s.get("finished") for s in got["state"]), "finished state", 30)
    await room.disconnect()
    names = sorted({e["name"] for e in got["events"]})
    print(f"\n==== ui smoke: {'PASS' if ok else 'FAIL'} ====\n  tutor lines {len(got['tutor'])}  states {len(got['state'])}  "
          f"events {len(got['events'])}  kinds {names}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
