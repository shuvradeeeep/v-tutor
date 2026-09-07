"""
Mint a LiveKit join token for a browser client and print a ready-to-open link.

    v-tutor/Scripts/python scripts/room_token.py --room demo --identity student

Open the printed URL (LiveKit Meet, custom tab) with the mic allowed; the
tutor worker started with `python main.py dev` joins the same room.
"""
from __future__ import annotations

import argparse
import sys
import urllib.parse
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402,F401  (loads .env)
import os  # noqa: E402

from livekit import api  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="demo")
    ap.add_argument("--identity", default="student")
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--agent", default=os.getenv("TUTOR_AGENT_NAME", "v-tutor"),
                    help="agent_name the worker registered with (main.py); '' for auto-dispatch only")
    args = ap.parse_args()
    url, key, secret = os.getenv("LIVEKIT_URL"), os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")
    if not (url and key and secret):
        sys.exit("LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET missing in .env")
    tok = (api.AccessToken(key, secret).with_identity(args.identity).with_name(args.identity)
           .with_ttl(timedelta(hours=args.hours))
           .with_grants(api.VideoGrants(room_join=True, room=args.room, can_publish=True,
                                        can_subscribe=True, can_publish_data=True)))
    if args.agent:
        # Dispatch the tutor to this room when this participant joins -- works
        # whether the room is new or already exists.
        tok = tok.with_room_config(api.RoomConfiguration(
            agents=[api.RoomAgentDispatch(agent_name=args.agent)]))
    token = tok.to_jwt()
    print(f"room:     {args.room}\nidentity: {args.identity}\nagent:    {args.agent or '(auto-dispatch)'}\n"
          f"url:      {url}\ntoken:    {token}\n")
    q = urllib.parse.urlencode({"liveKitUrl": url, "token": token})
    print(f"open:     https://meet.livekit.io/custom?{q}")


if __name__ == "__main__":
    main()
