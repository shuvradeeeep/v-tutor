"""
Serve the v-tutor web UI and mint LiveKit join tokens.

    v-tutor/Scripts/python web/server.py                 # http://localhost:8080  (laptop)
    v-tutor/Scripts/python web/server.py --https         # https://<lan-ip>:8443 (phone on the same Wi-Fi)

Run `python main.py dev` in another terminal: the token carries an agent
dispatch, so the tutor joins whichever room the page creates.

Phones only allow the microphone on HTTPS. `--https` generates a self-signed
certificate under .cache/certs on first use; accept the browser warning once.
The LiveKit secret never leaves this process -- the page only ever sees a
short-lived room token.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402  (loads .env)

from aiohttp import web  # noqa: E402
from livekit import api  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
CERT_DIR = ROOT / ".cache" / "certs"


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _self_signed(host: str) -> tuple[Path, Path]:
    """Generate (or reuse) a self-signed cert that names localhost and the LAN IP."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    import ipaddress

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    crt, key = CERT_DIR / f"{host}.crt", CERT_DIR / f"{host}.key"
    if crt.exists() and key.exists():
        return crt, key
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "v-tutor")])
    san = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        san.append(x509.IPAddress(ipaddress.ip_address(host)))
    except ValueError:
        san.append(x509.DNSName(host))
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=365)).add_extension(x509.SubjectAlternativeName(san), False)
            .sign(k, hashes.SHA256()))
    key.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                    serialization.NoEncryption()))
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return crt, key


def make_app(agent_name: str) -> web.Application:
    url, key, secret = os.getenv("LIVEKIT_URL"), os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")
    if not (url and key and secret):
        sys.exit("LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET missing in .env")

    async def token(req: web.Request) -> web.Response:
        room = (req.query.get("room") or "demo").strip()[:64] or "demo"
        identity = (req.query.get("identity") or "learner").strip()[:64] or "learner"
        tok = (api.AccessToken(key, secret).with_identity(identity).with_name(identity)
               .with_ttl(timedelta(hours=3))
               .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True,
                                            can_subscribe=True, can_publish_data=True)))
        if agent_name:
            tok = tok.with_room_config(api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=agent_name)]))
        return web.json_response({"url": url, "token": tok.to_jwt(), "room": room, "identity": identity})

    async def info(_: web.Request) -> web.Response:
        # What the page shows in its provider badge before the worker reports in.
        return web.json_response({
            "tts": {"provider": "rime" if os.getenv("RIME_API_KEY") else "fallback",
                    "model": config.RIME_MODEL_ID, "speakers": config.LANG_SPEAKER,
                    "format": f"{config.RIME_AUDIO_FORMAT} {config.RIME_SAMPLE_RATE} Hz",
                    "endpoint": config.RIME_HTTP_ENDPOINT},
            "stress_ms": config.STRESS_DELAY_MS,
            "langs": {k: config.LANG_NAMES.get(k, k) for k in ("en", "hi")},
        })

    async def upload(req: web.Request) -> web.Response:
        upload_dir = ROOT / ".cache" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        try:
            reader = await req.multipart()
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        saved_paths: list[str] = []
        saved_names: list[str] = []
        async for field in reader:
            if field.name != "pdf":
                continue
            filename = Path(field.filename or "upload.pdf").name
            dest = upload_dir / filename
            with open(dest, "wb") as fh:
                while True:
                    chunk = await field.read_chunk(65536)
                    if not chunk:
                        break
                    fh.write(chunk)
            saved_paths.append(str(dest))
            saved_names.append(filename)
        if not saved_paths:
            return web.json_response({"ok": False, "error": "No PDF field in request"}, status=400)
        return web.json_response({"ok": True, "paths": saved_paths, "names": saved_names})

    async def index(_: web.Request) -> web.FileResponse:
        return web.FileResponse(STATIC / "index.html")

    app = web.Application(client_max_size=100 * 1024 * 1024)  # 100 MB max upload
    app.router.add_get("/", index)
    app.router.add_get("/token", token)
    app.router.add_get("/info", info)
    app.router.add_post("/upload", upload)
    app.router.add_static("/static", STATIC)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--https", action="store_true", help="self-signed TLS so a phone can use its mic")
    ap.add_argument("--agent", default=os.getenv("TUTOR_AGENT_NAME", "v-tutor"))
    args = ap.parse_args()
    host_ip = _lan_ip()
    port = args.port or (8443 if args.https else 8080)
    ctx = None
    if args.https:
        crt, key = _self_signed(host_ip)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(crt, key)
    scheme = "https" if ctx else "http"
    print(f"\nv-tutor web UI\n  laptop:  {scheme}://localhost:{port}\n  phone:   {scheme}://{host_ip}:{port}"
          f"   (same Wi-Fi{'; accept the certificate warning once' if ctx else '; mic needs --https on phones'})"
          f"\n  worker:  python main.py dev   (agent '{args.agent}')\n")
    web.run_app(make_app(args.agent), host="0.0.0.0", port=port, ssl_context=ctx, print=None)


if __name__ == "__main__":
    main()
