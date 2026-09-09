// v-tutor web client. Joins the LiveKit room, plays the tutor's track, shows
// the transcript and live state, and sends button presses down the same path
// as spoken words (the worker feeds them to the bridge as transcripts).
import {
  Room, RoomEvent, Track, ParticipantEvent, ConnectionState,
} from "https://cdn.jsdelivr.net/npm/livekit-client@2/dist/livekit-client.esm.mjs";

const $ = (id) => document.getElementById(id);
const enc = new TextEncoder(), dec = new TextDecoder();

const ui = {
  join: $("join"), session: $("session"), form: $("join-form"), start: $("start"),
  identity: $("identity"), room: $("room"), hint: $("join-hint"),
  conn: $("conn"), provider: $("provider"), stress: $("stress"),
  topic: $("topic"), grade: $("grade"), bar: $("bar"), sections: $("sections"),
  orb: $("orb"), status: $("status"), metric: $("metric"), lang: $("lang-chips"),
  feed: $("feed"), mic: $("mic"), pause: $("pause"),
};

let room = null;
let tutor = null;

// A fresh room per visit. Two devices in the same room means two microphones
// on one tutor, and every utterance arrives twice. Editable for a shared demo.
ui.room.value = new URLSearchParams(location.search).get("room")
  || `lesson-${Math.random().toString(36).slice(2, 6)}`;
let pending = null;          // "you" entry shown while the transcript is on its way
let paused = false;

// ---------------------------------------------------------------- view
function setOrb(kind, status, metric) {
  ui.orb.className = `orb ${kind}`;
  if (status !== undefined) ui.status.textContent = status;
  if (metric !== undefined) ui.metric.textContent = metric;
}

function entry(cls, who, text) {
  const el = document.createElement("div");
  el.className = `entry ${cls}`;
  if (who) {
    const w = document.createElement("div");
    w.className = "who"; w.textContent = who; el.appendChild(w);
  }
  const t = document.createElement("div");
  t.className = "text"; t.textContent = text; el.appendChild(t);
  ui.feed.appendChild(el);
  ui.feed.scrollTop = ui.feed.scrollHeight;
  while (ui.feed.children.length > 200) ui.feed.firstChild.remove();
  return el;
}
const note = (text, cls = "") => entry(`note ${cls}`, null, text);

function clearPending() {
  if (pending) { pending.remove(); pending = null; }
}

async function send(topic, obj) {
  if (!room || room.state !== ConnectionState.Connected) return;
  await room.localParticipant.publishData(enc.encode(JSON.stringify(obj)), { reliable: true, topic });
}

function say(phrase) {
  clearPending();
  pending = entry("you pending", "You", phrase);
  send("control", { say: phrase });
}

function setPaused(p) {
  paused = p;
  ui.pause.querySelector("span").textContent = p ? "Continue" : "Pause";
  if (p) setOrb("idle", "Paused", "");
}

// ---------------------------------------------------------------- state from the worker
function applyState(s) {
  if (s.topic) {
    ui.topic.textContent = s.source_title || s.topic;
    ui.grade.textContent = s.grade ? `Class ${String(s.grade).replace(/^class\s*/i, "")}` : "";
  } else {
    ui.topic.textContent = s.onboarding === "language" ? "Choose a language"
      : s.onboarding === "source" ? "What shall we learn" : "Getting ready";
    ui.grade.textContent = "";
  }
  ui.lang.classList.toggle("hidden", s.onboarding !== "language");
  ui.bar.style.width = s.beats_total
    ? `${Math.min(100, (100 * (s.beat_index + (s.beat_spoken ? 1 : 0))) / s.beats_total)}%` : "0";
  if (Array.isArray(s.sections)) {
    ui.sections.replaceChildren(...s.sections.map((title, i) => {
      const el = document.createElement("span");
      el.textContent = title;
      if (i < s.section_index) el.classList.add("done");
      if (i === s.section_index) el.classList.add("now");
      return el;
    }));
    ui.sections.querySelector(".now")?.scrollIntoView({ inline: "center", block: "nearest" });
  }
  if (s.provider) setProvider(s.provider === "rime", s.model, s.speaker);
  if (s.stress_ms) { ui.stress.textContent = `+${s.stress_ms} ms tool delay`; ui.stress.classList.remove("hidden"); }
  if (s.paused !== undefined && s.paused !== paused) setPaused(s.paused);
  if (s.finished) { setOrb("idle", "Lesson over", ""); ui.lang.classList.add("hidden"); }
}

function setProvider(rime, model, speaker) {
  ui.provider.textContent = rime ? `Rime ${model} · ${speaker}` : "Fallback voice";
  ui.provider.classList.toggle("fallback", !rime);
}

// ---------------------------------------------------------------- events from the worker
function onEvent(e) {
  const p = e.payload || {};
  switch (e.name) {
    case "vad_start":
      setOrb("stopped", "Stopped", `${p.stop_ms} ms`);
      if (p.cursor && p.cursor.words_heard != null) note(`Stopped after ${p.cursor.words_heard} words`);
      clearPending(); pending = entry("you pending", "You", "…");
      break;
    case "barge_in_deferred":
      setOrb("listening", "Listening", "");
      clearPending(); pending = entry("you pending", "You", "…");
      break;
    case "transcript":
      clearPending();
      if (p.held) { pending = entry("you pending", "You", p.text); break; }
      if (p.text) entry("you", "You", p.text); else note("Nothing heard, carrying on");
      setOrb("thinking", "Thinking", p.whisper_ms ? `heard in ${p.whisper_ms} ms` : "");
      break;
    case "echo_drop":
      clearPending();
      note("Ignored an echo of the tutor");
      break;
    case "intent":
      if (p.session_cmd === "pause") setPaused(true);
      if (p.session_cmd === "continue") setPaused(false);
      break;
    case "filler":
      setOrb("thinking", "Looking that up", "");
      break;
    case "tts_drop_stale":
    case "fence_drop":
    case "discard":
      note("Dropped a stale reply", "warn");
      break;
    case "web_aborted":
      note("Cancelled a lookup you moved on from", "warn");
      break;
    case "tts_fallback":
      setProvider(false);
      note("Rime did not answer, line skipped", "warn");
      break;
    case "tts":
      if (!p.cached && p.first_audio_ms) ui.metric.textContent = `first audio ${p.first_audio_ms} ms`;
      break;
    case "second_learner_ignored":
      note(`Another microphone (${p.identity}) joined this room and is ignored. Use one device per room`, "warn");
      break;
    case "half_duplex_on":
      note("The mic hears the tutor. Barge-in is off for this session", "warn");
      break;
    case "playback_confirmed":
      if (!paused) setOrb("listening", "Listening", "");
      break;
  }
}

function onSpeak(m) {
  const kind = m.kind || "system";
  entry(kind, "Tutor", m.text);
  setOrb("speaking", "Speaking");
}

// ---------------------------------------------------------------- room
async function connect(ev) {
  ev?.preventDefault();
  const identity = (ui.identity.value || "learner").trim();
  const roomName = (ui.room.value || "demo").trim();
  ui.start.disabled = true; ui.start.textContent = "Connecting";
  ui.hint.textContent = "";
  let cfg;
  try {
    const r = await fetch(`/token?room=${encodeURIComponent(roomName)}&identity=${encodeURIComponent(identity)}`);
    if (!r.ok) throw new Error(await r.text());
    cfg = await r.json();
  } catch (err) {
    ui.start.disabled = false; ui.start.textContent = "Start";
    ui.hint.textContent = `Could not get a room token. ${err.message}`;
    return;
  }

  room = new Room({
    adaptiveStream: false, dynacast: false,
    // Web Audio mixing instead of a bare <audio> element: on phones this is
    // the path that most often lands on the loudspeaker rather than the
    // earpiece once the microphone is open.
    webAudioMix: true,
    audioCaptureDefaults: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });

  room.on(RoomEvent.ConnectionStateChanged, (st) => {
    ui.conn.className = `dot ${st === ConnectionState.Connected ? "ok" : st === ConnectionState.Disconnected ? "bad" : ""}`;
  });
  room.on(RoomEvent.TrackSubscribed, (track, pub, participant) => {
    if (track.kind !== Track.Kind.Audio) return;
    tutor = participant;
    const el = track.attach();
    el.autoplay = true; el.playsInline = true;
    document.body.appendChild(el);
    setOrb("listening", "Listening", "");
    participant.on(ParticipantEvent.IsSpeakingChanged, (speaking) => {
      if (speaking) setOrb("speaking", "Speaking");
      else if (!paused && ui.orb.classList.contains("speaking")) setOrb("listening", "Listening");
    });
  });
  room.on(RoomEvent.TrackUnsubscribed, (track) => track.detach().forEach((el) => el.remove()));
  room.on(RoomEvent.ParticipantDisconnected, (p) => {
    if (p === tutor) { tutor = null; setOrb("idle", "Tutor left", ""); }
  });
  room.on(RoomEvent.DataReceived, (payload, participant, kind, topic) => {
    let m; try { m = JSON.parse(dec.decode(payload)); } catch { return; }
    if (topic === "tutor") { if (m.role === "tutor") onSpeak(m); return; }
    if (topic === "events") onEvent(m);
    if (topic === "state") applyState(m);
  });
  room.on(RoomEvent.AudioPlaybackStatusChanged, () => {
    if (!room.canPlaybackAudio) {
      const el = note("Tap to enable sound", "tap");
      el.onclick = () => { room.startAudio(); el.remove(); };
    }
  });

  try {
    await room.connect(cfg.url, cfg.token);
    await room.localParticipant.setMicrophoneEnabled(true);
    await room.startAudio();
    preferLoudspeaker();
  } catch (err) {
    ui.start.disabled = false; ui.start.textContent = "Start";
    ui.hint.textContent = `Could not join. ${err.message}`;
    return;
  }

  ui.join.classList.add("hidden"); ui.session.classList.remove("hidden");
  setOrb("thinking", "Waiting for the tutor", "");
  setTimeout(() => { if (!tutor) note("No tutor yet. Is the worker running?"); }, 8000);

  const tick = () => {
    if (!room) return;
    const lvl = room.localParticipant.audioLevel || 0;
    ui.orb.style.setProperty("--level", (1 + Math.min(0.5, lvl * 2)).toFixed(3));
    requestAnimationFrame(tick);
  };
  tick();
}

// Phones treat an open microphone as a call and may route playback to the
// earpiece. Where the browser lets a page choose the output (Android Chrome),
// pick the loudspeaker. iOS offers no such choice; there the Web Audio mix
// above is the only lever, and earphones are the sure fix.
async function preferLoudspeaker() {
  try {
    const outs = await Room.getLocalDevices("audiooutput");
    const speaker = outs.find((d) => /speaker|loud/i.test(d.label || ""));
    if (speaker) {
      await room.switchActiveDevice("audiooutput", speaker.deviceId);
      console.info("audio output:", speaker.label);
    }
  } catch (err) {
    console.info("output selection not available here", err?.message || err);
  }
}

// ---------------------------------------------------------------- controls
ui.form.addEventListener("submit", connect);
document.querySelectorAll("[data-say]").forEach((b) => b.addEventListener("click", () => {
  const phrase = b.dataset.toggle && paused ? b.dataset.toggle : b.dataset.say;
  say(phrase);
  if (b.classList.contains("end")) setTimeout(() => room?.disconnect(), 6000);
}));
ui.mic.addEventListener("click", async () => {
  if (!room) return;
  const on = !room.localParticipant.isMicrophoneEnabled;
  await room.localParticipant.setMicrophoneEnabled(on);
  ui.mic.classList.toggle("on", on); ui.mic.classList.toggle("off", !on);
  ui.mic.querySelector("span").textContent = on ? "Mic" : "Muted";
});

// ---------------------------------------------------------------- PDF upload
const pdfBtn = document.getElementById("pdf-btn");
const pdfInput = document.getElementById("pdf-input");
pdfBtn.addEventListener("click", () => pdfInput.click());
pdfInput.addEventListener("change", async () => {
  const files = pdfInput.files;
  if (!files || files.length === 0) return;
  const formData = new FormData();
  for (const f of files) formData.append("pdf", f);
  pdfBtn.querySelector("span").textContent = "Uploading…";
  pdfBtn.disabled = true;
  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const json = await res.json();
    if (json.ok) {
      note(`📄 Loaded: ${json.names.join(", ")} — teaching from your document now`);
      send("control", { load_pdf: json.paths });
    } else {
      note(`PDF upload failed: ${json.error || "unknown error"}`, "warn");
    }
  } catch (err) {
    note(`PDF upload error: ${err.message}`, "warn");
  } finally {
    pdfBtn.querySelector("span").textContent = "PDF";
    pdfBtn.disabled = false;
    pdfInput.value = "";
  }
});

window.addEventListener("pagehide", () => room?.disconnect());

fetch("/info").then((r) => r.json()).then((i) => {
  setProvider(i.tts.provider === "rime", i.tts.model, i.tts.speakers.en);
  if (i.stress_ms) { ui.stress.textContent = `+${i.stress_ms} ms tool delay`; ui.stress.classList.remove("hidden"); }
  if (location.protocol === "http:" && !/localhost|127\.0\.0\.1/.test(location.hostname))
    ui.hint.textContent = "Phones need HTTPS for the microphone. Start the server with --https.";
}).catch(() => {});
