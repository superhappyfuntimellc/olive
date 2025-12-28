import os
import re
import math
import json
import hashlib
from io import BytesIO
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st

# ============================================================
# OLIVETTI DESK — one file, production-stable, paste+click
# ============================================================

# ============================================================
# ENV / METADATA HYGIENE
# ============================================================
os.environ.setdefault("MS_APP_ID", "olivetti-writing-desk")
os.environ.setdefault("ms-appid", "olivetti-writing-desk")

DEFAULT_MODEL = "gpt-4o-mini"

def _get_openai_key_or_empty() -> str:
    try:
        return str(st.secrets.get("OPENAI_API_KEY", ""))  # type: ignore[attr-defined]
    except Exception:
        return ""

def _get_openai_model() -> str:
    try:
        return str(st.secrets.get("OPENAI_MODEL", DEFAULT_MODEL))  # type: ignore[attr-defined]
    except Exception:
        return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

OPENAI_MODEL = _get_openai_model()

def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or _get_openai_key_or_empty())

def require_openai_key() -> str:
    """Stop the app with a clear message if no OpenAI key is configured."""
    key = os.getenv("OPENAI_API_KEY") or _get_openai_key_or_empty()
    if not key:
        st.error(
            "OPENAI_API_KEY is not set. Add it as an environment variable (OPENAI_API_KEY) or as a Streamlit secret."
        )
        st.stop()
    return key


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Olivetti Desk", layout="wide", initial_sidebar_state="expanded")

# Sexy, readable UI skin — no feature changes
st.markdown(
    """
<style>
:root{
  --bg0:#07090d;
  --bg1:#0b0f16;
  --panel:rgba(17,21,28,.82);
  --panel2:rgba(22,28,38,.78);
  --stroke:rgba(202,168,106,.22);
  --stroke2:rgba(255,255,255,.08);
  --accent:#caa86a;
  --text:#e9edf5;
  --muted:rgba(233,237,245,.72);
  --paper:#fbf7ee;
  --ink:#14161a;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"]{
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
}
h1, h2, h3, h4, h5, .stSubheader{
  font-family: ui-serif, Georgia, "Palatino Linotype", Palatino, serif;
}

.stApp{
  background:
    radial-gradient(1200px 800px at 20% 0%, rgba(109,214,255,.10), transparent 60%),
    radial-gradient(900px 700px at 90% 10%, rgba(202,168,106,.12), transparent 55%),
    linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
  color: var(--text);
}

header[data-testid="stHeader"]{ background: transparent; }

[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(13,16,23,.92), rgba(10,12,18,.86));
  border-right: 1px solid var(--stroke);
  box-shadow: 0 0 0 1px rgba(255,255,255,.03) inset, 10px 0 40px rgba(0,0,0,.35);
}
[data-testid="stSidebar"] *{ color: var(--text); }

section.main > div.block-container{
  padding-top: 1.25rem;
  padding-bottom: 2rem;
  max-width: 1500px;
}

details[data-testid="stExpander"]{
  background: var(--panel);
  border: 1px solid var(--stroke2);
  border-radius: 16px;
  box-shadow: 0 0 0 1px rgba(0,0,0,.25) inset, 0 12px 30px rgba(0,0,0,.28);
  overflow: hidden;
}
details[data-testid="stExpander"] > summary{
  padding: 0.65rem 0.9rem !important;
  font-size: 0.95rem !important;
  letter-spacing: .2px;
  color: var(--text);
  background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(0,0,0,0));
  border-bottom: 1px solid rgba(255,255,255,.06);
}
details[data-testid="stExpander"][open] > summary{ border-bottom: 1px solid rgba(202,168,106,.16); }

div[data-testid="stTextArea"] textarea{
  font-size: 18px !important;
  line-height: 1.65 !important;
  padding: 18px !important;
  resize: vertical !important;
  min-height: 520px !important;
  background: var(--paper) !important;
  color: var(--ink) !important;
  border: 1px solid rgba(20,22,26,.18) !important;
  box-shadow: 0 1px 0 rgba(255,255,255,.55) inset, 0 10px 24px rgba(0,0,0,.18) !important;
}
/* Compact scratch pads - expand with content */
div[data-testid="stTextArea"] textarea[aria-label="Notes"],
div[data-testid="stTextArea"] textarea[aria-label="Paste"]{
  min-height: 36px !important;
  height: auto !important;
  padding: 8px 12px !important;
  font-size: 14px !important;
  line-height: 1.4 !important;
  resize: vertical !important;
}
div[data-testid="stTextArea"] textarea:focus{
  outline: none !important;
  border: 1px solid rgba(202,168,106,.55) !important;
  box-shadow: 0 0 0 3px rgba(202,168,106,.18), 0 10px 24px rgba(0,0,0,.20) !important;
}

button[kind="secondary"], button[kind="primary"]{
  font-size: 16px !important;
  padding: 0.62rem 0.95rem !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,.10) !important;
  transition: transform .08s ease, box-shadow .12s ease, border-color .12s ease, filter .12s ease;
}
button[kind="primary"]{
  background: linear-gradient(180deg, rgba(202,168,106,.95), rgba(150,118,64,.92)) !important;
  color: #161616 !important;
  box-shadow: 0 10px 24px rgba(0,0,0,.28), 0 0 0 1px rgba(0,0,0,.22) inset !important;
}
button[kind="secondary"]{
  background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02)) !important;
  color: var(--text) !important;
  box-shadow: 0 8px 18px rgba(0,0,0,.22), 0 0 0 1px rgba(0,0,0,.25) inset !important;
}
button:hover{
  transform: translateY(-1px);
  border-color: rgba(202,168,106,.35) !important;
  box-shadow: 0 12px 28px rgba(0,0,0,.30), 0 0 0 1px rgba(202,168,106,.20) inset !important;
  filter: brightness(1.03);
}

[data-testid="stTabs"] [data-baseweb="tab-list"]{ gap: 8px; }
[data-testid="stTabs"] button[role="tab"]{
  border-radius: 999px !important;
  padding: 0.35rem 0.8rem !important;
  background: rgba(255,255,255,.04) !important;
  border: 1px solid rgba(255,255,255,.08) !important;
  color: var(--text) !important;
}
[data-testid="stTabs"] button[aria-selected="true"]{
  background: rgba(202,168,106,.18) !important;
  border-color: rgba(202,168,106,.28) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# GLOBALS
# ============================================================
LANES = ["Dialogue", "Narration", "Interiority", "Action"]
BAYS = ["NEW", "ROUGH", "EDIT", "FINAL"]


ENGINE_STYLES = ["NARRATIVE", "DESCRIPTIVE", "EMOTIONAL", "LYRICAL"]
AUTOSAVE_DIR = "autosave"
AUTOSAVE_PATH = os.path.join(AUTOSAVE_DIR, "olivetti_state.json")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB guardrail for large manuscripts

WORD_RE = re.compile(r"[A-Za-z']+")


# ============================================================
# UTILS
# ============================================================
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_text(s: str) -> str:
    t = (s or "").strip()
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def _split_paragraphs(text: str) -> List[str]:
    t = _normalize_text(text)
    if not t:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n", t, flags=re.MULTILINE) if p.strip()]


def _safe_filename(s: str, fallback: str = "olivetti") -> str:
    s = re.sub(r"[^\w\- ]+", "", (s or "").strip()).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:80] if s else fallback


def _clamp_text(s: str, max_chars: int = 12000) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 40] + "\n\n… (truncated) …"


# ============================================================
# VECTOR / VOICE VAULT (lightweight, no external deps)
# ============================================================
def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def _hash_vec(text: str, dims: int = 512) -> List[float]:
    vec = [0.0] * dims
    toks = _tokenize(text)
    for t in toks:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        vec[h % dims] += 1.0
    for i, v in enumerate(vec):
        if v > 0:
            vec[i] = 1.0 + math.log(v)
    return vec


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def default_voice_vault() -> Dict[str, Any]:
    ts = now_ts()
    return {
        "Voice A": {"created_ts": ts, "lanes": {ln: [] for ln in LANES}},
        "Voice B": {"created_ts": ts, "lanes": {ln: [] for ln in LANES}},
    }


def rebuild_vectors_in_voice_vault(compact_voices: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for vname, v in (compact_voices or {}).items():
        created_ts = v.get("created_ts") or now_ts()
        lanes_in = v.get("lanes", {}) or {}
        lanes_out: Dict[str, Any] = {ln: [] for ln in LANES}
        for ln in LANES:
            samples = lanes_in.get(ln, []) or []
            for s in samples:
                txt = _normalize_text(s.get("text", ""))
                if not txt:
                    continue
                lanes_out[ln].append({
                    "ts": s.get("ts") or now_ts(),
                    "text": txt,
                    "vec": _hash_vec(txt),
                })
        out[vname] = {"created_ts": created_ts, "lanes": lanes_out}
    return out


def compact_voice_vault(voices: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for vname, v in (voices or {}).items():
        lanes_out: Dict[str, Any] = {}
        for ln in LANES:
            samples = (v.get("lanes", {}) or {}).get(ln, []) or []
            lanes_out[ln] = [
                {"ts": s.get("ts"), "text": s.get("text", "")}
                for s in samples
                if (s.get("text") or "").strip()
            ]
        out[vname] = {"created_ts": v.get("created_ts"), "lanes": lanes_out}
    return out


def voice_names_for_selector() -> List[str]:
    base = ["— None —", "Voice A", "Voice B"]
    customs = sorted([k for k in (st.session_state.voices or {}).keys() if k not in ("Voice A", "Voice B")])
    return base + customs


def create_custom_voice(name: str) -> bool:
    n = (name or "").strip()
    if not n:
        return False
    if n in (st.session_state.voices or {}):
        return False
    st.session_state.voices[n] = {"created_ts": now_ts(), "lanes": {ln: [] for ln in LANES}}
    return True


def add_voice_sample(voice_name: str, lane: str, text: str) -> bool:
    vn = (voice_name or "").strip()
    if not vn:
        return False
    lane = lane if lane in LANES else "Narration"
    t = _normalize_text(text)
    if not t:
        return False
    v = (st.session_state.voices or {}).get(vn)
    if not v:
        # auto-create
        create_custom_voice(vn)
        v = st.session_state.voices.get(vn)
    v.setdefault("lanes", {ln: [] for ln in LANES})
    v["lanes"].setdefault(lane, [])
    v["lanes"][lane].append({"ts": now_ts(), "text": t, "vec": _hash_vec(t)})
    # cap per lane (keeps app fast)
    if len(v["lanes"][lane]) > 60:
        v["lanes"][lane] = v["lanes"][lane][-60:]
    st.session_state.voices[vn] = v
    return True


def delete_voice_sample(voice_name: str, lane: str, index_from_end: int = 0) -> bool:
    vn = (voice_name or "").strip()
    v = (st.session_state.voices or {}).get(vn)
    if not v:
        return False
    lane = lane if lane in LANES else "Narration"
    arr = (v.get("lanes", {}) or {}).get(lane, []) or []
    if not arr:
        return False
    idx = len(arr) - 1 - int(index_from_end)
    if idx < 0 or idx >= len(arr):
        return False
    arr.pop(idx)
    v["lanes"][lane] = arr
    st.session_state.voices[vn] = v
    return True


def retrieve_exemplars(voice_name: str, lane: str, query_text: str, k: int = 3) -> List[str]:
    v = (st.session_state.voices or {}).get(voice_name)
    if not v:
        return []
    lane = lane if lane in LANES else "Narration"
    pool = v.get("lanes", {}).get(lane, []) or []
    if not pool:
        return []
    qv = _hash_vec(query_text)
    scored = [(_cosine(qv, s.get("vec", [])), s.get("text", "")) for s in pool[-140:]]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [txt for score, txt in scored[:k] if score > 0.0 and txt][:k]


def retrieve_mixed_exemplars(voice_name: str, lane: str, query_text: str) -> List[str]:
    lane_ex = retrieve_exemplars(voice_name, lane, query_text, k=2)
    if lane == "Narration":
        return lane_ex if lane_ex else retrieve_exemplars(voice_name, "Narration", query_text, k=3)
    nar_ex = retrieve_exemplars(voice_name, "Narration", query_text, k=1)
    out = lane_ex + [x for x in nar_ex if x not in lane_ex]
    return out[:3]


# ============================================================
# LANE DETECTION (lightweight)
# ============================================================
THOUGHT_WORDS = {
    "think",
    "thought",
    "felt",
    "wondered",
    "realized",
    "remembered",
    "knew",
    "noticed",
    "decided",
    "hoped",
    "feared",
    "wanted",
    "imagined",
    "could",
    "should",
    "would",
}
ACTION_VERBS = {
    "run",
    "ran",
    "walk",
    "walked",
    "grab",
    "grabbed",
    "push",
    "pushed",
    "pull",
    "pulled",
    "slam",
    "slammed",
    "hit",
    "struck",
    "kick",
    "kicked",
    "turn",
    "turned",
    "snap",
    "snapped",
    "dive",
    "dived",
    "duck",
    "ducked",
    "rush",
    "rushed",
    "lunge",
    "lunged",
    "climb",
    "climbed",
    "drop",
    "dropped",
    "throw",
    "threw",
    "fire",
    "fired",
    "aim",
    "aimed",
    "break",
    "broke",
    "shatter",
    "shattered",
    "step",
    "stepped",
    "move",
    "moved",
    "reach",
    "reached",
}


def detect_lane(paragraph: str) -> str:
    p = (paragraph or "").strip()
    if not p:
        return "Narration"

    quote_count = p.count('"') + p.count("“") + p.count("”")
    has_dialogue_punct = p.startswith(("—", "- ", "“", '"'))

    dialogue_score = 0.0
    if quote_count >= 2:
        dialogue_score += 2.5
    if has_dialogue_punct:
        dialogue_score += 1.5

    toks = _tokenize(p)
    interior_score = 0.0
    if toks:
        first_person = sum(1 for t in toks if t in ("i", "me", "my", "mine", "myself"))
        thought_hits = sum(1 for t in toks if t in THOUGHT_WORDS)
        if first_person >= 2 and thought_hits >= 1:
            interior_score += 2.2

    action_score = 0.0
    if toks:
        verb_hits = sum(1 for t in toks if t in ACTION_VERBS)
        if verb_hits >= 2:
            action_score += 1.6
        if "!" in p:
            action_score += 0.3

    scores = {"Dialogue": dialogue_score, "Interiority": interior_score, "Action": action_score, "Narration": 0.25}
    lane = max(scores.items(), key=lambda kv: kv[1])[0]
    return "Narration" if scores[lane] < 0.9 else lane


def current_lane_from_draft(text: str) -> str:
    paras = _split_paragraphs(text)
    if not paras:
        return "Narration"
    return detect_lane(paras[-1])


# ============================================================
# INTENSITY (GLOBAL AI AGGRESSION KNOB)
# ============================================================
def intensity_profile(x: float) -> str:
    if x <= 0.25:
        return "LOW: conservative, literal, minimal invention, prioritize continuity and clarity."
    if x <= 0.60:
        return "MED: balanced creativity, stronger phrasing, controlled invention within canon."
    if x <= 0.85:
        return "HIGH: bolder choices, richer specificity; still obey canon and lane."
    return "MAX: aggressive originality and voice; still obey canon, no derailments."


def temperature_from_intensity(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return 0.15 + (x * 0.95)


# ============================================================
# PROJECT MODEL
# ============================================================
def _fingerprint_story_bible(sb: Dict[str, str]) -> str:
    parts = [
        (sb.get("synopsis", "") or "").strip(),
        (sb.get("genre_style_notes", "") or "").strip(),
        (sb.get("world", "") or "").strip(),
        (sb.get("characters", "") or "").strip(),
        (sb.get("outline", "") or "").strip(),
    ]
    blob = "\n\n---\n\n".join(parts)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def new_project_payload(title: str) -> Dict[str, Any]:
    ts = now_ts()
    title = title.strip() if title.strip() else "Untitled Project"
    story_bible_id = hashlib.md5(f"sb|{title}|{ts}".encode("utf-8")).hexdigest()[:12]
    return {
        "id": hashlib.md5(f"{title}|{ts}".encode("utf-8")).hexdigest()[:12],
        "title": title,
        "created_ts": ts,
        "updated_ts": ts,
        "bay": "NEW",
        "draft": "",
        "story_bible_id": story_bible_id,
        "story_bible_created_ts": ts,
        "story_bible_binding": {"locked": True, "locked_ts": ts, "source": "system", "source_story_bible_id": None},
        "story_bible_fingerprint": "",
        "story_bible": {"synopsis": "", "genre_style_notes": "", "world": "", "characters": "", "outline": ""},
        "voice_bible": {
            "vb_style_on": True,
            "vb_genre_on": True,
            "vb_trained_on": False,
            "vb_match_on": False,
            "vb_lock_on": False,
            "writing_style": "Neutral",
            "genre": "Literary",
            "trained_voice": "— None —",
            "voice_sample": "",
            "voice_lock_prompt": "",
            "style_intensity": 0.6,
            "genre_intensity": 0.6,
            "trained_intensity": 0.7,
            "match_intensity": 0.8,
            "lock_intensity": 1.0,
            "pov": "Close Third",
            "tense": "Past",
            "ai_intensity": 0.75,
        },
        "locks": {
            "story_bible_lock": True,  # relationship lock
            "sb_edit_unlocked": False,  # hard lock for edits (content)
            "voice_fingerprint_lock": True,
            "lane_lock": False,
            "forced_lane": "Narration",
        },
        "voices": default_voice_vault(),
        "style_banks": default_style_banks(),
    }


# ============================================================

# ============================================================
# ENGINE STYLE BANKS (project/workspace) — trainable writing styles
# ============================================================
def default_style_banks() -> Dict[str, Any]:
    ts = now_ts()
    return {s: {"created_ts": ts, "lanes": {ln: [] for ln in LANES}} for s in ENGINE_STYLES}


def rebuild_vectors_in_style_banks(banks: Dict[str, Any]) -> Dict[str, Any]:
    src = banks or {}
    out: Dict[str, Any] = {}
    for style in ENGINE_STYLES:
        b = (src.get(style) or {}) if isinstance(src, dict) else {}
        lanes = b.get("lanes") or {}
        new_lanes: Dict[str, List[Dict[str, Any]]] = {}
        for ln in LANES:
            samples = (lanes.get(ln) or []) if isinstance(lanes, dict) else []
            rebuilt: List[Dict[str, Any]] = []
            for it in samples:
                if isinstance(it, str):
                    t = it.strip()
                    if not t:
                        continue
                    rebuilt.append({"ts": now_ts(), "text": t, "vec": _hash_vec(t)})
                    continue
                if not isinstance(it, dict):
                    continue
                t = (it.get("text") or "").strip()
                if not t:
                    continue
                vec = it.get("vec") if isinstance(it.get("vec"), list) else None
                if not vec:
                    vec = _hash_vec(t)
                rebuilt.append({"ts": it.get("ts") or now_ts(), "text": t, "vec": vec})
            new_lanes[ln] = rebuilt
        out[style] = {"created_ts": b.get("created_ts") or now_ts(), "lanes": new_lanes}
    return out if out else default_style_banks()


def compact_style_banks(banks: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(banks, dict):
        return default_style_banks()
    out: Dict[str, Any] = {}
    for style in ENGINE_STYLES:
        b = banks.get(style) or {}
        lanes = b.get("lanes") or {}
        c_lanes: Dict[str, List[Dict[str, Any]]] = {}
        for ln in LANES:
            ss = (lanes.get(ln) or []) if isinstance(lanes, dict) else []
            cleaned: List[Dict[str, Any]] = []
            for it in ss:
                if not isinstance(it, dict):
                    continue
                t = (it.get("text") or "").strip()
                if not t:
                    continue
                cleaned.append({"ts": it.get("ts") or now_ts(), "text": _clamp_text(t, 9000)})
            c_lanes[ln] = cleaned
        out[style] = {"created_ts": b.get("created_ts") or now_ts(), "lanes": c_lanes}
    return out


def add_style_samples(style: str, lane: str, raw_text: str, split_mode: str = "Paragraphs", cap_per_lane: int = 250) -> int:
    style = (style or "").strip().upper()
    if style not in ENGINE_STYLES:
        return 0
    lane = lane if lane in LANES else "Narration"
    t = _normalize_text(raw_text)
    if not t.strip():
        return 0

    parts = _split_paragraphs(t) if split_mode == "Paragraphs" else [t.strip()]
    parts = [p for p in parts if len(p.strip()) >= 40]

    sb = st.session_state.get("style_banks")
    if not isinstance(sb, dict) or style not in sb:
        sb = rebuild_vectors_in_style_banks(default_style_banks())
        st.session_state.style_banks = sb

    bank = sb.get(style) or {}
    lanes = bank.get("lanes") or {}
    lane_list = list((lanes.get(lane) or [])) if isinstance(lanes, dict) else []

    added = 0
    for p in parts:
        p = _clamp_text(p.strip(), 9000)
        lane_list.append({"ts": now_ts(), "text": p, "vec": _hash_vec(p)})
        added += 1

    # cap: keep newest
    lane_list = lane_list[-cap_per_lane:]
    lanes[lane] = lane_list
    bank["lanes"] = lanes
    sb[style] = bank
    st.session_state.style_banks = sb
    return added


def delete_last_style_sample(style: str, lane: str) -> bool:
    style = (style or "").strip().upper()
    if style not in ENGINE_STYLES:
        return False
    lane = lane if lane in LANES else "Narration"
    sb = st.session_state.get("style_banks") or {}
    bank = (sb.get(style) or {})
    lanes = bank.get("lanes") or {}
    lane_list = (lanes.get(lane) or [])
    if not lane_list:
        return False
    lane_list.pop()
    lanes[lane] = lane_list
    bank["lanes"] = lanes
    sb[style] = bank
    st.session_state.style_banks = sb
    return True


def clear_style_lane(style: str, lane: str) -> None:
    style = (style or "").strip().upper()
    if style not in ENGINE_STYLES:
        return
    lane = lane if lane in LANES else "Narration"
    sb = st.session_state.get("style_banks") or rebuild_vectors_in_style_banks(default_style_banks())
    bank = sb.get(style) or {}
    lanes = bank.get("lanes") or {}
    lanes[lane] = []
    bank["lanes"] = lanes
    sb[style] = bank
    st.session_state.style_banks = sb


def retrieve_style_exemplars(style: str, lane: str, query: str, k: int = 2) -> List[str]:
    style = (style or "").strip().upper()
    if style not in ENGINE_STYLES:
        return []
    lane = lane if lane in LANES else "Narration"
    sb = st.session_state.get("style_banks") or {}
    bank = sb.get(style) or {}
    lanes = bank.get("lanes") or {}
    pool = (lanes.get(lane) or [])
    if not pool:
        # fallback: all lanes pooled
        pool = []
        for ln in LANES:
            pool.extend(lanes.get(ln) or [])
    if not pool:
        return []
    # favor newest slice for speed
    pool = pool[-160:]
    qv = _hash_vec(query or "")
    scored = []
    for it in pool:
        if not isinstance(it, dict):
            continue
        vec = it.get("vec")
        if not isinstance(vec, list):
            continue
        scored.append((_cosine(qv, vec), it.get("text") or ""))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [t.strip() for _, t in scored[: max(0, k)] if (t or "").strip()]
    return out


_ENGINE_STYLE_GUIDE = {
    "NARRATIVE": "Narrative clarity, clean cause→effect, confident pacing. Prioritize story logic and readability.",
    "DESCRIPTIVE": "Sensory precision, spatial clarity, vivid concrete nouns, controlled detail density (no purple bloat).",
    "EMOTIONAL": "Interior depth, subtext, emotional specificity. Show the feeling through behavior, sensation, and thought.",
    "LYRICAL": "Rhythm, musical syntax, image-forward language, elegant metaphor with restraint. Make prose sing without obscuring meaning.",
}


def engine_style_directive(style: str, intensity: float, lane: str) -> str:
    style = (style or "").strip().upper()
    base = _ENGINE_STYLE_GUIDE.get(style, "")
    x = float(intensity)
    if x <= 0.33:
        mod = "Keep it subtle and controlled. Minimal overt stylization."
    elif x <= 0.66:
        mod = "Medium stylization. Let the style clearly shape diction and cadence."
    else:
        mod = "High stylization. Strong stylistic fingerprint, but still professional and coherent."
    return f"{base}\\nLane: {lane}\\n{mod}"


# STORY BIBLE WORKSPACE (pre-project creation space)
# ============================================================
def default_story_bible_workspace() -> Dict[str, Any]:
    ts = now_ts()
    return {
        "workspace_story_bible_id": hashlib.md5(f"wsb|{ts}".encode("utf-8")).hexdigest()[:12],
        "workspace_story_bible_created_ts": ts,
        "title": "",
        "draft": "",
        "story_bible": {"synopsis": "", "genre_style_notes": "", "world": "", "characters": "", "outline": ""},
        "voice_sample": "",
        "ai_intensity": 0.75,
        "voices": default_voice_vault(),
        "style_banks": default_style_banks(),
    }


def in_workspace_mode() -> bool:
    return (st.session_state.active_bay == "NEW") and (st.session_state.project_id is None)


def save_workspace_from_session() -> None:
    w = st.session_state.sb_workspace or default_story_bible_workspace()
    w["title"] = st.session_state.get("workspace_title", w.get("title", ""))
    w["draft"] = st.session_state.main_text
    w["story_bible"] = {
        "synopsis": st.session_state.synopsis,
        "genre_style_notes": st.session_state.genre_style_notes,
        "world": st.session_state.world,
        "characters": st.session_state.characters,
        "outline": st.session_state.outline,
    }
    w["voice_sample"] = st.session_state.voice_sample
    w["ai_intensity"] = float(st.session_state.ai_intensity)
    w["voices"] = compact_voice_vault(st.session_state.voices)
    w["style_banks"] = compact_style_banks(st.session_state.get("style_banks") or rebuild_vectors_in_style_banks(default_style_banks()))
    st.session_state.sb_workspace = w


def set_ai_intensity(val: float) -> None:
    """Safely set ai_intensity without mutating widget-bound state after creation."""
    val = float(val)
    if "ai_intensity" not in st.session_state:
        st.session_state["ai_intensity"] = val
    else:
        st.session_state["ai_intensity_pending"] = val
        st.session_state["_apply_pending_widget_state"] = True


def load_workspace_into_session() -> None:
    w = st.session_state.sb_workspace or default_story_bible_workspace()
    sb = w.get("story_bible", {}) or {}
    st.session_state.project_id = None
    st.session_state.project_title = "—"
    st.session_state.main_text = w.get("draft", "") or ""
    st.session_state.synopsis = sb.get("synopsis", "") or ""
    st.session_state.genre_style_notes = sb.get("genre_style_notes", "") or ""
    st.session_state.world = sb.get("world", "") or ""
    st.session_state.characters = sb.get("characters", "") or ""
    st.session_state.outline = sb.get("outline", "") or ""
    st.session_state.voice_sample = w.get("voice_sample", "") or ""
    set_ai_intensity(float(w.get("ai_intensity", 0.75)))
    st.session_state.voices = rebuild_vectors_in_voice_vault(w.get("voices", default_voice_vault()))
    st.session_state.voices_seeded = True
    st.session_state.style_banks = rebuild_vectors_in_style_banks(w.get("style_banks", default_style_banks()))
    st.session_state.workspace_title = w.get("title", "") or ""


def reset_workspace_story_bible(keep_templates: bool = True) -> None:
    old = st.session_state.sb_workspace or default_story_bible_workspace()
    neww = default_story_bible_workspace()
    if keep_templates:
        neww["voice_sample"] = old.get("voice_sample", "")
        neww["ai_intensity"] = float(old.get("ai_intensity", 0.75))
        neww["voices"] = old.get("voices", default_voice_vault())
        neww["style_banks"] = old.get("style_banks", default_style_banks())
    st.session_state.sb_workspace = neww
    if in_workspace_mode():
        load_workspace_into_session()


# ============================================================
# SESSION INIT
# ============================================================
def init_state() -> None:
    defaults: Dict[str, Any] = {
        "active_bay": "NEW",
        "projects": {},
        "active_project_by_bay": {b: None for b in BAYS},
        "sb_workspace": default_story_bible_workspace(),
        "workspace_title": "",
        "project_id": None,
        "project_title": "—",
        "autosave_time": None,
        "last_action": "—",
        "voice_status": "—",
        "main_text": "",
        "synopsis": "",
        "genre_style_notes": "",
        "world": "",
        "characters": "",
        "outline": "",
        "junk": "",
        "tool_output": "",
        "scratch_pad": "",
        "scratch_pad_2": "",
        "find_count": 0,
        "find_term": "",
        "pending_action": None,
        "pending_load_project": None,
        "vb_style_on": True,
        "vb_genre_on": True,
        "vb_trained_on": False,
        "vb_match_on": False,
        "vb_lock_on": False,
        "writing_style": "Neutral",
        "genre": "Literary",
        "trained_voice": "— None —",
        "voice_sample": "",
        "voice_lock_prompt": "",
        "style_intensity": 0.6,
        "genre_intensity": 0.6,
        "trained_intensity": 0.7,
        "match_intensity": 0.8,
        "lock_intensity": 1.0,
        "pov": "Close Third",
        "tense": "Past",
        "ai_intensity": 0.75,
        "locks": {
            "story_bible_lock": True,
            "sb_edit_unlocked": False,
            "voice_fingerprint_lock": True,
            "lane_lock": False,
            "forced_lane": "Narration",
        },
        "voices": {},
        "voices_seeded": False,
        "style_banks": rebuild_vectors_in_style_banks(default_style_banks()),
        "last_saved_digest": "",

        # internal UI helpers (not widgets)
        "ui_notice": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# Apply pending widget values BEFORE widgets are created
if st.session_state.pop("_apply_pending_widget_state", False):
    if "ai_intensity_pending" in st.session_state:
        st.session_state["ai_intensity"] = float(st.session_state.pop("ai_intensity_pending"))

# Process pending project load BEFORE widgets render
pending_pid = st.session_state.get("pending_load_project")
if pending_pid:
    st.session_state.pending_load_project = None
    p = (st.session_state.projects or {}).get(pending_pid)
    if p:
        st.session_state.project_id = pending_pid
        st.session_state.project_title = p.get("title", "Untitled Project")
        st.session_state.main_text = p.get("draft", "")
        sb = p.get("story_bible", {}) or {}
        st.session_state.synopsis = sb.get("synopsis", "")
        st.session_state.genre_style_notes = sb.get("genre_style_notes", "")
        st.session_state.world = sb.get("world", "")
        st.session_state.characters = sb.get("characters", "")
        st.session_state.outline = sb.get("outline", "")
        st.session_state.voice_status = f"Loaded: {st.session_state.project_title}"



# ============================================================
# PROJECT <-> SESSION SYNC
# ============================================================
def load_project_into_session(pid: str) -> None:
    p = (st.session_state.projects or {}).get(pid)
    if not p:
        return

    st.session_state.project_id = pid
    st.session_state.project_title = p.get("title", "Untitled Project")
    st.session_state.main_text = p.get("draft", "")

    sb = p.get("story_bible", {}) or {}
    st.session_state.synopsis = sb.get("synopsis", "")
    st.session_state.genre_style_notes = sb.get("genre_style_notes", "")
    st.session_state.world = sb.get("world", "")
    st.session_state.characters = sb.get("characters", "")
    st.session_state.outline = sb.get("outline", "")

    vb = p.get("voice_bible", {}) or {}
    for k in [
        "vb_style_on",
        "vb_genre_on",
        "vb_trained_on",
        "vb_match_on",
        "vb_lock_on",
        "writing_style",
        "genre",
        "trained_voice",
        "voice_sample",
        "voice_lock_prompt",
        "style_intensity",
        "genre_intensity",
        "trained_intensity",
        "match_intensity",
        "lock_intensity",
        "pov",
        "tense",
        "ai_intensity",
    ]:
        if k in vb:
            st.session_state[k] = vb[k]

    locks = p.get("locks", {}) or {}
    if isinstance(locks, dict):
        # ensure new keys exist
        locks.setdefault("sb_edit_unlocked", False)
        st.session_state.locks = locks

    st.session_state.voices = rebuild_vectors_in_voice_vault(p.get("voices", default_voice_vault()))
    st.session_state.voices_seeded = True


def save_session_into_project() -> None:
    pid = st.session_state.project_id
    if not pid:
        return
    p = (st.session_state.projects or {}).get(pid)
    if not p:
        return
    # Don't save if pending load waiting - widget data is stale
    if st.session_state.get("pending_load_project"):
        return

    p["updated_ts"] = now_ts()
    p["draft"] = st.session_state.main_text
    p["story_bible"] = {
        "synopsis": st.session_state.synopsis,
        "genre_style_notes": st.session_state.genre_style_notes,
        "world": st.session_state.world,
        "characters": st.session_state.characters,
        "outline": st.session_state.outline,
    }
    p["voice_bible"] = {
        "vb_style_on": st.session_state.vb_style_on,
        "vb_genre_on": st.session_state.vb_genre_on,
        "vb_trained_on": st.session_state.vb_trained_on,
        "vb_match_on": st.session_state.vb_match_on,
        "vb_lock_on": st.session_state.vb_lock_on,
        "writing_style": st.session_state.writing_style,
        "genre": st.session_state.genre,
        "trained_voice": st.session_state.trained_voice,
        "voice_sample": st.session_state.voice_sample,
        "voice_lock_prompt": st.session_state.voice_lock_prompt,
        "style_intensity": st.session_state.style_intensity,
        "genre_intensity": st.session_state.genre_intensity,
        "trained_intensity": st.session_state.trained_intensity,
        "match_intensity": st.session_state.match_intensity,
        "lock_intensity": st.session_state.lock_intensity,
        "pov": st.session_state.pov,
        "tense": st.session_state.tense,
        "ai_intensity": float(st.session_state.ai_intensity),
    }
    p["locks"] = st.session_state.locks
    p["voices"] = compact_voice_vault(st.session_state.voices)
    p["style_banks"] = compact_style_banks(st.session_state.get("style_banks") or rebuild_vectors_in_style_banks(default_style_banks()))
    # keep fingerprint up to date
    try:
        p["story_bible_fingerprint"] = _fingerprint_story_bible(p["story_bible"])
    except Exception:
        pass


def list_projects_in_bay(bay: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for pid, p in (st.session_state.projects or {}).items():
        if isinstance(p, dict) and p.get("bay") == bay:
            items.append((pid, p.get("title", "Untitled")))
    items.sort(key=lambda x: (x[1] or "").lower())
    return items


def ensure_bay_has_active_project(bay: str) -> None:
    pid = (st.session_state.active_project_by_bay or {}).get(bay)
    if pid and pid in (st.session_state.projects or {}) and (st.session_state.projects[pid].get("bay") == bay):
        return
    items = list_projects_in_bay(bay)
    st.session_state.active_project_by_bay[bay] = items[0][0] if items else None


def switch_bay(target_bay: str) -> None:
    # Don't save - widget data may be stale/empty and would overwrite project content
    
    st.session_state.active_bay = target_bay
    ensure_bay_has_active_project(target_bay)
    pid = st.session_state.active_project_by_bay.get(target_bay)

    if pid:
        # Use pending pattern - load happens BEFORE widgets on next rerun
        st.session_state.pending_load_project = pid
        st.session_state.voice_status = f"{target_bay}: Loading..."
    else:
        st.session_state.project_id = None
        st.session_state.project_title = "—"
        if target_bay == "NEW":
            load_workspace_into_session()
            st.session_state.voice_status = "NEW: (Story Bible workspace)"
        else:
            st.session_state.main_text = ""
            st.session_state.synopsis = ""
            st.session_state.genre_style_notes = ""
            st.session_state.world = ""
            st.session_state.characters = ""
            st.session_state.outline = ""
            st.session_state.voice_sample = ""
            set_ai_intensity(0.75)
            st.session_state.voices = rebuild_vectors_in_voice_vault(default_voice_vault())
            st.session_state.voices_seeded = True
            st.session_state.voice_status = f"{target_bay}: (empty)"

    st.session_state.last_action = f"Bay → {target_bay}"


def create_project_from_current_bible(title: str) -> str:
    title = title.strip() if title.strip() else f"New Project {now_ts()}"
    source = "workspace" if in_workspace_mode() else "clone"

    source_story_bible_id = None
    source_story_bible_created_ts = None
    if source == "workspace":
        w = st.session_state.sb_workspace or default_story_bible_workspace()
        source_story_bible_id = w.get("workspace_story_bible_id")
        source_story_bible_created_ts = w.get("workspace_story_bible_created_ts")
    else:
        pid = st.session_state.project_id
        if pid and pid in (st.session_state.projects or {}):
            source_story_bible_id = st.session_state.projects[pid].get("story_bible_id")

    p = new_project_payload(title)
    p["bay"] = "NEW"
    p["draft"] = st.session_state.main_text
    p["story_bible"] = {
        "synopsis": st.session_state.synopsis,
        "genre_style_notes": st.session_state.genre_style_notes,
        "world": st.session_state.world,
        "characters": st.session_state.characters,
        "outline": st.session_state.outline,
    }

    if source == "workspace" and source_story_bible_id:
        p["story_bible_id"] = source_story_bible_id
        if source_story_bible_created_ts:
            p["story_bible_created_ts"] = source_story_bible_created_ts

    p["story_bible_binding"] = {
        "locked": True,
        "locked_ts": now_ts(),
        "source": source,
        "source_story_bible_id": source_story_bible_id,
    }
    p["story_bible_fingerprint"] = _fingerprint_story_bible(p["story_bible"])

    # Voice templates + intensity
    p["voice_bible"]["voice_sample"] = st.session_state.voice_sample
    p["voice_bible"]["ai_intensity"] = float(st.session_state.ai_intensity)
    p["voices"] = compact_voice_vault(st.session_state.voices)
    p["style_banks"] = compact_style_banks(st.session_state.get("style_banks") or rebuild_vectors_in_style_banks(default_style_banks()))

    st.session_state.projects[p["id"]] = p
    st.session_state.active_project_by_bay["NEW"] = p["id"]

    if source == "workspace":
        reset_workspace_story_bible(keep_templates=True)

    return p["id"]


def promote_project(pid: str, to_bay: str) -> None:
    p = (st.session_state.projects or {}).get(pid)
    if not p:
        return
    p["bay"] = to_bay
    p["updated_ts"] = now_ts()


def next_bay(bay: str) -> Optional[str]:
    if bay == "NEW":
        return "ROUGH"
    if bay == "ROUGH":
        return "EDIT"
    if bay == "EDIT":
        return "FINAL"
    return None


# ============================================================
# AUTOSAVE (atomic + backup)
# ============================================================
def _payload() -> Dict[str, Any]:
    if in_workspace_mode():
        save_workspace_from_session()
    else:
        save_session_into_project()
    return {
        "meta": {"saved_at": now_ts(), "version": "olivetti-prod-stable-v1"},
        "active_bay": st.session_state.active_bay,
        "active_project_by_bay": st.session_state.active_project_by_bay,
        "sb_workspace": st.session_state.sb_workspace,
        "projects": st.session_state.projects,
    }


def _digest(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def save_all_to_disk(force: bool = False) -> None:
    """Autosave state to disk with an atomic write and a simple backup."""
    try:
        os.makedirs(AUTOSAVE_DIR, exist_ok=True)
        payload = _payload()
        dig = _digest(payload)
        if (not force) and dig == st.session_state.last_saved_digest:
            return

        tmp_path = AUTOSAVE_PATH + ".tmp"
        bak_path = AUTOSAVE_PATH + ".bak"

        try:
            if os.path.exists(AUTOSAVE_PATH):
                import shutil

                shutil.copy2(AUTOSAVE_PATH, bak_path)
        except Exception:
            pass

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, AUTOSAVE_PATH)

        st.session_state.last_saved_digest = dig
    except Exception as e:
        st.session_state.voice_status = f"Autosave warning: {e}"


def load_all_from_disk() -> None:
    main_path = AUTOSAVE_PATH
    bak_path = AUTOSAVE_PATH + ".bak"

    def _boot_new() -> None:
        st.session_state.sb_workspace = st.session_state.get("sb_workspace") or default_story_bible_workspace()
        switch_bay("NEW")

    if (not os.path.exists(main_path)) and (not os.path.exists(bak_path)):
        _boot_new()
        return

    payload = None
    loaded_from = "primary"
    last_err = None
    for path, label in ((main_path, "primary"), (bak_path, "backup")):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            loaded_from = label
            break
        except Exception as e:
            last_err = e
            payload = None

    if payload is None:
        st.session_state.voice_status = f"Load warning: {last_err}"
        _boot_new()
        return

    try:
        projs = payload.get("projects", {})
        if isinstance(projs, dict):
            st.session_state.projects = projs

        apbb = payload.get("active_project_by_bay", {})
        if isinstance(apbb, dict):
            for b in BAYS:
                apbb.setdefault(b, None)
            st.session_state.active_project_by_bay = apbb

        w = payload.get("sb_workspace")
        if isinstance(w, dict) and w.get("workspace_story_bible_id"):
            st.session_state.sb_workspace = w
        else:
            st.session_state.sb_workspace = default_story_bible_workspace()

        # Migration guards
        for _, p in (st.session_state.projects or {}).items():
            if not isinstance(p, dict):
                continue
            ts = p.get("created_ts") or now_ts()
            title = p.get("title", "Untitled")
            p.setdefault("story_bible_id", hashlib.md5(f"sb|{title}|{ts}".encode("utf-8")).hexdigest()[:12])
            p.setdefault("story_bible_created_ts", ts)
            if not isinstance(p.get("story_bible_binding"), dict):
                p["story_bible_binding"] = {"locked": True, "locked_ts": ts, "source": "system", "source_story_bible_id": None}
            p.setdefault("locks", {})
            if isinstance(p["locks"], dict):
                p["locks"].setdefault("sb_edit_unlocked", False)
                p["locks"].setdefault("story_bible_lock", True)
            p.setdefault("voices", default_voice_vault())
            p.setdefault("style_banks", default_style_banks())
            if "story_bible_fingerprint" not in p:
                try:
                    p["story_bible_fingerprint"] = _fingerprint_story_bible(p.get("story_bible", {}) or {})
                except Exception:
                    p["story_bible_fingerprint"] = ""

        ab = payload.get("active_bay", "NEW")
        if ab not in BAYS:
            ab = "NEW"
        st.session_state.active_bay = ab

        ensure_bay_has_active_project(ab)
        pid = st.session_state.active_project_by_bay.get(ab)
        if pid:
            load_project_into_session(pid)
        else:
            if ab == "NEW":
                load_workspace_into_session()
                st.session_state.voice_status = "NEW: (Story Bible workspace)"
            else:
                switch_bay(ab)

        saved_at = (payload.get("meta", {}) or {}).get("saved_at", "")
        src = "autosave" if loaded_from == "primary" else "backup autosave"
        st.session_state.voice_status = f"Loaded {src} ({saved_at})."
        st.session_state.last_saved_digest = _digest(_payload())
    except Exception as e:
        st.session_state.voice_status = f"Load warning: {e}"
        _boot_new()


if "did_load_autosave" not in st.session_state:
    st.session_state.did_load_autosave = True
    load_all_from_disk()


def autosave() -> None:
    st.session_state.autosave_time = datetime.now().strftime("%H:%M:%S")
    save_all_to_disk()


# ============================================================
# IMPORT / EXPORT
# ============================================================
def _read_uploaded_text(uploaded) -> Tuple[str, str]:
    """Read .txt/.md/.docx from Streamlit UploadedFile."""
    if uploaded is None:
        return "", ""
    name = getattr(uploaded, "name", "") or ""
    raw = uploaded.getvalue()
    if raw is None:
        return "", name
    if len(raw) > MAX_UPLOAD_BYTES:
        return "", name
    ext = os.path.splitext(name.lower())[1]

    if ext in (".txt", ".md", ".markdown", ".text", ""):
        try:
            return raw.decode("utf-8"), name
        except Exception:
            return raw.decode("utf-8", errors="ignore"), name

    if ext == ".docx":
        try:
            from docx import Document  # python-docx

            doc = Document(BytesIO(raw))
            parts = []
            for p in doc.paragraphs:
                t = (p.text or "").strip()
                if t:
                    parts.append(t)
            return "\n\n".join(parts), name
        except Exception:
            try:
                return raw.decode("utf-8", errors="ignore"), name
            except Exception:
                return "", name

    try:
        return raw.decode("utf-8", errors="ignore"), name
    except Exception:
        return "", name


def _sb_sections_from_text_heuristic(text: str) -> Dict[str, str]:
    t = _normalize_text(text)
    if not t:
        return {"synopsis": "", "genre_style_notes": "", "world": "", "characters": "", "outline": ""}

    heading_map = {
        "synopsis": ["synopsis", "premise", "logline"],
        "genre_style_notes": ["genre", "style", "tone", "voice"],
        "world": ["world", "setting", "lore"],
        "characters": ["characters", "cast"],
        "outline": ["outline", "beats", "plot", "structure"],
    }

    lines = t.splitlines()
    buckets = {k: [] for k in heading_map.keys()}
    current = None

    def _match_heading(line: str) -> Optional[str]:
        l = re.sub(r"^[#\-\*\s]+", "", (line or "").strip()).lower()
        l = re.sub(r"[:\-\s]+$", "", l)
        for key, aliases in heading_map.items():
            if any(l == a or l.startswith(a + " ") for a in aliases):
                return key
        return None

    for line in lines:
        key = _match_heading(line)
        if key:
            current = key
            continue
        if current:
            buckets[current].append(line)
        else:
            buckets["synopsis"].append(line)

    return {k: _normalize_text("\n".join(v)) for k, v in buckets.items()}


def _extract_json_object(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s2 = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE | re.MULTILINE)
    m = re.search(r"\{.*\}", s2, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def story_bible_markdown(title: str, sb: Dict[str, str], meta: Dict[str, Any]) -> str:
    t = title or "Untitled"

    def sec(h: str, k: str) -> str:
        body = (sb.get(k, "") or "").strip()
        return f"## {h}\n\n{body}\n" if body else f"## {h}\n\n\n"

    front = f"# Story Bible — {t}\n\n- Exported: {now_ts()}\n"
    for mk, mv in (meta or {}).items():
        front += f"- {mk}: {mv}\n"
    front += "\n"
    return front + "\n".join(
        [
            sec("Synopsis", "synopsis"),
            sec("Genre / Style Notes", "genre_style_notes"),
            sec("World", "world"),
            sec("Characters", "characters"),
            sec("Outline", "outline"),
        ]
    )


def draft_markdown(title: str, draft: str, meta: Dict[str, Any]) -> str:
    t = title or "Untitled"
    front = f"# Draft — {t}\n\n- Exported: {now_ts()}\n"
    for mk, mv in (meta or {}).items():
        front += f"- {mk}: {mv}\n"
    front += "\n---\n\n"
    return front + (draft or "")


def make_project_bundle(pid: str) -> Dict[str, Any]:
    p = (st.session_state.projects or {}).get(pid, {}) or {}
    return {"meta": {"exported_at": now_ts(), "type": "project_bundle", "version": "1"}, "project": p}


def make_library_bundle() -> Dict[str, Any]:
    if in_workspace_mode():
        save_workspace_from_session()
    else:
        save_session_into_project()
    return {
        "meta": {"exported_at": now_ts(), "type": "library_bundle", "version": "1"},
        "active_bay": st.session_state.active_bay,
        "active_project_by_bay": st.session_state.active_project_by_bay,
        "sb_workspace": st.session_state.sb_workspace,
        "projects": st.session_state.projects,
    }


def _new_pid_like(seed: str) -> str:
    return hashlib.md5(f"{seed}|{now_ts()}".encode("utf-8")).hexdigest()[:12]


def import_project_bundle(bundle: Dict[str, Any], target_bay: str = "NEW", rename: str = "") -> Optional[str]:
    if not isinstance(bundle, dict):
        return None
    proj = bundle.get("project")
    if not isinstance(proj, dict):
        return None

    pid = str(proj.get("id") or _new_pid_like("import"))
    if pid in (st.session_state.projects or {}):
        pid = _new_pid_like(pid)

    proj = json.loads(json.dumps(proj))
    proj["id"] = pid
    if rename.strip():
        proj["title"] = rename.strip()
    if target_bay in BAYS:
        proj["bay"] = target_bay
    proj["updated_ts"] = now_ts()

    ts = proj.get("created_ts") or now_ts()
    title = proj.get("title", "Untitled")
    proj.setdefault("story_bible_id", hashlib.md5(f"sb|{title}|{ts}".encode("utf-8")).hexdigest()[:12])
    proj.setdefault("story_bible_created_ts", ts)
    if not isinstance(proj.get("story_bible_binding"), dict):
        proj["story_bible_binding"] = {"locked": True, "locked_ts": ts, "source": "import", "source_story_bible_id": None}
    proj.setdefault("locks", {})
    if isinstance(proj["locks"], dict):
        proj["locks"].setdefault("sb_edit_unlocked", False)
        proj["locks"].setdefault("story_bible_lock", True)
    proj.setdefault("voices", default_voice_vault())
    proj.setdefault("style_banks", default_style_banks())
    proj.setdefault("story_bible_fingerprint", "")

    st.session_state.projects[pid] = proj
    st.session_state.active_project_by_bay[proj.get("bay", "NEW")] = pid
    return pid


def import_library_bundle(bundle: Dict[str, Any]) -> int:
    if not isinstance(bundle, dict):
        return 0
    projs = bundle.get("projects")
    if not isinstance(projs, dict):
        return 0
    imported = 0
    for _, proj in projs.items():
        if not isinstance(proj, dict):
            continue
        pid = import_project_bundle({"project": proj}, target_bay=proj.get("bay", "NEW"), rename="")
        if pid:
            imported += 1

    w = bundle.get("sb_workspace")
    if isinstance(w, dict) and w.get("workspace_story_bible_id"):
        cur = st.session_state.sb_workspace or default_story_bible_workspace()
        cur_sb = (cur.get("story_bible", {}) or {})
        cur_empty = not any((cur_sb.get(k, "") or "").strip() for k in ["synopsis", "genre_style_notes", "world", "characters", "outline"])
        if cur_empty:
            st.session_state.sb_workspace = w
    return imported


# ============================================================
# JUNK DRAWER COMMANDS
# ============================================================
CMD_FIND = re.compile(r"^\s*/find\s*:\s*(.+)$", re.IGNORECASE)
CMD_CREATE = re.compile(r"^\s*/create\s*:\s*(.+)$", re.IGNORECASE)
CMD_PROMOTE = re.compile(r"^\s*/promote\s*$", re.IGNORECASE)


def _run_find(term: str) -> str:
    term = (term or "").strip()
    if not term:
        return "Find: missing search term. Use /find: word"

    def _hits(label: str, text: str) -> List[str]:
        lines = (text or "").splitlines()
        out = []
        for i, line in enumerate(lines, start=1):
            if term.lower() in (line or "").lower():
                out.append(f"{label} L{i}: {line.strip()}")
            if len(out) >= 20:
                break
        return out

    hits = []
    hits += _hits("DRAFT", st.session_state.main_text)
    hits += _hits("SYNOPSIS", st.session_state.synopsis)
    hits += _hits("WORLD", st.session_state.world)
    hits += _hits("CHARS", st.session_state.characters)
    hits += _hits("OUTLINE", st.session_state.outline)

    if not hits:
        return f"Find: no matches for '{term}'."
    return "\n".join(hits[:30])


def handle_junk_commands() -> None:
    raw = (st.session_state.junk or "").strip()
    if not raw:
        return

    m = CMD_CREATE.match(raw)
    if m:
        title = m.group(1).strip()
        pid = create_project_from_current_bible(title)
        st.session_state.active_project_by_bay["NEW"] = pid
        st.session_state.active_bay = "NEW"
        st.session_state.pending_load_project = pid
        st.session_state.voice_status = f"Created project in NEW: {title}"
        st.session_state.last_action = "Create Project"
        st.session_state.junk = ""
        autosave()
        st.rerun()

    if CMD_PROMOTE.match(raw):
        pid = st.session_state.project_id
        bay = st.session_state.active_bay
        nb = next_bay(bay)
        if not pid or not nb:
            st.session_state.scratch_pad = "Promote: no project selected, or already FINAL."
            st.session_state.voice_status = "Promote blocked."
            st.session_state.junk = ""
            autosave()
            return
        save_session_into_project()
        promote_project(pid, nb)
        st.session_state.active_project_by_bay[nb] = pid
        switch_bay(nb)
        st.session_state.voice_status = f"Promoted → {nb}: {st.session_state.project_title}"
        st.session_state.last_action = f"Promote → {nb}"
        st.session_state.junk = ""
        autosave()
        return

    m = CMD_FIND.match(raw)
    if m:
        st.session_state.scratch_pad = _clamp_text(_run_find(m.group(1)))
        st.session_state.voice_status = "Find complete"
        st.session_state.last_action = "Find"
        st.session_state.junk = ""
        autosave()
        return


# Run junk commands early (before widgets instantiate)
handle_junk_commands()


# ============================================================
# AI BRIEF + CALL
# ============================================================
def _story_bible_text() -> str:
    sb = []
    if (st.session_state.synopsis or "").strip():
        sb.append(f"SYNOPSIS:\n{st.session_state.synopsis.strip()}")
    if (st.session_state.genre_style_notes or "").strip():
        sb.append(f"GENRE/STYLE NOTES:\n{st.session_state.genre_style_notes.strip()}")
    if (st.session_state.world or "").strip():
        sb.append(f"WORLD:\n{st.session_state.world.strip()}")
    if (st.session_state.characters or "").strip():
        sb.append(f"CHARACTERS:\n{st.session_state.characters.strip()}")
    if (st.session_state.outline or "").strip():
        sb.append(f"OUTLINE:\n{st.session_state.outline.strip()}")
    return "\n\n".join(sb).strip() if sb else "— None provided —"


def build_partner_brief(action_name: str, lane: str) -> str:
    story_bible = _story_bible_text()
    vb = []
    if st.session_state.vb_style_on:
        vb.append(f"Writing Style: {st.session_state.writing_style} (intensity {st.session_state.style_intensity:.2f})")
    if st.session_state.vb_genre_on:
        vb.append(f"Genre Influence: {st.session_state.genre} (intensity {st.session_state.genre_intensity:.2f})")
    if st.session_state.vb_trained_on and st.session_state.trained_voice and st.session_state.trained_voice != "— None —":
        vb.append(f"Trained Voice: {st.session_state.trained_voice} (intensity {st.session_state.trained_intensity:.2f})")
    if st.session_state.vb_match_on and (st.session_state.voice_sample or "").strip():
        vb.append(f"Match Sample (intensity {st.session_state.match_intensity:.2f}):\n{st.session_state.voice_sample.strip()}")
    if st.session_state.vb_lock_on and (st.session_state.voice_lock_prompt or "").strip():
        vb.append(f"VOICE LOCK (strength {st.session_state.lock_intensity:.2f}):\n{st.session_state.voice_lock_prompt.strip()}")
    voice_controls = "\n\n".join(vb).strip() if vb else "— None enabled —"

    # Engine Style (trainable banks)
    style_name = (st.session_state.writing_style or "").strip().upper()
    style_directive = ""
    style_exemplars: List[str] = []
    if st.session_state.vb_style_on and style_name in ENGINE_STYLES:
        style_directive = engine_style_directive(style_name, float(st.session_state.style_intensity), lane)
        ctx2 = (st.session_state.main_text or "")[-2500:]
        q2 = ctx2 if ctx2.strip() else (st.session_state.synopsis or "")
        k = 1 + int(max(0.0, min(1.0, float(st.session_state.style_intensity))) * 2.0)
        style_exemplars = retrieve_style_exemplars(style_name, lane, q2, k=k)

    exemplars: List[str] = []
    tv = st.session_state.trained_voice
    if st.session_state.vb_trained_on and tv and tv != "— None —":
        ctx = (st.session_state.main_text or "")[-2500:]
        query = ctx if ctx.strip() else st.session_state.synopsis
        exemplars = retrieve_mixed_exemplars(tv, lane, query)
    ex_block = "\n\n---\n\n".join(exemplars) if exemplars else "— None —"
    style_ex_block = "\n\n---\n\n".join(style_exemplars) if style_exemplars else "— None —"

    ai_x = float(st.session_state.ai_intensity)
    return f"""
YOU ARE OLIVETTI: the author's personal writing and editing partner.
Professional output only. No UI talk. No process talk.

STORY BIBLE IS CANON + IDEA BANK.
When generating NEW material, pull at least 2 concrete specifics from the Story Bible.
Never contradict canon. Never add random characters/places unless Story Bible supports it.

LANE: {lane}

AI INTENSITY: {ai_x:.2f}
INTENSITY PROFILE: {intensity_profile(ai_x)}

VOICE CONTROLS:
{voice_controls}

ENGINE STYLE DIRECTIVE:
{style_directive if style_directive else "— None —"}

STYLE EXEMPLARS (mimic cadence/diction, not content):
{style_ex_block}

TRAINED EXEMPLARS (mimic patterns, not content):
{ex_block}

STORY BIBLE:
{story_bible}

ACTION: {action_name}
""".strip()


def call_openai(system_brief: str, user_task: str, text: str) -> str:
    key = require_openai_key()
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. Add to requirements.txt: openai") from e

    try:
        client = OpenAI(api_key=key, timeout=60)
    except TypeError:
        client = OpenAI(api_key=key)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_brief},
            {"role": "user", "content": f"{user_task}\n\nDRAFT:\n{text.strip()}"},
        ],
        temperature=temperature_from_intensity(st.session_state.ai_intensity),
    )
    return (resp.choices[0].message.content or "").strip()


def local_cleanup(text: str) -> str:
    t = (text or "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{4,}", "\n\n\n", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"([,.;:!?])([A-Za-z0-9])", r"\1 \2", t)
    t = re.sub(r"\.\.\.", "…", t)
    t = re.sub(r"\s*--\s*", " — ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def sb_breakdown_ai(text: str) -> Dict[str, str]:
    prompt = """Analyze this manuscript/draft and extract key Story Bible elements.
Return STRICT JSON with these exact keys:
synopsis, genre_style_notes, outline

EXTRACTION GUIDELINES:

**synopsis** (2-4 paragraphs):
- Core premise and central conflict
- Main character's goal and what's at stake
- Key turning points and narrative arc
- Emotional throughline

**genre_style_notes**:
- Genre classification (thriller, romance, literary, etc.)
- Tone and mood (dark, comedic, lyrical, gritty)
- Pacing style (fast-paced, slow burn, episodic)
- Narrative voice observations (first person intimate, distant third, etc.)
- Any distinctive stylistic elements

**outline** (scene/chapter level):
- Break down the narrative structure
- List major scenes or chapters with brief descriptions
- Note POV shifts if multiple viewpoints
- Track subplot threads
- Identify act breaks or major turning points

Rules:
- Extract EVERYTHING relevant from the text
- Use exact names, places, and terms from the manuscript
- If a section has no relevant content, return empty string
- Be comprehensive
- No commentary outside the JSON structure
"""
    try:
        out = call_openai(
            system_brief="You are an expert literary analyst. Extract synopsis, genre/style notes, and outline from the manuscript. Output only valid JSON.",
            user_task=prompt,
            text=text,
        )
        obj = _extract_json_object(out) or {}
        return {
            "synopsis": _normalize_text(str(obj.get("synopsis", ""))),
            "genre_style_notes": _normalize_text(str(obj.get("genre_style_notes", ""))),
            "outline": _normalize_text(str(obj.get("outline", ""))),
        }
    except Exception:
        return {"synopsis": "", "genre_style_notes": "", "outline": ""}


def _get_idea_context() -> str:
    """Gather idea content from Junk Drawer, Synopsis, and Genre/Style for AI generation."""
    parts = []
    if (st.session_state.junk or "").strip():
        parts.append(f"NOTES/IDEAS:\n{st.session_state.junk.strip()}")
    if (st.session_state.synopsis or "").strip():
        parts.append(f"SYNOPSIS:\n{st.session_state.synopsis.strip()}")
    if (st.session_state.genre_style_notes or "").strip():
        parts.append(f"GENRE/STYLE:\n{st.session_state.genre_style_notes.strip()}")
    return "\n\n".join(parts) if parts else ""


def generate_world_ai() -> str:
    """Generate World Elements from idea context."""
    context = _get_idea_context()
    if not context.strip():
        return ""
    prompt = """Based on the provided story ideas, synopsis, and genre notes, generate comprehensive WORLD ELEMENTS.

Include:
- Time period and era (specific dates/years if possible)
- Primary locations (cities, regions, countries, planets)
- Physical environment (climate, geography, architecture)
- Social structure (class systems, governments, organizations)
- Cultural elements (customs, beliefs, traditions)
- Technology or magic systems if applicable
- Economic conditions
- Any rules or laws that affect the story
- Sensory details (what does this world look, sound, smell like?)

Be specific and detailed. Use any names/places mentioned in the source material.
Output prose paragraphs, not JSON."""
    try:
        return call_openai(
            system_brief="You are a worldbuilding expert. Generate rich, detailed world elements for fiction.",
            user_task=prompt,
            text=context,
        )
    except Exception:
        return ""


def generate_characters_ai() -> str:
    """Generate Characters from idea context."""
    context = _get_idea_context()
    if not context.strip():
        return ""
    prompt = """Based on the provided story ideas, synopsis, and genre notes, generate a comprehensive CHARACTER BIBLE.

For each character include:
- Full name (and nicknames/aliases)
- Role: protagonist, antagonist, mentor, love interest, sidekick, etc.
- Age and physical description
- Personality traits (3-5 key traits)
- Background/history
- Motivation (what do they want?)
- Internal conflict (what holds them back?)
- Key relationships to other characters
- Character arc (how will they change?)
- Distinctive voice/speech patterns
- Secrets or hidden depths

Create main characters in detail, and sketch supporting characters.
Use any character names mentioned in the source material.
Output prose paragraphs organized by character, not JSON."""
    try:
        return call_openai(
            system_brief="You are a character development expert. Generate deep, compelling character profiles for fiction.",
            user_task=prompt,
            text=context,
        )
    except Exception:
        return ""


def generate_outline_ai() -> str:
    """Generate Outline from idea context."""
    context = _get_idea_context()
    if not context.strip():
        return ""
    prompt = """Based on the provided story ideas, synopsis, and genre notes, generate a detailed STORY OUTLINE.

Include:
- Act structure (3-act, 5-act, or genre-appropriate)
- Opening hook/inciting incident
- Key plot points and turning points
- Midpoint shift
- Rising action beats
- Climax setup and execution
- Resolution/denouement
- Subplot threads and how they weave in
- Chapter or scene breakdown (numbered)
- POV assignments if multiple viewpoints
- Pacing notes (fast/slow sections)
- Emotional arc alongside plot arc

Be specific about what happens in each section.
Use character names and locations from the source material.
Output as a structured outline with clear sections, not JSON."""
    try:
        return call_openai(
            system_brief="You are a story structure expert. Generate detailed, actionable outlines for fiction.",
            user_task=prompt,
            text=context,
        )
    except Exception:
        return ""


def _merge_section(existing: str, incoming: str, mode: str) -> str:
    ex = (existing or "").strip()
    inc = (incoming or "").strip()
    if mode == "Replace":
        return inc
    if not ex:
        return inc
    if not inc:
        return ex
    return (ex + "\n\n" + inc).strip()


# ============================================================
# ACTIONS (queued for Streamlit safety)
# ============================================================
def partner_action(action: str) -> None:
    text = st.session_state.scratch_pad or ""
    lane = current_lane_from_draft(text)
    brief = build_partner_brief(action, lane=lane)
    use_ai = has_openai_key()

    def apply_replace(result: str) -> None:
        if result and result.strip():
            st.session_state.scratch_pad_2 = result.strip()
            st.session_state.last_action = action
            st.session_state.voice_status = f"{action} complete"
            autosave()

    def apply_append(result: str) -> None:
        if result and result.strip():
            if (st.session_state.scratch_pad_2 or "").strip():
                st.session_state.scratch_pad_2 = (st.session_state.scratch_pad_2.rstrip() + "\n\n" + result.strip()).strip()
            else:
                st.session_state.scratch_pad_2 = result.strip()
            st.session_state.last_action = action
            st.session_state.voice_status = f"{action} complete"
            autosave()

    try:
        if action == "Write":
            if use_ai:
                task = (
                    f"Continue decisively in lane ({lane}). Add 1–3 paragraphs that advance the scene. "
                    "MANDATORY: incorporate at least 2 Story Bible specifics. "
                    "No recap. No planning. Just prose."
                )
                out = call_openai(brief, task, text if text.strip() else "Start the opening.")
                apply_append(out)
            else:
                st.session_state.voice_status = "Write requires OPENAI_API_KEY"
                autosave()
            return

        if action == "Rewrite":
            if use_ai:
                task = f"Rewrite for professional quality in lane ({lane}). Preserve meaning and canon. Return full revised text."
                out = call_openai(brief, task, text)
                apply_replace(out)
            else:
                st.session_state.voice_status = "Rewrite requires OPENAI_API_KEY"
                apply_replace(local_cleanup(text))
            return

        if action == "Expand":
            if use_ai:
                task = f"Expand with meaningful depth in lane ({lane}). No padding. Preserve canon. Return full revised text."
                out = call_openai(brief, task, text)
                apply_replace(out)
            else:
                st.session_state.voice_status = "Expand requires OPENAI_API_KEY"
                autosave()
            return

        if action == "Rephrase":
            if use_ai:
                task = f"Replace the final sentence with a stronger one (same meaning) in lane ({lane}). Return full text."
                out = call_openai(brief, task, text)
                apply_replace(out)
            else:
                st.session_state.voice_status = "Rephrase requires OPENAI_API_KEY"
                autosave()
            return

        if action == "Describe":
            if use_ai:
                task = f"Add vivid controlled description in lane ({lane}). Preserve pace and canon. Return full revised text."
                out = call_openai(brief, task, text)
                apply_replace(out)
            else:
                st.session_state.voice_status = "Describe requires OPENAI_API_KEY"
                autosave()
            return

        if action in ("Spell", "Grammar"):
            cleaned = local_cleanup(text)
            if use_ai:
                task = "Copyedit spelling/grammar/punctuation. Preserve voice. Return full revised text."
                out = call_openai(brief, task, cleaned)
                apply_replace(out if out else cleaned)
            else:
                apply_replace(cleaned)
            return

        if action == "Synonym":
            # AI reads from Notes, outputs to Paste
            last = ""
            m = re.search(r"([A-Za-z']{3,})\W*$", text.strip())
            if m:
                last = m.group(1)
            if not last:
                st.session_state.scratch_pad_2 = "Synonym: couldn't detect a target word (try ending with a word)."
                st.session_state.voice_status = "Synonym: no target"
                autosave()
                return
            if use_ai:
                task = (
                    f"Give 12 strong synonyms for '{last}'. "
                    "Group them by nuance (formal, punchy, poetic, archaic, etc). "
                    "No filler." 
                )
                out = call_openai(brief, task, text)
                st.session_state.scratch_pad_2 = f"Synonyms for '{last}':\n\n" + _clamp_text(out)
            else:
                st.session_state.scratch_pad_2 = f"Synonym requires OPENAI_API_KEY (target word: {last})."
            st.session_state.voice_status = f"Synonym: {last}"
            st.session_state.last_action = "Synonym"
            autosave()
            return

        if action == "Sentence":
            # AI reads from Notes, outputs to Paste
            last_sentence = ""
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            if sentences:
                last_sentence = sentences[-1].strip()
            if not last_sentence:
                st.session_state.scratch_pad_2 = "Sentence: couldn't detect a final sentence."
                st.session_state.voice_status = "Sentence: no target"
                autosave()
                return
            if use_ai:
                task = (
                    "Provide 8 alternative rewrites of the final sentence. "
                    "Keep meaning. Vary rhythm and diction. Return as a numbered list."
                )
                out = call_openai(brief, task, text)
                st.session_state.scratch_pad_2 = "Sentence alternatives:\n\n" + _clamp_text(out)
            else:
                st.session_state.scratch_pad_2 = "Sentence requires OPENAI_API_KEY."
            st.session_state.voice_status = "Sentence options"
            st.session_state.last_action = "Sentence"
            autosave()
            return

    except Exception as e:
        msg = str(e)
        if ("insufficient_quota" in msg) or ("exceeded your current quota" in msg.lower()):
            st.session_state.voice_status = "Engine: OpenAI quota exceeded."
            st.session_state.scratch_pad_2 = _clamp_text(
                "OpenAI returned a quota/billing error.\n\nFix:\n• Confirm your API key is correct\n• Check billing/usage limits\n• Or swap to a different key in Streamlit Secrets"
            )
        elif "OPENAI_API_KEY not set" in msg:
            st.session_state.voice_status = "Engine: missing OPENAI_API_KEY."
            st.session_state.scratch_pad_2 = "Set OPENAI_API_KEY in Streamlit Secrets (or environment) to enable AI."
        else:
            st.session_state.voice_status = f"Engine: {msg}"
            st.session_state.scratch_pad_2 = _clamp_text(f"ERROR:\n{msg}")
        autosave()


def queue_action(action: str) -> None:
    st.session_state.pending_action = action


def run_pending_action() -> None:
    action = st.session_state.get("pending_action")
    if not action:
        return
    st.session_state.pending_action = None

    # Find/Replace action - search term from Notes, results + AI suggestions to Paste
    if action == "__FIND__":
        term = (st.session_state.scratch_pad or "").strip()
        if not term:
            st.session_state.scratch_pad_2 = "Find: paste search term into Notes field first."
            st.session_state.voice_status = "Find: no search term"
            st.session_state.find_count = 0
            autosave()
            return
        
        # Search Draft only
        draft = st.session_state.main_text or ""
        lines = draft.splitlines()
        hits = []
        for i, line in enumerate(lines, start=1):
            if term.lower() in (line or "").lower():
                hits.append(f"L{i}: {line.strip()[:80]}")
            if len(hits) >= 6:
                break
        
        # Store count for visual indicator
        total_count = draft.lower().count(term.lower())
        st.session_state.find_count = total_count
        st.session_state.find_term = term
        
        # Build results for Paste
        if not hits:
            st.session_state.scratch_pad_2 = f"Find: no matches for '{term}' in Draft."
        else:
            result = f"Found {total_count} occurrence(s) of '{term}':\n\n" + "\n".join(hits[:6])
            
            # AI suggests replacements → append to Paste
            if has_openai_key():
                try:
                    from openai import OpenAI
                    key = require_openai_key()
                    client = OpenAI(api_key=key, timeout=30)
                    resp = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": "You suggest replacement words/phrases for editing. Be concise."},
                            {"role": "user", "content": f"Give 6 alternative words or phrases to replace '{term}'. Context: fiction writing. Just list them, one per line, no numbering or explanation."},
                        ],
                        temperature=0.7,
                    )
                    suggestions = (resp.choices[0].message.content or "").strip()
                    result += f"\n\nReplace '{term}' with:\n{suggestions}"
                except Exception:
                    pass
            
            st.session_state.scratch_pad_2 = result
        
        st.session_state.voice_status = f"Find: {total_count} matches"
        st.session_state.last_action = "Find"
        autosave()
        return

    if action == "__VAULT_CLEAR_SAMPLE__":
        # Clear the vault sample text area safely (pre-widget) and surface status.
        st.session_state.vault_sample_text = ""
        note = (st.session_state.get("ui_notice") or "").strip()
        if note:
            st.session_state.voice_status = note
            st.session_state.ui_notice = ""
        st.session_state.last_action = "Voice Vault"
        autosave()
        return

    if action == "__STYLE_CLEAR_PASTE__":
        # Clear the style trainer text area safely (pre-widget)
        st.session_state.style_train_paste = ""
        note = (st.session_state.get("ui_notice") or "").strip()
        if note:
            st.session_state.voice_status = note
            st.session_state.ui_notice = ""
        st.session_state.last_action = "Style Trainer"
        autosave()
        return

    partner_action(action)


# Run queued actions early (pre-widget)
run_pending_action()


# ============================================================
# UI — TOP BAR
# ============================================================
top = st.container()
with top:
    tab_rough, tab_edit, tab_final = st.tabs(["✏️ Rough", "🛠 Edit", "✅ Final"])

    def _bay_selector_ui(bay: str) -> None:
        with st.expander(f"Select {bay.title()} Project", expanded=False):
            bay_projects = list_projects_in_bay(bay)
            items: List[Tuple[Optional[str], str]] = [(None, "— (none) —")] + [(pid, title) for pid, title in bay_projects]

            # Disambiguate duplicate titles
            seen: Dict[str, int] = {}
            labels: List[str] = []
            ids: List[Optional[str]] = []
            for pid, title in items:
                base = title
                seen[base] = seen.get(base, 0) + 1
                label = base if seen[base] == 1 else f"{base}  ·  {str(pid)[-4:]}"
                labels.append(label)
                ids.append(pid)

            current_pid = (st.session_state.active_project_by_bay or {}).get(bay)
            current_idx = ids.index(current_pid) if current_pid in ids else 0

            sel_label = st.selectbox(
                f"{bay.title()} Projects",
                labels,
                index=current_idx,
                key=f"bay_select_{bay.lower()}"
            )
            sel_pid = ids[labels.index(sel_label)] if sel_label in labels else None

            if sel_pid != current_pid:
                st.session_state.active_project_by_bay[bay] = sel_pid
                if sel_pid:
                    # Use pending pattern - load BEFORE widgets on next rerun
                    st.session_state.pending_load_project = sel_pid
                    st.session_state.voice_status = f"{bay}: Loading..."
                else:
                    st.session_state.project_id = None
                    st.session_state.project_title = "—"
                    st.session_state.voice_status = f"{bay}: (none)"
                st.session_state.last_action = f"Select {bay.title()} Project"
                autosave()
                st.rerun()

    # Initialize visibility flags for selectors
    if "show_selector_rough" not in st.session_state:
        st.session_state.show_selector_rough = False
    if "show_selector_edit" not in st.session_state:
        st.session_state.show_selector_edit = False
    if "show_selector_final" not in st.session_state:
        st.session_state.show_selector_final = False

    with tab_rough:
        if st.button("Set Active: Rough", key="set_active_rough"):
            switch_bay("ROUGH")
            save_all_to_disk(force=True)
            st.session_state.show_selector_rough = True
        if st.session_state.get("show_selector_rough"):
            _bay_selector_ui("ROUGH")

    with tab_edit:
        if st.button("Set Active: Edit", key="set_active_edit"):
            switch_bay("EDIT")
            save_all_to_disk(force=True)
            st.session_state.show_selector_edit = True
        if st.session_state.get("show_selector_edit"):
            _bay_selector_ui("EDIT")

    with tab_final:
        if st.button("Set Active: Final", key="set_active_final"):
            switch_bay("FINAL")
            save_all_to_disk(force=True)
            st.session_state.show_selector_final = True
        if st.session_state.get("show_selector_final"):
            _bay_selector_ui("FINAL")

st.divider()

# ============================================================
# LOCKED LAYOUT (same ratios)
# ============================================================
left, center, right = st.columns([1.2, 3.2, 1.6])


# ============================================================
# LEFT — STORY BIBLE
# ============================================================
with left:
    st.subheader("📖 Story Bible")

    bay = st.session_state.active_bay
    # Current Bay Project selector removed per request

    # Hard lock: Story Bible edits are locked per-project unless explicitly unlocked
    is_project = bool(st.session_state.project_id)
    sb_edit_unlocked = bool((st.session_state.locks or {}).get("sb_edit_unlocked", False))
    sb_locked = is_project and (not sb_edit_unlocked)


    # ✅ AI Intensity lives in Story Bible panel (left)
    st.slider(
        "AI Intensity",
        0.0,
        1.0,
        key="ai_intensity",
        help="0.0 = conservative/precise, 1.0 = bold/creative. Applies to every AI generation.",
        on_change=autosave,
    )

    # Project controls
    action_cols = st.columns([1, 1])
    if bay == "NEW":
        label = "Start Project (Lock Bible → Project)" if in_workspace_mode() else "Create Project (from Bible)"
        if action_cols[0].button(label, key="create_project_btn"):
            title_guess = (st.session_state.synopsis.strip().splitlines()[0].strip() if st.session_state.synopsis.strip() else "New Project")
            pid = create_project_from_current_bible(title_guess)
            st.session_state.pending_load_project = pid
            st.session_state.voice_status = f"Created in NEW: {title_guess}"
            st.session_state.last_action = "Create Project"
            autosave()
            st.rerun()

        if in_workspace_mode() and action_cols[1].button("New Story Bible (fresh ID)", key="new_workspace_bible_btn"):
            reset_workspace_story_bible(keep_templates=True)
            st.session_state.voice_status = "Workspace: new Story Bible minted"
            st.session_state.last_action = "New Story Bible"
            autosave()
            st.rerun()

    # Import / Export hub (restored)
    with st.expander("📦 Import / Export", expanded=False):
        tab_imp, tab_exp = st.tabs(["Import", "Export"])

        with tab_imp:
            st.caption("Import a document into Draft or break it into Story Bible sections.")
            up = st.file_uploader("Upload (.txt, .md, .docx)", type=["txt", "md", "docx"], key="io_upload")
            paste = st.text_area("Paste text", key="io_paste", height=140)
            target = st.radio("Import target", ["Draft", "Story Bible"], horizontal=True, key="io_target")
            use_ai = st.checkbox(
                "Use AI Breakdown (Story Bible)",
                value=has_openai_key(),
                disabled=not has_openai_key(),
                help="Requires OPENAI_API_KEY. Falls back to heuristic if AI fails.",
                key="io_use_ai",
            )

            if st.button("Run Import", key="io_run_import"):
                src_file, name = _read_uploaded_text(up)
                src = _normalize_text(paste if (paste or "").strip() else src_file)
                if not src.strip():
                    st.session_state.scratch_pad_2 = "Import: no text provided (or file too large)."
                    st.session_state.voice_status = "Import blocked"
                    autosave()
                    st.rerun()
                elif target == "Draft":
                    # Import to Draft: create a new project in ROUGH bay (folder only, not draft window)
                    import_title = os.path.splitext(name)[0] if name else f"Imported {now_ts()[:10]}"
                    p = new_project_payload(import_title)
                    p["bay"] = "ROUGH"
                    p["draft"] = src
                    st.session_state.projects[p["id"]] = p
                    st.session_state.active_project_by_bay["ROUGH"] = p["id"]
                    st.session_state.voice_status = f"Imported → ROUGH: {import_title}"
                    st.session_state.last_action = "Import → Rough Draft"
                    autosave()
                    st.rerun()
                else:
                    # Story Bible: AI analyzes into synopsis, genre, outline only
                    if sb_locked:
                        st.session_state.scratch_pad_2 = "Story Bible is LOCKED. Unlock Story Bible Editing to import into it."
                        st.session_state.voice_status = "Import blocked (locked)"
                        autosave()
                        st.rerun()
                    else:
                        # Try AI breakdown first, fall back to heuristic if AI fails
                        if use_ai:
                            try:
                                sections = sb_breakdown_ai(src)
                                # Check if AI returned meaningful content
                                if not any((sections.get(k, "") or "").strip() for k in ["synopsis", "genre_style_notes", "outline"]):
                                    # AI returned empty results, fall back to heuristic
                                    sections = _sb_sections_from_text_heuristic(src)
                                    analysis_method = "AI + heuristic fallback"
                                else:
                                    analysis_method = "AI analyzed"
                            except Exception:
                                # AI breakdown failed completely, use heuristic
                                sections = _sb_sections_from_text_heuristic(src)
                                analysis_method = "heuristic (AI failed)"
                        else:
                            # User chose heuristic mode
                            sections = _sb_sections_from_text_heuristic(src)
                            analysis_method = "heuristic"
                        
                        # Update Story Bible sections
                        st.session_state.synopsis = sections.get("synopsis", "") or st.session_state.synopsis
                        st.session_state.genre_style_notes = sections.get("genre_style_notes", "") or st.session_state.genre_style_notes
                        st.session_state.outline = sections.get("outline", "") or st.session_state.outline
                        st.session_state.voice_status = f"Imported → Story Bible ({analysis_method})"
                        st.session_state.last_action = "Import → Story Bible"
                        autosave()
                        st.rerun()

        with tab_exp:
            title = "Workspace" if in_workspace_mode() else st.session_state.project_title
            stem = _safe_filename(title, "olivetti")

            # Ensure latest writes are saved into project/workspace
            if in_workspace_mode():
                save_workspace_from_session()
            else:
                save_session_into_project()

            draft_txt = st.session_state.main_text or ""

            st.download_button("Download Draft (.txt)", data=draft_txt, file_name=f"{stem}_draft.txt", mime="text/plain")
            st.download_button("Download Draft (.md)", data=f"# {title}\n\n{draft_txt}", file_name=f"{stem}_draft.md", mime="text/markdown")

            try:
                from docx import Document  # type: ignore

                def _docx_bytes(doc: "Document") -> bytes:
                    buf = BytesIO()
                    doc.save(buf)
                    return buf.getvalue()

                d = Document()
                d.add_heading(title, level=0)
                for para in _split_paragraphs(draft_txt):
                    d.add_paragraph(para)
                st.download_button(
                    "Download Draft (.docx)",
                    data=_docx_bytes(d),
                    file_name=f"{stem}_draft.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            except Exception:
                st.caption("DOCX export unavailable (python-docx not installed).")

            # FINAL bay gets fancy export formats
            if st.session_state.active_bay == "FINAL":
                st.divider()
                st.subheader("📚 Publication Formats")
                
                # Author metadata for formatted exports
                author_name = st.text_input("Author Name", key="export_author", value=st.session_state.get("export_author_saved", ""))
                if author_name != st.session_state.get("export_author_saved", ""):
                    st.session_state.export_author_saved = author_name
                
                # EPUB export
                try:
                    from ebooklib import epub  # type: ignore
                    
                    def _epub_bytes(title: str, author: str, content: str) -> bytes:
                        book = epub.EpubBook()
                        book.set_identifier(f"olivetti-{hashlib.md5(title.encode()).hexdigest()[:8]}")
                        book.set_title(title)
                        book.set_language("en")
                        book.add_author(author or "Unknown Author")
                        
                        # Split content into chapters by ## headers or --- dividers
                        chapters = []
                        parts = re.split(r'\n##\s+|\n---\n', content)
                        for i, part in enumerate(parts):
                            if not part.strip():
                                continue
                            ch = epub.EpubHtml(title=f"Chapter {i+1}", file_name=f"chap_{i+1}.xhtml", lang="en")
                            # Convert to basic HTML
                            html_content = "<h1>" + title + "</h1>" if i == 0 else ""
                            for para in part.strip().split("\n\n"):
                                if para.strip():
                                    html_content += f"<p>{para.strip()}</p>\n"
                            ch.content = f"<html><body>{html_content}</body></html>"
                            book.add_item(ch)
                            chapters.append(ch)
                        
                        # Table of contents and spine
                        book.toc = chapters
                        book.add_item(epub.EpubNcx())
                        book.add_item(epub.EpubNav())
                        book.spine = ["nav"] + chapters
                        
                        buf = BytesIO()
                        epub.write_epub(buf, book)
                        return buf.getvalue()
                    
                    st.download_button(
                        "📖 Download EPUB",
                        data=_epub_bytes(title, author_name, draft_txt),
                        file_name=f"{stem}.epub",
                        mime="application/epub+zip",
                    )
                except Exception:
                    st.caption("EPUB export unavailable (install: pip install ebooklib)")
                
                # HTML export (formatted for print/web)
                def _html_export(title: str, author: str, content: str) -> str:
                    html_body = ""
                    for para in content.split("\n\n"):
                        p = para.strip()
                        if not p:
                            continue
                        if p.startswith("# "):
                            html_body += f"<h1>{p[2:]}</h1>\n"
                        elif p.startswith("## "):
                            html_body += f"<h2>{p[3:]}</h2>\n"
                        elif p.startswith("### "):
                            html_body += f"<h3>{p[4:]}</h3>\n"
                        else:
                            html_body += f"<p>{p}</p>\n"
                    
                    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <meta name="author" content="{author}">
    <style>
        body {{
            font-family: Georgia, 'Times New Roman', serif;
            max-width: 700px;
            margin: 2em auto;
            padding: 1em;
            line-height: 1.8;
            color: #1a1a1a;
            background: #fefefe;
        }}
        h1 {{ font-size: 2.2em; margin-top: 1.5em; text-align: center; }}
        h2 {{ font-size: 1.6em; margin-top: 1.5em; border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }}
        h3 {{ font-size: 1.3em; margin-top: 1.2em; }}
        p {{ text-indent: 1.5em; margin: 0.8em 0; text-align: justify; }}
        p:first-of-type {{ text-indent: 0; }}
        .title-page {{ text-align: center; margin: 3em 0; }}
        .author {{ font-style: italic; font-size: 1.2em; }}
        @media print {{
            body {{ max-width: 100%; margin: 0; padding: 1in; }}
            h1, h2 {{ page-break-after: avoid; }}
            p {{ orphans: 3; widows: 3; }}
        }}
    </style>
</head>
<body>
    <div class="title-page">
        <h1>{title}</h1>
        <p class="author">by {author or 'Anonymous'}</p>
    </div>
    <hr>
    {html_body}
</body>
</html>"""
                
                st.download_button(
                    "🌐 Download HTML",
                    data=_html_export(title, author_name, draft_txt),
                    file_name=f"{stem}.html",
                    mime="text/html",
                )
                
                # RTF export (rich text for word processors)
                def _rtf_export(title: str, author: str, content: str) -> str:
                    # Basic RTF structure
                    rtf = r"{\rtf1\ansi\deff0 {\fonttbl{\f0 Times New Roman;}}"
                    rtf += r"{\info{\title " + title + r"}{\author " + (author or "Unknown") + r"}}"
                    rtf += r"\f0\fs24 "
                    
                    # Title
                    rtf += r"\qc\b\fs48 " + title + r"\b0\fs24\par\par"
                    if author:
                        rtf += r"\qc\i by " + author + r"\i0\par\par"
                    rtf += r"\ql "
                    
                    # Content
                    for para in content.split("\n\n"):
                        p = para.strip()
                        if not p:
                            continue
                        if p.startswith("# "):
                            rtf += r"\par\b\fs36 " + p[2:] + r"\b0\fs24\par\par"
                        elif p.startswith("## "):
                            rtf += r"\par\b\fs32 " + p[3:] + r"\b0\fs24\par\par"
                        else:
                            # Escape special RTF characters
                            p = p.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
                            rtf += r"\fi720 " + p + r"\par"
                    
                    rtf += "}"
                    return rtf
                
                st.download_button(
                    "📄 Download RTF",
                    data=_rtf_export(title, author_name, draft_txt),
                    file_name=f"{stem}.rtf",
                    mime="application/rtf",
                )
                
                # PDF-ready formatted DOCX (with proper styling)
                try:
                    from docx import Document  # type: ignore
                    from docx.shared import Inches, Pt  # type: ignore
                    from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore
                    
                    def _formatted_docx(title: str, author: str, content: str) -> bytes:
                        doc = Document()
                        
                        # Title page
                        title_para = doc.add_paragraph()
                        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        title_run = title_para.add_run(title)
                        title_run.bold = True
                        title_run.font.size = Pt(28)
                        
                        if author:
                            author_para = doc.add_paragraph()
                            author_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                            author_run = author_para.add_run(f"by {author}")
                            author_run.italic = True
                            author_run.font.size = Pt(14)
                        
                        doc.add_page_break()
                        
                        # Content
                        for para in content.split("\n\n"):
                            p = para.strip()
                            if not p:
                                continue
                            if p.startswith("# "):
                                h = doc.add_heading(p[2:], level=1)
                            elif p.startswith("## "):
                                h = doc.add_heading(p[3:], level=2)
                            elif p.startswith("### "):
                                h = doc.add_heading(p[4:], level=3)
                            else:
                                pg = doc.add_paragraph(p)
                                pg.paragraph_format.first_line_indent = Inches(0.5)
                                pg.paragraph_format.space_after = Pt(0)
                                pg.paragraph_format.line_spacing = 1.5
                        
                        buf = BytesIO()
                        doc.save(buf)
                        return buf.getvalue()
                    
                    st.download_button(
                        "📘 Download Formatted DOCX (print-ready)",
                        data=_formatted_docx(title, author_name, draft_txt),
                        file_name=f"{stem}_formatted.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                except Exception:
                    pass

    # Story Bible editor bundled under a single expander button
    with st.expander("📚 Story Bible", expanded=False):
        with st.expander("🗃 Junk Drawer"):
            st.text_area(
                "Junk Drawer",
                key="junk",
                height=80,
                on_change=autosave,
                label_visibility="collapsed",
                help="Commands: /create: Title  |  /promote  |  /find: term",
            )

        with st.expander("📝 Synopsis"):
            st.text_area("Synopsis", key="synopsis", height=100, on_change=autosave, label_visibility="collapsed", disabled=sb_locked)

        with st.expander("🎭 Genre / Style Notes"):
            st.text_area(
                "Genre / Style Notes",
                key="genre_style_notes",
                height=80,
                on_change=autosave,
                label_visibility="collapsed",
                disabled=sb_locked,
            )

        with st.expander("🌍 World Elements"):
            st.text_area("World", key="world", height=100, on_change=autosave, label_visibility="collapsed", disabled=sb_locked)
            if st.button("🤖 Generate World", key="btn_gen_world", disabled=sb_locked or not has_openai_key()):
                ctx = _get_idea_context()
                if not ctx.strip():
                    st.session_state.scratch_pad = "Add ideas to Junk Drawer, Synopsis, or Genre/Style first."
                    st.session_state.voice_status = "Generate World: no source content"
                else:
                    result = generate_world_ai()
                    if result.strip():
                        st.session_state.world = (st.session_state.world.rstrip() + "\n\n" + result.strip()).strip() if (st.session_state.world or "").strip() else result.strip()
                        st.session_state.voice_status = "Generated World Elements"
                        st.session_state.last_action = "Generate World"
                    else:
                        st.session_state.scratch_pad = "World generation failed or returned empty."
                        st.session_state.voice_status = "Generate World: failed"
                autosave()
                st.rerun()

        with st.expander("👤 Characters"):
            st.text_area(
                "Characters",
                key="characters",
                height=120,
                on_change=autosave,
                label_visibility="collapsed",
                disabled=sb_locked,
            )
            if st.button("🤖 Generate Characters", key="btn_gen_chars", disabled=sb_locked or not has_openai_key()):
                ctx = _get_idea_context()
                if not ctx.strip():
                    st.session_state.scratch_pad = "Add ideas to Junk Drawer, Synopsis, or Genre/Style first."
                    st.session_state.voice_status = "Generate Characters: no source content"
                else:
                    result = generate_characters_ai()
                    if result.strip():
                        st.session_state.characters = (st.session_state.characters.rstrip() + "\n\n" + result.strip()).strip() if (st.session_state.characters or "").strip() else result.strip()
                        st.session_state.voice_status = "Generated Characters"
                        st.session_state.last_action = "Generate Characters"
                    else:
                        st.session_state.scratch_pad = "Character generation failed or returned empty."
                        st.session_state.voice_status = "Generate Characters: failed"
                autosave()
                st.rerun()

        with st.expander("🧱 Outline"):
            st.text_area("Outline", key="outline", height=160, on_change=autosave, label_visibility="collapsed", disabled=sb_locked)
            if st.button("🤖 Generate Outline", key="btn_gen_outline", disabled=sb_locked or not has_openai_key()):
                ctx = _get_idea_context()
                if not ctx.strip():
                    st.session_state.scratch_pad = "Add ideas to Junk Drawer, Synopsis, or Genre/Style first."
                    st.session_state.voice_status = "Generate Outline: no source content"
                else:
                    result = generate_outline_ai()
                    if result.strip():
                        st.session_state.outline = (st.session_state.outline.rstrip() + "\n\n" + result.strip()).strip() if (st.session_state.outline or "").strip() else result.strip()
                        st.session_state.voice_status = "Generated Outline"
                        st.session_state.last_action = "Generate Outline"
                    else:
                        st.session_state.scratch_pad = "Outline generation failed or returned empty."
                        st.session_state.voice_status = "Generate Outline: failed"
                autosave()
                st.rerun()

        # Save Story Bible → Rough Draft (formatted sections, then clear Story Bible)
        if st.button("Save to Rough Draft", key="sb_save_to_rough"):
            parts = []
            if (st.session_state.synopsis or "").strip():
                parts.append(f"## Synopsis\n\n{(st.session_state.synopsis or '').strip()}")
            if (st.session_state.genre_style_notes or "").strip():
                parts.append(f"## Genre / Style Notes\n\n{(st.session_state.genre_style_notes or '').strip()}")
            if (st.session_state.world or "").strip():
                parts.append(f"## World Elements\n\n{(st.session_state.world or '').strip()}")
            if (st.session_state.characters or "").strip():
                parts.append(f"## Characters\n\n{(st.session_state.characters or '').strip()}")
            if (st.session_state.outline or "").strip():
                parts.append(f"## Outline\n\n{(st.session_state.outline or '').strip()}")

            if parts:
                seed = "# Rough Draft — from Story Bible\n\n" + "\n\n---\n\n".join(parts)
                st.session_state.main_text = seed.strip()
                # Clear Story Bible sections after saving
                st.session_state.synopsis = ""
                st.session_state.genre_style_notes = ""
                st.session_state.world = ""
                st.session_state.characters = ""
                st.session_state.outline = ""
                st.session_state.voice_status = "Saved Story Bible → Draft (cleared Bible)"
                st.session_state.last_action = "Save to Rough Draft"
            else:
                st.session_state.voice_status = "Nothing to save — Story Bible is empty"
            autosave()
            st.rerun()


# ============================================================
# CENTER — WRITING DESK
# ============================================================
with center:
    # Show find indicator if matches exist
    find_count = st.session_state.get("find_count", 0)
    find_term = st.session_state.get("find_term", "")
    if find_count > 0 and find_term:
        st.markdown(f"<span style='color:red;font-weight:bold;'>🔴 {find_count} match(es) for \"{find_term}\"</span>", unsafe_allow_html=True)
    st.subheader("✍️ Writing Desk")
    st.text_area("Draft", key="main_text", height=650, on_change=autosave, label_visibility="collapsed")
    st.text_area("Notes", key="scratch_pad", height=10, on_change=autosave, label_visibility="collapsed", placeholder="Notes...")
    st.text_area("Paste", key="scratch_pad_2", height=10, on_change=autosave, label_visibility="collapsed", placeholder="Paste...")

    b1 = st.columns(5)
    if b1[0].button("Write", key="btn_write"):
        queue_action("Write")
        st.rerun()
    if b1[1].button("Rewrite", key="btn_rewrite"):
        queue_action("Rewrite")
        st.rerun()
    if b1[2].button("Expand", key="btn_expand"):
        queue_action("Expand")
        st.rerun()
    if b1[3].button("Rephrase", key="btn_rephrase"):
        queue_action("Rephrase")
        st.rerun()
    if b1[4].button("Describe", key="btn_describe"):
        queue_action("Describe")
        st.rerun()

    b2 = st.columns(5)
    if b2[0].button("Spell", key="btn_spell"):
        queue_action("Spell")
        st.rerun()
    if b2[1].button("Grammar", key="btn_grammar"):
        queue_action("Grammar")
        st.rerun()
    if b2[2].button("Find", key="btn_find"):
        queue_action("__FIND__")
        st.rerun()
    if b2[3].button("Synonym", key="btn_synonym"):
        queue_action("Synonym")
        st.rerun()
    if b2[4].button("Sentence", key="btn_sentence"):
        queue_action("Sentence")
        st.rerun()


# ============================================================
# RIGHT — VOICE BIBLE
# ============================================================
with right:
    st.subheader("🎙 Voice Bible")

    with st.expander("✍️ Writing Style", expanded=False):
        st.checkbox("Enable Writing Style", key="vb_style_on", on_change=autosave)
        st.selectbox(
            "Writing Style",
            ["Neutral", "Minimal", "Expressive", "Hardboiled", "Poetic"] + ENGINE_STYLES,
            key="writing_style",
            disabled=not st.session_state.vb_style_on,
            on_change=autosave,
        )
        st.slider("Style Intensity", 0.0, 1.0, key="style_intensity", disabled=not st.session_state.vb_style_on, on_change=autosave)

        with st.expander("🎨 Style Trainer (Engine Styles)", expanded=False):
            st.caption("Train per-project style banks. These steer the engine styles across all actions.")
            s_cols = st.columns([1.2, 1.0, 1.0])
            st_style = s_cols[0].selectbox("Engine style", ENGINE_STYLES, key="style_train_style")
            st_lane = s_cols[1].selectbox("Lane", LANES, key="style_train_lane")
            split_mode = s_cols[2].selectbox("Split", ["Paragraphs", "Whole"], key="style_train_split")

            up = st.file_uploader("Upload training (.txt/.md/.docx)", type=["txt", "md", "docx"], key="style_train_upload")
            paste = st.text_area("Paste training text", key="style_train_paste", height=140)

            c1, c2, c3 = st.columns([1, 1, 1])
            if c1.button("Add Samples", key="style_train_add"):
                ftxt, fname = _read_uploaded_text(up)
                src = _normalize_text((paste or "").strip() if (paste or "").strip() else ftxt)
                if not src.strip():
                    st.session_state.scratch_pad = "Style Trainer: no text provided (or file too large)."
                    st.session_state.voice_status = "Style Trainer blocked"
                    autosave()
                else:
                    n = add_style_samples(st_style, st_lane, src, split_mode=split_mode)
                    st.session_state.voice_status = f"Style Trainer: added {n} sample(s) → {st_style} • {st_lane}"
                    st.session_state.scratch_pad = _clamp_text(f"Added {n} sample(s) to {st_style} / {st_lane}.\nSource: {fname or 'paste'}")
                    autosave()

            if c2.button("Delete last", key="style_train_del"):
                if delete_last_style_sample(st_style, st_lane):
                    st.session_state.voice_status = f"Style Trainer: deleted last → {st_style} • {st_lane}"
                    autosave()
                    st.rerun()
                else:
                    st.warning("Nothing to delete for that style/lane.")

            if c3.button("Clear trainer text", key="style_train_clear"):
                st.session_state.ui_notice = "Style trainer text cleared."
                queue_action("__STYLE_CLEAR_PASTE__")
                st.rerun()

            st.caption("Lane sample counts:")
            bank = (st.session_state.get("style_banks") or {}).get(st_style, {})
            lanes = bank.get("lanes") or {}
            counts = {ln: len((lanes.get(ln) or [])) for ln in LANES}
            st.code("  ".join([f"{ln}: {counts[ln]}" for ln in LANES]), language="")

            if st.button("Clear THIS lane", key="style_train_clear_lane"):
                clear_style_lane(st_style, st_lane)
                st.session_state.voice_status = f"Style Trainer: cleared lane → {st_style} • {st_lane}"
                autosave()
                st.rerun()

    with st.expander("🎭 Genre Influence", expanded=False):
        st.checkbox("Enable Genre Influence", key="vb_genre_on", on_change=autosave)
        st.selectbox(
            "Genre",
            ["Literary", "Noir", "Thriller", "Comedy", "Lyrical"],
            key="genre",
            disabled=not st.session_state.vb_genre_on,
            on_change=autosave,
        )
        st.slider("Genre Intensity", 0.0, 1.0, key="genre_intensity", disabled=not st.session_state.vb_genre_on, on_change=autosave)

    with st.expander("🎙 Trained Voice", expanded=False):
        st.checkbox("Enable Trained Voice", key="vb_trained_on", on_change=autosave)
        trained_options = voice_names_for_selector()
        if st.session_state.trained_voice not in trained_options:
            st.session_state.trained_voice = "— None —"
        st.selectbox(
            "Trained Voice",
            trained_options,
            key="trained_voice",
            disabled=not st.session_state.vb_trained_on,
            on_change=autosave,
        )
        st.slider(
            "Trained Voice Intensity",
            0.0,
            1.0,
            key="trained_intensity",
            disabled=not st.session_state.vb_trained_on,
            on_change=autosave,
        )

    with st.expander("🧬 Voice Vault (Training Samples)", expanded=False):
        st.caption("Drop in passages from your work. Olivetti retrieves the closest exemplars per lane.")

        existing_voices = [v for v in (st.session_state.voices or {}).keys()]
        existing_voices = sorted(existing_voices, key=lambda x: (x not in ("Voice A", "Voice B"), x))
        if not existing_voices:
            existing_voices = ["Voice A", "Voice B"]

        vcol1, vcol2 = st.columns([2, 1])
        vault_voice = vcol1.selectbox("Vault voice", existing_voices, key="vault_voice_sel")
        new_name = vcol2.text_input("New voice", key="vault_new_voice", label_visibility="collapsed", placeholder="New voice name")
        if vcol2.button("Create", key="vault_create_voice"):
            if create_custom_voice(new_name):
                st.session_state.voice_status = f"Voice created: {new_name.strip()}"
                autosave()
                st.rerun()
            else:
                st.warning("Could not create that voice (empty or already exists).")

        lane = st.selectbox("Lane", LANES, key="vault_lane_sel")
        sample = st.text_area("Sample", key="vault_sample_text", height=140, label_visibility="collapsed", placeholder="Paste a passage...")
        a1, a2 = st.columns([1, 1])
        if a1.button("Add sample", key="vault_add_sample"):
            if add_voice_sample(vault_voice, lane, sample):
                st.session_state.ui_notice = f"Added sample → {vault_voice} • {lane}"
                queue_action("__VAULT_CLEAR_SAMPLE__")
                autosave()
                st.rerun()
            else:
                st.warning("No sample text found.")

        # Quick stats + delete last sample
        v = (st.session_state.voices or {}).get(vault_voice, {})
        lane_counts = {ln: len((v.get("lanes", {}) or {}).get(ln, []) or []) for ln in LANES}
        st.caption("Samples: " + "  ".join([f"{ln}: {lane_counts[ln]}" for ln in LANES]))

        if a2.button("Delete last sample", key="vault_del_last"):
            if delete_voice_sample(vault_voice, lane, index_from_end=0):
                st.session_state.voice_status = f"Deleted last sample → {vault_voice} • {lane}"
                autosave()
                st.rerun()
            else:
                st.warning("Nothing to delete for that lane.")

    st.divider()

    with st.expander("🎭 Match My Style", expanded=False):
        st.checkbox("Enable Match My Style", key="vb_match_on", on_change=autosave)
        st.text_area(
            "Style Example",
            key="voice_sample",
            height=100,
            disabled=not st.session_state.vb_match_on,
            on_change=autosave,
        )
        st.slider("Match Intensity", 0.0, 1.0, key="match_intensity", disabled=not st.session_state.vb_match_on, on_change=autosave)

    st.divider()

    with st.expander("🔒 Voice Lock (Hard Constraint)", expanded=False):
        st.checkbox("Voice Lock (Hard Constraint)", key="vb_lock_on", on_change=autosave)
        st.text_area(
            "Voice Lock Prompt",
            key="voice_lock_prompt",
            height=80,
            disabled=not st.session_state.vb_lock_on,
            on_change=autosave,
        )
        st.slider("Lock Strength", 0.0, 1.0, key="lock_intensity", disabled=not st.session_state.vb_lock_on, on_change=autosave)

    st.divider()

    st.selectbox("POV", ["First", "Close Third", "Omniscient"], key="pov", on_change=autosave)
    st.selectbox("Tense", ["Past", "Present"], key="tense", on_change=autosave)


# ============================================================
# SAFETY NET SAVE EVERY RERUN
# ============================================================
save_all_to_disk()
