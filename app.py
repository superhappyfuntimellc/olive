# app.py — Olivetti Desk (one file, paste+click)
import os
import re
import math
import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st

# ============================================================
# ENV / METADATA HYGIENE
# ============================================================
os.environ.setdefault("MS_APP_ID", "olivetti-writing-desk")
os.environ.setdefault("ms-appid", "olivetti-writing-desk")

DEFAULT_MODEL = "gpt-4.1-mini"


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
    key = os.getenv("OPENAI_API_KEY") or _get_openai_key_or_empty()
    if not key:
        st.error(
            "OPENAI_API_KEY is not set. Add it as an environment variable (OPENAI_API_KEY) "
            "or as a Streamlit secret."
        )
        st.stop()
    return key


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Olivetti Desk", layout="wide", initial_sidebar_state="expanded"
)

# Keyboard shortcut handler for undo/redo
KEYBOARD_SHORTCUTS = """
<script>
document.addEventListener('keydown', function(e) {
    // Ctrl+Z or Cmd+Z for Undo
    if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        const undoButton = window.parent.document.querySelector('[data-testid="baseButton-secondary"]:not([disabled])');
        if (undoButton && undoButton.textContent.includes('Undo')) {
            undoButton.click();
        }
    }
    // Ctrl+Y or Cmd+Shift+Z for Redo
    if (((e.ctrlKey || e.metaKey) && e.key === 'y') || ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'z')) {
        e.preventDefault();
        const buttons = window.parent.document.querySelectorAll('[data-testid="baseButton-secondary"]:not([disabled])');
        for (let btn of buttons) {
            if (btn.textContent.includes('Redo')) {
                btn.click();
                break;
            }
        }
    }
});
</script>
"""
st.markdown(KEYBOARD_SHORTCUTS, unsafe_allow_html=True)

# ============================================================
# GLOBALS
# ============================================================
LANES = ["Dialogue", "Narration", "Interiority", "Action"]
BAYS = ["NEW", "ROUGH", "EDIT", "FINAL"]
ENGINE_STYLES = ["NARRATIVE", "DESCRIPTIVE", "EMOTIONAL", "LYRICAL"]
AUTOSAVE_DIR = "autosave"
AUTOSAVE_PATH = os.path.join(AUTOSAVE_DIR, "olivetti_state.json")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB guardrail
WORD_RE = re.compile(r"[A-Za-z']+")
AUTOSAVE_MIN_INTERVAL_S = 12.0  # throttle rerun autosaves


# ============================================================
# UTILS
# ============================================================
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now_unix() -> float:
    return time.time()


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


def mark_dirty() -> None:
    st.session_state["_dirty"] = True


# ============================================================
# VECTOR / VOICE VAULT
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
                lanes_out[ln].append(
                    {"ts": s.get("ts") or now_ts(), "text": txt, "vec": _hash_vec(txt)}
                )
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
    names = list((st.session_state.get("voices") or {}).keys())
    names = [n for n in names if n and n.strip()]
    return ["— None —"] + sorted(set(names), key=lambda x: x.lower())


def add_voice_sample(voice_name: str, lane: str, text: str) -> bool:
    voice_name = (voice_name or "").strip()
    lane = lane if lane in LANES else "Narration"
    text = _normalize_text(text)
    if not voice_name or voice_name == "— None —":
        return False
    if not text:
        return False
    if voice_name not in st.session_state.voices:
        st.session_state.voices[voice_name] = {
            "created_ts": now_ts(),
            "lanes": {ln: [] for ln in LANES},
        }
    st.session_state.voices[voice_name]["lanes"][lane].append(
        {"ts": now_ts(), "text": text, "vec": _hash_vec(text)}
    )
    mark_dirty()
    return True


# ============================================================
# STYLE BANKS
# ============================================================
def default_style_banks() -> Dict[str, Any]:
    ts = now_ts()
    return {
        s: {"created_ts": ts, "lanes": {ln: [] for ln in LANES}} for s in ENGINE_STYLES
    }


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
                cleaned.append(
                    {"ts": it.get("ts") or now_ts(), "text": _clamp_text(t, 9000)}
                )
            c_lanes[ln] = cleaned
        out[style] = {"created_ts": b.get("created_ts") or now_ts(), "lanes": c_lanes}
    return out


# ============================================================
# LANE DETECTION
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
    quote_count = p.count('"') + p.count('"') + p.count('"')
    has_dialogue_punct = p.startswith(("—", "- ", '"', '"'))
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
    scores = {
        "Dialogue": dialogue_score,
        "Interiority": interior_score,
        "Action": action_score,
        "Narration": 0.25,
    }
    lane = max(scores.items(), key=lambda kv: kv[1])[0]
    return "Narration" if scores[lane] < 0.9 else lane


def current_lane_from_draft(text: str) -> str:
    paras = _split_paragraphs(text)
    if not paras:
        return "Narration"
    return detect_lane(paras[-1])


# ============================================================
# INTENSITY
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
# STORY BIBLE WORKSPACE
# ============================================================
def default_story_bible_workspace() -> Dict[str, Any]:
    ts = now_ts()
    return {
        "workspace_story_bible_id": hashlib.md5(
            f"wsb|{ts}".encode("utf-8")
        ).hexdigest()[:12],
        "workspace_story_bible_created_ts": ts,
        "title": "",
        "draft": "",
        "story_bible": {
            "synopsis": "",
            "genre_style_notes": "",
            "world": "",
            "characters": "",
            "outline": "",
        },
        "voice_sample": "",
        "ai_intensity": 0.75,
        "voices": default_voice_vault(),
        "style_banks": default_style_banks(),
    }


def in_workspace_mode() -> bool:
    return (st.session_state.active_bay == "NEW") and (
        st.session_state.project_id is None
    )


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
    w["style_banks"] = compact_style_banks(
        st.session_state.get("style_banks")
        or rebuild_vectors_in_style_banks(default_style_banks())
    )
    st.session_state.sb_workspace = w


def load_workspace_into_session() -> None:
    w = st.session_state.sb_workspace or default_story_bible_workspace()
    sb = w.get("story_bible", {}) or {}
    st.session_state.project_id = None
    st.session_state.project_title = "—"
    st.session_state.main_text = w.get("draft", "") or ""
    # Initialize undo history with loaded text
    st.session_state._undo_history = [st.session_state.main_text]
    st.session_state._redo_history = []
    st.session_state.synopsis = sb.get("synopsis", "") or ""
    st.session_state.genre_style_notes = sb.get("genre_style_notes", "") or ""
    st.session_state.world = sb.get("world", "") or ""
    st.session_state.characters = sb.get("characters", "") or ""
    st.session_state.outline = sb.get("outline", "") or ""
    st.session_state.voice_sample = w.get("voice_sample", "") or ""
    st.session_state.ai_intensity = float(w.get("ai_intensity", 0.75))
    st.session_state.voices = rebuild_vectors_in_voice_vault(
        w.get("voices", default_voice_vault())
    )
    st.session_state.style_banks = rebuild_vectors_in_style_banks(
        w.get("style_banks", default_style_banks())
    )
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
    mark_dirty()


# ============================================================
# UNDO / REDO SUPPORT
# ============================================================
def push_undo_history(text: str) -> None:
    """Push current text to undo history stack."""
    history = st.session_state.get("_undo_history", [])
    # Avoid duplicate consecutive entries
    if not history or history[-1] != text:
        history.append(text)
        # Keep history bounded to 50 states
        if len(history) > 50:
            history.pop(0)
        st.session_state._undo_history = history
        # Clear redo stack when new change is made
        st.session_state._redo_history = []


def undo_text() -> None:
    """Undo last text change."""
    undo_history = st.session_state.get("_undo_history", [])
    if len(undo_history) > 1:
        # Pop current state and save to redo
        current = undo_history.pop()
        redo_history = st.session_state.get("_redo_history", [])
        redo_history.append(current)
        st.session_state._redo_history = redo_history
        # Restore previous state
        st.session_state.main_text = undo_history[-1]
        st.session_state._undo_history = undo_history
        mark_dirty()


def redo_text() -> None:
    """Redo previously undone text change."""
    redo_history = st.session_state.get("_redo_history", [])
    if redo_history:
        # Pop from redo and restore
        text = redo_history.pop()
        undo_history = st.session_state.get("_undo_history", [])
        undo_history.append(text)
        st.session_state._undo_history = undo_history
        st.session_state._redo_history = redo_history
        st.session_state.main_text = text
        mark_dirty()


def can_undo() -> bool:
    """Check if undo is available."""
    return len(st.session_state.get("_undo_history", [])) > 1


def can_redo() -> bool:
    """Check if redo is available."""
    return len(st.session_state.get("_redo_history", [])) > 0


def on_main_text_change() -> None:
    """Callback for main text changes to track history."""
    push_undo_history(st.session_state.main_text)
    mark_dirty()


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
        "_autosave_unix": 0.0,
        "_dirty": False,
        "last_action": "—",
        "main_text": "",
        "synopsis": "",
        "genre_style_notes": "",
        "world": "",
        "characters": "",
        "outline": "",
        "tool_output": "",
        "vb_style_on": True,
        "writing_style": "Neutral",
        "ai_intensity": 0.75,
        "voices": rebuild_vectors_in_voice_vault(default_voice_vault()),
        "style_banks": rebuild_vectors_in_style_banks(default_style_banks()),
        "story_bible_lock": True,
        "_undo_history": [""],
        "_redo_history": [],
        "_autosave_checked": False,
        "_show_recovery_dialog": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# AUTOSAVE / PERSISTENCE
# ============================================================
def ensure_autosave_dir() -> None:
    if not os.path.exists(AUTOSAVE_DIR):
        os.makedirs(AUTOSAVE_DIR, exist_ok=True)


def autosave_state() -> None:
    ensure_autosave_dir()
    snapshot = {
        "saved_ts": now_ts(),
        "session": {
            k: (
                compact_voice_vault(v)
                if k == "voices"
                else (
                    compact_style_banks(v)
                    if k == "style_banks"
                    else st.session_state.get(k)
                )
            )
            for k, v in st.session_state.items()
            if k not in ("_rerun_count",)
        },
    }
    with open(AUTOSAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    st.session_state.autosave_time = snapshot["saved_ts"]
    st.session_state["_autosave_unix"] = now_unix()
    st.session_state["_dirty"] = False


def maybe_autosave_throttled() -> None:
    if not st.session_state.get("_dirty"):
        return
    last = float(st.session_state.get("_autosave_unix", 0.0) or 0.0)
    if now_unix() - last < AUTOSAVE_MIN_INTERVAL_S:
        return
    autosave_state()


def get_autosave_info() -> Optional[Dict[str, Any]]:
    """Get autosave details without loading it."""
    if not os.path.exists(AUTOSAVE_PATH):
        return None
    try:
        with open(AUTOSAVE_PATH, "r", encoding="utf-8") as f:
            snap = json.load(f)
        sess = snap.get("session", {}) or {}
        main_text = sess.get("main_text", "")
        word_count = len(WORD_RE.findall(main_text)) if main_text else 0
        return {
            "saved_ts": snap.get("saved_ts", "Unknown"),
            "word_count": word_count,
            "preview": main_text[:100] + "..." if len(main_text) > 100 else main_text,
            "has_content": bool(main_text and main_text.strip()),
        }
    except Exception:
        return None


def load_autosave() -> bool:
    if not os.path.exists(AUTOSAVE_PATH):
        return False
    try:
        with open(AUTOSAVE_PATH, "r", encoding="utf-8") as f:
            snap = json.load(f)
        sess = snap.get("session", {}) or {}
        for k in (
            "voices",
            "style_banks",
            "sb_workspace",
            "main_text",
            "synopsis",
            "genre_style_notes",
            "world",
            "characters",
            "outline",
            "ai_intensity",
            "workspace_title",
            "story_bible_lock",
        ):
            if k not in sess:
                continue
            if k == "voices":
                st.session_state.voices = rebuild_vectors_in_voice_vault(
                    sess.get("voices", default_voice_vault())
                )
            elif k == "style_banks":
                st.session_state.style_banks = rebuild_vectors_in_style_banks(
                    sess.get("style_banks", default_style_banks())
                )
            else:
                st.session_state[k] = sess.get(k)
        st.session_state.autosave_time = snap.get("saved_ts", now_ts())
        st.session_state["_autosave_unix"] = now_unix()
        st.session_state["_dirty"] = False
        # Reset undo history after load
        st.session_state._undo_history = [st.session_state.get("main_text", "")]
        st.session_state._redo_history = []
        return True
    except Exception:
        return False


# ============================================================
# OPENAI CALLS (optional, safe fallback)
# ============================================================
def _openai_system_prompt(action: str, lane: str) -> str:
    sb = _normalize_text(st.session_state.synopsis)
    notes = _normalize_text(st.session_state.genre_style_notes)
    world = _normalize_text(st.session_state.world)
    chars = _normalize_text(st.session_state.characters)
    outline = _normalize_text(st.session_state.outline)
    style_on = bool(st.session_state.vb_style_on)
    style = st.session_state.writing_style if style_on else "Neutral"
    intensity = float(st.session_state.ai_intensity)
    profile = intensity_profile(intensity)
    prompt_text = f"""You are Olivetti Desk, a writing assistant.

Task: {action}
Lane focus: {lane}
Intensity: {profile}
Writing style: {style}

Rules:
- Preserve continuity and canon.
- Respect the lane (Dialogue/Narration/Interiority/Action).
- Output only the rewritten/continued prose, no commentary.

Story bible:
Synopsis: {sb}
Genre/style notes: {notes}
World: {world}
Characters: {chars}
Outline: {outline}
"""
    return _normalize_text(prompt_text.strip())


def call_openai_text(user_prompt: str, system_prompt: str, temperature: float) -> str:
    if not has_openai_key():
        return "OpenAI key not configured. Set OPENAI_API_KEY to enable model features."
    key = require_openai_key()
    # Try the modern "OpenAI-compatible" Responses API over HTTPS (no extra deps besides httpx).
    try:
        import httpx  # type: ignore

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": OPENAI_MODEL,
            "temperature": float(temperature),
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
        }
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                "https://api.openai.com/v1/responses", headers=headers, json=payload
            )
            r.raise_for_status()
            data = r.json()
            # Extract text from response output
            out_texts: List[str] = []
            for item in data.get("output") or []:
                for c in item.get("content") or []:
                    if c.get("type") == "output_text":
                        out_texts.append(c.get("text", ""))
            text = _normalize_text("\n".join(out_texts))
            return text or "(No text returned.)"
    except Exception as e:
        return f"Model call failed. (Details: {e})"


def do_action(action: str) -> None:
    draft = st.session_state.main_text or ""
    lane = current_lane_from_draft(draft)
    system_prompt = _openai_system_prompt(action=action, lane=lane)
    temperature = temperature_from_intensity(float(st.session_state.ai_intensity))
    if action == "WRITE":
        user_prompt = f"Continue this draft naturally:\n\n{draft}".strip()
    elif action == "REWRITE":
        user_prompt = f"Rewrite this to improve clarity, rhythm, and voice while preserving meaning:\n\n{draft}".strip()
    elif action == "EXPAND":
        user_prompt = f"Expand this with richer specificity and sensory detail while preserving intent:\n\n{draft}".strip()
    else:
        user_prompt = draft
    st.session_state.last_action = action
    st.session_state.tool_output = call_openai_text(
        user_prompt=user_prompt, system_prompt=system_prompt, temperature=temperature
    )
    mark_dirty()


# ============================================================
# UI
# ============================================================
def main_ui() -> None:
    st.sidebar.title("Olivetti Desk")
    st.sidebar.markdown("**Workspace**")
    st.sidebar.selectbox(
        "Active Bay",
        BAYS,
        index=BAYS.index(st.session_state.active_bay),
        key="active_bay",
        on_change=mark_dirty,
    )

    # Autosave controls with recovery dialog
    autosave_info = get_autosave_info()

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("Recover autosave", disabled=autosave_info is None):
            st.session_state._show_recovery_dialog = True
    with c2:
        if st.button("Save now"):
            autosave_state()
            st.sidebar.success("Saved")

    # Show autosave status
    if autosave_info:
        st.sidebar.caption(
            f"💾 Last autosave: {autosave_info['saved_ts']} "
            f"({autosave_info['word_count']} words)"
        )
    else:
        st.sidebar.caption("💾 No autosave available")

    # Recovery dialog
    if st.session_state.get("_show_recovery_dialog") and autosave_info:
        with st.sidebar.expander("⚠️ Autosave Recovery", expanded=True):
            st.warning("Loading autosave will replace your current work!")
            st.markdown("**Autosave details:**")
            st.text(f"Saved: {autosave_info['saved_ts']}")
            st.text(f"Words: {autosave_info['word_count']}")
            if autosave_info["preview"]:
                st.text("Preview:")
                st.code(autosave_info["preview"], language=None)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✓ Load it", key="confirm_recovery"):
                    ok = load_autosave()
                    st.session_state._show_recovery_dialog = False
                    if ok:
                        st.rerun()
            with col2:
                if st.button("✗ Cancel", key="cancel_recovery"):
                    st.session_state._show_recovery_dialog = False
                    st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Voice controls**")
    st.sidebar.checkbox(
        "Enable writing style",
        value=st.session_state.vb_style_on,
        key="vb_style_on",
        on_change=mark_dirty,
    )
    st.sidebar.selectbox(
        "Writing style",
        ["Neutral", "Crisp", "Flowing"],
        index=0,
        key="writing_style",
        on_change=mark_dirty,
    )
    st.sidebar.slider(
        "AI intensity",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.ai_intensity),
        step=0.01,
        key="ai_intensity",
        on_change=mark_dirty,
    )

    st.sidebar.markdown("---")
    st.sidebar.checkbox(
        "Lock story bible editing",
        value=bool(st.session_state.story_bible_lock),
        key="story_bible_lock",
        on_change=mark_dirty,
    )

    # Top bar
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(st.session_state.get("project_title", "Olivetti Desk"))
    with col2:
        if st.button("Autosave now"):
            autosave_state()
            st.success("Autosaved")

    # Main layout
    left, center, right = st.columns([1.2, 2.4, 1.2])

    with left:
        st.subheader("Story bible")
        st.text_input(
            "Workspace title",
            value=st.session_state.workspace_title,
            key="workspace_title",
            on_change=mark_dirty,
        )
        locked = bool(st.session_state.story_bible_lock)
        st.text_area(
            "Synopsis",
            value=st.session_state.synopsis,
            height=110,
            key="synopsis",
            disabled=locked,
            on_change=mark_dirty,
        )
        st.text_area(
            "Genre/style notes",
            value=st.session_state.genre_style_notes,
            height=110,
            key="genre_style_notes",
            disabled=locked,
            on_change=mark_dirty,
        )
        st.text_area(
            "World",
            value=st.session_state.world,
            height=110,
            key="world",
            disabled=locked,
            on_change=mark_dirty,
        )
        st.text_area(
            "Characters",
            value=st.session_state.characters,
            height=110,
            key="characters",
            disabled=locked,
            on_change=mark_dirty,
        )
        st.text_area(
            "Outline",
            value=st.session_state.outline,
            height=110,
            key="outline",
            disabled=locked,
            on_change=mark_dirty,
        )
        a, b = st.columns(2)
        with a:
            if st.button("Reset workspace"):
                reset_workspace_story_bible()
                st.rerun()
        with b:
            if st.button("Save workspace"):
                save_workspace_from_session()
                autosave_state()
                st.success("Workspace saved")

    with center:
        st.subheader("Writing desk")

        # Undo/Redo controls
        undo_col1, undo_col2, undo_col3 = st.columns([1, 1, 4])
        with undo_col1:
            if st.button("↶ Undo", disabled=not can_undo(), help="Undo (Ctrl+Z)"):
                undo_text()
                st.rerun()
        with undo_col2:
            if st.button("↷ Redo", disabled=not can_redo(), help="Redo (Ctrl+Y)"):
                redo_text()
                st.rerun()

        st.text_area(
            "Main text",
            value=st.session_state.main_text,
            height=440,
            key="main_text",
            on_change=on_main_text_change,
        )
        st.markdown("**Actions**")
        action_col1, action_col2, action_col3 = st.columns(3)
        if action_col1.button("WRITE"):
            do_action("WRITE")
        if action_col2.button("REWRITE"):
            do_action("REWRITE")
        if action_col3.button("EXPAND"):
            do_action("EXPAND")
        st.markdown("**Tool output**")
        st.text_area(
            "Tool output",
            value=st.session_state.tool_output,
            height=160,
            key="tool_output_display",
        )

    with right:
        st.subheader("Voice bible")
        # Voice selection
        names = voice_names_for_selector()
        idx = 0
        if st.session_state.get("trained_voice") in names:
            idx = names.index(st.session_state.get("trained_voice"))
        st.selectbox(
            "Trained voice",
            options=names,
            index=idx,
            key="trained_voice",
            on_change=mark_dirty,
        )
        st.selectbox(
            "Lane",
            options=LANES,
            index=LANES.index("Narration"),
            key="voice_lane",
            on_change=mark_dirty,
        )
        st.text_area(
            "Voice sample",
            value=st.session_state.voice_sample,
            height=140,
            key="voice_sample",
            on_change=mark_dirty,
        )
        if st.button("Add voice sample"):
            ok = add_voice_sample(
                st.session_state.trained_voice,
                st.session_state.voice_lane,
                st.session_state.voice_sample,
            )
            if ok:
                st.success("Sample added")
            else:
                st.warning("Pick a voice (not None) and add non-empty text.")
        st.markdown("---")
        st.caption(
            f"Detected lane: {current_lane_from_draft(st.session_state.main_text)}"
        )
        st.markdown("---")
        st.write(
            f"Autosave: {st.session_state.get('autosave_time', 'never')} • "
            f"Dirty: {bool(st.session_state.get('_dirty'))} • "
            f"Last action: {st.session_state.get('last_action', '—')}"
        )


# ============================================================
# STARTUP
# ============================================================
def main() -> None:
    init_state()

    # Automatic recovery prompt on first run
    if not st.session_state.get("_autosave_checked"):
        st.session_state._autosave_checked = True
        autosave_info = get_autosave_info()
        # Only prompt if autosave exists and has content
        if autosave_info and autosave_info.get("has_content"):
            # Check if current session is empty (fresh start)
            current_text = st.session_state.get("main_text", "")
            if not current_text or not current_text.strip():
                # Auto-load on fresh start
                if load_autosave():
                    st.toast("✓ Recovered autosave from " + autosave_info["saved_ts"])
            else:
                # Show dialog if there's existing work
                st.session_state._show_recovery_dialog = True

    # Keep workspace mirror consistent
    save_workspace_from_session()
    # Throttled background autosave for edits
    maybe_autosave_throttled()
    main_ui()


if __name__ == "__main__":
    main()
