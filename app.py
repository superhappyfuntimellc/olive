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
# CLOUD SYNC SUPPORT (S3-Compatible)
# ============================================================
def get_s3_credentials() -> Optional[Dict[str, str]]:
    """Get S3 credentials from secrets or environment."""
    try:
        # Try Streamlit secrets first
        if hasattr(st, "secrets"):
            aws_key = st.secrets.get("AWS_ACCESS_KEY_ID", "")
            aws_secret = st.secrets.get("AWS_SECRET_ACCESS_KEY", "")
            aws_bucket = st.secrets.get("AWS_S3_BUCKET", "")
            aws_region = st.secrets.get("AWS_REGION", "us-east-1")
            aws_endpoint = st.secrets.get("AWS_ENDPOINT_URL", "")  # For MinIO/custom
        else:
            aws_key = ""
            aws_secret = ""
            aws_bucket = ""
            aws_region = ""
            aws_endpoint = ""

        # Fall back to environment variables
        aws_key = aws_key or os.getenv("AWS_ACCESS_KEY_ID", "")
        aws_secret = aws_secret or os.getenv("AWS_SECRET_ACCESS_KEY", "")
        aws_bucket = aws_bucket or os.getenv("AWS_S3_BUCKET", "")
        aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")
        aws_endpoint = aws_endpoint or os.getenv("AWS_ENDPOINT_URL", "")

        if not (aws_key and aws_secret and aws_bucket):
            return None

        return {
            "aws_access_key_id": aws_key,
            "aws_secret_access_key": aws_secret,
            "bucket": aws_bucket,
            "region": aws_region,
            "endpoint_url": aws_endpoint if aws_endpoint else None,
        }
    except Exception:
        return None


def has_cloud_sync() -> bool:
    """Check if cloud sync is configured."""
    return get_s3_credentials() is not None


def list_cloud_saves() -> List[Dict[str, Any]]:
    """List available saves in cloud storage."""
    try:
        import boto3
        from botocore.exceptions import ClientError

        creds = get_s3_credentials()
        if not creds:
            return []

        # Create S3 client
        s3_config = {
            "aws_access_key_id": creds["aws_access_key_id"],
            "aws_secret_access_key": creds["aws_secret_access_key"],
            "region_name": creds["region"],
        }
        if creds["endpoint_url"]:
            s3_config["endpoint_url"] = creds["endpoint_url"]

        s3 = boto3.client("s3", **s3_config)

        # List objects with prefix
        response = s3.list_objects_v2(Bucket=creds["bucket"], Prefix="olivetti/")

        saves = []
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".json"):
                save_name = key.replace("olivetti/", "").replace(".json", "")
                saves.append(
                    {
                        "name": save_name,
                        "key": key,
                        "size": obj["Size"],
                        "modified": obj["LastModified"].strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

        # Sort by modified date (newest first)
        saves.sort(key=lambda x: x["modified"], reverse=True)
        return saves
    except ImportError:
        st.error("boto3 not installed. Run: pip install boto3")
        return []
    except Exception as e:
        st.error(f"Failed to list cloud saves: {e}")
        return []


def upload_to_cloud(save_name: str) -> bool:
    """Upload current state to cloud storage."""
    try:
        import boto3
        from botocore.exceptions import ClientError

        creds = get_s3_credentials()
        if not creds:
            st.error("Cloud sync not configured")
            return False

        # Create snapshot
        snapshot = {
            "saved_ts": now_ts(),
            "save_name": save_name,
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

        # Serialize to JSON
        json_data = json.dumps(snapshot, ensure_ascii=False, indent=2)

        # Upload to S3
        s3_config = {
            "aws_access_key_id": creds["aws_access_key_id"],
            "aws_secret_access_key": creds["aws_secret_access_key"],
            "region_name": creds["region"],
        }
        if creds["endpoint_url"]:
            s3_config["endpoint_url"] = creds["endpoint_url"]

        s3 = boto3.client("s3", **s3_config)

        key = f"olivetti/{save_name}.json"
        s3.put_object(
            Bucket=creds["bucket"],
            Key=key,
            Body=json_data.encode("utf-8"),
            ContentType="application/json",
        )

        return True
    except ImportError:
        st.error("boto3 not installed. Run: pip install boto3")
        return False
    except Exception as e:
        st.error(f"Failed to upload to cloud: {e}")
        return False


def download_from_cloud(save_key: str) -> bool:
    """Download state from cloud storage."""
    try:
        import boto3
        from botocore.exceptions import ClientError

        creds = get_s3_credentials()
        if not creds:
            st.error("Cloud sync not configured")
            return False

        # Download from S3
        s3_config = {
            "aws_access_key_id": creds["aws_access_key_id"],
            "aws_secret_access_key": creds["aws_secret_access_key"],
            "region_name": creds["region"],
        }
        if creds["endpoint_url"]:
            s3_config["endpoint_url"] = creds["endpoint_url"]

        s3 = boto3.client("s3", **s3_config)

        response = s3.get_object(
            Bucket=creds["bucket"],
            Key=save_key,
        )

        # Parse JSON
        json_data = response["Body"].read().decode("utf-8")
        snapshot = json.loads(json_data)

        # Load session state
        sess = snapshot.get("session", {}) or {}
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

        # Reset undo history after cloud load
        st.session_state._undo_history = [st.session_state.get("main_text", "")]
        st.session_state._redo_history = []
        st.session_state["_dirty"] = False

        return True
    except ImportError:
        st.error("boto3 not installed. Run: pip install boto3")
        return False
    except Exception as e:
        st.error(f"Failed to download from cloud: {e}")
        return False


def delete_from_cloud(save_key: str) -> bool:
    """Delete a save from cloud storage."""
    try:
        import boto3
        from botocore.exceptions import ClientError

        creds = get_s3_credentials()
        if not creds:
            st.error("Cloud sync not configured")
            return False

        s3_config = {
            "aws_access_key_id": creds["aws_access_key_id"],
            "aws_secret_access_key": creds["aws_secret_access_key"],
            "region_name": creds["region"],
        }
        if creds["endpoint_url"]:
            s3_config["endpoint_url"] = creds["endpoint_url"]

        s3 = boto3.client("s3", **s3_config)

        s3.delete_object(
            Bucket=creds["bucket"],
            Key=save_key,
        )

        return True
    except ImportError:
        st.error("boto3 not installed. Run: pip install boto3")
        return False
    except Exception as e:
        st.error(f"Failed to delete from cloud: {e}")
        return False


# ============================================================
# IMPORT / EXPORT SUPPORT
# ============================================================
def import_text_from_txt(uploaded_file) -> Optional[str]:
    """Import text from TXT file."""
    try:
        content = uploaded_file.read().decode("utf-8")
        return _normalize_text(content)
    except Exception as e:
        st.error(f"Failed to import TXT: {e}")
        return None


def import_text_from_md(uploaded_file) -> Optional[str]:
    """Import text from Markdown file."""
    try:
        content = uploaded_file.read().decode("utf-8")
        return _normalize_text(content)
    except Exception as e:
        st.error(f"Failed to import MD: {e}")
        return None


def import_text_from_docx(uploaded_file) -> Optional[str]:
    """Import text from DOCX file using python-docx."""
    try:
        # Import dynamically to avoid hard dependency
        import docx

        doc = docx.Document(uploaded_file)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        content = "\n\n".join(paragraphs)
        return _normalize_text(content)
    except ImportError:
        st.error("python-docx not installed. Run: pip install python-docx")
        return None
    except Exception as e:
        st.error(f"Failed to import DOCX: {e}")
        return None


def import_text_from_pdf(uploaded_file) -> Optional[str]:
    """Import text from PDF file using pypdf."""
    try:
        # Import dynamically to avoid hard dependency
        import pypdf

        reader = pypdf.PdfReader(uploaded_file)
        paragraphs = []
        for page in reader.pages:
            text = page.extract_text()
            if text.strip():
                paragraphs.append(text.strip())
        content = "\n\n".join(paragraphs)
        return _normalize_text(content)
    except ImportError:
        st.error("pypdf not installed. Run: pip install pypdf")
        return None
    except Exception as e:
        st.error(f"Failed to import PDF: {e}")
        return None


def export_to_txt(text: str, filename: str) -> bytes:
    """Export text to TXT format."""
    return text.encode("utf-8")


def export_to_md(text: str, filename: str) -> bytes:
    """Export text to Markdown format."""
    # Add a title if we have one
    title = st.session_state.get("workspace_title", "")
    if title:
        md_content = f"# {title}\n\n{text}"
    else:
        md_content = text
    return md_content.encode("utf-8")


def export_to_docx(text: str, filename: str) -> Optional[bytes]:
    """Export text to DOCX format using python-docx."""
    try:
        import docx
        from io import BytesIO

        doc = docx.Document()

        # Add title if available
        title = st.session_state.get("workspace_title", "")
        if title:
            doc.add_heading(title, level=0)

        # Add paragraphs
        paragraphs = _split_paragraphs(text)
        for para in paragraphs:
            doc.add_paragraph(para)

        # Save to bytes
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.read()
    except ImportError:
        st.error("python-docx not installed. Run: pip install python-docx")
        return None
    except Exception as e:
        st.error(f"Failed to export DOCX: {e}")
        return None


def export_to_pdf(text: str, filename: str) -> Optional[bytes]:
    """Export text to PDF format using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from io import BytesIO

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()
        story = []

        # Add title if available
        title = st.session_state.get("workspace_title", "")
        if title:
            title_style = styles["Title"]
            story.append(Paragraph(title, title_style))
            story.append(Spacer(1, 0.2 * inch))

        # Add paragraphs
        paragraphs = _split_paragraphs(text)
        body_style = styles["BodyText"]
        for para in paragraphs:
            # Escape XML special characters
            para_escaped = (
                para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
            story.append(Paragraph(para_escaped, body_style))
            story.append(Spacer(1, 0.15 * inch))

        doc.build(story)
        buffer.seek(0)
        return buffer.read()
    except ImportError:
        st.error("reportlab not installed. Run: pip install reportlab")
        return None
    except Exception as e:
        st.error(f"Failed to export PDF: {e}")
        return None


# ============================================================
# BAY TRANSFER & EXPORT
# ============================================================
def can_transfer_bay() -> bool:
    """Check if current bay can be transferred to next bay."""
    current_bay = st.session_state.get("active_bay", "NEW")
    # Can transfer from ROUGH to EDIT, or EDIT to FINAL
    return current_bay in ["ROUGH", "EDIT"]


def get_next_bay() -> Optional[str]:
    """Get the next bay in the workflow."""
    current_bay = st.session_state.get("active_bay", "NEW")
    if current_bay == "ROUGH":
        return "EDIT"
    elif current_bay == "EDIT":
        return "FINAL"
    return None


def transfer_to_next_bay() -> bool:
    """Transfer current draft to the next bay in the workflow."""
    if not can_transfer_bay():
        return False

    current_bay = st.session_state.get("active_bay", "NEW")
    next_bay = get_next_bay()

    if not next_bay:
        return False

    # Check if next bay already has content
    next_project = st.session_state.active_project_by_bay.get(next_bay)
    if next_project:
        # Next bay has content - need confirmation
        st.session_state._transfer_needs_confirmation = True
        st.session_state._transfer_target_bay = next_bay
        return False

    # Perform transfer
    _execute_bay_transfer(next_bay)
    return True


def _execute_bay_transfer(target_bay: str) -> None:
    """Execute the bay transfer (internal helper)."""
    current_bay = st.session_state.get("active_bay", "NEW")

    # Save current state to workspace
    save_workspace_from_session()

    # Create a copy of the current workspace for the target bay
    current_workspace = st.session_state.sb_workspace.copy()

    # Store in the target bay
    st.session_state.active_project_by_bay[target_bay] = current_workspace

    # Switch to the target bay
    st.session_state.active_bay = target_bay

    # Load the workspace (which is the same content we just transferred)
    load_workspace_into_session()

    mark_dirty()


def get_export_filename() -> str:
    """Generate export filename based on workspace title and bay."""
    title = st.session_state.get("workspace_title", "")
    bay = st.session_state.get("active_bay", "NEW")

    if title:
        base = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
        return f"{base}_{bay.lower()}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"olivetti_{bay.lower()}_{timestamp}"


# ============================================================
# BAY TRANSFER & EXPORT
# ============================================================
def can_transfer_bay() -> bool:
    """Check if current bay can be transferred to next bay."""
    current_bay = st.session_state.get("active_bay", "NEW")
    # Can transfer from ROUGH to EDIT, or EDIT to FINAL
    return current_bay in ["ROUGH", "EDIT"]


def get_next_bay() -> Optional[str]:
    """Get the next bay in the workflow."""
    current_bay = st.session_state.get("active_bay", "NEW")
    if current_bay == "ROUGH":
        return "EDIT"
    elif current_bay == "EDIT":
        return "FINAL"
    return None


def transfer_to_next_bay() -> bool:
    """Transfer current draft to the next bay in the workflow."""
    if not can_transfer_bay():
        return False

    current_bay = st.session_state.get("active_bay", "NEW")
    next_bay = get_next_bay()

    if not next_bay:
        return False

    # Check if next bay already has content
    next_project = st.session_state.active_project_by_bay.get(next_bay)
    if next_project:
        # Next bay has content - need confirmation
        st.session_state._transfer_needs_confirmation = True
        st.session_state._transfer_target_bay = next_bay
        return False

    # Perform transfer
    _execute_bay_transfer(next_bay)
    return True


def _execute_bay_transfer(target_bay: str) -> None:
    """Execute the bay transfer (internal helper)."""
    current_bay = st.session_state.get("active_bay", "NEW")

    # Save current state to workspace
    save_workspace_from_session()

    # Create a copy of the current workspace for the target bay
    current_workspace = st.session_state.sb_workspace.copy()

    # Store in the target bay
    st.session_state.active_project_by_bay[target_bay] = current_workspace

    # Switch to the target bay
    st.session_state.active_bay = target_bay

    # Load the workspace (which is the same content we just transferred)
    load_workspace_into_session()

    mark_dirty()


def get_export_filename() -> str:
    """Generate export filename based on workspace title and bay."""
    title = st.session_state.get("workspace_title", "")
    bay = st.session_state.get("active_bay", "NEW")

    if title:
        base = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
        return f"{base}_{bay.lower()}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"olivetti_{bay.lower()}_{timestamp}"


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
        "_confirm_import": False,
        "_show_cloud_upload": False,
        "_show_cloud_download": False,
        "_cloud_save_name": "",
        "_selected_cloud_save": None,
        "_transfer_needs_confirmation": False,
        "_transfer_target_bay": None,
        "_show_export_dialog": False,
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
    st.sidebar.markdown("**Import / Export**")

    # Import section
    st.sidebar.markdown("*Import text:*")
    uploaded_file = st.sidebar.file_uploader(
        "Choose file",
        type=["txt", "md", "docx", "pdf"],
        key="file_uploader",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()

        # Show confirmation dialog if current work exists
        current_text = st.session_state.get("main_text", "")
        if current_text and current_text.strip():
            if not st.session_state.get("_confirm_import"):
                st.sidebar.warning("⚠️ This will replace your current text!")
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    if st.button("✓ Import", key="confirm_import_btn"):
                        st.session_state._confirm_import = True
                        st.rerun()
                with col2:
                    if st.button("✗ Cancel", key="cancel_import_btn"):
                        st.session_state._confirm_import = False
                        st.rerun()
        else:
            st.session_state._confirm_import = True

        # Proceed with import if confirmed
        if st.session_state.get("_confirm_import"):
            imported_text = None

            if file_ext == "txt":
                imported_text = import_text_from_txt(uploaded_file)
            elif file_ext == "md":
                imported_text = import_text_from_md(uploaded_file)
            elif file_ext == "docx":
                imported_text = import_text_from_docx(uploaded_file)
            elif file_ext == "pdf":
                imported_text = import_text_from_pdf(uploaded_file)

            if imported_text:
                st.session_state.main_text = imported_text
                # Reset undo history after import
                st.session_state._undo_history = [imported_text]
                st.session_state._redo_history = []
                st.session_state._confirm_import = False
                mark_dirty()
                st.sidebar.success(f"✓ Imported {uploaded_file.name}")
                st.rerun()

    # Export section
    st.sidebar.markdown("*Export text:*")
    export_cols = st.sidebar.columns(4)

    current_text = st.session_state.get("main_text", "")
    export_disabled = not current_text or not current_text.strip()

    # Generate base filename
    title = st.session_state.get("workspace_title", "")
    base_filename = title if title else "olivetti_export"
    # Clean filename
    base_filename = re.sub(r"[^\w\s-]", "", base_filename).strip().replace(" ", "_")

    with export_cols[0]:
        if current_text and not export_disabled:
            txt_data = export_to_txt(current_text, base_filename)
            st.download_button(
                label="TXT",
                data=txt_data,
                file_name=f"{base_filename}.txt",
                mime="text/plain",
                disabled=export_disabled,
                key="export_txt",
            )
        else:
            st.button("TXT", disabled=True, key="export_txt_disabled")

    with export_cols[1]:
        if current_text and not export_disabled:
            md_data = export_to_md(current_text, base_filename)
            st.download_button(
                label="MD",
                data=md_data,
                file_name=f"{base_filename}.md",
                mime="text/markdown",
                disabled=export_disabled,
                key="export_md",
            )
        else:
            st.button("MD", disabled=True, key="export_md_disabled")

    with export_cols[2]:
        if current_text and not export_disabled:
            docx_data = export_to_docx(current_text, base_filename)
            if docx_data:
                st.download_button(
                    label="DOCX",
                    data=docx_data,
                    file_name=f"{base_filename}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    disabled=export_disabled,
                    key="export_docx",
                )
            else:
                st.button("DOCX", disabled=True, key="export_docx_disabled")
        else:
            st.button("DOCX", disabled=True, key="export_docx_disabled2")

    with export_cols[3]:
        if current_text and not export_disabled:
            pdf_data = export_to_pdf(current_text, base_filename)
            if pdf_data:
                st.download_button(
                    label="PDF",
                    data=pdf_data,
                    file_name=f"{base_filename}.pdf",
                    mime="application/pdf",
                    disabled=export_disabled,
                    key="export_pdf",
                )
            else:
                st.button("PDF", disabled=True, key="export_pdf_disabled")
        else:
            st.button("PDF", disabled=True, key="export_pdf_disabled2")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Cloud Sync (S3)**")

    # Check if cloud sync is configured
    cloud_configured = has_cloud_sync()

    if not cloud_configured:
        st.sidebar.info(
            "☁️ Cloud sync not configured\n\n"
            "Set environment variables or Streamlit secrets:\n"
            "• AWS_ACCESS_KEY_ID\n"
            "• AWS_SECRET_ACCESS_KEY\n"
            "• AWS_S3_BUCKET\n"
            "• AWS_REGION (optional)\n"
            "• AWS_ENDPOINT_URL (optional)"
        )
    else:
        # Cloud sync controls
        cloud_col1, cloud_col2 = st.sidebar.columns(2)

        with cloud_col1:
            if st.button("☁️ Upload", key="cloud_upload_btn"):
                st.session_state._show_cloud_upload = True

        with cloud_col2:
            if st.button("📥 Download", key="cloud_download_btn"):
                st.session_state._show_cloud_download = True

        # Upload dialog
        if st.session_state.get("_show_cloud_upload"):
            with st.sidebar.expander("☁️ Upload to Cloud", expanded=True):
                save_name = st.text_input(
                    "Save name",
                    value=st.session_state.get(
                        "workspace_title",
                        f"save_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    ),
                    key="cloud_save_name_input",
                )

                # Sanitize save name
                save_name_clean = (
                    re.sub(r"[^\w\s-]", "", save_name).strip().replace(" ", "_")
                )

                if save_name_clean:
                    st.caption(f"Will save as: {save_name_clean}.json")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(
                        "✓ Upload",
                        key="confirm_cloud_upload",
                        disabled=not save_name_clean,
                    ):
                        if upload_to_cloud(save_name_clean):
                            st.success(f"✓ Uploaded to cloud: {save_name_clean}")
                            st.session_state._show_cloud_upload = False
                            time.sleep(1)
                            st.rerun()
                with col2:
                    if st.button("✗ Cancel", key="cancel_cloud_upload"):
                        st.session_state._show_cloud_upload = False
                        st.rerun()

        # Download dialog
        if st.session_state.get("_show_cloud_download"):
            with st.sidebar.expander("📥 Download from Cloud", expanded=True):
                # List available saves
                cloud_saves = list_cloud_saves()

                if not cloud_saves:
                    st.info("No cloud saves found")
                else:
                    st.markdown("**Available saves:**")

                    # Display saves
                    for save in cloud_saves:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if st.button(
                                f"📄 {save['name']}",
                                key=f"select_{save['key']}",
                                help=f"Modified: {save['modified']}\nSize: {save['size']} bytes",
                            ):
                                st.session_state._selected_cloud_save = save
                        with col2:
                            if st.button(
                                "🗑️", key=f"delete_{save['key']}", help="Delete"
                            ):
                                if delete_from_cloud(save["key"]):
                                    st.success("Deleted")
                                    time.sleep(0.5)
                                    st.rerun()

                # Confirmation for selected save
                selected = st.session_state.get("_selected_cloud_save")
                if selected:
                    st.warning(
                        f"⚠️ Download '{selected['name']}'?\nThis will replace your current work!"
                    )
                    st.caption(f"Modified: {selected['modified']}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✓ Download", key="confirm_cloud_download"):
                            if download_from_cloud(selected["key"]):
                                st.success(f"✓ Downloaded: {selected['name']}")
                                st.session_state._selected_cloud_save = None
                                st.session_state._show_cloud_download = False
                                time.sleep(1)
                                st.rerun()
                    with col2:
                        if st.button("✗ Cancel", key="cancel_cloud_download_confirm"):
                            st.session_state._selected_cloud_save = None
                            st.rerun()

                st.markdown("---")
                if st.button("Close", key="close_cloud_download"):
                    st.session_state._show_cloud_download = False
                    st.session_state._selected_cloud_save = None
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

        # Bay indicator with visual styling
        current_bay = st.session_state.get("active_bay", "NEW")
        bay_colors = {
            "NEW": "#4A90E2",  # Blue
            "ROUGH": "#F5A623",  # Orange
            "EDIT": "#7ED321",  # Green
            "FINAL": "#BD10E0",  # Purple
        }
        bay_color = bay_colors.get(current_bay, "#888888")
        st.markdown(
            f"""
            <div style="
                background-color: {bay_color};
                color: white;
                padding: 8px 16px;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                📍 Current Bay: {current_bay}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Top bar: Undo/Redo + Bay Transfer + Export
        top_col1, top_col2, top_col3, top_col4 = st.columns([1, 1, 2, 2])

        # Undo/Redo controls
        with top_col1:
            if st.button("↶ Undo", disabled=not can_undo(), help="Undo (Ctrl+Z)"):
                undo_text()
                st.rerun()
        with top_col2:
            if st.button("↷ Redo", disabled=not can_redo(), help="Redo (Ctrl+Y)"):
                redo_text()
                st.rerun()

        # Bay Transfer button
        with top_col3:
            current_bay = st.session_state.get("active_bay", "NEW")
            next_bay = get_next_bay()
            if next_bay:
                transfer_label = f"{current_bay} → {next_bay}"
                if st.button(transfer_label, help=f"Move draft to {next_bay} bay"):
                    if not transfer_to_next_bay():
                        # Need confirmation - will show dialog below
                        st.rerun()
                    else:
                        st.success(f"✓ Moved to {next_bay} bay")
                        st.rerun()
            else:
                st.button("Transfer", disabled=True, help="No next bay available")

        # Export button
        with top_col4:
            has_text = bool(st.session_state.get("main_text", "").strip())
            if has_text:
                export_filename = get_export_filename()
                if st.button("📤 Export", help=f"Export as {export_filename}"):
                    st.session_state._show_export_dialog = True
            else:
                st.button("📤 Export", disabled=True, help="No text to export")

        # Transfer confirmation dialog
        if st.session_state.get("_transfer_needs_confirmation"):
            target_bay = st.session_state.get("_transfer_target_bay")
            with st.expander(f"⚠️ {target_bay} Bay Already Has Content", expanded=True):
                st.warning(
                    f"The {target_bay} bay already contains a draft. Transfer will overwrite it."
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✓ Overwrite", key="confirm_transfer"):
                        _execute_bay_transfer(target_bay)
                        st.session_state._transfer_needs_confirmation = False
                        st.session_state._transfer_target_bay = None
                        st.success(f"✓ Moved to {target_bay} bay")
                        st.rerun()
                with col2:
                    if st.button("✗ Cancel", key="cancel_transfer"):
                        st.session_state._transfer_needs_confirmation = False
                        st.session_state._transfer_target_bay = None
                        st.rerun()

        # Export dialog
        if st.session_state.get("_show_export_dialog"):
            with st.expander("📤 Export Draft", expanded=True):
                export_filename = get_export_filename()
                st.markdown(f"**Export as:** {export_filename}")
                st.caption(f"Current bay: {st.session_state.get('active_bay', 'NEW')}")

                # Export format selection
                export_format = st.radio(
                    "Format",
                    ["TXT", "MD", "DOCX", "PDF"],
                    horizontal=True,
                    key="export_format_choice",
                )

                # Generate export data
                current_text = st.session_state.get("main_text", "")
                export_data = None
                mime_type = "text/plain"
                file_ext = export_format.lower()

                if export_format == "TXT":
                    export_data = export_to_txt(current_text, export_filename)
                    mime_type = "text/plain"
                elif export_format == "MD":
                    export_data = export_to_md(current_text, export_filename)
                    mime_type = "text/markdown"
                elif export_format == "DOCX":
                    export_data = export_to_docx(current_text, export_filename)
                    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif export_format == "PDF":
                    export_data = export_to_pdf(current_text, export_filename)
                    mime_type = "application/pdf"

                col1, col2 = st.columns(2)
                with col1:
                    if export_data:
                        st.download_button(
                            label=f"⬇ Download {export_format}",
                            data=export_data,
                            file_name=f"{export_filename}.{file_ext}",
                            mime=mime_type,
                            key="export_download_btn",
                        )
                    else:
                        st.button(f"⬇ Download {export_format}", disabled=True)
                with col2:
                    if st.button("Close", key="close_export_dialog"):
                        st.session_state._show_export_dialog = False
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
