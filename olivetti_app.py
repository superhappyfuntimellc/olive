import os
import re
import math
import json
import hashlib
import sqlite3
import threading
import time
import tempfile
from io import BytesIO
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st
import streamlit.components.v1

# ============================================================
# OLIVETTI DESK — Complete Writing Application
# ============================================================

# ============================================================
# ENV / METADATA
# ============================================================
os.environ.setdefault("MS_APP_ID", "olivetti-writing-desk")
os.environ.setdefault("ms-appid", "olivetti-writing-desk")

DEFAULT_MODEL = "gpt-4o-mini"

def _get_openai_key_or_empty() -> str:
    try:
        return str(st.secrets.get("OPENAI_API_KEY", ""))
    except Exception:
        return ""

def _get_openai_model() -> str:
    try:
        return str(st.secrets.get("OPENAI_MODEL", DEFAULT_MODEL))
    except Exception:
        return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

OPENAI_MODEL = _get_openai_model()

def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY") or _get_openai_key_or_empty())

def require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY") or _get_openai_key_or_empty()
    if not key:
        st.error("OPENAI_API_KEY is not set. Add it as an environment variable or Streamlit secret.")
        st.stop()
    return key

# ============================================================
# CONSTANTS
# ============================================================
BAYS = ["NEW", "ROUGH", "EDIT", "FINAL"]
AUTOSAVE_DIR = os.path.join(os.path.dirname(__file__), "autosave")
DB_PATH = os.path.join(AUTOSAVE_DIR, "olivetti_projects.db")

# ============================================================
# UTILITIES
# ============================================================
def now_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"

def ensure_autosave_dir():
    os.makedirs(AUTOSAVE_DIR, exist_ok=True)

def _safe_filename(s: str) -> str:
    return re.sub(r'[^\w\-_\. ]', '_', s)

def _clamp_text(text: str, max_len: int) -> str:
    return text[:max_len] if len(text) > max_len else text

def temperature_from_intensity(intensity: float) -> float:
    return 0.3 + (intensity * 0.9)

def intensity_profile(ai_val: float) -> str:
    if ai_val <= 0.25:
        return "Conservative, factual, minimal creativity"
    elif ai_val <= 0.6:
        return "Balanced creativity and coherence"
    elif ai_val <= 0.85:
        return "High creativity, expressive"
    else:
        return "Maximum creativity, experimental"

def current_lane_from_draft(draft: str) -> str:
    word_count = len(draft.split())
    if word_count < 500:
        return "NEW"
    elif word_count < 2000:
        return "ROUGH"
    elif word_count < 5000:
        return "EDIT"
    else:
        return "FINAL"

# ============================================================
# SQLite PERSISTENCE
# ============================================================
def ensure_db():
    ensure_autosave_dir()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_ts TEXT,
        updated_ts TEXT,
        bay TEXT,
        draft TEXT,
        story_bible TEXT,
        voices TEXT,
        style_banks TEXT,
        settings TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_project_to_db(payload: Dict[str, Any]):
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Extract story_bible components
    story_bible = payload.get("story_bible", {})
    if not isinstance(story_bible, dict):
        story_bible = {"synopsis": "", "characters": "", "world": ""}
    
    # Extract settings
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        settings = {"ai_intensity": 0.75, "style_intensity": 0.6}
    
    cur.execute("""
    INSERT OR REPLACE INTO projects (id,title,created_ts,updated_ts,bay,draft,story_bible,voices,style_banks,settings)
    VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        payload.get("id"),
        payload.get("title"),
        payload.get("created_ts"),
        now_ts(),
        payload.get("bay"),
        payload.get("draft"),
        json.dumps(story_bible),
        json.dumps(payload.get("voices", {})),
        json.dumps(payload.get("style_banks", {})),
        json.dumps(settings)
    ))
    conn.commit()
    conn.close()

def list_projects_from_db() -> List[Dict[str,str]]:
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id,title,created_ts,updated_ts FROM projects ORDER BY updated_ts DESC")
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_ts": r[2], "updated_ts": r[3]} for r in rows]

def load_project_from_db(pid: str) -> Optional[Dict[str, Any]]:
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id,title,created_ts,updated_ts,bay,draft,story_bible,voices,style_banks,settings FROM projects WHERE id=?", (pid,))
    r = cur.fetchone()
    conn.close()
    if not r:
        return None
    return {
        "id": r[0],
        "title": r[1],
        "created_ts": r[2],
        "updated_ts": r[3],
        "bay": r[4],
        "draft": r[5],
        "story_bible": json.loads(r[6] or '{"synopsis":"","characters":"","world":""}'),
        "voices": json.loads(r[7] or "{}"),
        "style_banks": json.loads(r[8] or "{}"),
        "settings": json.loads(r[9] or '{"ai_intensity":0.75,"style_intensity":0.6}')
    }

def autosave_state():
    """Autosave current session state to database"""
    if st.session_state.get("project_id"):
        save_project_to_db({
            "id": st.session_state.get("project_id"),
            "title": st.session_state.get("project_title", "Untitled"),
            "created_ts": st.session_state.get("created_ts", now_ts()),
            "bay": st.session_state.get("active_bay", "NEW"),
            "draft": st.session_state.get("main_text", ""),
            "story_bible": {
                "synopsis": st.session_state.get("synopsis", ""),
                "characters": st.session_state.get("characters", ""),
                "world": st.session_state.get("world", "")
            },
            "voices": st.session_state.get("voices", {}),
            "style_banks": st.session_state.get("style_banks", {}),
            "settings": {
                "ai_intensity": st.session_state.get("ai_intensity", 0.75),
                "style_intensity": st.session_state.get("style_intensity", 0.6)
            }
        })

# ============================================================
# UNDO / REDO
# ============================================================
def push_undo():
    stack = st.session_state.get("_undo_stack", [])
    stack.append({
        "main_text": st.session_state.get("main_text", ""),
        "synopsis": st.session_state.get("synopsis", ""),
        "voice_sample": st.session_state.get("voice_sample", "")
    })
    st.session_state["_undo_stack"] = stack[-60:]
    st.session_state["_redo_stack"] = []

def do_undo():
    stack = st.session_state.get("_undo_stack", [])
    if not stack:
        return False
    state = stack.pop()
    rstack = st.session_state.get("_redo_stack", [])
    rstack.append({
        "main_text": st.session_state.get("main_text", ""),
        "synopsis": st.session_state.get("synopsis", ""),
        "voice_sample": st.session_state.get("voice_sample", "")
    })
    st.session_state["_redo_stack"] = rstack
    st.session_state["_undo_stack"] = stack
    st.session_state.main_text = state.get("main_text","")
    st.session_state.synopsis = state.get("synopsis","")
    st.session_state.voice_sample = state.get("voice_sample","")
    return True

def do_redo():
    rstack = st.session_state.get("_redo_stack", [])
    if not rstack:
        return False
    state = rstack.pop()
    ustack = st.session_state.get("_undo_stack", [])
    ustack.append({
        "main_text": st.session_state.get("main_text", ""),
        "synopsis": st.session_state.get("synopsis", ""),
        "voice_sample": st.session_state.get("voice_sample", "")
    })
    st.session_state["_undo_stack"] = ustack
    st.session_state["_redo_stack"] = rstack
    st.session_state.main_text = state.get("main_text","")
    st.session_state.synopsis = state.get("synopsis","")
    st.session_state.voice_sample = state.get("voice_sample","")
    return True

# ============================================================
# OPENAI INTEGRATION
# ============================================================
def engine_style_directive(style_name: str, style_intensity: float, lane: str) -> str:
    """Generate style directive based on style settings"""
    base_styles = {
        "Neutral": "Write in a clear, balanced style without strong stylistic flourishes.",
        "Crisp": "Write with short, punchy sentences. Be direct and economical with words.",
        "Flowing": "Write with longer, flowing sentences that create rhythm and immersion."
    }
    
    base = base_styles.get(style_name, base_styles["Neutral"])
    intensity_note = ""
    if style_intensity < 0.3:
        intensity_note = " Keep stylistic elements subtle."
    elif style_intensity > 0.7:
        intensity_note = " Apply the style strongly and consistently."
    
    lane_note = f" This is the {lane} stage of writing."
    return base + intensity_note + lane_note

def compose_prompt(action: str, draft: str, story_bible: dict, style_name: str, style_intensity: float, exemplars: List[str], lane: str) -> str:
    """Compose a comprehensive prompt for the AI model"""
    sb_short = (story_bible.get("synopsis","") or "")[:1200]
    style_directive = engine_style_directive(style_name, style_intensity, lane)
    exemplar_block = "\n\n".join(f"EXAMPLE: {e}" for e in exemplars) if exemplars else ""
    prompt = (
        f"System: You are a professional writing assistant.\n"
        f"Story Bible Summary:\n{sb_short}\n\n"
        f"Style Directive:\n{style_directive}\n\n"
        f"{exemplar_block}\n\n"
        f"Draft (last 1200 chars):\n{draft[-1200:]}\n\n"
        f"Instruction: {action}. Preserve canon and lane. Return only the edited text."
    )
    return prompt

def call_openai_real(prompt: str, max_tokens: int = 300) -> str:
    try:
        import openai
    except Exception:
        return "openai package not installed. pip install openai to enable model features."
    key = require_openai_key()
    openai.api_key = key
    model = OPENAI_MODEL
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"system","content":"You are a helpful creative writing assistant."},
                      {"role":"user","content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature_from_intensity(float(st.session_state.get("ai_intensity",0.75)))
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Model call failed: {str(e)}"

def perform_action(action: str):
    push_undo()
    st.session_state.pending_action = action
    st.session_state.last_action = action
    draft = st.session_state.get("main_text","") or ""
    
    # Get context
    story_bible = st.session_state.get("story_bible", {
        "synopsis": st.session_state.get("synopsis", ""),
        "characters": st.session_state.get("characters", ""),
        "world": st.session_state.get("world", "")
    })
    style_name = st.session_state.get("writing_style", "Neutral")
    style_intensity = float(st.session_state.get("style_intensity", 0.6))
    lane = current_lane_from_draft(draft)
    
    # Get exemplars (placeholder - would load from style_banks)
    exemplars = []
    
    # Compose comprehensive prompt
    prompt = compose_prompt(action, draft, story_bible, style_name, style_intensity, exemplars, lane)
    
    if has_openai_key():
        st.session_state.tool_output = "Calling model..."
        out = call_openai_real(prompt, max_tokens=400)
        if action == "WRITE":
            st.session_state.main_text = draft + "\n\n" + out
        elif action == "EXPAND":
            st.session_state.main_text = draft + "\n\n" + out
        elif action == "REWRITE":
            st.session_state.main_text = out
        else:
            st.session_state.tool_output = out
        st.session_state.tool_output = f"{action} completed at {now_ts()}"
    else:
        placeholder = f"[{action} placeholder generated at {now_ts()}]"
        if action in ("WRITE","EXPAND"):
            st.session_state.main_text = draft + "\n\n" + placeholder
        elif action == "REWRITE":
            st.session_state.main_text = placeholder
        st.session_state.tool_output = f"{action} placeholder applied"
    
    autosave_state()

# ============================================================
# KEYBOARD SHORTCUTS
# ============================================================
SHORTCUT_JS = """
<script>
document.addEventListener('keydown', function(e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    const btn = [...document.querySelectorAll('button')].find(b => b.innerText && b.innerText.trim() === 'WRITE');
    if (btn) { btn.click(); e.preventDefault(); }
  }
  if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key === 'z') {
    const btn = [...document.querySelectorAll('button')].find(b => b.innerText && b.innerText.trim() === 'UNDO');
    if (btn) { btn.click(); e.preventDefault(); }
  }
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'Z') {
    const btn = [...document.querySelectorAll('button')].find(b => b.innerText && b.innerText.trim() === 'REDO');
    if (btn) { btn.click(); e.preventDefault(); }
  }
});
</script>
"""

# ============================================================
# VOICE HELPERS
# ============================================================
def voice_names_for_selector() -> List[str]:
    return ["— None —", "Neutral", "Crisp", "Flowing"]

def default_story_bible_workspace():
    return {"story_bible": {}, "story_bible_binding": {"locked": False}}

def load_workspace_into_session():
    st.session_state.sb_workspace = default_story_bible_workspace()

def add_style_samples(style: str, context: str, text: str) -> int:
    return 1  # placeholder

# ============================================================
# UI COMPONENTS
# ============================================================
def render_top_bays():
    """Top bay selector row: NEW / ROUGH / EDIT / FINAL"""
    cols = st.columns([1,1,1,1])
    for i, bay in enumerate(BAYS):
        is_active = (st.session_state.get("active_bay") == bay)
        btn_label = bay
        if cols[i].button(btn_label, key=f"bay_{bay}"):
            st.session_state.active_bay = bay
            if bay == "NEW" and st.session_state.get("project_id") is None:
                load_workspace_into_session()
            st.rerun()
        if is_active:
            cols[i].markdown(f"<div style='text-align:center;color:#caa86a;font-weight:600'>{bay} (active)</div>", unsafe_allow_html=True)

def render_story_bible_panel():
    """Left column: Story Bible controls and workspace"""
    st.subheader("Story Bible")
    projects = list_projects_from_db()
    project_list = ["— Workspace —"] + [p["title"] for p in projects]
    sel = st.selectbox("Current Project", project_list, index=0)
    
    lock_col1, lock_col2 = st.columns([1,1])
    if lock_col1.button("Story Bible Hard Lock"):
        sb = st.session_state.get("sb_workspace") or default_story_bible_workspace()
        sb_binding = sb.get("story_bible_binding", {})
        sb_binding["locked"] = not sb_binding.get("locked", True)
        sb_binding["locked_ts"] = now_ts()
        sb["story_bible_binding"] = sb_binding
        st.session_state.sb_workspace = sb
        st.success("Toggled story bible lock")
    
    st.markdown("**AI Intensity (Master Control)**")
    ai_val = st.slider("AI Intensity", min_value=0.0, max_value=1.0, 
                       value=float(st.session_state.get("ai_intensity", 0.75)), 
                       step=0.01, key="ai_intensity")
    st.markdown(f"*Mode:* **{('LOW','MED','HIGH','MAX')[0 if ai_val<=0.25 else 1 if ai_val<=0.6 else 2 if ai_val<=0.85 else 3]}**")
    st.markdown(f"*Profile:* {intensity_profile(ai_val)}")
    
    with st.expander("Import / Export", expanded=False):
        st.button("Import Story Bible")
        st.button("Export Story Bible")
    with st.expander("Junk Drawer", expanded=False):
        st.text_area("Junk", value=st.session_state.get("junk",""), key="junk")
    with st.expander("Synopsis", expanded=False):
        st.text_area("Synopsis (quick view)", value=st.session_state.get("synopsis",""), height=120, key="synopsis_preview")

def render_writing_desk():
    """Center column: main editor and active controls"""
    st.subheader("Writing Desk")
    ac1, ac2, ac3, ac4 = st.columns([1,1,1,1])
    
    st.session_state.canon_guardian = ac1.checkbox("Canon Guardian", 
                                                    value=st.session_state.get("canon_guardian", True))
    st.session_state.voice_heatmap = ac2.checkbox("Voice Heatmap", 
                                                   value=st.session_state.get("voice_heatmap", False))
    
    if ac3.button("CHECK"):
        st.session_state.last_action = "CHECK"
        st.session_state.tool_output = "CHECK performed (placeholder)"
    if ac4.button("ANALYZE"):
        st.session_state.last_action = "ANALYZE"
        st.session_state.tool_output = "ANALYZE performed (placeholder)"
    
    st.text_area("Main text", value=st.session_state.get("main_text",""), height=520, key="main_text")
    
    lane = current_lane_from_draft(st.session_state.get("main_text",""))
    st.markdown(f"**Detected lane:** {lane}  •  **Project:** {st.session_state.get('project_title','—')}  •  **AI intensity:** {st.session_state.get('ai_intensity',0.75)}")

def render_voice_bible_panel():
    """Right column: voice/style controls and trainer"""
    st.subheader("Voice Bible")
    st.checkbox("Enable Writing Style", value=st.session_state.get("vb_style_on", True), key="vb_style_on")
    st.selectbox("Writing Style", options=["Neutral","Crisp","Flowing"], index=0, key="writing_style")
    st.markdown("**Style: Balanced**")
    st.slider("Style Intensity", min_value=0.0, max_value=1.0, 
              value=float(st.session_state.get("style_intensity", 0.6)), 
              step=0.01, key="style_intensity")
    
    st.checkbox("Genre Intelligence", value=st.session_state.get("vb_genre_on", True), key="vb_genre_on")
    st.checkbox("Trained Voice (Vector Matching)", value=st.session_state.get("vb_match_on", False), key="vb_match_on")
    st.selectbox("Trained Voice", options=voice_names_for_selector(), index=0, key="trained_voice")
    
    with st.expander("Style Trainer (Adaptive Learning)", expanded=False):
        st.text_area("Paste sample text to train", value="", key="style_trainer_input", height=120)
        if st.button("Add to Style Bank"):
            added = add_style_samples(st.session_state.get("writing_style","Neutral"), 
                                     "Narration", 
                                     st.session_state.get("style_trainer_input",""))
            st.success(f"Added {added} sample(s) to style bank")

def render_bottom_toolbar_upgraded():
    st.markdown("---")
    st.markdown("### Tools")
    tools = ["WRITE","REWRITE","EXPAND","REPHRASE","DESCRIBE","SPELL","GRAMMAR","FIND","SYNONYM","SENTENCE"]
    cols = st.columns(len(tools) + 3)
    
    if cols[0].button("UNDO"):
        ok = do_undo()
        if ok:
            st.success("Undid last change")
        else:
            st.info("Nothing to undo")
    if cols[1].button("REDO"):
        ok = do_redo()
        if ok:
            st.success("Redid change")
        else:
            st.info("Nothing to redo")
    if cols[2].button("EXPORT"):
        txt = st.session_state.get("main_text","")
        b = txt.encode("utf-8")
        st.download_button("Download draft as TXT", data=b, 
                          file_name=f"{_safe_filename(st.session_state.get('project_title','draft'))}.txt", 
                          mime="text/plain")
    
    for i, t in enumerate(tools):
        if cols[i+3].button(t):
            perform_action(t)
            st.rerun()
    
    st.components.v1.html(SHORTCUT_JS, height=0)
    st.text_area("Tool output", value=st.session_state.get("tool_output",""), height=140, key="tool_output_display")

# ============================================================
# MAIN UI
# ============================================================
def main_ui():
    """Assembles the full UI"""
    render_top_bays()
    
    left_col, center_col, right_col = st.columns([1.1, 2.4, 1.1])
    with left_col:
        render_story_bible_panel()
    with center_col:
        render_writing_desk()
    with right_col:
        render_voice_bible_panel()
    
    render_bottom_toolbar_upgraded()

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    st.set_page_config(page_title="Olivetti Desk", layout="wide", initial_sidebar_state="collapsed")
    
    # Initialize session state
    if "active_bay" not in st.session_state:
        st.session_state.active_bay = "NEW"
    if "main_text" not in st.session_state:
        st.session_state.main_text = ""
    if "ai_intensity" not in st.session_state:
        st.session_state.ai_intensity = 0.75
    if "_undo_stack" not in st.session_state:
        st.session_state["_undo_stack"] = []
    if "_redo_stack" not in st.session_state:
        st.session_state["_redo_stack"] = []
    
    main_ui()
