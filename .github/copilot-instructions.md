# Olivetti AI Writing Partner - Copilot Instructions

## Architecture Overview

Olivetti is a **single-file Streamlit app** (app.py, ~2960 lines) for AI-assisted fiction writing. All code lives in one file by design for "paste+click" deployment.

### Core Components (in order)
| Section | Lines | Key Functions |
|---------|-------|---------------|
| Config/Styling | 1-200 | CSS theming, `DEFAULT_MODEL`, API key helpers |
| Vector/Voice Vault | 240-400 | `_hash_vec()`, `retrieve_exemplars()` - lightweight semantic search |
| Project Model | 525-590 | `new_project_payload()` - Story Bible, Voice Bible, locks structure |
| Style Banks | 590-850 | `retrieve_style_exemplars()` - trainable per-lane writing styles |
| Session State | 858-950 | `init_state()` - all state in `st.session_state` |
| AI Brief + Call | 1630-1740 | `build_partner_brief()`, `call_openai()` |
| Actions | 1926-2080 | `partner_action()` - Write/Rewrite/Expand handlers |
| UI Layout | 2175-end | Three-column layout, tabs, action buttons |

### Data Flow
```
Notes (scratch_pad) → AI Action → Paste (scratch_pad_2) → User review → main_text
Projects dict → autosave/olivetti_state.json
```

## Critical Patterns

### Streamlit Widget State (MUST follow)
Widgets with `key=` own their state. **Never directly assign to widget-bound state after render.**
```python
# WRONG - breaks after widget renders
st.session_state.main_text = "new"
st.rerun()

# RIGHT - use pending pattern, processed BEFORE widgets at ~line 930
st.session_state.pending_load_project = pid
st.rerun()
```

### Action Queue Pattern
Actions that modify state must be queued, not executed inline:
```python
# In UI: queue_action("MyAction")
# Processed by run_pending_action() at line 2175 (before widget render)
# Handler in partner_action() at line 1926
```

### AI Integration
All AI flows through unified brief:
```python
brief = build_partner_brief(action, lane)  # Assembles Story Bible + Voice Bible + exemplars
result = call_openai(brief, task, text)     # Uses temperature_from_intensity(ai_intensity)
```

### Bay System
Four stages: `NEW → ROUGH → EDIT → FINAL` (defined in `BAYS` constant line 199)
- `active_project_by_bay` - dict mapping bay to project ID
- `list_projects_in_bay(bay)` - returns `[(pid, title), ...]`
- `switch_bay()`, `promote_project()` handle transitions

## Development Commands
```bash
streamlit run app.py              # Run app (port 8501)
lsof -ti:8501 | xargs kill -9     # Kill stuck process
```

### API Key Config
```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o-mini"  # optional, defaults to gpt-4o-mini
```

## Common Modifications

### Adding a New Action Button
1. Add button in UI (~line 2760): `if st.button("MyAction"): queue_action("MyAction")`
2. Handle in `partner_action()` (~line 1926):
   ```python
   if action == "MyAction":
       task = "Your AI task instruction"
       out = call_openai(brief, task, text)
       apply_replace(out)  # or apply_append(out)
       return
   ```

### Adding Session State Fields
Add to `defaults` dict in `init_state()` (~line 860). Include in `new_project_payload()` if project-specific.

### Modifying AI Behavior
- System prompt: modify `build_partner_brief()` return string
- Temperature: adjust `temperature_from_intensity()` mapping
- Model: set `OPENAI_MODEL` in secrets or env

## Files
(Plain text file references used for GitHub Copilot compatibility)
- app.py - Entire application (single file by design)
- autosave/olivetti_state.json - Persisted state
- .streamlit/secrets.toml - API keys (gitignored)
- OLIVETTI_FEATURES_MANUAL.md - User documentation
