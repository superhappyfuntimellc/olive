# Olivetti Desk

A single-file Streamlit writing application with AI-powered creative assistance.

## 🚀 Quick Start

**Single-file distribution** - Just one Python file to run:

```bash
streamlit run app.py
```

## 📋 Requirements

- Python 3.11+
- Dependencies (install once):
  ```bash
  pip install -r requirements.txt
  ```

## 📦 What's Included

- **app.py** - Complete standalone application (935 lines)
- **requirements.txt** - Runtime dependencies only
- **pyproject.toml** - Black formatter configuration
- **requirements-dev.txt** - Development tools (optional)

## ✨ Features

- **Story Bible Workspace** - Track synopsis, characters, world-building, outline
- **Voice Vault** - Train and apply custom writing voices per lane
- **Lane Detection** - Auto-detect Dialogue, Narration, Interiority, Action
- **AI Intensity Controls** - Adjust creativity from conservative to maximum
- **Writing Actions** - WRITE (continue), REWRITE (improve), EXPAND (detail)
- **Autosave** - Automatic state persistence with throttling
- **Session Management** - Workspace persistence across sessions

## 🔧 Configuration

Create `.streamlit/secrets.toml` for OpenAI API access:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4.1-mini"  # optional
```

## 📖 Architecture

**Single-file design** - All functionality in `app.py`:
- No auxiliary Python modules required
- All imports from standard library + Streamlit
- Self-contained with no external dependencies beyond `requirements.txt`
- Can be deployed by copying just `app.py` + installing requirements

## 🛠️ Development

```bash
# Install dev dependencies (Black formatter)
pip install -r requirements-dev.txt

# Format code (automatic on commit via pre-commit hook)
black app.py
```

## 📄 License

MIT
