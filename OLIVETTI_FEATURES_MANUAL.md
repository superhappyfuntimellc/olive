# OLIVETTI CREATIVE EDITING PARTNER
## Complete Features & Functionality Manual
**Current Build Status** | December 26, 2025

---

## 🚀 LOCAL SETUP INSTRUCTIONS

### Quick Start (5 minutes)
1. **Install Dependencies**
   ```bash
   cd /workspaces/superhappyfuntimellc
   pip install -r requirements.txt
   ```

2. **Configure API Key**
   - Your OpenAI API key is stored in: `streamlit/.secrets.toml`
   - Current format (✅ Fixed):
     ```toml
     OPENAI_API_KEY = "sk-proj-..."
     ```

3. **Launch the App**
   ```bash
   cd /workspaces/superhappyfuntimellc
   streamlit run app.py
   ```
   - App runs on: `http://localhost:8501`
   - Access via your Codespace forwarded port

4. **Stop/Restart**
   ```bash
   # Kill existing process on port 8501
   lsof -ti:8501 | xargs kill -9
   
   # Restart
   streamlit run app.py
   ```

---

## 🏗️ SYSTEM ARCHITECTURE

### The Spiderweb (Unified Intelligence)
Every component feeds into a single AI engine with trainable controls:

```
USER INPUTS (Training)
├── Style Banks → Per-lane writing style samples
├── Voice Vault → Voice training samples (semantic vectors)
├── Story Bible → Canon/worldbuilding enforcement
└── Voice Bible → Real-time AI modulation controls

↓ INTELLIGENT RETRIEVAL ↓
├── retrieve_style_exemplars() → Context-aware semantic search
├── retrieve_mixed_exemplars() → Voice vault vector matching
├── _story_bible_text() → Canon assembly
└── engine_style_directive() → Style guidance generation

↓ UNIFIED AI BRIEF ↓
build_partner_brief(action, lane):
✓ AI Intensity → temperature conversion
✓ Writing Style → engine directives + trained exemplars
✓ Genre Influence → mood/pacing directives
✓ Trained Voice → semantic retrieval of user samples
✓ Match My Style → one-shot adaptation
✓ Voice Lock → hard constraints (MANDATORY)
✓ POV/Tense → technical specs
✓ Story Bible → canon enforcement
✓ Lane Detection → context-aware mode (Dialogue/Narration/Action/Interiority)

↓ UNIFIED AI GENERATION ↓
call_openai(brief, task, text):
→ OpenAI API with temperature from AI Intensity
→ System prompt = full Voice Bible brief
→ User prompt = action-specific task
→ Returns professional-grade prose

↓ ALL ACTIONS USE SAME ENGINE ↓
✓ Write/Rewrite/Expand/etc → Writing desk actions
✓ Generate Story Bible sections → Story Bible generation
✓ Import document parsing → sb_breakdown_ai()
✓ Canon checking → analyze_canon_conformity()
✓ Voice analysis → analyze_voice_conformity()
```

**Key Principle**: More training = Better adaptation. The system learns from your samples.

---

## 📂 PROJECT BAY SYSTEM

### The Four Bays
Your workspace is organized into four development stages:

1. **🆕 NEW** (Story Bible Workspace)
   - Pre-project ideation space
   - Build Story Bible first
   - Train Voice Bible settings
   - No project file yet (just workspace state)
   - **Use**: `/create: [Title]` to spawn project

2. **✏️ ROUGH** (First Draft Bay)
   - Active writing/generation
   - Heavy AI assistance
   - Story Bible locked to project
   - Voice Bible fully active

3. **🛠 EDIT** (Revision Bay)
   - Polish and refine
   - Structural changes
   - Promote from ROUGH when draft complete

4. **✅ FINAL** (Publication Ready)
   - Last pass for submission
   - Clean export formats
   - Promote from EDIT when polished

### Bay Navigation
- Click bay buttons at top: `🆕 New` `✏️ Rough` `🛠 Edit` `✅ Final`
- Each bay remembers its last active project
- Switching bays auto-saves current work

### Project Commands (Junk Drawer)
```
/create: My Novel Title    # Create new project in NEW bay
/promote                    # Move current project to next bay (NEW→ROUGH→EDIT→FINAL)
/find: searchterm          # Search across all Story Bible sections + draft
```

---

## ✍️ THE WRITING DESK

### Bottom Action Bar (The Core Engine)
All actions respect Voice Bible settings and Story Bible canon:

**Content Generation**
- **Write** — Continue draft with 1-3 new paragraphs (pulls from Story Bible specifics)
- **Expand** — Add depth/detail to current text without changing meaning
- **Describe** — Add vivid sensory description while preserving pace

**Editing Actions**
- **Rewrite** — Improve quality while preserving meaning/canon
- **Rephrase** — Replace final sentence with stronger alternative
- **Spell/Grammar** — Copyedit spelling/grammar/punctuation

**Tool Outputs** (Display in Tool Output panel, don't modify draft)
- **Synonym** — 12 strong alternatives for last word (grouped by nuance)
- **Sentence** — 8 rewrites of final sentence (varied rhythm/diction)

### Lane Detection (Automatic Context Awareness)
The AI automatically detects which type of prose you're writing:

- **Narration** — Default descriptive/narrative prose
- **Dialogue** — Quoted speech, conversational patterns
- **Interiority** — Internal thoughts, character psychology
- **Action** — Physical movement, kinetic sequences

**How it works**: System analyzes final paragraph and adjusts AI brief accordingly.

---

## 📖 STORY BIBLE (The Canon Engine)

### Five Core Sections
All sections feed into every AI action to ensure continuity:

1. **Synopsis** — Core conflict, characters, stakes (2-3 paragraphs)
2. **Genre/Style Notes** — Tone, voice, stylistic markers
3. **World** — Setting, rules, atmosphere, time period, systems
4. **Characters** — Names, roles, relationships, motivations, traits
5. **Outline** — Acts, beats, key scenes, turning points

### Story Bible Features

**AI Generation** (Uses Voice Bible)
- Click "✨ Generate" in each section
- Pulls from existing context (draft + other sections)
- Respects all Voice Bible controls
- Incremental build: generates one section at a time

**Import Documents**
- Upload `.txt`, `.md`, `.docx` files
- AI automatically breaks down into sections
- Choose merge mode:
  - **Replace** — Overwrite existing content
  - **Append** — Add to existing content
  - **Heuristic** — Smart pattern matching (no AI needed)

**Canon Enforcement** (Alpha Feature)
- Enable "Canon Guardian" toggle
- Scans draft for continuity violations
- Flags issues with confidence scores:
  - Character trait contradictions (eye color, etc.)
  - Dead character mentions
  - Timeline contradictions
  - Technology level conflicts
- Resolution options: Update Bible / Fix Draft / Ignore

**Export Story Bible**
- Download as formatted Markdown
- Includes metadata and all sections
- Shareable reference document

---

## 🎙️ VOICE BIBLE (The AI Control Center)

### The Six Control Systems
Think of these as mixing board sliders for AI behavior:

#### 1. AI INTENSITY (Master Control)
**Location**: Sidebar → Voice Bible → Top slider  
**Range**: 0.0 (LOW) → 1.0 (MAX)

**What it does**: 
- Directly controls OpenAI temperature (0.15 + intensity × 0.95)
- Affects creativity, risk-taking, and stylistic boldness
- **LOW (≤0.25)**: Conservative, literal, minimal invention
- **MED (0.26-0.60)**: Balanced creativity, controlled invention
- **HIGH (0.61-0.85)**: Bolder choices, richer specificity
- **MAX (0.86-1.0)**: Aggressive originality, maximum voice

**Use cases**:
- Brainstorming? → HIGH/MAX
- Canon-sensitive scenes? → LOW/MED
- Final polish? → MED

#### 2. STYLE ENGINE (Trainable Writing Styles)
**Toggle**: `[x] Style Engine`  
**Styles**: Neutral / Narrative / Descriptive / Emotional / Lyrical / Sparse / Ornate

**How it works**:
1. Select a style (e.g., "LYRICAL")
2. Add training samples in Style Banks (sidebar)
3. AI semantically retrieves best exemplars during generation
4. More samples = Better adaptation

**Style Intensity Slider**: Controls how strongly style is applied (0.0-1.0)

**Style Banks Training**:
```
1. Navigate to: Sidebar → "Style Banks (Trainable)"
2. Select style: LYRICAL (or any other)
3. Select lane: Narration/Dialogue/Action/Interiority
4. Paste sample text
5. Choose split mode: Paragraphs / Whole Block
6. Click "Add Samples"
7. Repeat with more samples (up to 250 per lane)
```

**Lane-Specific Training**: Each style can have different samples per lane (Dialogue vs Narration)

#### 3. GENRE INTELLIGENCE
**Toggle**: `[x] Genre Intelligence`  
**Genres**: Literary / Thriller / Noir / Horror / Romance / Fantasy / Sci-Fi / Historical / Contemporary

**What it does**:
- Applies genre-specific mood directives
- Adjusts pacing and tension patterns
- Influences vocabulary selection

**Genre Intensity Slider**: Controls strength of genre influence

**Example**: Noir + HIGH intensity = hardboiled tone, shadowy atmosphere, terse dialogue

#### 4. TRAINED VOICE (Your Personal Style)
**Toggle**: `[x] Trained Voice`  
**Voices**: Voice A / Voice B / Custom voices

**How it works**:
1. Create or select a voice
2. Add voice samples (your own writing or target author)
3. System builds semantic vectors
4. AI retrieves similar patterns during generation
5. Adapts to your unique rhythm/syntax/diction

**Voice Vault Training**:
```
1. Sidebar → "Voice Vault (Your Voices)"
2. Click "Create Custom Voice" (optional)
3. Select voice (Voice A/B/custom)
4. Select lane: Narration/Dialogue/Action/Interiority
5. Paste sample of YOUR writing
6. Click "Add Sample"
7. Add 5-20 samples for best results
8. System auto-builds semantic vectors
```

**Vector Matching**: Uses cosine similarity on hash vectors for fast retrieval

**Trained Intensity Slider**: Controls how strongly trained voice is applied

#### 5. MATCH MY STYLE (One-Shot Adaptation)
**Toggle**: `[x] Match My Style`  
**Input**: Paste sample text in "Voice Sample" field

**What it does**:
- Instantly adapts to provided sample (no training needed)
- One-shot style transfer
- Good for quick experiments or guest passages

**Match Intensity Slider**: Controls adaptation strength

**Use case**: Paste a paragraph from target author → AI mimics that style

#### 6. VOICE LOCK (Hard Constraints)
**Toggle**: `[x] Voice Lock`  
**Input**: Write strict rules in "Voice Lock Prompt"

**What it does**:
- **MANDATORY enforcement** of rules
- Highest priority in AI brief
- No exceptions allowed

**Lock Intensity Slider**: Should stay at 1.0 for true enforcement

**Example rules**:
```
NEVER use adverbs ending in -ly
NEVER use passive voice (was/were/been)
NEVER use abstract nouns (love, hate, fear)
ALWAYS use active verbs
ALWAYS ground in sensory detail
```

**Use case**: Technical constraints (screenplay format, style guide rules, house style)

#### 7. TECHNICAL CONTROLS
**Toggle**: `[x] Technical Controls`  
**Settings**: 
- **POV**: First / Close Third / Omniscient
- **Tense**: Past / Present

**What it does**:
- Enforces narrative perspective
- Maintains consistent verb tense
- Applied to ALL generated content

**Technical Intensity Slider**: Controls enforcement strictness

---

## 🗂️ VOICE BIBLE STATUS DISPLAY

**Location**: Top bar, right side  
**Example**: `AI:HIGH • Style:LYRICAL • Genre:Noir • Voice:Voice A • Tech:Close Third/Past`

**What you see**:
- Active AI Intensity level
- Enabled Voice Bible controls
- Current settings at a glance
- Updates after every action

**After each action**:
```
Status shows: "Write complete (AI:HIGH • Style:LYRICAL • Genre:Literary)"
```
This confirms Voice Bible was applied.

---

## 📊 ANALYSIS TOOLS (Alpha Features)

### Style Sample Analysis
**Location**: Sidebar → Style Banks → "🔍 Analyze Text"

**What it does**:
- Scores text for writing strength
- Metrics:
  - Vocabulary richness (unique word ratio)
  - Sensory language density
  - Sentence length optimization
  - Action verb count
  - Thought/interiority depth
- Returns top 10 strongest samples
- Use to identify best training material

### Voice Conformity Analysis
**What it does**:
- Scores each paragraph (0-100) for Voice Bible conformity
- Identifies specific violations:
  - POV mismatches
  - Tense shifts
  - Style deviations (missing sensory language for LYRICAL, etc.)
  - Genre element gaps
  - Voice Lock violations (adverbs when banned, etc.)
- Higher score = better conformity

**Use case**: Quality check after AI generation

### Canon Conformity (Canon Guardian)
**Toggle**: Enable in Story Bible section  
**What it scans**:
- Character trait contradictions
- Dead character continuity errors
- Timeline/day inconsistencies
- World-building conflicts (tech level, etc.)

**Output**:
- Issue type, severity (error/warning)
- Confidence score (60-85%)
- Paragraph location
- Resolution options: Update Bible / Fix Draft / Ignore

**Ignored flags**: System remembers ignored issues

---

## 💾 DATA PERSISTENCE

### Auto-Save System
- **Frequency**: After every action
- **Location**: `autosave/olivetti_state.json`
- **Backup**: `autosave/olivetti_state.json.bak`
- **What's saved**:
  - All projects (all bays)
  - Story Bible workspace
  - Voice Vault vectors
  - Style Banks vectors
  - Active bay state
  - Last active project per bay

**Manual save**: Auto-triggers on bay switch, project creation, action completion

### Project Data Model
Each project contains:
```json
{
  "id": "unique_hash",
  "title": "Project Title",
  "bay": "ROUGH",
  "draft": "full draft text",
  "story_bible": { /* 5 sections */ },
  "voice_bible": { /* all control settings */ },
  "voices": { /* Voice Vault samples */ },
  "style_banks": { /* Style Bank samples */ },
  "story_bible_id": "unique_hash",
  "story_bible_binding": { /* relationship lock */ },
  "locks": { /* edit permissions */ }
}
```

---

## 📤 IMPORT / EXPORT

### Project Export/Import
**Location**: Sidebar → "Import/Export"

**Export Single Project**:
1. Select project from dropdown
2. Click "📦 Export Project"
3. Downloads: `olivetti_project_[Title]_[timestamp].json`

**Export Entire Library**:
1. Click "📚 Export Library"
2. Downloads: `olivetti_library_[timestamp].json`
3. Includes ALL projects + workspace

**Import Project**:
1. Upload `.json` project bundle
2. Choose target bay (NEW/ROUGH/EDIT/FINAL)
3. Optionally rename
4. Click "Import"

**Import Library**:
1. Upload library bundle
2. Merges into current workspace
3. Workspace preserved if empty

### Draft Export Formats

**Markdown**:
- Click "📄 Export Draft (Markdown)"
- Includes metadata header
- Clean formatting
- Ready for version control

**Manuscript Standard**:
- Click "📄 Export Manuscript"
- Industry-standard formatting:
  - Title page with word count
  - Proper indentation
  - Chapter breaks
  - Page headers
- Ready for submission

**HTML (eBook)**:
- Click "📄 Export eBook HTML"
- Semantic HTML5 markup
- Embedded CSS styling
- Chapter detection
- Ready for Calibre/Pandoc conversion

**DOCX**:
- Requires `python-docx` library
- Preserves paragraph formatting
- Compatible with MS Word

---

## 🔧 TECHNICAL DETAILS

### Dependencies
```
streamlit>=1.28.0
openai>=1.0.0
python-docx>=1.0.0  # Optional, for DOCX support
```

### File Structure
```
superhappyfuntimellc/
├── app.py                      # Main application (4183 lines)
├── requirements.txt            # Python dependencies
├── streamlit/.secrets.toml     # API key storage (✅ Fixed format)
├── autosave/
│   ├── olivetti_state.json     # Primary save file
│   ├── olivetti_state.json.bak # Backup
│   └── olivetti_state.json.bak[1-3]  # Historical backups
└── OLIVETTI_FEATURES_MANUAL.md # This file
```

### Environment Variables
```bash
# Primary method: Streamlit secrets
# File: streamlit/.secrets.toml
OPENAI_API_KEY = "sk-proj-..."
OPENAI_MODEL = "gpt-4.1-mini"  # Optional, defaults to gpt-4.1-mini

# Alternative: Shell environment
export OPENAI_API_KEY="sk-proj-..."
export OPENAI_MODEL="gpt-4.1-mini"
```

### Vector Storage (No External Dependencies)
- **Hash vectors**: MD5-based bag-of-words (512 dimensions)
- **Similarity**: Cosine distance
- **Storage**: JSON-serialized in project state
- **No embedding API needed**: Fully local/offline capable

### OpenAI API Usage
- **Model**: gpt-4.1-mini (configurable)
- **Temperature**: Dynamic (AI Intensity → 0.15 to 1.1)
- **Timeout**: 60 seconds
- **Error handling**: Quota detection, graceful fallback

---

## 🎯 CURRENT BUILD STATUS

### ✅ Fully Implemented (Production Stable)
1. **Project Bay System** — 4-bay workflow (NEW/ROUGH/EDIT/FINAL)
2. **Story Bible** — 5 sections with AI generation + import
3. **Voice Bible** — 6 control systems with real-time modulation
4. **Writing Desk** — 9 actions (Write/Rewrite/Expand/Rephrase/Describe/Spell/Grammar/Synonym/Sentence)
5. **Style Banks** — Trainable per-lane style exemplars with semantic retrieval
6. **Voice Vault** — Custom voice training with vector storage
7. **Lane Detection** — Automatic context awareness (4 lanes)
8. **Unified AI Brief** — All controls feed into single generation engine
9. **Auto-Save** — Atomic writes with backup rotation
10. **Import/Export** — Projects, library, drafts (Markdown/Manuscript/HTML/DOCX)
11. **Canon Guardian** — Continuity checking (alpha)
12. **Voice Conformity** — Quality scoring (alpha)
13. **Style Analysis** — Sample strength scoring (alpha)

### 🔄 In Progress / Alpha
1. **Canon Guardian** — Needs refinement for false positives
2. **Voice Conformity** — Scoring algorithm improvements
3. **DOCX Export** — Requires optional dependency

### 📋 Known Issues
1. **UI** — Some layout quirks (user noted "a little funky")
2. **Story Bible Lock** — Edit lock system partially implemented
3. **Mobile Responsiveness** — Desktop-optimized only

### 🎨 Design Philosophy
- **Olivetti Quaderno Aesthetic** — Vintage typewriter meets modern LCD
- **Cream/Bronze/Charcoal Palette** — Professional, not garish
- **Paper-Texture Editing** — Mimics typewriter experience
- **Monospace Headers** — IBM Plex Mono for technical feel
- **Serif Body** — Libre Baskerville for elegance

---

## 🚨 TROUBLESHOOTING

### App Won't Start
```bash
# Check if port 8501 is in use
lsof -ti:8501

# Kill existing process
lsof -ti:8501 | xargs kill -9

# Restart
streamlit run app.py
```

### OpenAI API Errors
1. **"insufficient_quota"** → Check billing at platform.openai.com
2. **"OPENAI_API_KEY not set"** → Verify `streamlit/.secrets.toml` format
3. **"Rate limit"** → Wait 60s or reduce AI Intensity

### Lost Work
1. Check `autosave/olivetti_state.json.bak` (backup)
2. Check historical backups: `.bak1`, `.bak2`, `.bak3`
3. Manual recovery: Load backup JSON in Python, extract projects

### Slow Performance
1. **Style Banks bloated?** → Clear unused lanes (keeps newest 250)
2. **Voice Vault large?** → Cap is 60 samples per lane (auto-trimmed)
3. **Draft too long?** → System clamps context to 12K chars for AI calls

---

## 💡 WORKFLOW EXAMPLES

### Example 1: Start a New Novel
```
1. Launch app → You're in NEW bay (Story Bible workspace)
2. Fill Story Bible sections:
   - Synopsis: Write or generate with AI
   - Characters: List main cast
   - World: Define setting
3. Set Voice Bible:
   - AI Intensity: 0.75 (HIGH)
   - Style Engine: LYRICAL (enable + add samples)
   - Genre: Literary
   - Trained Voice: Voice A (add your writing samples)
4. Junk Drawer: /create: My Novel Title
5. Start writing in draft area
6. Click "Write" to generate with AI
7. System pulls from Story Bible specifics automatically
8. When draft chapter complete: /promote (moves to ROUGH)
```

### Example 2: Train Your Voice
```
1. Sidebar → Voice Vault
2. Select "Voice A" (or create custom)
3. For Narration lane:
   - Paste 5-10 strong paragraphs of YOUR prose
   - Click "Add Sample" each time
4. For Dialogue lane:
   - Paste 5-10 dialogue exchanges
   - Click "Add Sample"
5. Enable in Voice Bible: [x] Trained Voice
6. Set intensity: 0.7
7. Write or generate → AI adapts to your patterns
```

### Example 3: Match Author Style
```
1. Find passage from target author (100-300 words)
2. Sidebar → Voice Bible → Match My Style section
3. Paste passage in "Voice Sample" field
4. Enable: [x] Match My Style
5. Set intensity: 0.8
6. Generate new content → AI mimics that style
```

### Example 4: Enforce Style Rules
```
1. Sidebar → Voice Bible → Voice Lock
2. Enable: [x] Voice Lock
3. Enter rules:
   NEVER use adverbs (-ly words)
   NEVER use "very" or "really"
   ALWAYS use active voice
   ALWAYS ground in sensory detail
4. Set intensity: 1.0 (maximum enforcement)
5. All generated content obeys these rules
```

---

## 🎓 BEST PRACTICES

### Training Recommendations
1. **Style Banks**: 20-50 samples per lane minimum (250 max)
2. **Voice Vault**: 10-20 samples per lane for good adaptation
3. **Variety**: Mix paragraph lengths and contexts
4. **Quality**: Use your strongest prose as training data
5. **Analysis**: Run "🔍 Analyze Text" to find best samples

### AI Intensity Guidelines
- **First draft**: 0.7-0.9 (HIGH/MAX) for creativity
- **Editing pass**: 0.5-0.7 (MED/HIGH) for control
- **Final polish**: 0.4-0.6 (MED) for consistency
- **Canon-sensitive**: 0.3-0.5 (LOW/MED) for safety

### Story Bible Strategy
1. **Start broad**: Synopsis first, then narrow to specifics
2. **Names matter**: Use proper nouns consistently
3. **Detail density**: More detail = better AI generation
4. **Update frequently**: Bible grows with your story
5. **Canon check**: Enable Canon Guardian during revision

### Voice Bible Stacking
- **Moderate**: 2-3 controls active (Style + Genre + AI Intensity)
- **Heavy**: 4-5 controls (add Trained Voice or Match)
- **Maximum**: All 6 controls (only for specific passages)
- **Remember**: More controls = more constraint = less freedom

---

## 📞 SUPPORT & MAINTENANCE

### Quick Reference Commands
```bash
# Start app
streamlit run app.py

# Stop app
lsof -ti:8501 | xargs kill -9

# View logs (terminal running streamlit)
# Look for "Olivetti - INFO" messages

# Backup state manually
cp autosave/olivetti_state.json autosave/backup_$(date +%Y%m%d_%H%M%S).json

# Check file size (should be < 5MB for good performance)
du -h autosave/olivetti_state.json
```

### Data Locations
- **Config**: `streamlit/.secrets.toml`
- **State**: `autosave/olivetti_state.json`
- **Backups**: `autosave/*.bak*`
- **Code**: `app.py` (single file, 4183 lines)

### Version Info
- **Build**: Production Stable V1
- **Last Updated**: December 26, 2025
- **Model**: OpenAI gpt-4.1-mini (configurable)
- **Python**: 3.11+ recommended
- **Streamlit**: 1.28.0+

---

## 🎉 READY TO WRITE

You now have:
- ✅ Complete feature documentation
- ✅ Local setup instructions
- ✅ Workflow examples
- ✅ Best practices guide
- ✅ Troubleshooting reference

**Your app is running at**: http://localhost:8501

**Next steps**:
1. Open the app in your browser
2. Explore the NEW bay (Story Bible workspace)
3. Set up Voice Bible controls
4. Add some training samples
5. Start writing!

---

*Built by superhappyfuntimellc*  
*"Professional-Grade AI Author Engine"*
