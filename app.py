from flask import Flask, request, jsonify, render_template, stream_with_context, Response
import json
import os
from pathlib import Path

app = Flask(__name__)

# Path to todos storage
TODOS_FILE = Path(__file__).parent / 'todos.json'

# ============================================================
# TODO APP ROUTES
# ============================================================

def load_todos():
    """Load todos from JSON file."""
    if not TODOS_FILE.exists():
        return []
    try:
        with open(TODOS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []

def save_todos(todos):
    """Save todos to JSON file."""
    with open(TODOS_FILE, 'w') as f:
        json.dump(todos, f, indent=2)

def next_id(todos):
    """Generate next available ID."""
    return max([t.get('id', 0) for t in todos], default=0) + 1

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/api/todos', methods=['GET'])
def get_todos():
    """Get all todos."""
    return jsonify(load_todos())

@app.route('/api/todos', methods=['POST'])
def create_todo():
    """Create a new todo."""
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'text is required'}), 400
    
    todos = load_todos()
    new_todo = {
        'id': next_id(todos),
        'text': text,
        'completed': False
    }
    todos.append(new_todo)
    save_todos(todos)
    
    return jsonify(new_todo), 201

@app.route('/api/todos/<int:todo_id>', methods=['PUT'])
def update_todo(todo_id):
    """Update an existing todo."""
    data = request.get_json()
    todos = load_todos()
    
    todo = next((t for t in todos if t['id'] == todo_id), None)
    if not todo:
        return jsonify({'error': 'Todo not found'}), 404
    
    if 'text' in data:
        todo['text'] = data['text']
    if 'completed' in data:
        todo['completed'] = bool(data['completed'])
    
    save_todos(todos)
    return jsonify(todo)

@app.route('/api/todos/<int:todo_id>', methods=['DELETE'])
def delete_todo(todo_id):
    """Delete a todo."""
    todos = load_todos()
    todos = [t for t in todos if t['id'] != todo_id]
    save_todos(todos)
    
    return '', 204

@app.route('/api/todos/clear_completed', methods=['POST'])
def clear_completed():
    """Remove all completed todos."""
    todos = load_todos()
    todos = [t for t in todos if not t.get('completed', False)]
    save_todos(todos)
    
    return jsonify({'success': True})

# ============================================================
# OLIVETTI PROJECT API ROUTES
# ============================================================

def detect_lane_from_draft(draft: str) -> str:
    """Detect writing lane based on draft length"""
    word_count = len(draft.split())
    if word_count < 500:
        return "NEW"
    elif word_count < 2000:
        return "ROUGH"
    elif word_count < 5000:
        return "EDIT"
    else:
        return "FINAL"

def load_project(pid: str):
    """Load project from olivetti database - placeholder"""
    # TODO: Integrate with olivetti_app.py database functions
    return {
        "id": pid,
        "draft": "",
        "story_bible": {"synopsis": "", "characters": "", "world": ""}
    }

def retrieve_style_exemplars(style: str, lane: str, draft: str, k: int = 3):
    """Retrieve style exemplars - placeholder"""
    # TODO: Implement vector similarity search
    return []

def compose_prompt(action: str, draft: str, story_bible: dict, style: str, intensity: float, exemplars: list, lane: str) -> str:
    """Compose AI prompt with full context"""
    sb_short = (story_bible.get("synopsis","") or "")[:1200]
    
    style_directives = {
        "Neutral": "Write in a clear, balanced style without strong stylistic flourishes.",
        "Crisp": "Write with short, punchy sentences. Be direct and economical with words.",
        "Flowing": "Write with longer, flowing sentences that create rhythm and immersion."
    }
    style_directive = style_directives.get(style, style_directives["Neutral"])
    
    if intensity < 0.3:
        style_directive += " Keep stylistic elements subtle."
    elif intensity > 0.7:
        style_directive += " Apply the style strongly and consistently."
    
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

def temperature_from_intensity(intensity: float) -> float:
    """Convert AI intensity to model temperature"""
    return 0.3 + (intensity * 0.9)

def call_model_stream(prompt: str, temperature: float = 0.7):
    """Call OpenAI API with streaming - placeholder"""
    # TODO: Implement actual OpenAI streaming call
    import time
    words = ["This", "is", "a", "streaming", "response", "placeholder."]
    for word in words:
        time.sleep(0.1)
        yield f"data: {json.dumps({'text': word + ' '})}\n\n"
    yield "data: [DONE]\n\n"

@app.route("/api/projects/<pid>/action", methods=["POST"])
def project_action(pid):
    """Execute writing action on project with streaming response"""
    payload = request.json
    action = payload.get("action")
    lane = payload.get("lane")
    style = payload.get("style") or "Neutral"
    intensity = float(payload.get("intensity", 0.6))
    
    project = load_project(pid)
    
    if not lane:
        lane = detect_lane_from_draft(project.get("draft", ""))
    
    exemplars = retrieve_style_exemplars(style, lane, project.get("draft", ""), k=3)
    prompt = compose_prompt(action, project.get("draft", ""), project.get("story_bible", {}), 
                           style, intensity, exemplars, lane)
    
    # Return streaming response
    def generate():
        for chunk in call_model_stream(prompt, temperature=temperature_from_intensity(intensity)):
            yield chunk
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
