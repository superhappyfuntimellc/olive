from flask import Flask, request, jsonify, render_template
import json
import os
from pathlib import Path

app = Flask(__name__)

# Path to todos storage
TODOS_FILE = Path(__file__).parent / 'todos.json'

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
