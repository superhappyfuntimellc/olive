const api = {
  list: () => fetch('/api/todos').then(r => r.json()),
  create: text => fetch('/api/todos', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text})
  }).then(r => r.json()),
  update: (id, patch) => fetch(`/api/todos/${id}`, {
    method: 'PUT',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(patch)
  }).then(r => r.json()),
  delete: id => fetch(`/api/todos/${id}`, { method: 'DELETE' })
    .then(r => { if (!r.ok) throw new Error('Delete failed'); })
};

const $newTodo = document.getElementById('new-todo');
const $list = document.getElementById('todo-list');
const $count = document.getElementById('count');
const $filters = document.querySelectorAll('.filters button');
const $clearCompleted = document.getElementById('clear-completed');

let todos = [];
let filter = 'all';

function render() {
  $list.innerHTML = '';
  const visible = todos.filter(t => {
    if (filter === 'active') return !t.completed;
    if (filter === 'completed') return t.completed;
    return true;
  });

  visible.forEach(t => {
    const li = document.createElement('li');
    li.className = 'todo-item';
    li.dataset.id = t.id;

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = !!t.completed;
    checkbox.addEventListener('change', async () => {
      await api.update(t.id, { completed: checkbox.checked });
      await load();
    });

    const span = document.createElement('span');
    span.textContent = t.text;
    span.contentEditable = true;
    span.className = t.completed ? 'completed' : '';
    span.addEventListener('blur', async () => {
      const newText = span.textContent.trim();
      if (newText && newText !== t.text) {
        await api.update(t.id, { text: newText });
        await load();
      } else {
        span.textContent = t.text;
      }
    });

    const del = document.createElement('button');
    del.textContent = '✕';
    del.className = 'delete';
    del.addEventListener('click', async () => {
      await api.delete(t.id);
      await load();
    });

    li.appendChild(checkbox);
    li.appendChild(span);
    li.appendChild(del);
    $list.appendChild(li);
  });

  const remaining = todos.filter(t => !t.completed).length;
  $count.textContent = `${remaining} item${remaining !== 1 ? 's' : ''} left`;
}

async function load() {
  try {
    todos = await api.list();
    render();
  } catch (err) {
    console.error('Failed to load todos', err);
  }
}

$newTodo.addEventListener('keydown', async (e) => {
  if (e.key === 'Enter') {
    const text = $newTodo.value.trim();
    if (!text) return;
    await api.create(text);
    $newTodo.value = '';
    await load();
  }
});

$filters.forEach(btn => {
  btn.addEventListener('click', () => {
    filter = btn.dataset.filter;
    $filters.forEach(b => b.classList.toggle('active', b === btn));
    render();
  });
});

$clearCompleted.addEventListener('click', async () => {
  try {
    await fetch('/api/todos/clear_completed', { method: 'POST' });
    await load();
  } catch (err) {
    console.error('Failed to clear completed', err);
  }
});

// initial load
load();
