// Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Model Library
// -------------------------------------------------------------------------
async function loadModels() {
  const list = document.getElementById('model-list');
  try {
    const res = await apiFetch('/api/models');
    allModels = await res.json();
    renderModels();
  } catch (e) {
    list.innerHTML = '<div id="model-empty">Error loading models</div>';
  }
}

function renderModels() {
  const list = document.getElementById('model-list');
  if (!list) return;
  const query = document.getElementById('model-search').value.toLowerCase().trim();

  const filtered = query
    ? allModels.filter(m => m.name.toLowerCase().includes(query) || m.path.toLowerCase().includes(query) || (m.quant && m.quant.toLowerCase().includes(query)))
    : allModels;

  list.innerHTML = '';
  if (filtered.length === 0) {
    list.innerHTML = `<div id="model-empty">${allModels.length === 0 ? 'No models found in /models' : 'No matches'}</div>`;
    return;
  }
  filtered.forEach(m => {
    const el = document.createElement('div');
    el.className = 'model-item' + (m.path === selectedModelPath ? ' selected' : '');
    const quantBadge = m.quant ? `<span class="badge badge-quant">${escHtml(m.quant)}</span>` : '';
    el.innerHTML = `
      <span class="name">${escHtml(m.name)}</span>
      <div class="badges">
        <span class="badge">${m.type.toUpperCase()}</span>
        ${quantBadge}
        <span class="badge badge-size">${escHtml(m.size_display)}</span>
      </div>
      <span class="path">${escHtml(m.path)}</span>
      <button class="btn-delete-model" title="Delete model from disk"><i class="fa-solid fa-trash"></i></button>
    `;
    el.querySelector('.btn-delete-model').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteModel(m);
    });
    el.addEventListener('click', () => selectModel(m, el));
    list.appendChild(el);
  });
}

async function selectModel(model, el) {
  document.querySelectorAll('.model-item').forEach(x => x.classList.remove('selected'));
  el.classList.add('selected');
  selectedModelPath = model.path;
  document.getElementById('f-model-path').value = model.path;
  updatePortSuggestion();
  // Load preset if one exists
  try {
    const res = await apiFetch(`/api/presets${encodePathForUrl(model.path)}`);
    if (res.ok) {
      const p = await res.json();
      if (p.n_gpu_layers != null) document.getElementById('f-gpu-layers').value = p.n_gpu_layers;
      if (p.ctx_size != null) document.getElementById('f-ctx-size').value = p.ctx_size;
      document.getElementById('f-threads').value = p.threads || '';
      document.getElementById('f-parallel').value = p.parallel || '';
      document.getElementById('f-extra').value = p.extra_args || '';
      document.getElementById('f-gpu-devices').value = p.gpu_devices || '';
      document.getElementById('f-idle-timeout').value = p.idle_timeout_min || 0;
      document.getElementById('f-max-concurrent').value = p.max_concurrent || 0;
      document.getElementById('f-max-queue-depth').value = p.max_queue_depth || 200;
      document.getElementById('f-share-queue').checked = !!p.share_queue;
      document.getElementById('f-embedding-model').checked = !!p.embedding_model;
      toast('Preset loaded', 'info');
    }
  } catch (e) { /* no preset, use defaults */ }
  // Detect layer count for model
  await updateGpuLayersTotal(model.path);
}

async function deleteModel(model) {
  const ok = await showConfirm('Delete Model', `Delete "${model.name}" (${model.size_display}) from disk?\n\n${model.path}\n\nThis cannot be undone.`);
  if (!ok) return;
  try {
    const res = await apiFetch('/api/models/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: model.path }),
    });
    const data = await res.json();
    if (res.ok) {
      toast('Model deleted', 'info');
      if (selectedModelPath === model.path) selectedModelPath = null;
      await loadModels();
    } else {
      toast(`Cannot delete: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error deleting model: ' + e.message, 'error');
  }
}

async function updateGpuLayersTotal(modelPath) {
  const label = document.getElementById('gpu-layers-total');
  label.textContent = '';
  if (!modelPath || !modelPath.toLowerCase().endsWith('.gguf')) return;
  try {
    const res = await apiFetch(`/api/model-layers?path=${encodeURIComponent(modelPath)}`);
    const data = await res.json();
    if (data.layers && data.layers > 0) {
      label.textContent = `/ ${data.layers}`;
    }
  } catch (e) { /* ignore */ }
}

// Detect layers when model path is changed manually
const modelPathField = document.getElementById('f-model-path');
if (modelPathField) {
  modelPathField.addEventListener('change', function() {
    updateGpuLayersTotal(this.value.trim());
  });
}
