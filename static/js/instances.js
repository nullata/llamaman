// Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Instance polling
// -------------------------------------------------------------------------
function queueToneClass(pct) {
  return pct > 80 ? 'progress-tone-danger' : pct > 50 ? 'progress-tone-warning' : 'progress-tone-success';
}

async function pollInstances() {
  try {
    const res = await apiFetch('/api/instances');
    const list = await res.json();
    const map = {};
    list.forEach(i => map[i.id] = i);
    instances = map;
    renderInstances();
  } catch (e) { /* ignore */ }
}


function renderInstances() {
  const container = document.getElementById('instance-container');
  const count = document.getElementById('instance-count');

  const all = Object.values(instances);
  const active = all.filter(i => i.status !== 'stopped' && i.status !== 'sleeping');
  if (count) {
    count.textContent = `${active.length} instance${active.length !== 1 ? 's' : ''}`;
  }
  if (!container) return;

  // Update heading
  document.getElementById('instances-heading').textContent =
    `Running Instances (${all.length})`;

  if (all.length === 0) {
    container.innerHTML = '<div id="no-instances">No instances yet. Launch one above.</div>';
    return;
  }

  // Build ordered list: active first, then sleeping, then stopped
  const sleeping = all.filter(i => i.status === 'sleeping');
  const stopped = all.filter(i => i.status === 'stopped');
  const ordered = [...active, ...sleeping, ...stopped];

  // Preserve existing cards or build fresh
  const existingIds = new Set([...container.querySelectorAll('.instance-card')].map(el => el.dataset.id));
  const newIds = new Set(ordered.map(i => i.id));

  // Remove cards no longer present
  existingIds.forEach(id => {
    if (!newIds.has(id)) container.querySelector(`[data-id="${id}"]`)?.remove();
  });

  ordered.forEach((inst, idx) => {
    let card = container.querySelector(`[data-id="${inst.id}"]`);
    if (!card) {
      card = document.createElement('div');
      card.className = 'instance-card';
      card.dataset.id = inst.id;
      container.appendChild(card);
    }

    const uptime = (inst.status === 'stopped' || inst.status === 'sleeping')
      ? (inst.status === 'sleeping' ? 'Sleeping' : 'Down') : `Up ${formatUptime(inst.started_at)}`;
    const statusClass = `status-${inst.status}`;

    const s = inst.stats || {};
    const statsItems = [];
    if (s.model_load_time_s != null) statsItems.push(`Load ${s.model_load_time_s}s`);
    if (s.last_tokens_per_sec != null) statsItems.push(`${s.last_tokens_per_sec} t/s`);
    if (s.last_ttft_ms != null) statsItems.push(`TTFT ${s.last_ttft_ms}ms`);
    if (s.total_requests) statsItems.push(`${s.total_requests} req`);
    if (s.crash_count) statsItems.push(`<span class="text-danger">${s.crash_count} crash${s.crash_count > 1 ? 'es' : ''}</span>`);
    if (inst.last_request_at) {
      const ago = Math.round((Date.now() / 1000) - inst.last_request_at);
      if (ago < 60) statsItems.push(`last req ${ago}s ago`);
      else if (ago < 3600) statsItems.push(`last req ${Math.round(ago/60)}m ago`);
      else statsItems.push(`last req ${Math.round(ago/3600)}h ago`);
    }
    const statsLine = statsItems.length > 0
      ? `<div class="meta inst-meta-accent">${statsItems.join(' · ')}</div>` : '';

    // Queue indicator
    const q = inst.queue;
    let queueLine = '';
    if (q) {
      const qPct = q.max_queue_depth > 0 ? Math.round((q.queued / q.max_queue_depth) * 100) : 0;
      const qMax = Math.max(q.max_queue_depth || 0, 1);
      queueLine = `<div class="meta instance-queue-row">
        <span class="instance-queue-label">Queue</span>
        <progress class="instance-queue-progress ${queueToneClass(qPct)}" max="${qMax}" value="${Math.min(q.queued, qMax)}"></progress>
        <span class="instance-queue-text">${q.active}/${q.max_concurrent} active · ${q.queued} queued</span>
      </div>`;
    }

    const portLine = inst.internal_port != null
      ? `Public ${inst.port} -> llama-server ${inst.internal_port}`
      : `Port ${inst.port}`;

    card.innerHTML = `
    <div class="inst-info">
      <div class="model">${escHtml(inst.model_name)}</div>
      <div class="meta">${portLine} &nbsp;·&nbsp; PID ${inst.pid} &nbsp;·&nbsp; ${uptime}</div>
      ${statsLine}
      ${queueLine}
    </div>
    <span class="status-badge ${statusClass}">${inst.status}</span>
    <div class="inst-actions">
      <button class="btn btn-secondary btn-logs" data-id="${inst.id}"><i class="fa-solid fa-terminal"></i> Logs</button>
      ${inst.status !== 'stopped' && inst.status !== 'sleeping' ? `<button class="btn btn-danger btn-stop" data-id="${inst.id}"><i class="fa-solid fa-stop"></i> Stop</button>` : ''}
      ${inst.status === 'sleeping' ? `<button class="btn btn-danger btn-stop" data-id="${inst.id}"><i class="fa-solid fa-stop"></i> Stop</button>` : ''}
      ${inst.status === 'stopped' || inst.status === 'sleeping' ? `<button class="btn btn-primary btn-restart" data-id="${inst.id}"><i class="fa-solid fa-rotate-right"></i> Restart</button>` : ''}
      ${inst.status === 'stopped' ? `<button class="btn btn-danger btn-remove" data-id="${inst.id}" title="Remove from list"><i class="fa-solid fa-trash"></i></button>` : ''}
    </div>
  `;
  });

  // Move cards to correct order in DOM
  ordered.forEach(inst => {
    const card = container.querySelector(`[data-id="${inst.id}"]`);
    if (card) container.appendChild(card);
  });

  // Remove stale no-instances placeholder
  container.querySelector('#no-instances')?.remove();

  // Bind buttons
  container.querySelectorAll('.btn-stop').forEach(btn => {
    btn.addEventListener('click', () => stopInstance(btn.dataset.id));
  });
  container.querySelectorAll('.btn-remove').forEach(btn => {
    btn.addEventListener('click', () => removeInstance(btn.dataset.id));
  });
  container.querySelectorAll('.btn-restart').forEach(btn => {
    btn.addEventListener('click', () => restartInstance(btn.dataset.id));
  });
  container.querySelectorAll('.btn-logs').forEach(btn => {
    btn.addEventListener('click', () => openLogModal('instance', btn.dataset.id));
  });
}

// -------------------------------------------------------------------------
// Instance actions
// -------------------------------------------------------------------------
async function stopInstance(id) {
  try {
    const res = await apiFetch(`/api/instances/${id}`, { method: 'DELETE' });
    if (res.ok) {
      toast('Instance stopped', 'info');
      await pollInstances();
      await updatePortSuggestion();
    } else {
      toast('Failed to stop instance', 'error');
    }
  } catch (e) {
    toast('Error stopping instance: ' + e.message, 'error');
  }
}

async function restartInstance(id) {
  try {
    const attemptRestart = async (confirmOvercommit = false) => {
      const res = await apiFetch(`/api/instances/${id}/restart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(confirmOvercommit ? { confirm_overcommit: true } : {}),
      });
      const data = await readApiResponse(res);
      if (!res.ok && data.confirm_required) {
        const ok = await showConfirm('Launch Beyond Limit', data.error);
        if (!ok) return { cancelled: true };
        return await attemptRestart(true);
      }
      return { res, data };
    };

    const result = await attemptRestart();
    if (result.cancelled) return;

    const { res, data } = result;
    if (res.ok) {
      const msg = data.internal_port != null
        ? `Instance restarted: public ${data.port}, llama-server ${data.internal_port}`
        : `Instance restarted on port ${data.port}`;
      toast(msg, 'success');
      await pollInstances();
      await updatePortSuggestion();
    } else {
      toast(`Restart failed: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error restarting: ' + e.message, 'error');
  }
}

async function removeInstance(id) {
  try {
    const res = await apiFetch(`/api/instances/${id}/remove`, { method: 'DELETE' });
    if (res.ok) {
      toast('Instance removed', 'info');
      await pollInstances();
    } else {
      const data = await res.json();
      toast(`Cannot remove: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error removing instance: ' + e.message, 'error');
  }
}

// -------------------------------------------------------------------------
// Launch form
// -------------------------------------------------------------------------
async function updatePortSuggestion() {
  const portField = document.getElementById('f-port');
  if (!portField) return;
  portField.value = await nextAvailablePort();
}

function readLaunchForm() {
  const ctxSizeRaw = document.getElementById('f-ctx-size').value.trim();
  if (!ctxSizeRaw) {
    throw new Error('Context size is required');
  }
  const ctxSize = parseInt(ctxSizeRaw, 10);
  if (!Number.isInteger(ctxSize) || ctxSize <= 0) {
    throw new Error('Context size must be a positive integer');
  }

  const body = {
    n_gpu_layers: parseInt(document.getElementById('f-gpu-layers').value),
    ctx_size: ctxSize,
    extra_args: document.getElementById('f-extra').value.trim(),
    gpu_devices: document.getElementById('f-gpu-devices').value.trim(),
    idle_timeout_min: parseInt(document.getElementById('f-idle-timeout').value) || 0,
    max_concurrent: parseInt(document.getElementById('f-max-concurrent').value) || 0,
    max_queue_depth: parseInt(document.getElementById('f-max-queue-depth').value) || 200,
    share_queue: document.getElementById('f-share-queue').checked,
    embedding_model: document.getElementById('f-embedding-model').checked,
  };
  const threads = document.getElementById('f-threads').value.trim();
  if (threads) body.threads = parseInt(threads);
  const parallel = document.getElementById('f-parallel').value.trim();
  if (parallel) body.parallel = parseInt(parallel);
  return body;
}

const launchForm = document.getElementById('launch-form');
if (launchForm) launchForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('btn-launch');
  const status = document.getElementById('launch-status');
  btn.disabled = true;
  status.textContent = 'Launching…';

  try {
    const body = readLaunchForm();
    body.model_path = document.getElementById('f-model-path').value.trim();
    body.port = parseInt(document.getElementById('f-port').value);

    const attemptLaunch = async (confirmOvercommit = false) => {
      const launchBody = {
        ...body,
        ...(confirmOvercommit ? { confirm_overcommit: true } : {}),
      };
      const res = await apiFetch('/api/instances', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(launchBody),
      });
      const data = await readApiResponse(res);
      if (!res.ok && data.confirm_required) {
        const ok = await showConfirm('Launch Beyond Limit', data.error);
        if (!ok) return { cancelled: true };
        return await attemptLaunch(true);
      }
      return { res, data };
    };

    const result = await attemptLaunch();
    if (result.cancelled) {
      status.textContent = '';
      return;
    }

    const { res, data } = result;
    if (res.ok) {
      const msg = data.internal_port != null
        ? `Instance launched: public ${data.port}, llama-server ${data.internal_port}`
        : `Instance launched on port ${data.port}`;
      toast(msg, 'success');
      status.textContent = '';
      updatePortSuggestion();
      await pollInstances();
    } else {
      toast(`Launch failed: ${data.error}`, 'error');
      status.textContent = '';
    }
  } catch (e) {
    toast('Launch error: ' + e.message, 'error');
    status.textContent = '';
  } finally {
    btn.disabled = false;
  }
});

// -------------------------------------------------------------------------
// Preset save
// -------------------------------------------------------------------------
const savePresetBtn = document.getElementById('btn-save-preset');
if (savePresetBtn) savePresetBtn.addEventListener('click', async () => {
  const modelPath = document.getElementById('f-model-path').value.trim();
  if (!modelPath) {
    toast('Select a model first', 'error');
    return;
  }

  try {
    const body = readLaunchForm();
    const res = await apiFetch(`/api/presets${encodePathForUrl(modelPath)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (res.ok) {
      toast('Preset saved', 'success');
    } else {
      const data = await readApiResponse(res);
      toast(`Failed to save preset: ${data.error || 'unknown error'}`, 'error');
    }
  } catch (e) {
    toast('Error saving preset: ' + e.message, 'error');
  }
});
