// Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// System info (CPU & RAM)
// -------------------------------------------------------------------------
async function loadSystemInfo() {
  try {
    const card = document.getElementById('system-info-card');
    const container = document.getElementById('system-info-bars');
    if (!card || !container) return;
    const res = await apiFetch('/api/system-info');
    const d = await res.json();
    if (d.error) return;
    card.style.display = '';

    const coresLabel = document.getElementById('system-cores');
    coresLabel.textContent = `${d.cpu_cores} cores`;

    const cpuPct = Math.round(d.cpu_percent);
    const ramPct = Math.round(d.ram_percent);
    const cpuColor = cpuPct > 90 ? 'var(--red)' : cpuPct > 70 ? 'var(--yellow)' : 'var(--green)';
    const ramColor = ramPct > 90 ? 'var(--red)' : ramPct > 70 ? 'var(--yellow)' : 'var(--green)';

    const ramUsedGB = (d.ram_used_mb / 1024).toFixed(1);
    const ramTotalGB = (d.ram_total_mb / 1024).toFixed(1);

    container.innerHTML = `
      <div class="gpu-bar-row">
        <span class="gpu-bar-label">CPU</span>
        <div class="gpu-bar-track">
          <div class="gpu-bar-fill" style="width:${cpuPct}%;background:${cpuColor};"></div>
        </div>
        <span class="gpu-bar-text">${cpuPct}%</span>
      </div>
      <div class="gpu-bar-row">
        <span class="gpu-bar-label">RAM</span>
        <div class="gpu-bar-track">
          <div class="gpu-bar-fill" style="width:${ramPct}%;background:${ramColor};"></div>
        </div>
        <span class="gpu-bar-text">${ramUsedGB} / ${ramTotalGB} GB (${ramPct}%)</span>
      </div>
    `;
  } catch (e) {
    // hide on error
  }
}

// -------------------------------------------------------------------------
// GPU VRAM indicator
// -------------------------------------------------------------------------
async function loadGpuInfo() {
  try {
    const card = document.getElementById('gpu-vram-card');
    const container = document.getElementById('gpu-vram-bars');
    if (!card || !container) return;
    const res = await apiFetch('/api/gpu-info');
    const data = await res.json();

    if (!data.gpus || data.gpus.length === 0) {
      card.style.display = 'none';
      showGpuWarning();
      return;
    }

    card.style.display = '';
    container.innerHTML = '';

    data.gpus.forEach(gpu => {
      const vramPct = Math.round((gpu.memory_used_mb / gpu.memory_total_mb) * 100);
      const vramColor = vramPct > 90 ? 'var(--red)' : vramPct > 70 ? 'var(--yellow)' : 'var(--green)';
      const corePct = gpu.utilization_pct ?? 0;
      const coreColor = corePct > 90 ? 'var(--red)' : corePct > 70 ? 'var(--yellow)' : 'var(--green)';
      const row = document.createElement('div');
      row.className = 'gpu-bar-row';
      row.innerHTML = `
        <span class="gpu-bar-label" title="${escHtml(gpu.name)}">GPU ${gpu.index}</span>
        <div style="flex:1;display:flex;flex-direction:column;gap:3px;">
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:0.75em;width:3em;color:var(--muted);">core</span>
            <div class="gpu-bar-track" style="flex:1;">
              <div class="gpu-bar-fill" style="width:${corePct}%;background:${coreColor};"></div>
            </div>
            <span class="gpu-bar-text" style="width:3.5em;">${corePct}%</span>
          </div>
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:0.75em;width:3em;color:var(--muted);">VRAM</span>
            <div class="gpu-bar-track" style="flex:1;">
              <div class="gpu-bar-fill" style="width:${vramPct}%;background:${vramColor};"></div>
            </div>
            <span class="gpu-bar-text" style="width:3.5em;">${gpu.memory_used_mb} / ${gpu.memory_total_mb} MB</span>
          </div>
        </div>
      `;
      container.appendChild(row);
    });
  } catch (e) {
    const card = document.getElementById('gpu-vram-card');
    if (card) card.style.display = 'none';
    showGpuWarning();
  }
}
function showGpuWarning() {
  document.getElementById('gpu-warning').style.display = 'block';
}

// -------------------------------------------------------------------------
// Cleanup Settings
// -------------------------------------------------------------------------
async function loadSettings() {
  try {
    if (!document.getElementById('s-dl-cleanup-enabled')) return;
    const res = await apiFetch('/api/settings');
    if (!res) return;
    const s = await res.json();
    const c = s.cleanup || {};
    document.getElementById('s-dl-cleanup-enabled').checked = !!c.downloads_enabled;
    document.getElementById('s-dl-cleanup-age').value = c.downloads_max_age_hours ?? 24;
    document.getElementById('s-inst-cleanup-enabled').checked = !!c.instances_enabled;
    document.getElementById('s-inst-cleanup-age').value = c.instances_max_age_hours ?? 24;

    const authToggle = document.getElementById('s-require-auth');
    if (authToggle) {
      authToggle.checked = s.require_auth !== false; // default ON
      updateAuthHint();
    }
  } catch (e) {}
}

async function saveSettings() {
  if (!document.getElementById('s-dl-cleanup-enabled')) return;
  const payload = {
    cleanup: {
      downloads_enabled: document.getElementById('s-dl-cleanup-enabled').checked,
      downloads_max_age_hours: parseInt(document.getElementById('s-dl-cleanup-age').value) || 24,
      instances_enabled: document.getElementById('s-inst-cleanup-enabled').checked,
      instances_max_age_hours: parseInt(document.getElementById('s-inst-cleanup-age').value) || 24,
    }
  };
  try {
    const res = await apiFetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (res && res.ok) {
      const status = document.getElementById('settings-status');
      status.textContent = 'Saved.';
      setTimeout(() => { status.textContent = ''; }, 2000);
    }
  } catch (e) {}
}

function updateAuthHint() {
  const hint = document.getElementById('auth-hint');
  const toggle = document.getElementById('s-require-auth');
  if (!hint || !toggle) return;
  hint.textContent = toggle.checked
    ? 'All API requests (including model loading) require a valid bearer token.'
    : 'Model loading endpoints are open. Only management endpoints require authentication.';
}

async function saveRequireAuth() {
  const toggle = document.getElementById('s-require-auth');
  if (!toggle) return;
  try {
    const res = await apiFetch('/api/settings');
    if (!res) return;
    const current = await res.json();
    current.require_auth = toggle.checked;
    const save = await apiFetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(current),
    });
    if (save && save.ok) {
      updateAuthHint();
      toast(toggle.checked ? 'Authentication required for all endpoints' : 'Model endpoints are now open', 'info');
    }
  } catch (e) {
    toast('Error saving auth setting: ' + e.message, 'error');
  }
}

const requireAuthToggle = document.getElementById('s-require-auth');
if (requireAuthToggle) {
  requireAuthToggle.addEventListener('change', saveRequireAuth);
}

const saveSettingsBtn = document.getElementById('btn-save-settings');
if (saveSettingsBtn) saveSettingsBtn.addEventListener('click', saveSettings);

// -------------------------------------------------------------------------
// API Keys
// -------------------------------------------------------------------------
async function loadApiKeys() {
  const list = document.getElementById('api-keys-list');
  if (!list) return;
  try {
    const res = await apiFetch('/api/api-keys');
    const keys = await res.json();
    if (keys.length === 0) {
      list.innerHTML = '<div style="color:var(--muted);font-size:12px;text-align:center;padding:12px;">No API keys yet. API is open to all requests.</div>';
      return;
    }
    list.innerHTML = '';
    keys.forEach(k => {
      const date = new Date(k.created_at * 1000).toLocaleDateString();
      const item = document.createElement('div');
      item.className = 'dl-item';
      item.innerHTML = `
        <div class="dl-item-top">
          <span class="dl-item-name"><strong>${escHtml(k.name)}</strong></span>
          <code style="font-size:11px;color:var(--muted);">${escHtml(k.prefix)}</code>
          <span style="font-size:11px;color:var(--muted);">${date}</span>
          <button class="btn-xs danger btn-ak-delete" data-id="${k.id}"><i class="fa-solid fa-trash"></i> Revoke</button>
        </div>
      `;
      list.appendChild(item);
    });
    list.querySelectorAll('.btn-ak-delete').forEach(btn => {
      btn.addEventListener('click', () => revokeApiKey(btn.dataset.id));
    });
  } catch (e) { /* ignore */ }
}

async function createApiKey() {
  const nameInput = document.getElementById('ak-name');
  const name = nameInput.value.trim() || 'Untitled';
  try {
    const res = await apiFetch('/api/api-keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    const data = await res.json();
    if (res.ok) {
      toast('API key created', 'success');
      nameInput.value = '';
      // Show the key in a modal
      document.getElementById('api-key-value').textContent = data.key;
      document.getElementById('api-key-modal').classList.add('open');
      await loadApiKeys();
    } else {
      toast(`Failed: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error creating key: ' + e.message, 'error');
  }
}

async function revokeApiKey(id) {
  const ok = await showConfirm('Revoke API Key', 'Revoke this API key? Any tools using it will lose access.');
  if (!ok) return;
  try {
    const res = await apiFetch(`/api/api-keys/${id}`, { method: 'DELETE' });
    if (res.ok) {
      toast('API key revoked', 'info');
      await loadApiKeys();
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

const btnCreateKey = document.getElementById('btn-create-api-key');
if (btnCreateKey) btnCreateKey.addEventListener('click', createApiKey);

const btnCopyKey = document.getElementById('btn-copy-api-key');
if (btnCopyKey) btnCopyKey.addEventListener('click', () => {
  const key = document.getElementById('api-key-value').textContent;
  const icon = btnCopyKey.querySelector('i');

  const onCopied = () => {
    if (icon) {
      icon.className = 'fa-solid fa-check';
      icon.style.color = 'var(--green)';
      setTimeout(() => { icon.className = 'fa-solid fa-copy'; icon.style.color = ''; }, 2000);
    }
  };

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(key).then(onCopied);
  } else {
    const ta = document.createElement('textarea');
    ta.value = key;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    onCopied();
  }
});

const apiKeyModal = document.getElementById('api-key-modal');
const btnCloseApiKeyModal = document.getElementById('btn-close-api-key-modal');
if (btnCloseApiKeyModal) btnCloseApiKeyModal.addEventListener('click', () => apiKeyModal.classList.remove('open'));
if (apiKeyModal) apiKeyModal.addEventListener('click', (e) => {
  if (e.target === apiKeyModal) apiKeyModal.classList.remove('open');
});
