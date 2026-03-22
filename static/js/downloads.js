// Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Download Modal
// -------------------------------------------------------------------------
async function refreshDownloadDiskSpace() {
  try {
    const res = await apiFetch('/api/disk-space');
    const data = await res.json();
    if (data.free_gb != null) {
      document.getElementById('dl-disk-space').textContent =
        `${data.free_gb} GB free / ${data.total_gb} GB total`;
    }
  } catch (e) { /* ignore */ }
}

async function openDownloadModal() {
  document.getElementById('download-modal').classList.add('open');
  document.getElementById('d-repo-id').focus();
  refreshDownloadDiskSpace();
}

function closeDownloadModal() {
  document.getElementById('download-modal').classList.remove('open');
  document.getElementById('dl-form-status').textContent = '';
}

const btnOpenDownload = document.getElementById('btn-open-download');
if (btnOpenDownload) btnOpenDownload.addEventListener('click', openDownloadModal);

const btnCloseDownload = document.getElementById('btn-close-download');
if (btnCloseDownload) btnCloseDownload.addEventListener('click', closeDownloadModal);

const downloadModalEl = document.getElementById('download-modal');
if (downloadModalEl) downloadModalEl.addEventListener('click', (e) => {
  if (e.target === downloadModalEl) closeDownloadModal();
});

const downloadForm = document.getElementById('download-form');
if (downloadForm) downloadForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('btn-start-download');
  const status = document.getElementById('dl-form-status');
  btn.disabled = true;
  status.textContent = 'Starting…';

  const body = {
    repo_id:  document.getElementById('d-repo-id').value.trim(),
    filename: document.getElementById('d-filename').value.trim(),
    hf_token: document.getElementById('d-token').value.trim(),
    speed_limit_mbps: parseFloat(document.getElementById('d-speed-limit').value) || 0,
  };

  try {
    const res = await apiFetch('/api/downloads', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (res.ok) {
      toast(`Download started: ${body.repo_id}`, 'success');
      closeDownloadModal();
      document.getElementById('download-form').reset();
      await pollDownloads();
      // Expand downloads panel if collapsed
      const panel = document.getElementById('downloads-panel');
      const hdr = document.getElementById('dl-section-toggle');
      if (panel && hdr) {
        panel.classList.remove('hidden');
        hdr.classList.remove('collapsed');
      }
    } else {
      toast(`Download failed: ${data.error}`, 'error');
      status.textContent = '';
    }
  } catch (err) {
    toast('Error: ' + err.message, 'error');
    status.textContent = '';
  } finally {
    btn.disabled = false;
  }
});

// -------------------------------------------------------------------------
// Downloads panel
// -------------------------------------------------------------------------
async function pollDownloads() {
  try {
    const res = await apiFetch('/api/downloads');
    const list = await res.json();
    const map = {};
    list.forEach(d => map[d.id] = d);
    downloads = map;
    renderDownloads();
  } catch (e) { /* ignore */ }
}

function renderDownloads() {
  const panel = document.getElementById('downloads-panel');
  if (!panel) return;
  const all = Object.values(downloads);

  if (all.length === 0) {
    panel.innerHTML = '<div id="dl-empty">No downloads yet.</div>';
    return;
  }

  // Sort: active first, then by started_at desc
  all.sort((a, b) => {
    const aActive = a.status === 'downloading' ? 1 : 0;
    const bActive = b.status === 'downloading' ? 1 : 0;
    return bActive - aActive || b.started_at - a.started_at;
  });

  panel.innerHTML = '';
  all.forEach(dl => {
    const item = document.createElement('div');
    item.className = 'dl-item';
    item.dataset.id = dl.id;

    const label = dl.filename ? dl.filename : dl.repo_id.split('/').pop();
    const spinner = dl.status === 'downloading'
      ? '<span class="spinner"></span>' : '';

    item.innerHTML = `
      <div class="dl-item-top">
        ${spinner}
        <span class="dl-item-name" title="${escHtml(dl.repo_id)}">${escHtml(label)}</span>
        <span class="dl-status dl-status-${dl.status}">${dl.status}</span>
      </div>
      <div class="dl-item-actions">
        <button class="btn-xs btn-dl-logs" data-id="${dl.id}"><i class="fa-solid fa-terminal"></i> Logs</button>
        ${dl.status === 'downloading'
          ? `<button class="btn-xs danger btn-dl-cancel" data-id="${dl.id}"><i class="fa-solid fa-ban"></i> Cancel</button>`
          : ''}
        ${dl.status === 'completed'
          ? `<button class="btn-xs btn-dl-use" data-path="${escHtml(dl.dest_path)}" data-filename="${escHtml(dl.filename)}"><i class="fa-solid fa-arrow-right"></i> Use</button>`
          : ''}
        ${['failed', 'cancelled', 'completed'].includes(dl.status)
          ? `<button class="btn-xs danger btn-dl-remove" data-id="${dl.id}"><i class="fa-solid fa-trash"></i> Remove</button>`
          : ''}
      </div>
    `;
    panel.appendChild(item);
  });

  // Bind buttons
  panel.querySelectorAll('.btn-dl-logs').forEach(btn => {
    btn.addEventListener('click', () => openLogModal('download', btn.dataset.id));
  });
  panel.querySelectorAll('.btn-dl-cancel').forEach(btn => {
    btn.addEventListener('click', () => cancelDownload(btn.dataset.id));
  });
  panel.querySelectorAll('.btn-dl-remove').forEach(btn => {
    btn.addEventListener('click', () => removeDownload(btn.dataset.id));
  });
  panel.querySelectorAll('.btn-dl-use').forEach(btn => {
    btn.addEventListener('click', () => {
      const fullPath = btn.dataset.filename
        ? btn.dataset.path + '/' + btn.dataset.filename
        : btn.dataset.path;
      window.location.href = `/?model_path=${encodeURIComponent(fullPath)}`;
    });
  });

  // Auto-refresh models list when a download just completed
  all.forEach(dl => {
    if (dl.status === 'completed' && !dl.hasNotified) {
      dl.hasNotified = true;
      loadModels();
    }
  });
}

async function cancelDownload(id) {
  try {
    const res = await apiFetch(`/api/downloads/${id}`, { method: 'DELETE' });
    if (res.ok) {
      toast('Download cancelled', 'info');
      await pollDownloads();
    } else {
      toast('Failed to cancel download', 'error');
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}

async function removeDownload(id) {
  try {
    const res = await apiFetch(`/api/downloads/${id}/remove`, { method: 'DELETE' });
    if (res.ok) {
      toast('Download removed', 'info');
      await pollDownloads();
    } else {
      const data = await res.json();
      toast(`Cannot remove: ${data.error}`, 'error');
    }
  } catch (e) {
    toast('Error: ' + e.message, 'error');
  }
}
