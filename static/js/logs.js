// Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Log Modal (shared by instances and downloads) - SSE live streaming
// -------------------------------------------------------------------------
let logModalInstanceId = null;
let logRefreshTimer = null;
let logFetchUrl = null;       // current URL used by the log modal
let logEventSource = null;    // SSE EventSource for live log streaming

function shouldAutoScroll() {
  return document.getElementById('log-autoscroll').checked;
}

function startLogStream(streamUrl, fallbackUrl) {
  const out = document.getElementById('log-output');
  out.textContent = '';

  // Try SSE first
  try {
    logEventSource = new EventSource(streamUrl);
    logEventSource.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.lines) {
          out.textContent += data.lines.join('');
          if (shouldAutoScroll()) out.parentElement.scrollTop = out.parentElement.scrollHeight;
        }
      } catch (e) { /* ignore parse errors */ }
    };
    logEventSource.onerror = () => {
      // SSE failed, fall back to polling
      if (logEventSource) { logEventSource.close(); logEventSource = null; }
      logFetchUrl = fallbackUrl;
      fetchLogsPoll();
      logRefreshTimer = setInterval(fetchLogsPoll, 3000);
    };
  } catch (e) {
    // EventSource not supported, use polling
    logFetchUrl = fallbackUrl;
    fetchLogsPoll();
    logRefreshTimer = setInterval(fetchLogsPoll, 3000);
  }
}

async function fetchLogsPoll() {
  if (!logFetchUrl) return;
  try {
    const res = await apiFetch(logFetchUrl);
    const data = await res.json();
    const out = document.getElementById('log-output');
    if (data.lines) {
      out.textContent = data.lines.join('');
      if (shouldAutoScroll()) out.parentElement.scrollTop = out.parentElement.scrollHeight;
    } else {
      out.textContent = data.error || '(empty)';
    }
  } catch (e) {
    document.getElementById('log-output').textContent = 'Error fetching logs';
  }
}

function openLogModal(type, id) {
  let title;
  if (type === 'instance') {
    const inst = instances[id];
    title = inst ? `Logs - ${inst.model_name} :${inst.port}` : 'Logs';
  } else {
    const dl = downloads[id];
    title = dl ? `Download Logs - ${dl.repo_id}` : 'Download Logs';
  }
  const endpoint = type === 'instance' ? 'instances' : 'downloads';
  document.getElementById('log-modal-title').textContent = title;
  document.getElementById('log-output').textContent = 'Loading…';
  document.getElementById('log-modal').classList.add('open');
  logModalInstanceId = id;
  startLogStream(`/api/${endpoint}/${id}/logs/stream`, `/api/${endpoint}/${id}/logs`);
}

function closeLogs() {
  document.getElementById('log-modal').classList.remove('open');
  logModalInstanceId = null;
  logFetchUrl = null;
  if (logEventSource) { logEventSource.close(); logEventSource = null; }
  if (logRefreshTimer) { clearInterval(logRefreshTimer); logRefreshTimer = null; }
}

const closeLogsBtn = document.getElementById('btn-close-logs');
const logModalEl = document.getElementById('log-modal');
if (closeLogsBtn && logModalEl) {
  closeLogsBtn.addEventListener('click', closeLogs);
  logModalEl.addEventListener('click', (e) => {
    if (e.target === logModalEl) closeLogs();
  });
}
