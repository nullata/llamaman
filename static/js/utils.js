// Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Shared state
// -------------------------------------------------------------------------
let instances = {};          // id -> instance obj
let downloads = {};          // id -> download obj
let selectedModelPath = null;
let allModels = [];           // cached model list for filtering

// -------------------------------------------------------------------------
// Collapsible sections (persisted in localStorage)
// -------------------------------------------------------------------------
const SECTION_STORAGE_KEY = 'llamaman-sections';

function loadSectionStates() {
  try { return JSON.parse(localStorage.getItem(SECTION_STORAGE_KEY)) || {}; }
  catch { return {}; }
}

function saveSectionState(section, collapsed) {
  const states = loadSectionStates();
  states[section] = collapsed;
  localStorage.setItem(SECTION_STORAGE_KEY, JSON.stringify(states));
}

function toggleSection(section) {
  const heading = document.querySelector(`.collapsible-heading[data-section="${section}"]`);
  const body = document.querySelector(`.collapsible-body[data-section="${section}"]`);
  if (!heading || !body) return;
  const collapsed = body.classList.toggle('hidden');
  heading.classList.toggle('collapsed', collapsed);
  saveSectionState(section, collapsed);
}

function restoreSectionStates() {
  const states = loadSectionStates();
  // Main content sections
  document.querySelectorAll('.collapsible-heading[data-section]').forEach(heading => {
    const section = heading.dataset.section;
    const body = document.querySelector(`.collapsible-body[data-section="${section}"]`);
    if (!body) return;
    if (states[section]) {
      body.classList.add('hidden');
      heading.classList.add('collapsed');
    }
    heading.addEventListener('click', () => toggleSection(section));
  });
  // Sidebar downloads section
  const dlToggle = document.getElementById('dl-section-toggle');
  const dlPanel = document.getElementById('downloads-panel');
  if (dlToggle && dlPanel) {
    if (states.downloads) {
      dlPanel.classList.add('hidden');
      dlToggle.classList.add('collapsed');
    }
  }
}

restoreSectionStates();

// -------------------------------------------------------------------------
// Confirm Modal
// -------------------------------------------------------------------------
function showConfirm(title, message) {
  return new Promise((resolve) => {
    const overlay = document.getElementById('confirm-modal');
    document.getElementById('confirm-modal-title').textContent = title;
    document.getElementById('confirm-modal-message').textContent = message;
    overlay.classList.add('open');

    const cleanup = (result) => {
      overlay.classList.remove('open');
      btnOk.removeEventListener('click', onOk);
      btnCancel.removeEventListener('click', onCancel);
      btnClose.removeEventListener('click', onCancel);
      overlay.removeEventListener('click', onOverlay);
      resolve(result);
    };
    const onOk = () => cleanup(true);
    const onCancel = () => cleanup(false);
    const onOverlay = (e) => { if (e.target === overlay) cleanup(false); };

    const btnOk = document.getElementById('btn-confirm-ok');
    const btnCancel = document.getElementById('btn-confirm-cancel');
    const btnClose = document.getElementById('btn-confirm-close');
    btnOk.addEventListener('click', onOk);
    btnCancel.addEventListener('click', onCancel);
    btnClose.addEventListener('click', onCancel);
    overlay.addEventListener('click', onOverlay);
  });
}

// -------------------------------------------------------------------------
// Utilities
// -------------------------------------------------------------------------
async function apiFetch(url, opts) {
  const res = await fetch(url, opts);
  if (res.status === 401) { window.location.href = '/login'; return null; }
  return res;
}

// Encode a file path for use in a URL - encodes each segment but keeps slashes.
function encodePathForUrl(p) {
  return p.split('/').map(encodeURIComponent).join('/');
}

function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

function formatUptime(startedAt) {
  const secs = Math.floor(Date.now() / 1000 - startedAt);
  if (secs < 60) return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  return `${h}h ${m}m`;
}

async function nextAvailablePort() {
  try {
    const res = await apiFetch('/api/next-port');
    const data = await res.json();
    return data.port || 8000;
  } catch (e) {
    return 8000;
  }
}

function escHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
