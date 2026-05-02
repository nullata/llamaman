// Copyright (c) LlamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

// -------------------------------------------------------------------------
// Boot - event wiring, intervals, initial loads
// -------------------------------------------------------------------------
const refreshModelsBtn = document.getElementById('btn-refresh-models');
if (refreshModelsBtn) refreshModelsBtn.addEventListener('click', loadModels);

// Downloads sidebar section toggle
const dlSectionToggle = document.getElementById('dl-section-toggle');
if (dlSectionToggle) dlSectionToggle.addEventListener('click', () => {
  const panel = document.getElementById('downloads-panel');
  const collapsed = panel.classList.toggle('hidden');
  dlSectionToggle.classList.toggle('collapsed', collapsed);
  saveSectionState('downloads', collapsed);
});

const modelSearch = document.getElementById('model-search');
if (modelSearch) modelSearch.addEventListener('input', renderModels);

const refreshGpuBtn = document.getElementById('btn-refresh-gpu');
if (refreshGpuBtn) refreshGpuBtn.addEventListener('click', loadGpuInfo);

const refreshSystemBtn = document.getElementById('btn-refresh-system');
if (refreshSystemBtn) refreshSystemBtn.addEventListener('click', loadSystemInfo);

// Instance list refresh (every 5s, includes status from background poller)
setInterval(pollInstances, 5000);

// Container resource usage (CPU/memory) refresh (every 3s)
setInterval(pollContainerStats, 3000);

// Download status refresh (every 3s)
setInterval(pollDownloads, 3000);

// System info refresh (every 10s)
setInterval(loadSystemInfo, 10000);

// GPU VRAM refresh (every 10s)
setInterval(loadGpuInfo, 10000);

// Cleanup metadata refresh (every 60s)
setInterval(refreshCleanupLastRan, 60000);

// Initial load
loadModels();
loadSystemInfo();
loadGpuInfo();
pollInstances().then(() => { updatePortSuggestion(); pollContainerStats(); });

const params = new URLSearchParams(window.location.search);
const presetModelPath = params.get('model_path');
if (presetModelPath && modelPathField) {
  modelPathField.value = presetModelPath;
  if (typeof setActiveTab === 'function') setActiveTab('settings', 'launch');
  updateGpuLayersTotal(presetModelPath);
}

pollDownloads();
loadSettings();
loadApiKeys();
loadImages();

// Info-tip clipping fallback: when a centered tooltip would extend past the
// viewport, switch to anchoring it on the icon's right (or left) edge.
// Tooltip max-width is 240px; centered means up to 120px on either side.
function updateInfoTipClipping(tip) {
  const rect = tip.getBoundingClientRect();
  const center = rect.left + rect.width / 2;
  const halfMax = 120;
  const overflowsRight = center + halfMax > window.innerWidth;
  const overflowsLeft = center - halfMax < 0;
  tip.classList.toggle('clip-right', overflowsRight);
  tip.classList.toggle('clip-left', !overflowsRight && overflowsLeft);
}
document.addEventListener('mouseover', (e) => {
  const tip = e.target.closest && e.target.closest('.info-tip');
  if (tip) updateInfoTipClipping(tip);
});
document.addEventListener('focusin', (e) => {
  const tip = e.target.closest && e.target.closest('.info-tip');
  if (tip) updateInfoTipClipping(tip);
});
