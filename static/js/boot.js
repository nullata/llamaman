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

// Download status refresh (every 3s)
setInterval(pollDownloads, 3000);

// System info refresh (every 10s)
setInterval(loadSystemInfo, 10000);

// GPU VRAM refresh (every 15s)
setInterval(loadGpuInfo, 15000);

// Initial load
loadModels();
loadSystemInfo();
loadGpuInfo();
pollInstances().then(updatePortSuggestion);

const params = new URLSearchParams(window.location.search);
const presetModelPath = params.get('model_path');
if (presetModelPath && modelPathField) {
  modelPathField.value = presetModelPath;
  updateGpuLayersTotal(presetModelPath);
}

pollDownloads();
loadSettings();
loadApiKeys();
