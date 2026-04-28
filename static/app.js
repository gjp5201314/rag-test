'use strict';

const serviceBase = '';
const statusBadge = document.getElementById('statusBadge');
const currentKbNameEl = document.getElementById('currentKbName');
const loadedStateEl = document.getElementById('loadedState');
const documentCountEl = document.getElementById('documentCount');
const apiStateEl = document.getElementById('apiState');
const buildMessageEl = document.getElementById('buildMessage');
const answerMetaEl = document.getElementById('answerMeta');
const answerContentEl = document.getElementById('answerContent');
const answerBlockEl = document.getElementById('answerBlock');
const resultsListEl = document.getElementById('resultsList');
const askForm = document.getElementById('askForm');
const refreshStatusBtn = document.getElementById('refreshStatusBtn');

const progressContainer = document.getElementById('progressContainer');
const progressLabel = document.getElementById('progressLabel');
const progressPercent = document.getElementById('progressPercent');
const progressFill = document.getElementById('progressFill');

const answerProgressContainer = document.getElementById('answerProgressContainer');
const answerProgressLabel = document.getElementById('answerProgressLabel');
const answerProgressPercent = document.getElementById('answerProgressPercent');
const answerProgressFill = document.getElementById('answerProgressFill');

const kbTabsEl = document.getElementById('kbTabs');
const askKbTabsEl = document.getElementById('askKbTabs');
const createKbBtn = document.getElementById('createKbBtn');
const deleteKbBtn = document.getElementById('deleteKbBtn');
const createKbForm = document.getElementById('createKbForm');
const newKbNameInput = document.getElementById('newKbName');
const newKbDescInput = document.getElementById('newKbDesc');
const confirmCreateKb = document.getElementById('confirmCreateKb');
const cancelCreateKb = document.getElementById('cancelCreateKb');
const currentKbTitleEl = document.getElementById('currentKbTitle');
const currentKbDescEl = document.getElementById('currentKbDesc');
const fileInput = document.getElementById('fileInput');
const uploadDropZone = document.getElementById('uploadDropZone');
const fileListEl = document.getElementById('fileList');
const buildBtn = document.getElementById('buildBtn');
const chunkSizeInput = document.getElementById('chunkSize');
const chunkOverlapInput = document.getElementById('chunkOverlap');
const questionInput = document.getElementById('questionInput');
const charCountEl = document.getElementById('charCount');
const copyAnswerBtn = document.getElementById('copyAnswerBtn');
const collapseAllBtn = document.getElementById('collapseAllBtn');

let currentKbId = 'default';
let askKbId = 'default';
let activeBuildJobId = null;
let activeBuildPollPromise = null;
let buildProgressInterval = null;
let answerProgressInterval = null;
let isUploading = false;

// ============ Ensure Sandbox Modal Exists ============
(function ensureSandboxModal() {
  if (!document.getElementById('sandboxModal')) {
    const modalHtml = `
      <div id="sandboxModal" class="sandbox-modal hidden">
        <div class="sandbox-backdrop"></div>
        <div class="sandbox-dialog">
          <div class="sandbox-header">
            <span class="sandbox-title"></span>
            <button type="button" class="sandbox-close" onclick="hideSandboxModal()">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          </div>
          <div class="sandbox-body"></div>
          <div class="sandbox-footer"></div>
        </div>
      </div>
    `;
    const toastContainer = document.getElementById('toastContainer');
    if (toastContainer) {
      toastContainer.insertAdjacentHTML('afterend', modalHtml);
    }
  }
})();

// ============ Toast System ============

function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;

  const icons = {
    success: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',
    error: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
    info: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type]}</span>
    <span class="toast-message">${message}</span>
    <button class="toast-close" onclick="this.parentElement.remove()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
  `;

  container.appendChild(toast);

  if (duration > 0) {
    setTimeout(() => {
      toast.classList.add('toast-out');
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }
}

// ============ Sandbox Modal ============

let sandboxResolve = null;

function showSandboxModal(options = {}) {
  const {
    title = '提示',
    content = '',
    confirmText = '确定',
    cancelText = '取消',
    confirmClass = 'action-btn primary',
    showCancel = true,
    dangerous = false
  } = options;

  const modal = document.getElementById('sandboxModal');
  const titleEl = modal.querySelector('.sandbox-title');
  const bodyEl = modal.querySelector('.sandbox-body');
  const footerEl = modal.querySelector('.sandbox-footer');

  titleEl.textContent = title;
  bodyEl.innerHTML = `<div class="sandbox-body-content">${content}</div>`;

  const confirmClassFinal = dangerous ? 'action-btn danger' : confirmClass;
  footerEl.innerHTML = `
    ${showCancel ? `<button type="button" class="ghost-btn" onclick="hideSandboxModal(false)">${cancelText}</button>` : ''}
    <button type="button" class="${confirmClassFinal}" id="sandboxConfirmBtn">${confirmText}</button>
  `;

  document.getElementById('sandboxConfirmBtn').addEventListener('click', () => hideSandboxModal(true));

  modal.classList.remove('hidden');
  document.getElementById('sandboxConfirmBtn').focus();

  return new Promise((resolve) => {
    sandboxResolve = resolve;
  });
}

function hideSandboxModal(result = false) {
  const modal = document.getElementById('sandboxModal');
  modal.classList.add('hidden');

  if (sandboxResolve) {
    sandboxResolve(result);
    sandboxResolve = null;
  }
}

window.hideSandboxModal = hideSandboxModal;

// ============ Status & Progress ============

function setStatusBadge(text, type) {
  const statusText = statusBadge.querySelector('.status-text') || statusBadge;
  const spinner = statusBadge.querySelector('.spinner');

  if (statusText !== statusBadge) {
    statusText.textContent = text;
  }

  if (spinner) {
    if (type === 'warn') {
      spinner.classList.remove('hidden');
    } else {
      spinner.classList.add('hidden');
    }
  }

  statusBadge.className = `status-badge ${type}`;
}

function showBuildProgress(label = '正在构建索引...') {
  progressContainer.classList.remove('hidden');
  progressLabel.textContent = label;
  progressPercent.textContent = '0%';
  progressFill.style.width = '0%';

  let progress = 0;
  const phases = ['正在扫描文件...', '正在解析文档...', '正在创建索引...', '正在保存...'];

  if (buildProgressInterval) clearInterval(buildProgressInterval);

  buildProgressInterval = setInterval(() => {
    progress += Math.random() * 15;
    if (progress > 90) progress = 90;

    const phaseIndex = Math.min(Math.floor(progress / 25), phases.length - 1);
    progressLabel.textContent = phases[phaseIndex];
    progressPercent.textContent = `${Math.round(progress)}%`;
    progressFill.style.width = `${progress}%`;
  }, 500);
}

function hideBuildProgress() {
  if (buildProgressInterval) {
    clearInterval(buildProgressInterval);
    buildProgressInterval = null;
  }
  progressContainer.classList.add('hidden');
}

function setBuildProgress(percent, label) {
  if (percent !== undefined) {
    progressFill.style.width = `${percent}%`;
    progressPercent.textContent = `${percent}%`;
  }
  if (label !== undefined) {
    progressLabel.textContent = label;
  }
}

function showAnswerProgress(label = '正在检索并生成回答...') {
  answerProgressContainer.classList.remove('hidden');
  answerProgressLabel.textContent = label;
  answerProgressPercent.textContent = '0%';
  answerProgressFill.style.width = '0%';

  let progress = 0;

  if (answerProgressInterval) clearInterval(answerProgressInterval);

  answerProgressInterval = setInterval(() => {
    progress += Math.random() * 10;
    if (progress > 85) progress = 85;

    const phase = progress < 30 ? '正在检索文档...' :
                  progress < 60 ? '正在分析上下文...' :
                  progress < 80 ? '正在生成回答...' : '正在完成...';
    answerProgressLabel.textContent = phase;
    answerProgressPercent.textContent = `${Math.round(progress)}%`;
    answerProgressFill.style.width = `${progress}%`;
  }, 400);
}

function hideAnswerProgress() {
  if (answerProgressInterval) {
    clearInterval(answerProgressInterval);
    answerProgressInterval = null;
  }
  answerProgressContainer.classList.add('hidden');
}

// ============ API Helpers ============

async function requestJson(url, options = {}) {
  const res = await fetch(`${serviceBase}${url}`, {
    ...options,
    headers: {
      ...(options.body && !(options.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}),
      ...options.headers,
    },
    body: options.body,
  });

  const data = await res.json().catch(() => ({}));

  if (!res.ok) {
    throw new Error(data.error || `请求失败 (${res.status})`);
  }

  return data;
}

async function loadStatus() {
  try {
    const data = await requestJson(`/api/status?kb_id=${encodeURIComponent(currentKbId)}`);

    const isLoaded = data.loaded && data.index_exists;
    if (isLoaded) {
      setStatusBadge('就绪', 'ok');
    } else if (data.api_key_configured === false) {
      setStatusBadge('待配置', 'warn');
    } else {
      setStatusBadge('待配置', 'warn');
    }

    currentKbNameEl.textContent = (data.kb_info && data.kb_info.name) || currentKbId;
    loadedStateEl.textContent = data.loaded ? '已加载' : '未加载';
    documentCountEl.textContent = (data.kb_info && data.kb_info.chunk_count) ?? 0;
    apiStateEl.textContent = data.api_key_configured ? '已配置' : '未配置';

    return data;
  } catch (error) {
    setStatusBadge('异常', 'error');
    console.error('[Status]', error);
    return null;
  }
}

// ============ Knowledge Base Management ============

async function loadKbList() {
  try {
    const data = await requestJson('/api/kb/list');

    kbTabsEl.innerHTML = '';
    askKbTabsEl.innerHTML = '';

    const kbList = Array.isArray(data) ? data : (data.knowledge_bases || []);

    if (kbList.length === 0) {
      kbList.push({ id: 'default', name: '默认知识库', chunk_count: 0, description: '' });
    }

    kbList.forEach((kb) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `kb-tab${kb.id === currentKbId ? ' active' : ''}`;
      btn.dataset.kbId = kb.id;
      btn.innerHTML = `
        <span class="kb-tab-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
        </span>
        <span class="kb-tab-name">${kb.name}</span>
        <span class="kb-tab-count">${kb.chunk_count} 分片</span>
      `;
      btn.addEventListener('click', () => switchKb(kb.id));
      kbTabsEl.appendChild(btn);

      const askBtn = document.createElement('button');
      askBtn.type = 'button';
      askBtn.className = `kb-tab${kb.id === askKbId ? ' active' : ''}`;
      askBtn.dataset.kbId = kb.id;
      askBtn.innerHTML = `<span class="kb-tab-name">${kb.name}</span>`;
      askBtn.addEventListener('click', () => switchAskKb(kb.id));
      askKbTabsEl.appendChild(askBtn);
    });

    const currentKb = kbList.find((kb) => kb.id === currentKbId) || kbList[0];
    if (currentKb) {
      currentKbTitleEl.textContent = currentKb.name;
      currentKbDescEl.textContent = currentKb.description || '';
      currentKbDescEl.style.display = currentKb.description ? 'inline' : 'none';
      deleteKbBtn.classList.toggle('hidden', currentKbId === 'default');
    }

    await loadStatus();
    await loadFiles();
  } catch (error) {
    console.error('[KB List]', error);
  }
}

async function switchKb(kbId) {
  if (kbId === currentKbId) return;
  currentKbId = kbId;
  askKbId = kbId;

  kbTabsEl.querySelectorAll('.kb-tab').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.kbId === kbId);
  });

  await loadStatus();
  await loadFiles();
}

function switchAskKb(kbId) {
  if (kbId === askKbId) return;
  askKbId = kbId;

  askKbTabsEl.querySelectorAll('.kb-tab').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.kbId === kbId);
  });
}

// ============ KB Create / Delete ============

createKbBtn.addEventListener('click', () => {
  createKbForm.classList.remove('hidden');
  createKbBtn.style.display = 'none';
  newKbNameInput.focus();
});

cancelCreateKb.addEventListener('click', () => {
  createKbForm.classList.add('hidden');
  createKbBtn.style.display = '';
  newKbNameInput.value = '';
  newKbDescInput.value = '';
});

confirmCreateKb.addEventListener('click', async () => {
  const name = newKbNameInput.value.trim();
  if (!name) {
    showToast('请输入知识库名称', 'error');
    return;
  }

  try {
    await requestJson('/api/kb/create', {
      method: 'POST',
      body: JSON.stringify({ name, description: newKbDescInput.value.trim() }),
    });

    showToast(`知识库「${name}」创建成功`, 'success');
    cancelCreateKb.click();
    await loadKbList();

    const newKb = kbList?.find((kb) => kb.name === name);
    if (newKb) {
      await switchKb(newKb.id);
    }
  } catch (error) {
    showToast(`创建失败：${error.message}`, 'error');
  }
});

deleteKbBtn.addEventListener('click', async () => {
  if (currentKbId === 'default') return;

  const confirmed = await showSandboxModal({
    title: '删除知识库',
    content: `确定要删除知识库「${currentKbTitleEl.textContent}」吗？此操作不可恢复。`,
    confirmText: '删除',
    dangerous: true
  });

  if (!confirmed) return;

  try {
    await requestJson(`/api/kb/${encodeURIComponent(currentKbId)}`, { method: 'DELETE' });
    showToast('知识库已删除', 'success');
    currentKbId = 'default';
    askKbId = 'default';
    await loadKbList();
  } catch (error) {
    showToast(`删除失败：${error.message}`, 'error');
  }
});

// ============ File Upload (Drag & Drop) ============

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function getFileIcon(filename) {
  const ext = filename.split('.').pop().toLowerCase();
  const icons = {
    pdf: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    doc: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    docx: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>',
    txt: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    md: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    py: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
    js: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
  };
  return icons[ext] || icons.txt;
}

uploadDropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadDropZone.classList.add('dragover');
});

uploadDropZone.addEventListener('dragleave', (e) => {
  e.preventDefault();
  uploadDropZone.classList.remove('dragover');
});

uploadDropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadDropZone.classList.remove('dragover');
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFileUpload(files);
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    handleFileUpload(fileInput.files);
  }
});

async function handleFileUpload(files) {
  if (isUploading) {
    showToast('正在上传中，请稍候', 'info');
    return;
  }

  isUploading = true;
  const totalFiles = files.length;
  let uploadedCount = 0;

  for (const file of files) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      await requestJson(`/api/kb/${encodeURIComponent(currentKbId)}/upload`, {
        method: 'POST',
        body: formData,
      });

      uploadedCount++;
      showToast(`已上传 ${uploadedCount}/${totalFiles}: ${file.name}`, 'success', 2000);
    } catch (error) {
      showToast(`上传失败：${file.name} - ${error.message}`, 'error');
    }
  }

  isUploading = false;
  fileInput.value = '';
  await loadFiles();
}

// ============ File List ============

async function loadFiles() {
  try {
    const data = await requestJson(`/api/kb/${encodeURIComponent(currentKbId)}/files`);
    const files = Array.isArray(data) ? data : (data.files || []);

    if (files.length === 0) {
      fileListEl.innerHTML = `
        <div class="empty-state">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          <span>暂无上传文件</span>
        </div>
      `;
      return;
    }

    fileListEl.innerHTML = files
      .map(
        (f) => `
      <div class="file-item" data-filename="${f.filename}" data-kb-id="${currentKbId}">
        <div class="file-icon">${getFileIcon(f.filename)}</div>
        <div class="file-info">
          <div class="file-name">${f.filename}</div>
          <div class="file-meta">
            <span>${formatFileSize(f.size)}</span>
          </div>
        </div>
        <button type="button" class="delete-file-btn ghost-btn small danger" data-filename="${f.filename}" data-kb-id="${currentKbId}" title="删除文件">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
        </button>
      </div>
    `
      )
      .join('');

    fileListEl.querySelectorAll('.delete-file-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const filename = btn.dataset.filename;
        const kbId = btn.dataset.kbId;
        const confirmed = await showSandboxModal({
          title: '删除文件',
          content: `确定删除文件「${filename}」？`,
          confirmText: '删除',
          dangerous: true
        });
        if (!confirmed) return;
        try {
          await requestJson(`/api/kb/${kbId}/files?filename=${encodeURIComponent(filename)}`, {
            method: 'DELETE',
          });
          showToast('文件已删除', 'success');
          await loadFiles();
          await loadStatus();
        } catch (error) {
          showToast(`删除失败：${error.message}`, 'error');
        }
      });
    });
  } catch (error) {
    console.error('[Files]', error);
  }
}

// ============ Build Index ============

function setBuildMessage(text, loading = false) {
  buildMessageEl.textContent = text;
  buildMessageEl.className = `message-card ${loading ? 'info' : 'muted'}`;
}

buildBtn.addEventListener('click', async () => {
  const chunkSize = parseInt(chunkSizeInput.value, 10) || 1000;
  const chunkOverlap = parseInt(chunkOverlapInput.value, 10) || 200;

  setStatusBadge('构建中', 'warn');
  setBuildMessage('正在提交构建任务，请稍候...', true);
  buildBtn.disabled = true;
  showBuildProgress();

  try {
    const data = await requestJson('/api/build', {
      method: 'POST',
      body: JSON.stringify({
        kb_id: currentKbId,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
      }),
    });

    const jobId = data.job_id;
    if (!jobId) {
      throw new Error('服务端未返回构建任务编号');
    }

    setBuildProgress(95, '正在完成...');

    const job = await pollBuildJob(jobId);

    setBuildProgress(100, '完成！');
    setBuildMessage(job.message || '索引构建完成');
    hideBuildProgress();
    showToast('索引构建完成', 'success');

    await loadStatus();
    await loadKbList();
  } catch (error) {
    hideBuildProgress();
    setBuildMessage(`构建失败：${error.message}`);
    setStatusBadge('异常', 'error');
    showToast(`构建失败：${error.message}`, 'error');
  } finally {
    buildBtn.disabled = false;
  }
});

async function pollBuildJob(jobId) {
  if (activeBuildJobId === jobId && activeBuildPollPromise) {
    return activeBuildPollPromise;
  }

  activeBuildJobId = jobId;
  activeBuildPollPromise = (async () => {
    while (true) {
      const job = await requestJson(`/api/build-status?job_id=${encodeURIComponent(jobId)}`);

      if (job.message) {
        setBuildMessage(job.message, !['completed', 'failed'].includes(job.status));
      }

      if (job.status === 'completed') {
        return job;
      }

      if (job.status === 'failed') {
        throw new Error(job.error || job.message || '索引构建失败');
      }

      setStatusBadge('构建中', 'warn');
      setBuildProgress(undefined, job.message || '正在构建索引...');
      await sleep(1500);
    }
  })();

  try {
    return await activeBuildPollPromise;
  } finally {
    activeBuildJobId = null;
    activeBuildPollPromise = null;
  }
}

// ============ Ask Question ============

// Character count
questionInput.addEventListener('input', () => {
  const len = questionInput.value.length;
  charCountEl.textContent = `${len} / 1000`;
  charCountEl.style.color = len > 900 ? 'var(--accent-primary)' : '';
});

// Ctrl+Enter to submit
askForm.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    askForm.dispatchEvent(new Event('submit'));
  }
});

askForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(askForm);
  const payload = Object.fromEntries(formData.entries());
  payload.kb_id = askKbId;

  answerBlockEl.classList.remove('empty');
  answerMetaEl.textContent = '检索中';
  answerContentEl.textContent = '正在检索本地片段并请求千问模型，请稍候...';
  copyAnswerBtn.classList.add('hidden');
  showAnswerProgress();

  try {
    const data = await requestJson('/api/ask', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    answerProgressFill.style.width = '100%';
    answerProgressPercent.textContent = '100%';
    answerProgressLabel.textContent = '完成！';

    setTimeout(() => {
      hideAnswerProgress();
    }, 800);

    answerMetaEl.textContent = `已命中 ${data.results.length} 个分片`;
    answerContentEl.textContent = data.answer;
    copyAnswerBtn.classList.remove('hidden');
    renderResults(data.results);
    await loadStatus();
  } catch (error) {
    hideAnswerProgress();
    answerMetaEl.textContent = '请求失败';
    answerContentEl.textContent = error.message;
    renderResults([]);
    setStatusBadge('异常', 'error');
    showToast(`请求失败：${error.message}`, 'error');
  }
});

// Copy answer
copyAnswerBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(answerContentEl.textContent);
    showToast('回答已复制到剪贴板', 'success');
  } catch (error) {
    showToast('复制失败', 'error');
  }
});

// ============ Results ============

function renderResults(results) {
  collapseAllBtn.classList.toggle('hidden', results.length === 0);

  if (results.length === 0) {
    resultsListEl.innerHTML = `
      <div class="empty-state">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <span>未找到相关结果</span>
      </div>
    `;
    resultsListEl.classList.add('empty');
    return;
  }

  resultsListEl.classList.remove('empty');
  resultsListEl.innerHTML = results
    .map(
      (r, i) => `
    <div class="result-item collapsed" data-index="${i}">
      <div class="result-header">
        <div class="result-title">
          <span class="result-filename">${r.path}</span>
          <span class="result-chunk">#${r.chunk_id}</span>
        </div>
        <div class="result-score" title="相似度">
          <div class="score-bar">
            <div class="score-fill" style="width: ${(r.score * 100).toFixed(0)}%"></div>
          </div>
          <span class="score-value">${(r.score * 100).toFixed(1)}%</span>
        </div>
      </div>
      <div class="result-text">${r.text}</div>
      <button type="button" class="result-expand-btn">
        <span class="expand-text">展开</span>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
      <div class="result-actions">
        <button type="button" class="result-copy-btn" data-content="${(r.text || '').replace(/"/g, '&quot;')}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          复制片段
        </button>
      </div>
    </div>
  `
    )
    .join('');

  resultsListEl.querySelectorAll('.result-item').forEach((item) => {
    const expandBtn = item.querySelector('.result-expand-btn');
    expandBtn.addEventListener('click', () => {
      item.classList.toggle('collapsed');
      item.classList.toggle('expanded');
      const expandText = expandBtn.querySelector('.expand-text');
      expandText.textContent = item.classList.contains('collapsed') ? '展开' : '收起';
    });
  });

  resultsListEl.querySelectorAll('.result-copy-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const content = btn.dataset.content;
      try {
        await navigator.clipboard.writeText(content);
        btn.classList.add('copied');
        btn.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>
          已复制
        `;
        setTimeout(() => {
          btn.classList.remove('copied');
          btn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            复制片段
          `;
        }, 2000);
      } catch (error) {
        showToast('复制失败', 'error');
      }
    });
  });
}

// Collapse all
collapseAllBtn.addEventListener('click', () => {
  const allExpanded = resultsListEl.querySelectorAll('.result-item.expanded').length === resultsListEl.querySelectorAll('.result-item').length;

  resultsListEl.querySelectorAll('.result-item').forEach((item) => {
    if (allExpanded) {
      item.classList.remove('expanded');
      item.classList.add('collapsed');
      item.querySelector('.expand-text').textContent = '展开';
    } else {
      item.classList.remove('collapsed');
      item.classList.add('expanded');
      item.querySelector('.expand-text').textContent = '收起';
    }
  });

  collapseAllBtn.textContent = allExpanded ? '折叠全部' : '展开全部';
});

// ============ Refresh ============

refreshStatusBtn.addEventListener('click', async () => {
  await loadStatus();
  await loadFiles();
  showToast('状态已刷新', 'info', 2000);
});

// ============ Utilities ============

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ============ Init ============

loadKbList();
