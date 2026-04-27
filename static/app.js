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
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const fileListEl = document.getElementById('fileList');
const buildBtn = document.getElementById('buildBtn');
const chunkSizeInput = document.getElementById('chunkSize');
const chunkOverlapInput = document.getElementById('chunkOverlap');

let currentKbId = 'default';
let askKbId = 'default';
let activeBuildJobId = null;
let activeBuildPollPromise = null;

function setStatusBadge(text, type) {
  statusBadge.textContent = text;
  statusBadge.className = `status-badge ${type}`;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = String(text);
  return div.innerHTML;
}

function setBuildMessage(message, muted = false) {
  buildMessageEl.className = muted ? 'message-card muted' : 'message-card';
  buildMessageEl.textContent = message;
}

function sleep(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  });

  const rawText = await response.text();
  let data = {};

  if (rawText) {
    try {
      data = JSON.parse(rawText);
    } catch (error) {
      if (!response.ok) {
        throw new Error(`请求失败（HTTP ${response.status}）`);
      }
      throw new Error('服务端返回了无效的 JSON 响应');
    }
  }

  if (!response.ok) {
    throw new Error(data.error || `请求失败（HTTP ${response.status}）`);
  }
  return data;
}

function renderResults(results) {
  if (!results || results.length === 0) {
    resultsListEl.className = 'results-list empty';
    resultsListEl.innerHTML = '<div class="empty-state">当前没有命中的文档分片。</div>';
    return;
  }

  resultsListEl.className = 'results-list';
  resultsListEl.innerHTML = results
    .map(
      (item) => `
        <article class="result-item">
          <div class="result-top">
            <div class="result-path">${escapeHtml(item.path)}#chunk-${item.chunk_id}</div>
            <div class="result-score">score ${Number(item.score).toFixed(4)}</div>
          </div>
          <div class="result-text">${escapeHtml(item.text)}</div>
        </article>
      `,
    )
    .join('');
}

function renderFileList(files) {
  if (!files || files.length === 0) {
    fileListEl.innerHTML = '<div class="empty-state">暂无上传文件</div>';
    return;
  }

  fileListEl.innerHTML = files
    .map(
      (file) => `
        <div class="file-item">
          <span class="file-name">${escapeHtml(file.filename)}</span>
          <span class="file-size">${(file.size / 1024).toFixed(1)} KB</span>
          <button class="delete-file-btn" data-filename="${escapeHtml(file.filename)}">删除</button>
        </div>
      `,
    )
    .join('');

  fileListEl.querySelectorAll('.delete-file-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const filename = btn.dataset.filename;
      if (confirm(`确定删除文件 "${filename}"？`)) {
        try {
          await requestJson(`/api/kb/${currentKbId}/files/${encodeURIComponent(filename)}`, {
            method: 'DELETE',
          });
          await loadFiles();
        } catch (error) {
          alert(`删除失败：${error.message}`);
        }
      }
    });
  });
}

function renderKbTabs(bases) {
  kbTabsEl.innerHTML = bases
    .map(
      (kb) => `
        <button type="button" class="kb-tab ${kb.id === currentKbId ? 'active' : ''}" data-kb-id="${escapeHtml(kb.id)}">
          <span class="kb-tab-name">${escapeHtml(kb.name)}</span>
          <span class="kb-tab-count">${kb.chunk_count} 分片</span>
        </button>
      `,
    )
    .join('');

  askKbTabsEl.innerHTML = bases
    .map(
      (kb) => `
        <button type="button" class="kb-tab ${kb.id === askKbId ? 'active' : ''}" data-kb-id="${escapeHtml(kb.id)}">
          <span class="kb-tab-name">${escapeHtml(kb.name)}</span>
        </button>
      `,
    )
    .join('');

  kbTabsEl.querySelectorAll('.kb-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      currentKbId = tab.dataset.kbId;
      renderKbTabs(bases);
      loadStatus();
      loadFiles();
    });
  });

  askKbTabsEl.querySelectorAll('.kb-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      askKbId = tab.dataset.kbId;
      renderKbTabs(bases);
    });
  });
}

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
      await sleep(1500);
    }
  })();

  try {
    return await activeBuildPollPromise;
  } finally {
    if (activeBuildJobId === jobId) {
      activeBuildJobId = null;
      activeBuildPollPromise = null;
    }
  }
}

function updateStatusView(data) {
  currentKbNameEl.textContent = data.kb_info?.name || currentKbId;
  loadedStateEl.textContent = data.loaded ? '已加载' : '未加载';
  documentCountEl.textContent = String(data.document_count ?? 0);
  apiStateEl.textContent = data.api_key_configured ? '已配置' : '未配置';

  if (data.kb_info) {
    currentKbTitleEl.textContent = data.kb_info.name;
    if (data.kb_info.id === 'default') {
      deleteKbBtn.classList.add('hidden');
    } else {
      deleteKbBtn.classList.remove('hidden');
    }
  }

  if (data.build_job && ['queued', 'running'].includes(data.build_job.status)) {
    setStatusBadge('构建中', 'warn');
    if (data.build_job.message) {
      setBuildMessage(data.build_job.message, true);
    }
    if (activeBuildJobId !== data.build_job.job_id) {
      pollBuildJob(data.build_job.job_id)
        .then(async () => {
          await loadStatus();
          await loadKbList();
        })
        .catch((error) => {
          setBuildMessage(`构建失败：${error.message}`);
          setStatusBadge('异常', 'error');
        });
    }
    return;
  }

  if (data.warming_up) {
    setStatusBadge('预热中', 'warn');
    setBuildMessage('正在后台预加载索引，首次提问会更快，请稍候...', true);
    return;
  }

  if (data.loaded && data.api_key_configured) {
    setStatusBadge('就绪', 'ok');
  } else if (data.index_exists) {
    setStatusBadge('待配置', 'warn');
  } else {
    setStatusBadge('未建立', 'idle');
  }
}

async function loadStatus() {
  try {
    const data = await requestJson(`/api/status?kb_id=${encodeURIComponent(currentKbId)}`);
    updateStatusView(data);
  } catch (error) {
    setStatusBadge('异常', 'error');
    setBuildMessage(error.message);
  }
}

async function loadKbList() {
  try {
    const bases = await requestJson('/api/kb/list');
    renderKbTabs(bases);
  } catch (error) {
    console.error('加载知识库列表失败:', error);
  }
}

async function loadFiles() {
  try {
    const files = await requestJson(`/api/kb/${currentKbId}/files`);
    renderFileList(files);
  } catch (error) {
    fileListEl.innerHTML = '<div class="empty-state">加载文件列表失败</div>';
  }
}

createKbBtn.addEventListener('click', () => {
  createKbForm.classList.remove('hidden');
  newKbNameInput.focus();
});

cancelCreateKb.addEventListener('click', () => {
  createKbForm.classList.add('hidden');
  newKbNameInput.value = '';
  newKbDescInput.value = '';
});

confirmCreateKb.addEventListener('click', async () => {
  const name = newKbNameInput.value.trim();
  const description = newKbDescInput.value.trim();

  if (!name) {
    alert('请输入知识库名称');
    return;
  }

  try {
    const kb = await requestJson('/api/kb/create', {
      method: 'POST',
      body: JSON.stringify({ name, description }),
    });

    createKbForm.classList.add('hidden');
    newKbNameInput.value = '';
    newKbDescInput.value = '';

    await loadKbList();
    currentKbId = kb.id;
    await loadStatus();
    await loadFiles();
  } catch (error) {
    alert(`创建失败：${error.message}`);
  }
});

deleteKbBtn.addEventListener('click', async () => {
  if (currentKbId === 'default') {
    alert('不能删除默认知识库');
    return;
  }

  const kbName = currentKbTitleEl.textContent;
  if (!confirm(`确定删除知识库 "${kbName}"？`)) {
    return;
  }

  try {
    await requestJson(`/api/kb/${currentKbId}`, { method: 'DELETE' });
    currentKbId = 'default';
    await loadKbList();
    await loadStatus();
    await loadFiles();
  } catch (error) {
    alert(`删除失败：${error.message}`);
  }
});

uploadBtn.addEventListener('click', async () => {
  const files = fileInput.files;
  if (!files || files.length === 0) {
    alert('请选择要上传的文件');
    return;
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append('file', file);
  }

  try {
    setBuildMessage('正在上传文件...', true);

    const response = await fetch(`/api/kb/${currentKbId}/upload`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || '上传失败');
    }

    setBuildMessage(`文件上传成功`, true);
    fileInput.value = '';
    await loadFiles();
  } catch (error) {
    setBuildMessage(`上传失败：${error.message}`);
  }
});

buildBtn.addEventListener('click', async () => {
  const chunkSize = parseInt(chunkSizeInput.value, 10) || 1000;
  const chunkOverlap = parseInt(chunkOverlapInput.value, 10) || 200;

  setStatusBadge('构建中', 'warn');
  setBuildMessage('正在提交构建任务，请稍候...', true);
  buildBtn.disabled = true;

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

    const job = await pollBuildJob(jobId);
    setBuildMessage(job.message || '索引构建完成');
    await loadStatus();
    await loadKbList();
  } catch (error) {
    setBuildMessage(`构建失败：${error.message}`);
    setStatusBadge('异常', 'error');
  } finally {
    buildBtn.disabled = false;
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

  try {
    const data = await requestJson('/api/ask', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    answerMetaEl.textContent = `已命中 ${data.results.length} 个分片`;
    answerContentEl.textContent = data.answer;
    renderResults(data.results);
    await loadStatus();
  } catch (error) {
    answerMetaEl.textContent = '请求失败';
    answerContentEl.textContent = error.message;
    renderResults([]);
    setStatusBadge('异常', 'error');
  }
});

refreshStatusBtn.addEventListener('click', () => {
  loadStatus();
  loadKbList();
});

loadKbList();
loadStatus();
loadFiles();
