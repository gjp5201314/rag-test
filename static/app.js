const statusBadge = document.getElementById('statusBadge');
const indexPathEl = document.getElementById('indexPath');
const loadedStateEl = document.getElementById('loadedState');
const documentCountEl = document.getElementById('documentCount');
const apiStateEl = document.getElementById('apiState');
const buildMessageEl = document.getElementById('buildMessage');
const answerMetaEl = document.getElementById('answerMeta');
const answerContentEl = document.getElementById('answerContent');
const answerBlockEl = document.getElementById('answerBlock');
const resultsListEl = document.getElementById('resultsList');
const buildForm = document.getElementById('buildForm');
const askForm = document.getElementById('askForm');
const refreshStatusBtn = document.getElementById('refreshStatusBtn');

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
  indexPathEl.textContent = data.index_path || './data/index.json';
  loadedStateEl.textContent = data.loaded ? '已加载' : '未加载';
  documentCountEl.textContent = String(data.document_count ?? 0);
  apiStateEl.textContent = data.api_key_configured ? '已配置' : '未配置';

  if (data.build_job && ['queued', 'running'].includes(data.build_job.status)) {
    setStatusBadge('构建中', 'warn');
    if (data.build_job.message) {
      setBuildMessage(data.build_job.message, true);
    }
    if (activeBuildJobId !== data.build_job.job_id) {
      pollBuildJob(data.build_job.job_id)
        .then(async (job) => {
          const result = job.result || {};
          const targetIndex = result.index || data.build_job.index || data.index_path || './data/index.json';
          askForm.elements.index.value = targetIndex;
          await loadStatus(targetIndex);
        })
        .catch((error) => {
          setBuildMessage(`构建失败：${error.message}`);
          setStatusBadge('异常', 'error');
        });
    }
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

async function loadStatus(indexPath = './data/index.json') {
  try {
    const data = await requestJson(`/api/status?index=${encodeURIComponent(indexPath)}`);
    updateStatusView(data);
  } catch (error) {
    setStatusBadge('异常', 'error');
    setBuildMessage(error.message);
  }
}

buildForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(buildForm);
  const payload = Object.fromEntries(formData.entries());
  const submitButton = buildForm.querySelector('button[type="submit"]');
  const targetIndex = payload.index || './data/index.json';

  askForm.elements.index.value = targetIndex;
  setStatusBadge('构建中', 'warn');
  setBuildMessage('正在提交构建任务，请稍候...', true);
  if (submitButton) {
    submitButton.disabled = true;
  }

  try {
    const data = await requestJson('/api/build', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    const jobId = data.job_id;
    if (!jobId) {
      throw new Error('服务端未返回构建任务编号');
    }

    const job = await pollBuildJob(jobId);
    const result = job.result || {};
    const builtIndex = result.index || targetIndex;

    setBuildMessage(job.message || `索引构建完成：共 ${result.chunks ?? 0} 个分片，索引文件为 ${builtIndex}`);
    askForm.elements.index.value = builtIndex;
    await loadStatus(builtIndex);
  } catch (error) {
    setBuildMessage(`构建失败：${error.message}`);
    setStatusBadge('异常', 'error');
  } finally {
    if (submitButton) {
      submitButton.disabled = false;
    }
  }
});

askForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(askForm);
  const payload = Object.fromEntries(formData.entries());

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
    await loadStatus(payload.index || './data/index.json');
  } catch (error) {
    answerMetaEl.textContent = '请求失败';
    answerContentEl.textContent = error.message;
    renderResults([]);
    setStatusBadge('异常', 'error');
  }
});

refreshStatusBtn.addEventListener('click', () => {
  const currentIndex = askForm.elements.index.value || './data/index.json';
  loadStatus(currentIndex);
});

loadStatus();
