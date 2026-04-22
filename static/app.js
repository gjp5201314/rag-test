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

function setStatusBadge(text, type) {
  statusBadge.textContent = text;
  statusBadge.className = `status-badge ${type}`;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = String(text);
  return div.innerHTML;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || '请求失败');
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

function updateStatusView(data) {
  indexPathEl.textContent = data.index_path || './data/index.json';
  loadedStateEl.textContent = data.loaded ? '已加载' : '未加载';
  documentCountEl.textContent = String(data.document_count ?? 0);
  apiStateEl.textContent = data.api_key_configured ? '已配置' : '未配置';

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
    buildMessageEl.textContent = error.message;
  }
}

buildForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(buildForm);
  const payload = Object.fromEntries(formData.entries());

  buildMessageEl.className = 'message-card muted';
  buildMessageEl.textContent = '正在扫描文件并构建索引，请稍候...';

  try {
    const data = await requestJson('/api/build', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    buildMessageEl.className = 'message-card';
    buildMessageEl.textContent = `索引构建完成：共 ${data.chunks} 个分片，索引文件为 ${data.index}`;
    askForm.elements.index.value = data.index;
    await loadStatus(data.index);
  } catch (error) {
    buildMessageEl.className = 'message-card';
    buildMessageEl.textContent = `构建失败：${error.message}`;
    setStatusBadge('异常', 'error');
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
