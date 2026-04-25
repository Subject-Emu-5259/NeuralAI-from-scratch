// NeuralAI Chat UI — Full Feature Set

const chatContainer = document.getElementById('chatContainer');
const messagesEl = document.getElementById('messages');
const welcomeScreen = document.getElementById('welcomeScreen');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const clearBtn = document.getElementById('clearBtn');
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const searchInput = document.getElementById('searchInput');
const chatHistory = document.getElementById('chatHistory');
const exportBtn = document.getElementById('exportBtn');
const infoBtn = document.getElementById('infoBtn');
const themeBtn = document.getElementById('themeBtn');
const attachBtn = document.getElementById('attachBtn');
const msgCountEl = document.getElementById('msgCount');
const modelLabel = document.getElementById('modelLabel');

let conversation = [];
let isStreaming = false;
let isDark = true;
let historyItems = [{ id: 'new', text: 'New Conversation' }];

// ── Theme Toggle ───────────────────────────────────────────────
themeBtn?.addEventListener('click', () => {
  isDark = !isDark;
  document.documentElement.style.setProperty('--bg', isDark ? '#07070c' : '#f5f5f5');
  document.documentElement.style.setProperty('--surface', isDark ? '#0f0f17' : '#ffffff');
  document.documentElement.style.setProperty('--text', isDark ? '#e4e4e7' : '#1a1a1a');
  showToast(isDark ? 'Dark mode' : 'Light mode', 'success');
});

// ── Info Modal ─────────────────────────────────────────────────
infoBtn?.addEventListener('click', () => {
  const modal = document.createElement('div');
  modal.className = 'toast';
  modal.style.cssText = 'bottom:auto;top:80px;right:24px;max-width:320px;animation:none;opacity:1;';
  modal.innerHTML = `
    <div style="font-weight:700;font-size:14px;margin-bottom:8px;">🧠 NeuralAI Model</div>
    <div style="font-size:12px;color:var(--text-muted);line-height:1.7;">
      <div><strong>Base:</strong> SmolLM2-360M-Instruct</div>
      <div><strong>Training:</strong> QLoRA fine-tuning</div>
      <div><strong>Format:</strong> ChatML template</div>
      <div><strong>Device:</strong> CPU inference</div>
      <div><strong>Context:</strong> 2048 tokens</div>
    </div>
  `;
  document.body.appendChild(modal);
  setTimeout(() => modal.remove(), 5000);
});

// ── Export Chat ────────────────────────────────────────────────
exportBtn?.addEventListener('click', () => {
  if (conversation.length === 0) { showToast('No conversation to export', 'error'); return; }
  const md = conversation.map(m => `**${m.role}:** ${m.content}`).join('\n\n');
  const blob = new Blob([`# NeuralAI Chat\n\n${md}\n\n*Exported: ${new Date().toLocaleString()}*`], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = `neuralai-chat-${Date.now()}.md`; a.click();
  URL.revokeObjectURL(url);
  showToast('Chat exported!', 'success');
});

// ── Sidebar Toggle ─────────────────────────────────────────────
function openSidebar() { sidebar.classList.add('open'); sidebarOverlay.classList.add('open'); }
function closeSidebar() { sidebar.classList.remove('open'); sidebarOverlay.classList.remove('open'); }
sidebarToggle?.addEventListener('click', openSidebar);
sidebarOverlay?.addEventListener('click', closeSidebar);
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSidebar(); });

// ── Search Conversations ───────────────────────────────────────
searchInput?.addEventListener('input', () => {
  const q = searchInput.value.toLowerCase();
  document.querySelectorAll('.history-item').forEach(item => {
    const text = item.querySelector('.history-item-text')?.textContent || '';
    item.style.display = text.toLowerCase().includes(q) ? 'flex' : 'none';
  });
});

// ── Add History Item ────────────────────────────────────────────
function addHistory(text) {
  const label = historyLabel();
  let existing = document.querySelector(`.history-item[data-id="${label}"]`);
  if (existing) existing.remove();
  const div = document.createElement('button');
  div.className = 'history-item';
  div.dataset.id = label;
  div.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg><span class="history-item-text">${escHtml(text.slice(0, 40))}</span>`;
  div.addEventListener('click', () => {
    showToast('Conversation loaded from history', 'success');
    closeSidebar();
  });
  const group = chatHistory.querySelector(`.history-label`)?.nextElementSibling;
  if (group) chatHistory.insertBefore(div, group.nextElementSibling);
}

function historyLabel() {
  const now = new Date();
  if (now.getHours() < 12) return 'today-morning';
  if (now.getHours() < 18) return 'today-afternoon';
  return 'today-evening';
}

// ── Toast Notifications ───────────────────────────────────────
function showToast(msg, type = '') {
  const old = document.querySelector('.toast');
  if (old) old.remove();
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span> ${escHtml(msg)}`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

// ── Input Handling ─────────────────────────────────────────────
chatInput?.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
  const len = chatInput.value.length;
  const tokens = Math.ceil(len / 4);
  sendBtn.disabled = chatInput.value.trim() === '' || isStreaming;
});

chatInput?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!sendBtn.disabled) sendMessage(); }
});

sendBtn?.addEventListener('click', () => { if (!sendBtn.disabled) sendMessage(); });

// ── Quick Prompts ─────────────────────────────────────────────
document.querySelectorAll('.prompt-card').forEach(btn => {
  btn.addEventListener('click', () => {
    chatInput.value = btn.dataset.prompt;
    chatInput.dispatchEvent(new Event('input'));
    sendMessage();
  });
});

// ── New Chat ───────────────────────────────────────────────────
newChatBtn?.addEventListener('click', () => {
  conversation = [];
  messagesEl.innerHTML = '';
  welcomeScreen.style.display = 'flex';
  chatInput.value = '';
  chatInput.style.height = 'auto';
  chatInput.dispatchEvent(new Event('input'));
  closeSidebar();
});

// ── Clear Chat ─────────────────────────────────────────────────
document.getElementById('clearChatBtn')?.addEventListener('click', () => {
  conversation = [];
  messagesEl.innerHTML = '';
  welcomeScreen.style.display = 'flex';
  showToast('Chat cleared', 'success');
  closeSidebar();
});

// ── RAG Attach ──────────────────────────────────────────────────
attachBtn?.addEventListener('click', () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.pdf,.docx,.txt,.md';
  input.onchange = () => {
    if (input.files[0]) {
      showToast(`📄 "${input.files[0].name}" attached — RAG processing coming soon!`, 'success');
    }
  };
  input.click();
});

// ── Send Message ───────────────────────────────────────────────
async function sendMessage() {
  const userMsg = chatInput.value.trim();
  if (!userMsg || isStreaming) return;

  welcomeScreen.style.display = 'none';
  addMsg('user', userMsg);
  conversation.push({ role: 'user', content: userMsg });
  chatInput.value = '';
  chatInput.style.height = 'auto';
  chatInput.dispatchEvent(new Event('input'));
  updateMsgCount();

  isStreaming = true;
  sendBtn.disabled = false;
  sendBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';

  const assistantEl = addMsg('assistant', '');
  const bubbleEl = assistantEl.querySelector('.msg-bubble');
  bubbleEl.innerHTML = '<div class="thinking"><div class="typing-dots"><span></span><span></span><span></span></div> Thinking...</div>';
  scrollBottom();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: userMsg, temperature: 0.7, max_tokens: 256, messages: conversation })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let full = '';

    bubbleEl.innerHTML = '';
    isStreaming = true;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = dec.decode(value, { stream: true });
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim();
          if (raw === '[DONE]' || raw === 'data: [DONE]') { isStreaming = false; break; }
          try {
            const parsed = JSON.parse(raw);
            if (parsed.content) {
              full += parsed.content;
              bubbleEl.innerHTML = fmt(full) + copyBtn();
              scrollBottom();
            }
          } catch { /* skip invalid */ }
        }
      }
    }

    conversation.push({ role: 'assistant', content: full });
    addHistory(userMsg);
    updateMsgCount();

  } catch (err) {
    bubbleEl.innerHTML = `<span style="color:#ef4444">⚠️ Error: ${err.message}</span>`;
  } finally {
    isStreaming = false;
    sendBtn.disabled = chatInput.value.trim() === '';
    sendBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';
    scrollBottom();
  }
}

// ── Add Message ────────────────────────────────────────────────
function addMsg(role, content, placeholder = false) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  const avatar = role === 'assistant'
    ? `<div class="msg-avatar"><svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M12 2a4 4 0 0 1 4 4c0 1.1-.45 2.1-1.17 2.83L12 12l-2.83-3.17A4 4 0 0 1 12 2z"/><path d="M12 12v10"/><circle cx="12" cy="8" r="1.5" fill="white" stroke="none"/><path d="M8 6a4 4 0 0 1 4-4"/><path d="M16 6a4 4 0 0 0-4-4"/></svg></div>`
    : `<div class="msg-avatar"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>`;
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  div.innerHTML = `
    ${avatar}
    <div class="msg-content">
      <div class="msg-meta"><span class="msg-role">${role === 'assistant' ? 'NeuralAI' : 'You'}</span><span class="msg-time">${time}</span></div>
      <div class="msg-bubble">${placeholder ? '' : escHtml(content)}</div>
    </div>`;
  messagesEl.appendChild(div);
  scrollBottom();
  return div;
}

function copyBtn() {
  return `<div class="msg-actions"><button class="copy-btn" onclick="copyMsg(this)"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy</button></div>`;
}

function copyMsg(btn) {
  const text = btn.closest('.msg-bubble').innerText.replace(/Copy$/, '').trim();
  navigator.clipboard.writeText(text).then(() => {
    btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>Copied!';
    setTimeout(() => btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy', 2000);
  });
}

function updateMsgCount() {
  if (msgCountEl) msgCountEl.textContent = `${conversation.filter(m => m.role !== 'system').length} messages`;
}

// ── Markdown Formatter ──────────────────────────────────────────
function escHtml(text) {
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function fmt(text) {
  let out = escHtml(text);
  out = out.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`);
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  out = out.replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>');
  out = out.replace(/(<li>.*<\/li>)/s, '<ol>$1</ol>');
  out = out.replace(/\n/g, '<br>');
  return out;
}

function scrollBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Init
chatInput?.dispatchEvent(new Event('input'));