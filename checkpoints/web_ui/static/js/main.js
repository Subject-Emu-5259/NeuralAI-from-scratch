// NeuralAI Chat UI — Full Feature Set
const chatContainer = document.getElementById('chatContainer');
const messagesEl = document.getElementById('messages');
const welcomeScreen = document.getElementById('welcomeScreen');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const searchInput = document.getElementById('searchInput');
const exportBtn = document.getElementById('exportBtn');
const infoBtn = document.getElementById('infoBtn');
const themeBtn = document.getElementById('themeBtn');
const attachBtn = document.getElementById('attachBtn');
const msgCountEl = document.getElementById('msgCount');

let conversation = [];
let isStreaming = false;
let isDark = true;
let attachedFiles = {};

function escHtml(text) { return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function fmt(text) {
  let out = escHtml(text);
  out = out.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, l, c) => `<pre><code class="lang-${l}">${c.trim()}</code></pre>`);
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  out = out.replace(/\n/g, '<br>');
  return out;
}

function scrollBottom() { chatContainer.scrollTop = chatContainer.scrollHeight; }

function showToast(msg, type = '') {
  const old = document.querySelector('.toast');
  if (old) old.remove();
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span> ${msg}`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

// Theme
themeBtn?.addEventListener('click', () => {
  isDark = !isDark;
  document.documentElement.style.setProperty('--bg', isDark ? '#07070c' : '#f5f5f5');
  document.documentElement.style.setProperty('--surface', isDark ? '#0f0f17' : '#ffffff');
  document.documentElement.style.setProperty('--text', isDark ? '#e4e4e7' : '#1a1a1a');
  showToast(isDark ? 'Dark mode' : 'Light mode', 'success');
});

// Info
infoBtn?.addEventListener('click', () => {
  fetch('/api/status').then(r => r.json()).then(data => {
    const m = document.createElement('div');
    m.style.cssText = 'position:fixed;top:70px;right:16px;width:300px;background:var(--surface,#0f0f17);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:16px;z-index:200;box-shadow:0 8px 32px rgba(0,0,0,0.5);animation:none;opacity:1;';
    m.innerHTML = `<div style="font-weight:700;font-size:15px;margin-bottom:12px;">🧠 NeuralAI</div><div style="font-size:12px;color:var(--text-muted,#6b6b76);line-height:2;"><div><strong>Base:</strong> ${data.model||'SmolLM2-360M-Instruct'}</div><div><strong>Type:</strong> ${data.model_type||'base'}</div><div><strong>Device:</strong> ${data.device||'CPU'}</div><div><strong>Version:</strong> ${data.version||'2.1'}</div><div><strong>RAG:</strong> ${data.rag?'✅ Active':'❌ Off'}</div><div><strong>Files:</strong> ${data.indexed_files||0}</div><div><strong>Attached:</strong> ${Object.keys(attachedFiles).length}</div></div>`;
    document.body.appendChild(m);
    setTimeout(() => m.remove(), 6000);
  });
});

// Export
exportBtn?.addEventListener('click', () => {
  if (conversation.length === 0) { showToast('No conversation to export', 'error'); return; }
  const md = conversation.filter(m => m.role !== 'system').map(m => `**${m.role}:** ${m.content}`).join('\n\n');
  const blob = new Blob([`# NeuralAI Chat\n\n${md}\n\n*Exported: ${new Date().toLocaleString()}*`], { type: 'text/markdown' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `neuralai-chat-${Date.now()}.md`; a.click();
  URL.revokeObjectURL(a.href);
  showToast('Chat exported!', 'success');
});

// Sidebar
function openSidebar() { sidebar.classList.add('open'); sidebarOverlay.classList.add('open'); }
function closeSidebar() { sidebar.classList.remove('open'); sidebarOverlay.classList.remove('open'); }
sidebarToggle?.addEventListener('click', openSidebar);
sidebarOverlay?.addEventListener('click', closeSidebar);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeSidebar(); });

// Search
searchInput?.addEventListener('input', () => {
  const q = searchInput.value.toLowerCase();
  document.querySelectorAll('.history-item').forEach(item => {
    const t = item.querySelector('.history-item-text')?.textContent || '';
    item.style.display = t.toLowerCase().includes(q) ? 'flex' : 'none';
  });
});

// Attached files bar
function updateAttachedUI() {
  const existing = document.getElementById('attachedFiles');
  if (existing) existing.remove();
  const ids = Object.keys(attachedFiles);
  if (ids.length === 0) return;
  const div = document.createElement('div');
  div.id = 'attachedFiles';
  div.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px;padding:8px 16px;background:var(--surface-2,#14141f);border-top:1px solid var(--border,rgba(255,255,255,0.06));';
  ids.forEach(fid => {
    const name = attachedFiles[fid];
    const chip = document.createElement('span');
    chip.style.cssText = 'display:inline-flex;align-items:center;gap:4px;padding:3px 10px;background:var(--accent-glow,rgba(124,58,237,0.15));border:1px solid rgba(124,58,237,0.3);border-radius:20px;font-size:11px;color:var(--accent-text,#a78bfa);';
    chip.textContent = '📄 ' + (name.length > 20 ? name.slice(0,18) + '…' : name);
    div.appendChild(chip);
  });
  document.querySelector('.input-area')?.parentElement?.insertBefore(div, document.querySelector('.input-area'));
}

// File upload / RAG
attachBtn?.addEventListener('click', () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = '.pdf,.docx,.doc,.txt,.md';
  input.onchange = async () => {
    if (!input.files[0]) return;
    const file = input.files[0];
    const formData = new FormData();
    formData.append('file', file);
    showToast(`📄 Uploading "${file.name}"…`);
    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.error) { showToast('❌ ' + data.error, 'error'); return; }
      attachedFiles[data.file_id] = data.filename;
      showToast(data.message, 'success');
      updateAttachedUI();
      updateMsgCount();
    } catch (err) { showToast('❌ Upload failed: ' + err.message, 'error'); }
  };
  input.click();
});

// New chat
newChatBtn?.addEventListener('click', () => {
  conversation = [];
  messagesEl.innerHTML = '';
  welcomeScreen.style.display = 'flex';
  chatInput.value = '';
  chatInput.style.height = 'auto';
  attachedFiles = {};
  const af = document.getElementById('attachedFiles');
  if (af) af.remove();
  chatInput.dispatchEvent(new Event('input'));
  closeSidebar();
  updateMsgCount();
});

// Input
chatInput?.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
  sendBtn.disabled = chatInput.value.trim() === '' || isStreaming;
});

chatInput?.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!sendBtn.disabled) sendMessage(); }
});

sendBtn?.addEventListener('click', () => { if (!sendBtn.disabled) sendMessage(); });

// Quick prompts
document.querySelectorAll('.prompt-card').forEach(btn => {
  btn.addEventListener('click', () => {
    chatInput.value = btn.dataset.prompt;
    chatInput.dispatchEvent(new Event('input'));
    sendMessage();
  });
});

// Send message
async function sendMessage() {
  const userMsg = chatInput.value.trim();
  if (!userMsg || isStreaming) return;

  welcomeScreen.style.display = 'none';
  const fileIdsThisMsg = Object.keys(attachedFiles);

  addMsg('user', userMsg, fileIdsThisMsg);
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
  bubbleEl.innerHTML = '<div class="thinking"><div class="typing-dots"><span></span><span></span><span></span></div> Thinking…</div>';
  scrollBottom();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: userMsg, temperature: 0.7, max_tokens: 512, messages: conversation, file_ids: fileIdsThisMsg })
    });

    if (!res.ok) throw new Error('HTTP ' + res.status);

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let full = '';
    bubbleEl.innerHTML = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = dec.decode(value, { stream: true });
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') { isStreaming = false; break; }
          try {
            const parsed = JSON.parse(raw);
            if (parsed.content) {
              full += parsed.content;
              bubbleEl.innerHTML = fmt(full) + copyBtn();
              scrollBottom();
            }
          } catch {}
        }
      }
    }

    conversation.push({ role: 'assistant', content: full });

  } catch (err) {
    bubbleEl.innerHTML = '<span style="color:#ef4444">⚠️ Error: ' + err.message + '</span>';
  } finally {
    isStreaming = false;
    sendBtn.disabled = chatInput.value.trim() === '';
    sendBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';
    scrollBottom();
    closeSidebar();
  }
}

// Add message
function addMsg(role, content, fileIds) {
  const div = document.createElement('div');
  div.className = 'message ' + role;

  let attachedInfo = '';
  if (role === 'user' && fileIds && fileIds.length > 0) {
    const names = fileIds.map(fid => attachedFiles[fid] || fid);
    attachedInfo = '<div style="font-size:11px;color:var(--text-dim,#3e3e50);margin-top:4px;display:flex;flex-wrap:wrap;gap:4px;">' +
      names.map(n => '<span style="background:var(--accent-glow,rgba(124,58,237,0.1));padding:1px 8px;border-radius:10px;">📄 ' + escHtml(n.length > 22 ? n.slice(0,20)+'…' : n) + '</span>').join('') + '</div>';
  }

  const avatar = role === 'assistant'
    ? '<div class="msg-avatar"><svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2"><path d="M12 2a4 4 0 0 1 4 4c0 1.1-.45 2.1-1.17 2.83L12 12l-2.83-3.17A4 4 0 0 1 12 2z"/><path d="M12 12v10"/><circle cx="12" cy="8" r="1.5" fill="white" stroke="none"/><path d="M8 6a4 4 0 0 1 4-4"/><path d="M16 6a4 4 0 0 0-4-4"/></svg></div>'
    : '<div class="msg-avatar"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>';

  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  div.innerHTML = avatar + '<div class="msg-content"><div class="msg-meta"><span class="msg-role">' + (role === 'assistant' ? 'NeuralAI' : 'You') + '</span><span class="msg-time">' + time + '</span></div><div class="msg-bubble">' + (content ? escHtml(content) : '') + '</div>' + attachedInfo + '</div>';
  messagesEl.appendChild(div);
  scrollBottom();
  return div;
}

function copyBtn() {
  return '<div class="msg-actions"><button class="copy-btn" onclick="copyMsg(this)"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy</button></div>';
}

window.copyMsg = function(btn) {
  const text = btn.closest('.msg-bubble').innerText.replace(/Copy$/, '').trim();
  navigator.clipboard.writeText(text).then(() => {
    btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>Copied!';
    setTimeout(() => { btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy'; }, 2000);
  });
};

function updateMsgCount() {
  if (msgCountEl) msgCountEl.textContent = conversation.filter(m => m.role !== 'system').length + ' messages';
}

chatInput?.dispatchEvent(new Event('input'));
