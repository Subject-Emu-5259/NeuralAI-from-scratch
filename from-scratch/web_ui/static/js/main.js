// NeuralAI Chat UI v4.0 — Full Feature Set with Persistence, Memory, Settings
// Includes: Conversation Management, Memory System, Model Rules, User Bio, Settings

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
const darkModeBtn = document.getElementById('darkModeBtn');
const queryInput = document.getElementById('queryInput');
const searchBtn = document.getElementById('searchBtn');

// State
let conversation = [];
let isStreaming = false;
let isDark = false;
let attachedFiles = {};
let currentConversationId = null;
let conversations = [];
let userSettings = {};
let memoryFacts = [];
let modelRules = [];

// ========================================
// UTILITIES
// ========================================

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

function scrollBottom() { if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight; }

function showToast(msg, type = '') {
  const old = document.querySelector('.toast');
  if (old) old.remove();
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span> ${msg}`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

function formatTime(iso) {
  try { return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
  catch { return ''; }
}

// ========================================
// DARK MODE
// ========================================

function toggleDarkMode() {
  isDark = !isDark;
  document.body.classList.toggle('dark-mode', isDark);
  localStorage.setItem('neuralai_dark_mode', isDark);
  if (darkModeBtn) darkModeBtn.textContent = isDark ? '☀️ Light' : '🌙 Dark';
  updateSetting('theme', isDark ? 'dark' : 'light');
  showToast(isDark ? 'Dark mode enabled' : 'Light mode enabled', 'success');
}

function loadDarkMode() {
  const saved = localStorage.getItem('neuralai_dark_mode');
  if (saved === 'true') {
    isDark = true;
    document.body.classList.add('dark-mode');
    if (darkModeBtn) darkModeBtn.textContent = '☀️ Light';
  }
}

darkModeBtn?.addEventListener('click', toggleDarkMode);
themeBtn?.addEventListener('click', toggleDarkMode);

// ========================================
// SETTINGS API
// ========================================

async function loadSettings() {
  try {
    const res = await fetch('/api/settings');
    const data = await res.json();
    userSettings = data.settings || {};
  } catch (e) {
    console.error('Failed to load settings:', e);
  }
}

async function updateSetting(key, value) {
  try {
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ [key]: value })
    });
    userSettings[key] = value;
  } catch (e) {
    console.error('Failed to update setting:', e);
  }
}

// ========================================
// MEMORY API
// ========================================

async function loadMemory() {
  try {
    const res = await fetch('/api/memory');
    const data = await res.json();
    memoryFacts = data.facts || [];
  } catch (e) {
    console.error('Failed to load memory:', e);
  }
}

async function addMemory(fact, category = 'general') {
  try {
    const res = await fetch('/api/memory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fact, category })
    });
    const data = await res.json();
    if (data.success) {
      memoryFacts.unshift({ id: data.id, fact, category });
      showToast('Memory saved!', 'success');
      return true;
    }
  } catch (e) {
    showToast('Failed to save memory', 'error');
  }
  return false;
}

async function deleteMemory(id) {
  try {
    await fetch(`/api/memory/${id}`, { method: 'DELETE' });
    memoryFacts = memoryFacts.filter(m => m.id !== id);
    showToast('Memory deleted', 'success');
  } catch (e) {
    showToast('Failed to delete memory', 'error');
  }
}

// ========================================
// RULES API
// ========================================

async function loadRules() {
  try {
    const res = await fetch('/api/rules');
    const data = await res.json();
    modelRules = data.rules || [];
  } catch (e) {
    console.error('Failed to load rules:', e);
  }
}

async function addRule(rule) {
  try {
    const res = await fetch('/api/rules', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rule, is_active: 1 })
    });
    const data = await res.json();
    if (data.success) {
      modelRules.unshift({ id: data.id, rule, is_active: 1 });
      showToast('Rule added!', 'success');
      return true;
    }
  } catch (e) {
    showToast('Failed to add rule', 'error');
  }
  return false;
}

async function deleteRule(id) {
  try {
    await fetch(`/api/rules/${id}`, { method: 'DELETE' });
    modelRules = modelRules.filter(r => r.id !== id);
    showToast('Rule deleted', 'success');
  } catch (e) {
    showToast('Failed to delete rule', 'error');
  }
}

async function toggleRule(id) {
  try {
    await fetch(`/api/rules/${id}/toggle`, { method: 'POST' });
    const rule = modelRules.find(r => r.id === id);
    if (rule) rule.is_active = rule.is_active ? 0 : 1;
  } catch (e) {
    showToast('Failed to toggle rule', 'error');
  }
}

// ========================================
// CONVERSATIONS API
// ========================================

async function loadConversations() {
  try {
    const res = await fetch('/api/conversations');
    const data = await res.json();
    conversations = data.conversations || [];
    renderHistoryDropdown();
  } catch (e) {
    console.error('Failed to load conversations:', e);
  }
}

async function createNewConversation() {
  try {
    const res = await fetch('/api/conversations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: 'New Chat' })
    });
    const data = await res.json();
    if (data.success) {
      conversations.unshift({ id: data.id, title: data.title, message_count: 0 });
      currentConversationId = data.id;
      conversation = [];
      messagesEl.innerHTML = '';
      welcomeScreen.style.display = 'flex';
      renderConversationList();
      return data.id;
    }
  } catch (e) {
    showToast('Failed to create conversation', 'error');
  }
  return null;
}

async function loadConversation(convId) {
  try {
    const res = await fetch(`/api/conversations/${convId}`);
    const data = await res.json();
    if (data.conversation) {
      currentConversationId = convId;
      conversation = data.messages.map(m => ({ role: m.role, content: m.content }));
      messagesEl.innerHTML = '';
      welcomeScreen.style.display = conversation.length ? 'none' : 'flex';
      conversation.forEach(m => addMsg(m.role, m.content));
      updateMsgCount();
      
      // Update active state in list
      document.querySelectorAll('.history-item').forEach(item => {
        item.classList.toggle('active', item.dataset.id === convId);
      });
    }
  } catch (e) {
    showToast('Failed to load conversation', 'error');
  }
}

async function deleteConversation(convId) {
  try {
    await fetch(`/api/conversations/${convId}`, { method: 'DELETE' });
    conversations = conversations.filter(c => c.id !== convId);
    if (currentConversationId === convId) {
      currentConversationId = null;
      conversation = [];
      messagesEl.innerHTML = '';
      welcomeScreen.style.display = 'flex';
    }
    renderConversationList();
    showToast('Conversation deleted', 'success');
  } catch (e) {
    showToast('Failed to delete conversation', 'error');
  }
}

function renderConversationList() {
  const container = document.getElementById('historyList');
  if (!container) return;
  
  container.innerHTML = '';
  
  if (conversations.length === 0) {
    container.innerHTML = '<div style="color:#6b7280;font-size:13px;padding:8px;">No past conversations</div>';
    return;
  }
  
  conversations.forEach(conv => {
    const item = document.createElement('div');
    item.className = `history-item ${conv.id === currentConversationId ? 'active' : ''}`;
    item.dataset.id = conv.id;
    item.innerHTML = `
      <div class="history-item-text" onclick="loadConversation('${conv.id}')">${escHtml(conv.title)}</div>
      <div class="history-item-meta">${conv.message_count || 0} msgs</div>
      <button class="history-item-delete" onclick="event.stopPropagation(); deleteConversation('${conv.id}')">×</button>
    `;
    container.appendChild(item);
  });
}

// ========================================
// HISTORY DROPDOWN (in top search bar)
// ========================================

let hideDropdownTimeout = null;

function showHistoryDropdown() {
  clearTimeout(hideDropdownTimeout);
  const dropdown = document.getElementById('historyDropdown');
  if (dropdown) {
    dropdown.style.display = 'block';
    renderHistoryDropdown();
  }
}

function hideHistoryDropdownDelayed() {
  hideDropdownTimeout = setTimeout(() => {
    const dropdown = document.getElementById('historyDropdown');
    if (dropdown) dropdown.style.display = 'none';
  }, 200);
}

function renderHistoryDropdown() {
  const list = document.getElementById('historyDropdownList');
  const empty = document.getElementById('historyDropdownEmpty');
  if (!list) return;
  
  list.innerHTML = '';
  
  if (conversations.length === 0) {
    list.style.display = 'none';
    if (empty) empty.style.display = 'block';
    return;
  }
  
  list.style.display = 'block';
  if (empty) empty.style.display = 'none';
  
  conversations.slice(0, 10).forEach(conv => {
    const item = document.createElement('div');
    item.className = 'history-item';
    item.style.cssText = 'padding:10px 12px;border-bottom:1px solid rgba(255,255,255,0.05);display:flex;justify-content:space-between;align-items:center;';
    item.innerHTML = `
      <span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escHtml(conv.title)}</span>
      <span style="font-size:11px;color:#6b7280;margin-left:8px;">${conv.message_count || 0} msgs</span>
      <button onclick="event.stopPropagation();deleteConversation('${conv.id}')" style="background:none;border:none;color:#ef4444;cursor:pointer;font-size:14px;margin-left:8px;opacity:0.5;">×</button>
    `;
    item.onclick = (e) => {
      if (e.target.tagName !== 'BUTTON') {
        loadConversation(conv.id);
        const dropdown = document.getElementById('historyDropdown');
        if (dropdown) dropdown.style.display = 'none';
        const queryInput = document.getElementById('queryInput');
        if (queryInput) queryInput.blur();
      }
    };
    list.appendChild(item);
  });
  
  // Add "New Chat" option at bottom
  const newChatItem = document.createElement('div');
  newChatItem.style.cssText = 'padding:10px 12px;text-align:center;color:#e4e4e7;cursor:pointer;font-weight:500;border-top:1px solid rgba(255,255,255,0.1);';
  newChatItem.textContent = '+ New Chat';
  newChatItem.onclick = () => {
    createNewConversation();
    const dropdown = document.getElementById('historyDropdown');
    if (dropdown) dropdown.style.display = 'none';
  };
  list.appendChild(newChatItem);
}

window.showHistoryDropdown = showHistoryDropdown;
window.hideHistoryDropdownDelayed = hideHistoryDropdownDelayed;

// ========================================
// INFO PANEL
// ========================================

infoBtn?.addEventListener('click', () => {
  fetch('/api/status').then(r => r.json()).then(data => {
    const m = document.createElement('div');
    m.style.cssText = 'position:fixed;top:70px;right:16px;width:320px;background:var(--surface,#0f0f17);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:16px;z-index:200;box-shadow:0 8px 32px rgba(0,0,0,0.5);';
    m.innerHTML = `
      <div style="font-weight:700;font-size:15px;margin-bottom:12px;">🧠 NeuralAI v${data.version}</div>
      <div style="font-size:12px;color:var(--text-muted,#6b6b76);line-height:2;">
        <div><strong>Base:</strong> ${data.model||'SmolLM2-360M-Instruct'}</div>
        <div><strong>Type:</strong> ${data.model_type||'base'}</div>
        <div><strong>Device:</strong> ${data.device||'CPU'}</div>
        <div><strong>RAG:</strong> ${data.rag?'✅ Active':'❌ Off'}</div>
        <div><strong>Files:</strong> ${data.indexed_files||0}</div>
        <div><strong>Memory:</strong> ${memoryFacts.length} facts</div>
        <div><strong>Rules:</strong> ${modelRules.filter(r=>r.is_active).length} active</div>
      </div>
      <div style="margin-top:12px;display:flex;gap:8px;">
        <button onclick="showMemoryPanel()" style="flex:1;padding:8px;background:#3b82f6;border:none;border-radius:8px;color:#fff;cursor:pointer;">🧠 Memory</button>
        <button onclick="showRulesPanel()" style="flex:1;padding:8px;background:#10b981;border:none;border-radius:8px;color:#fff;cursor:pointer;">📜 Rules</button>
      </div>
    `;
    document.body.appendChild(m);
    setTimeout(() => m.remove(), 8000);
  });
});


// ========================================
// MEMORY PANEL
// ========================================

function showMemoryPanel() {
  const existing = document.getElementById('memoryPanel');
  if (existing) { existing.remove(); return; }
  
  const panel = document.createElement('div');
  panel.id = 'memoryPanel';
  panel.style.cssText = 'position:fixed;top:0;right:0;width:400px;height:100vh;background:#0f0f17;border-left:1px solid rgba(255,255,255,0.1);z-index:1000;overflow-y:auto;padding:24px;';
  
  panel.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;">
      <h2 style="margin:0;font-size:20px;">🧠 Memory</h2>
      <button onclick="this.closest('#memoryPanel').remove()" style="background:none;border:none;color:#888;font-size:24px;cursor:pointer;">×</button>
    </div>
    
    <div style="margin-bottom:20px;">
      <textarea id="newMemoryInput" placeholder="Add something for NeuralAI to remember about you..." style="width:100%;height:60px;background:#161b22;border:1px solid rgba(255,255,255,0.1);border-radius:8px;padding:12px;color:#e4e4e7;resize:vertical;font-family:inherit;"></textarea>
      <button id="addMemoryBtn" style="width:100%;padding:10px;background:#3b82f6;border:none;border-radius:8px;color:#fff;margin-top:8px;cursor:pointer;">+ Add Memory</button>
    </div>
    
    <div style="font-size:13px;color:#888;margin-bottom:12px;">Saved Memories:</div>
    <div id="memoryList" style="display:flex;flex-direction:column;gap:8px;"></div>
  `;
  
  document.body.appendChild(panel);
  
  renderMemoryList();
  
  document.getElementById('addMemoryBtn').addEventListener('click', async () => {
    const input = document.getElementById('newMemoryInput');
    const fact = input.value.trim();
    if (fact && await addMemory(fact)) {
      input.value = '';
      renderMemoryList();
    }
  });
}

function renderMemoryList() {
  const container = document.getElementById('memoryList');
  if (!container) return;
  
  container.innerHTML = '';
  
  if (memoryFacts.length === 0) {
    container.innerHTML = '<div style="color:#6b7280;font-size:13px;padding:8px;">No memories saved yet</div>';
    return;
  }
  
  memoryFacts.forEach(m => {
    const item = document.createElement('div');
    item.style.cssText = 'display:flex;justify-content:space-between;align-items:flex-start;background:#161b22;border-radius:8px;padding:12px;';
    item.innerHTML = `
      <div style="flex:1;font-size:13px;">${escHtml(m.fact)}</div>
      <button onclick="deleteMemory(${m.id});this.closest('#memoryPanel')&&showMemoryPanel();" style="background:none;border:none;color:#ef4444;cursor:pointer;font-size:16px;">×</button>
    `;
    container.appendChild(item);
  });
}

// ========================================
// RULES PANEL
// ========================================

function showRulesPanel() {
  const existing = document.getElementById('rulesPanel');
  if (existing) { existing.remove(); return; }
  
  const panel = document.createElement('div');
  panel.id = 'rulesPanel';
  panel.style.cssText = 'position:fixed;top:0;right:0;width:400px;height:100vh;background:#0f0f17;border-left:1px solid rgba(255,255,255,0.1);z-index:1000;overflow-y:auto;padding:24px;';
  
  panel.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;">
      <h2 style="margin:0;font-size:20px;">📜 Model Rules</h2>
      <button onclick="this.closest('#rulesPanel').remove()" style="background:none;border:none;color:#888;font-size:24px;cursor:pointer;">×</button>
    </div>
    
    <div style="font-size:12px;color:#888;margin-bottom:16px;">Rules guide how the AI responds. Active rules are injected into the system prompt.</div>
    
    <div style="margin-bottom:20px;">
      <textarea id="newRuleInput" placeholder="Add a behavioral rule (e.g., 'Always be concise', 'Use simple language')..." style="width:100%;height:60px;background:#161b22;border:1px solid rgba(255,255,255,0.1);border-radius:8px;padding:12px;color:#e4e4e7;resize:vertical;font-family:inherit;"></textarea>
      <button id="addRuleBtn" style="width:100%;padding:10px;background:#10b981;border:none;border-radius:8px;color:#fff;margin-top:8px;cursor:pointer;">+ Add Rule</button>
    </div>
    
    <div style="font-size:13px;color:#888;margin-bottom:12px;">Active Rules:</div>
    <div id="rulesList" style="display:flex;flex-direction:column;gap:8px;"></div>
  `;
  
  document.body.appendChild(panel);
  
  renderRulesList();
  
  document.getElementById('addRuleBtn').addEventListener('click', async () => {
    const input = document.getElementById('newRuleInput');
    const rule = input.value.trim();
    if (rule && await addRule(rule)) {
      input.value = '';
      renderRulesList();
    }
  });
}

function renderRulesList() {
  const container = document.getElementById('rulesList');
  if (!container) return;
  
  container.innerHTML = '';
  
  if (modelRules.length === 0) {
    container.innerHTML = '<div style="color:#6b7280;font-size:13px;padding:8px;">No rules defined yet</div>';
    return;
  }
  
  modelRules.forEach(r => {
    const item = document.createElement('div');
    item.style.cssText = `display:flex;justify-content:space-between;align-items:center;background:#161b22;border-radius:8px;padding:12px;opacity:${r.is_active ? 1 : 0.5}`;
    item.innerHTML = `
      <div style="flex:1;font-size:13px;">${escHtml(r.rule)}</div>
      <div style="display:flex;gap:8px;align-items:center;">
        <button onclick="toggleRule(${r.id});renderRulesList();" style="background:none;border:none;color:${r.is_active ? '#10b981' : '#6b7280'};cursor:pointer;font-size:12px;">${r.is_active ? '✓' : '○'}</button>
        <button onclick="deleteRule(${r.id});renderRulesList();" style="background:none;border:none;color:#ef4444;cursor:pointer;font-size:16px;">×</button>
      </div>
    `;
    container.appendChild(item);
  });
}

// Make functions global for onclick
window.showMemoryPanel = showMemoryPanel;
window.showRulesPanel = showRulesPanel;
window.loadConversation = loadConversation;
window.deleteConversation = deleteConversation;
window.deleteMemory = deleteMemory;
window.toggleRule = toggleRule;
window.deleteRule = deleteRule;
window.renderMemoryList = renderMemoryList;
window.renderRulesList = renderRulesList;

// ========================================
// EXPORT
// ========================================

exportBtn?.addEventListener('click', () => {
  if (conversation.length === 0) { showToast('No conversation to export', 'error'); return; }
  const md = conversation.filter(m => m.role !== 'system').map(m => `**${m.role}:** ${m.content}`).join('\n\n');
  const blob = new Blob([`# NeuralAI Chat\n\n${md}\n\n*Exported: ${new Date().toLocaleString()}*`], { type: 'text/markdown' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `neuralai-chat-${Date.now()}.md`; a.click();
  URL.revokeObjectURL(a.href);
  showToast('Chat exported!', 'success');
});

// ========================================
// SIDEBAR
// ========================================

function openSidebar() { sidebar.classList.add('open'); sidebarOverlay.classList.add('open'); }
function closeSidebar() { sidebar.classList.remove('open'); sidebarOverlay.classList.remove('open'); }
sidebarToggle?.addEventListener('click', openSidebar);
sidebarOverlay?.addEventListener('click', closeSidebar);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeSidebar(); });

// ========================================
// SEARCH
// ========================================

searchInput?.addEventListener('input', () => {
  const q = searchInput.value.toLowerCase();
  document.querySelectorAll('.history-item').forEach(item => {
    const t = item.querySelector('.history-item-text')?.textContent || '';
    item.style.display = t.toLowerCase().includes(q) ? 'flex' : 'none';
  });
});

// ========================================
// FILE UPLOAD
// ========================================

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
    chip.style.cssText = 'display:inline-flex;align-items:center;gap:4px;padding:3px 10px;background:var(--accent-glow,rgba(59,130,246,0.15));border:1px solid rgba(59,130,246,0.3);border-radius:20px;font-size:11px;color:var(--accent-text,#e4e4e7);';
    chip.textContent = '📄 ' + (name.length > 20 ? name.slice(0,18) + '…' : name);
    div.appendChild(chip);
  });
  document.querySelector('.input-area')?.parentElement?.insertBefore(div, document.querySelector('.input-area'));
}

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

// ========================================
// NEW CHAT
// ========================================

newChatBtn?.addEventListener('click', async () => {
  await createNewConversation();
  chatInput.value = '';
  chatInput.style.height = 'auto';
  attachedFiles = {};
  const af = document.getElementById('attachedFiles');
  if (af) af.remove();
  chatInput.dispatchEvent(new Event('input'));
  closeSidebar();
  updateMsgCount();
});

// ========================================
// INPUT HANDLING
// ========================================

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

// ========================================
// SEND MESSAGE
// ========================================

async function sendMessage() {
  const userMsg = chatInput.value.trim();
  if (!userMsg || isStreaming) return;

  welcomeScreen.style.display = 'none';
  const fileIdsThisMsg = Object.keys(attachedFiles);

  // Create conversation if needed
  if (!currentConversationId) {
    await createNewConversation();
  }

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
      body: JSON.stringify({ 
        prompt: userMsg, 
        temperature: parseFloat(userSettings.model_temperature || 0.7), 
        max_tokens: parseInt(userSettings.model_max_tokens || 512), 
        messages: conversation, 
        file_ids: fileIdsThisMsg,
        conversation_id: currentConversationId
      })
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
    
    // Refresh conversation list to update title/count
    loadConversations();

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

// ========================================
// ADD MESSAGE
// ========================================

function addMsg(role, content, fileIds) {
  const div = document.createElement('div');
  div.className = 'message ' + role;

  let attachedInfo = '';
  if (role === 'user' && fileIds && fileIds.length > 0) {
    const names = fileIds.map(fid => attachedFiles[fid] || fid);
    attachedInfo = '<div style="font-size:11px;color:var(--text-dim,#3e3e50);margin-top:4px;display:flex;flex-wrap:wrap;gap:4px;">' +
      names.map(n => '<span style="background:var(--accent-glow,rgba(59,130,246,0.1));padding:1px 8px;border-radius:10px;">📄 ' + escHtml(n.length > 22 ? n.slice(0,20)+'…' : n) + '</span>').join('') + '</div>';
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

// ========================================
// INITIALIZATION
// ========================================

async function initialize() {
  loadDarkMode();
  await loadSettings();
  await loadMemory();
  await loadRules();
  await loadConversations();
  chatInput?.dispatchEvent(new Event('input'));
  console.log('[NeuralAI] v4.0 initialized with persistence, memory, and settings');
}

initialize();

// Build: 1777526000

// ========================================
// SETTINGS TAB FUNCTIONS
// ========================================

async function saveUserBio() {
  const bio = document.getElementById('userBioInput').value.trim();
  try {
    await updateSetting('user_bio', bio);
    showToast('Bio saved!', 'success');
  } catch (e) {
    showToast('Failed to save bio', 'error');
  }
}

async function loadBio() {
  try {
    const res = await fetch('/api/settings');
    const data = await res.json();
    const bioInput = document.getElementById('userBioInput');
    if (bioInput && data.settings && data.settings.user_bio) {
      bioInput.value = data.settings.user_bio;
    }
  } catch (e) {
    console.error('Failed to load bio:', e);
  }
}

async function addMemoryFromTab() {
  const input = document.getElementById('memoryInput');
  const fact = input.value.trim();
  if (!fact) return;
  
  try {
    await fetch('/api/memory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fact })
    });
    input.value = '';
    loadMemoryList();
    showToast('Memory added!', 'success');
  } catch (e) {
    showToast('Failed to add memory', 'error');
  }
}

async function loadMemoryList() {
  try {
    const res = await fetch('/api/memory');
    const data = await res.json();
    const list = document.getElementById('memoryList');
    if (!list) return;
    
    list.innerHTML = '';
    (data.facts || []).forEach(m => {
      const div = document.createElement('div');
      div.className = 'memory-item';
      div.innerHTML = `
        <span>${escHtml(m.fact)}</span>
        <button class="delete-btn" onclick="deleteMemoryItem(${m.id})">×</button>
      `;
      list.appendChild(div);
    });
  } catch (e) {
    console.error('Failed to load memory:', e);
  }
}

async function deleteMemoryItem(id) {
  try {
    await fetch(`/api/memory/${id}`, { method: 'DELETE' });
    loadMemoryList();
    showToast('Memory deleted', 'success');
  } catch (e) {
    showToast('Failed to delete memory', 'error');
  }
}

async function addRuleFromTab() {
  const input = document.getElementById('ruleInput');
  const rule = input.value.trim();
  if (!rule) return;
  
  try {
    await fetch('/api/rules', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rule, is_active: 1 })
    });
    input.value = '';
    loadRulesList();
    showToast('Rule added!', 'success');
  } catch (e) {
    showToast('Failed to add rule', 'error');
  }
}

async function loadRulesList() {
  try {
    const res = await fetch('/api/rules');
    const data = await res.json();
    const list = document.getElementById('rulesList');
    if (!list) return;
    
    list.innerHTML = '';
    (data.rules || []).forEach(r => {
      const div = document.createElement('div');
      div.className = 'rule-item';
      div.innerHTML = `
        <span>${escHtml(r.rule)}</span>
        <button class="rule-toggle ${r.is_active ? 'on' : 'off'}" onclick="toggleRuleItem(${r.id}, ${r.is_active})">${r.is_active ? 'ON' : 'OFF'}</button>
        <button class="delete-btn" onclick="deleteRuleItem(${r.id})">×</button>
      `;
      list.appendChild(div);
    });
  } catch (e) {
    console.error('Failed to load rules:', e);
  }
}

async function toggleRuleItem(id, currentState) {
  try {
    await fetch(`/api/rules/${id}/toggle`, { method: 'POST' });
    loadRulesList();
  } catch (e) {
    showToast('Failed to toggle rule', 'error');
  }
}

async function deleteRuleItem(id) {
  try {
    await fetch(`/api/rules/${id}`, { method: 'DELETE' });
    loadRulesList();
    showToast('Rule deleted', 'success');
  } catch (e) {
    showToast('Failed to delete rule', 'error');
  }
}

// Make functions global for onclick handlers
window.saveUserBio = saveUserBio;
window.addMemoryFromTab = addMemoryFromTab;
window.loadMemoryList = loadMemoryList;
window.deleteMemoryItem = deleteMemoryItem;
window.addRuleFromTab = addRuleFromTab;
window.loadRulesList = loadRulesList;
window.toggleRuleItem = toggleRuleItem;
window.deleteRuleItem = deleteRuleItem;

// Load settings when tab is shown
document.addEventListener('DOMContentLoaded', () => {
  // Watch for settings tab
  document.querySelectorAll('.nav-item[data-tab]').forEach(btn => {
    btn.addEventListener('click', () => {
      if (btn.dataset.tab === 'settings') {
        setTimeout(() => {
          loadBio();
          loadMemoryList();
          loadRulesList();
        }, 100);
      }
    });
  });
});

