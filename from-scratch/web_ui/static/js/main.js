// NeuralAI Chat UI — Frontend Logic

const chatArea = document.getElementById('chatArea');
const messagesEl = document.getElementById('messages');
const welcomeScreen = document.getElementById('welcomeScreen');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const clearBtn = document.getElementById('clearBtn');
const sidebar = document.getElementById('sidebar');
const menuBtn = document.getElementById('menuBtn');
const settingsBtn = document.getElementById('settingsBtn');
const settingsPanel = document.getElementById('settingsPanel');
const settingsOverlay = document.getElementById('settingsOverlay');
const settingsCloseBtn = document.getElementById('settingsCloseBtn');

let conversation = [];
let isStreaming = false;

// ─── Input ───────────────────────────────────────────────
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
  sendBtn.disabled = chatInput.value.trim() === '' || isStreaming;
});

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

sendBtn.addEventListener('click', () => { if (!sendBtn.disabled) sendMessage(); });

// ─── Quick Prompts ───────────────────────────────────────
document.querySelectorAll('.prompt-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    chatInput.value = btn.dataset.prompt;
    chatInput.dispatchEvent(new Event('input'));
    sendMessage();
  });
});

// ─── Chat Actions ────────────────────────────────────────
newChatBtn.addEventListener('click', () => {
  conversation = [];
  messagesEl.innerHTML = '';
  welcomeScreen.style.display = 'flex';
  chatInput.value = '';
  chatInput.dispatchEvent(new Event('input'));
  closeSidebar();
});

clearBtn.addEventListener('click', () => {
  conversation = [];
  messagesEl.innerHTML = '';
  welcomeScreen.style.display = 'flex';
  closeSidebar();
});

// ─── Sidebar (mobile) ─────────────────────────────────────
menuBtn.addEventListener('click', () => sidebar.classList.toggle('open'));
document.addEventListener('click', (e) => {
  if (sidebar.classList.contains('open') && !sidebar.contains(e.target) && !menuBtn.contains(e.target)) {
    sidebar.classList.remove('open');
  }
});

// ─── Settings Panel ───────────────────────────────────────
settingsBtn.addEventListener('click', () => {
  settingsPanel.classList.add('open');
  settingsOverlay.classList.add('open');
  fetchModelStatus();
});
settingsCloseBtn.addEventListener('click', closeSettingsPanel);
settingsOverlay.addEventListener('click', closeSettingsPanel);

function closeSettingsPanel() {
  settingsPanel.classList.remove('open');
  settingsOverlay.classList.remove('open');
}

function closeSidebar() {
  sidebar.classList.remove('open');
}

// ─── Settings Controls ───────────────────────────────────
const tempRange = document.getElementById('tempRange');
const tempValue = document.getElementById('tempValue');
tempRange.addEventListener('input', () => { tempValue.textContent = (tempRange.value / 100).toFixed(2); });

const maxTokensRange = document.getElementById('maxTokensRange');
const maxTokensValue = document.getElementById('maxTokensValue');
maxTokensRange.addEventListener('input', () => { maxTokensValue.textContent = maxTokensRange.value; });

async function fetchModelStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    document.getElementById('statusModel').textContent = data.model.split('/').pop();
    document.getElementById('statusType').textContent = data.model_type;
    document.getElementById('statusDevice').textContent = data.device;
  } catch {
    document.getElementById('statusModel').textContent = 'Error';
  }
}

// ─── Send Message ─────────────────────────────────────────
async function sendMessage() {
  const userMsg = chatInput.value.trim();
  if (!userMsg || isStreaming) return;

  welcomeScreen.style.display = 'none';
  addMessage('user', userMsg);
  conversation.push({ role: 'user', content: userMsg });
  chatInput.value = '';
  chatInput.style.height = 'auto';
  sendBtn.disabled = true;
  isStreaming = true;

  const assistantEl = addMessage('assistant', '', true);
  const bubbleEl = assistantEl.querySelector('.message-bubble');
  bubbleEl.innerHTML = '<div class="typing"><div class="typing-dots"><span></span><span></span><span></span></div> Thinking...</div>';
  scrollBottom();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: conversation,
        prompt: userMsg,
        max_tokens: parseInt(maxTokensRange.value),
        temperature: parseFloat(tempRange.value) / 100
      })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';

    bubbleEl.innerHTML = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim();
          if (raw === '[DONE]') break;
          try {
            const parsed = JSON.parse(raw);
            if (parsed.content) {
              fullText += parsed.content;
              bubbleEl.innerHTML = formatText(fullText);
              scrollBottom();
            } else if (parsed.error) {
              bubbleEl.innerHTML = `<span style="color:#ef4444">Error: ${parsed.error}</span>`;
            }
          } catch {}
        }
      }
    }

    conversation.push({ role: 'assistant', content: fullText });

  } catch (err) {
    bubbleEl.innerHTML = `<span style="color:#ef4440">Error: ${err.message}</span>`;
  } finally {
    isStreaming = false;
    sendBtn.disabled = chatInput.value.trim() === '';
    scrollBottom();
  }
}

// ─── Message Helpers ──────────────────────────────────────
function addMessage(role, content, placeholder = false) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  const avatar = role === 'assistant'
    ? '<div class="message-avatar"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg></div>'
    : '<div class="message-avatar"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>';
  div.innerHTML = `${avatar}<div class="message-bubble">${placeholder ? '' : escapeHtml(content)}</div>`;
  messagesEl.appendChild(div);
  scrollBottom();
  return div;
}

function scrollBottom() { chatArea.scrollTop = chatArea.scrollHeight; }

function escapeHtml(text) {
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function formatText(text) {
  let out = escapeHtml(text);
  out = out.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code>${code.trim()}</code></pre>`);
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  out = out.replace(/\n/g, '<br>');
  return out;
}

// Init
chatInput.dispatchEvent(new Event('input'));
