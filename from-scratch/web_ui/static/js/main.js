// NeuralAI Chat UI — Frontend Logic

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
const settingsBtn = document.getElementById('settingsBtn');
const settingsMenu = document.getElementById('settingsMenu');
const tempSlider = document.getElementById('tempSlider');
const tempVal = document.getElementById('tempVal');
const maxTokensSlider = document.getElementById('maxTokensSlider');
const maxTokensVal = document.getElementById('maxTokensVal');

let conversation = [];
let isStreaming = false;

// ── Settings ──────────────────────────────────────────────────
tempSlider.addEventListener('input', () => {
  tempVal.textContent = (tempSlider.value / 100).toFixed(1);
});
maxTokensSlider.addEventListener('input', () => {
  maxTokensVal.textContent = maxTokensSlider.value;
});

settingsBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  settingsMenu.classList.toggle('open');
});

document.addEventListener('click', () => {
  settingsMenu.classList.remove('open');
});
settingsMenu.addEventListener('click', (e) => e.stopPropagation());

// ── Sidebar ───────────────────────────────────────────────────
function openSidebar() {
  sidebar.classList.add('open');
  sidebarOverlay.classList.add('open');
}
function closeSidebar() {
  sidebar.classList.remove('open');
  sidebarOverlay.classList.remove('open');
}
sidebarToggle.addEventListener('click', openSidebar);
sidebarOverlay.addEventListener('click', closeSidebar);

// ── Input ─────────────────────────────────────────────────────
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + 'px';
  sendBtn.disabled = chatInput.value.trim() === '' || isStreaming;
});

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage();
  }
});

sendBtn.addEventListener('click', () => { if (!sendBtn.disabled) sendMessage(); });

// Quick prompts
document.querySelectorAll('.prompt').forEach(btn => {
  btn.addEventListener('click', () => {
    chatInput.value = btn.dataset.prompt;
    chatInput.dispatchEvent(new Event('input'));
    sendMessage();
  });
});

// ── New Chat & Clear ──────────────────────────────────────────
newChatBtn.addEventListener('click', () => {
  conversation = [];
  messagesEl.innerHTML = '';
  welcomeScreen.style.display = 'flex';
  chatInput.value = '';
  chatInput.style.height = 'auto';
  chatInput.dispatchEvent(new Event('input'));
  closeSidebar();
});

clearBtn.addEventListener('click', () => {
  conversation = [];
  messagesEl.innerHTML = '';
  welcomeScreen.style.display = 'flex';
  closeSidebar();
  settingsMenu.classList.remove('open');
});

// ── Send Message ──────────────────────────────────────────────
async function sendMessage() {
  const userMsg = chatInput.value.trim();
  if (!userMsg || isStreaming) return;

  welcomeScreen.style.display = 'none';
  addMsg('user', userMsg);
  conversation.push({ role: 'user', content: userMsg });
  chatInput.value = '';
  chatInput.style.height = 'auto';
  chatInput.dispatchEvent(new Event('input'));

  isStreaming = true;
  sendBtn.disabled = false;
  sendBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';

  const assistantEl = addMsg('assistant', '');
  const bubbleEl = assistantEl.querySelector('.msg-bubble');
  bubbleEl.innerHTML = '<div class="thinking"><div class="dots"><span></span><span></span><span></span></div> Thinking...</div>';
  scrollBottom();

  const temperature = parseFloat(tempSlider.value) / 100;
  const maxTokens = parseInt(maxTokensSlider.value);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: userMsg, temperature, max_tokens: maxTokens, messages: conversation })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

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
          const data = line.slice(6);
          if (data === '[DONE]' || data.trim() === '[DONE]') { isStreaming = false; break; }
          try {
            const parsed = JSON.parse(data);
            if (parsed.content) {
              full += parsed.content;
              bubbleEl.innerHTML = fmt(full);
              scrollBottom();
            }
          } catch { /* skip */ }
        }
      }
    }

    conversation.push({ role: 'assistant', content: full });

  } catch (err) {
    bubbleEl.innerHTML = `<span style="color:#ef4444">Error: ${err.message}</span>`;
  } finally {
    isStreaming = false;
    sendBtn.disabled = chatInput.value.trim() === '';
    sendBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';
    scrollBottom();
    closeSidebar();
  }
}

// ── Helpers ────────────────────────────────────────────────────
function addMsg(role, content, placeholder = false) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  const avatar = role === 'assistant'
    ? '<div class="msg-avatar">🧠</div>'
    : '<div class="msg-avatar"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>';
  div.innerHTML = `${avatar}<div class="msg-bubble">${placeholder ? '' : escHtml(content)}</div>`;
  messagesEl.appendChild(div);
  scrollBottom();
  return div;
}

function scrollBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escHtml(text) {
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function fmt(text) {
  let out = escHtml(text);
  out = out.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, l, c) => `<pre><code class="lang-${l}">${c.trim()}</code></pre>`);
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  out = out.replace(/\n/g, '<br>');
  return out;
}

// Init
chatInput.dispatchEvent(new Event('input'));
