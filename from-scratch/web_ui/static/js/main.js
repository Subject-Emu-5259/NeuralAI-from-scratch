// NeuralAI Chat UI — Frontend Logic

const chatContainer = document.getElementById('chatContainer');
const messagesEl = document.getElementById('messages');
const welcomeScreen = document.getElementById('welcomeScreen');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const charCount = document.getElementById('charCount');
const modelSelect = document.getElementById('modelSelect');
const newChatBtn = document.getElementById('newChatBtn');
const clearBtn = document.getElementById('clearBtn');
const settingsBtn = document.getElementById('settingsBtn');
const settingsPanel = document.getElementById('settingsPanel');
const settingsClose = document.getElementById('settingsClose');
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');

let conversation = [];
let isStreaming = false;

// ─── Settings Panel ────────────────────────────────────────
settingsBtn.addEventListener('click', () => {
  settingsPanel.classList.add('open');
  settingsPanel.style.display = 'flex';
});
settingsClose.addEventListener('click', () => {
  settingsPanel.classList.remove('open');
  settingsPanel.style.display = 'none';
});
// Close on backdrop click (main area)
document.addEventListener('click', (e) => {
  if (settingsPanel.classList.contains('open') &&
      !settingsPanel.contains(e.target) && !settingsBtn.contains(e.target)) {
    settingsPanel.classList.remove('open');
    settingsPanel.style.display = 'none';
  }
  if (sidebar.classList.contains('open') && !sidebar.contains(e.target) && !sidebarToggle.contains(e.target)) {
    sidebar.classList.remove('open');
  }
});

// ─── Input ─────────────────────────────────────────────────
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
  charCount.textContent = chatInput.value.length;
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
document.querySelectorAll('.prompt-chip').forEach(btn => {
  btn.addEventListener('click', () => {
    chatInput.value = btn.dataset.prompt;
    chatInput.dispatchEvent(new Event('input'));
    sendMessage();
  });
});

// New chat
newChatBtn.addEventListener('click', () => {
  conversation = [];
  messagesEl.innerHTML = '';
  welcomeScreen.style.display = 'flex';
  chatInput.value = '';
  chatInput.dispatchEvent(new Event('input'));
});

// Clear all
clearBtn.addEventListener('click', () => {
  conversation = [];
  messagesEl.innerHTML = '';
  welcomeScreen.style.display = 'flex';
});

// Sidebar toggle (mobile)
sidebarToggle.addEventListener('click', () => {
  sidebar.classList.toggle('open');
});

// ─── Send Message ───────────────────────────────────────────
async function sendMessage() {
  const userMsg = chatInput.value.trim();
  if (!userMsg || isStreaming) return;

  // Show welcome fade out
  welcomeScreen.style.display = 'none';

  // Add user message
  addMessage('user', userMsg);
  conversation.push({ role: 'user', content: userMsg });
  chatInput.value = '';
  chatInput.style.height = 'auto';
  charCount.textContent = '0';
  sendBtn.disabled = true;

  isStreaming = true;
  sendBtn.disabled = false; // allow cancel
  sendBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';

  // Add assistant placeholder
  const assistantEl = addMessage('assistant', '', true);
  const bubbleEl = assistantEl.querySelector('.message-bubble');
  bubbleEl.innerHTML = '<div class="thinking"><div class="thinking-dots"><span></span><span></span><span></span></div> Thinking...</div>';

  scrollBottom();

  try {
    const model = modelSelect.value;
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: conversation, model })
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';

    bubbleEl.innerHTML = '';
    isStreaming = true;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') { isStreaming = false; break; }
          try {
            const parsed = JSON.parse(data);
            if (parsed.content) {
              fullText += parsed.content;
              bubbleEl.innerHTML = formatMarkdown(fullText);
              scrollBottom();
            }
          } catch {}
        }
      }
    }

    conversation.push({ role: 'assistant', content: fullText });

  } catch (err) {
    bubbleEl.innerHTML = `<span style="color:#ef4444">Error: ${err.message}</span>`;
  } finally {
    isStreaming = false;
    sendBtn.disabled = chatInput.value.trim() === '';
    sendBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';
    scrollBottom();
  }
}

function addMessage(role, content, placeholder = false) {
  const div = document.createElement('div');
  div.className = `message ${role}`;

  const avatar = role === 'assistant'
    ? '<div class="message-avatar">🧠</div>'
    : '<div class="message-avatar"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>';

  div.innerHTML = `${avatar}<div class="message-bubble">${placeholder ? '' : escapeHtml(content)}</div>`;
  messagesEl.appendChild(div);
  scrollBottom();
  return div;
}

function scrollBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ─── Format Markdown ───────────────────────────────────────
function escapeHtml(text) {
  const esc = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  return esc;
}

function formatMarkdown(text) {
  let out = escapeHtml(text);
  // Code blocks
  out = out.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code class="lang-${lang}">${code.trim()}</code></pre>`);
  // Inline code
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  // Italic
  out = out.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  // Line breaks
  out = out.replace(/\n/g, '<br>');
  return out;
}

// Initialize
chatInput.dispatchEvent(new Event('input'));
