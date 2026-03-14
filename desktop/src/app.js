/**
 * OpenHydra Desktop — app.js
 * Vanilla JS frontend for the Tauri v2 desktop application.
 *
 * Responsibilities:
 *  - Node lifecycle: start / stop / poll via Tauri commands
 *  - Chat: streaming SSE inference against the local node API
 *  - Network tab: fetch /v1/network/status and render peer table
 *  - Settings persistence: localStorage
 */

'use strict';

/* ──────────────────────────────────────────────────────────────────
   Constants
────────────────────────────────────────────────────────────────── */

const NODE_API_BASE  = 'http://127.0.0.1:8080';
const POLL_INTERVAL  = 2000;   // ms — how often we check node liveness
const STORAGE_KEY    = 'openhydra_settings';

/* ──────────────────────────────────────────────────────────────────
   Tauri bridge (graceful fallback in browser preview)
────────────────────────────────────────────────────────────────── */

const isTauri = typeof window.__TAURI__ !== 'undefined';

async function tauriInvoke(cmd, args = {}) {
  if (!isTauri) {
    console.warn('[OpenHydra] Not in Tauri context — mocking command:', cmd, args);
    // Provide useful stubs so the UI can be previewed in a browser
    if (cmd === 'is_node_running') return false;
    if (cmd === 'get_node_pid')    return null;
    if (cmd === 'start_node')      return `Node started (mock): peer_id=${args.peer_id}`;
    if (cmd === 'stop_node')       return;
    return null;
  }
  return window.__TAURI__.core.invoke(cmd, args);
}

/* ──────────────────────────────────────────────────────────────────
   State
────────────────────────────────────────────────────────────────── */

const state = {
  nodeRunning:  false,
  nodeStarting: false,
  nodeStopping: false,
  peerId:       '',
  modelId:      'openhydra-qwen3.5-0.8b',
  dhtUrl:       'http://127.0.0.1:8468',
  messages:     [],          // { role, content, timestamp }
  streaming:    false,
  abortCtrl:    null,        // AbortController for active SSE stream
  pollTimer:    null,
};

/* ──────────────────────────────────────────────────────────────────
   DOM references
────────────────────────────────────────────────────────────────── */

const $ = id => document.getElementById(id);

const dom = {
  // sidebar
  statusDot:       $('status-dot'),
  statusLabel:     $('status-label'),
  btnNodeToggle:   $('btn-node-toggle'),
  peerIdContainer: $('peer-id-container'),
  peerIdDisplay:   $('peer-id-display'),
  networkInfo:     $('network-info'),
  statPeers:       $('stat-peers'),
  statModel:       $('stat-model'),
  statBalance:     $('stat-balance'),
  statPid:         $('stat-pid'),
  // settings
  inputPeerId:     $('input-peer-id'),
  inputModelId:    $('input-model-id'),
  inputDhtUrl:     $('input-dht-url'),
  // tabs
  tabBtns:         document.querySelectorAll('.tab-btn'),
  tabPanels:       document.querySelectorAll('.tab-panel'),
  // chat
  chatMessages:    $('chat-messages'),
  chatEmpty:       $('chat-empty'),
  chatInput:       $('chat-input'),
  btnSend:         $('btn-send'),
  chatHint:        $('chat-hint'),
  chatModelBadge:  $('chat-model-badge'),
  nodeWarning:     $('node-warning'),
  // network
  btnRefreshNet:   $('btn-refresh-network'),
  ncTotalPeers:    $('nc-total-peers'),
  ncModels:        $('nc-models'),
  ncRegion:        $('nc-region'),
  ncAvgRep:        $('nc-avg-rep'),
  peerTableBody:   $('peer-table-body'),
  // toast
  toastContainer:  $('toast-container'),
};

/* ──────────────────────────────────────────────────────────────────
   Settings persistence
────────────────────────────────────────────────────────────────── */

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    if (saved.peerId)   dom.inputPeerId.value  = saved.peerId;
    if (saved.modelId)  dom.inputModelId.value = saved.modelId;
    if (saved.dhtUrl)   dom.inputDhtUrl.value  = saved.dhtUrl;
  } catch (_) { /* ignore */ }
}

function saveSettings() {
  const settings = {
    peerId:  dom.inputPeerId.value.trim(),
    modelId: dom.inputModelId.value,
    dhtUrl:  dom.inputDhtUrl.value.trim(),
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

function readSettings() {
  state.peerId  = dom.inputPeerId.value.trim()  || generatePeerId();
  state.modelId = dom.inputModelId.value        || 'openhydra-qwen3.5-0.8b';
  state.dhtUrl  = dom.inputDhtUrl.value.trim()  || 'http://127.0.0.1:8468';
  // If we generated a peer ID, write it back into the field
  if (!dom.inputPeerId.value.trim()) {
    dom.inputPeerId.value = state.peerId;
  }
  saveSettings();
}

function generatePeerId() {
  const hex = () => Math.random().toString(16).slice(2, 10);
  return `${hex()}-${hex()}-${hex()}`;
}

/* ──────────────────────────────────────────────────────────────────
   Node lifecycle
────────────────────────────────────────────────────────────────── */

async function startNode() {
  if (state.nodeRunning || state.nodeStarting) return;
  readSettings();
  state.nodeStarting = true;
  updateNodeUI();
  dom.btnNodeToggle.disabled = true;

  try {
    const result = await tauriInvoke('start_node', {
      peerId:         state.peerId,
      dhtUrl:         state.dhtUrl,
      modelId:        state.modelId,
      compactionMode: state.compactionMode,
    });
    showToast(result || 'Node started', 'success');
    // Wait a moment for the process to bind its API port
    await sleep(1200);
    state.nodeStarting = false;
    state.nodeRunning  = true;
    startPolling();
    updateNodeUI();
    updateChips();
  } catch (err) {
    state.nodeStarting = false;
    showToast(`Failed to start node: ${err}`, 'error');
    updateNodeUI();
  } finally {
    dom.btnNodeToggle.disabled = false;
  }
}

async function stopNode() {
  if (!state.nodeRunning || state.nodeStopping) return;
  state.nodeStopping = true;
  dom.btnNodeToggle.disabled = true;

  // Cancel any active stream
  if (state.abortCtrl) {
    state.abortCtrl.abort();
    state.abortCtrl = null;
    state.streaming = false;
  }

  try {
    await tauriInvoke('stop_node');
    showToast('Node stopped', 'info');
  } catch (err) {
    showToast(`Stop error: ${err}`, 'error');
  }

  state.nodeStopping = false;
  state.nodeRunning  = false;
  stopPolling();
  updateNodeUI();
  dom.btnNodeToggle.disabled = false;
}

async function checkNodeStatus() {
  try {
    const running = await tauriInvoke('is_node_running');
    if (state.nodeRunning && !running) {
      // Process died unexpectedly
      state.nodeRunning = false;
      stopPolling();
      updateNodeUI();
      showToast('Node process stopped unexpectedly', 'error');
      return;
    }
    if (running) {
      const pid = await tauriInvoke('get_node_pid');
      dom.statPid.textContent = pid ?? '—';
      // Fetch live balance
      fetchBalance();
      // Fetch peer count from network status
      fetchNetworkSummary();
    }
  } catch (_) { /* process might be starting */ }
}

function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(checkNodeStatus, POLL_INTERVAL);
  // Immediate first check
  checkNodeStatus();
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

/* ──────────────────────────────────────────────────────────────────
   UI state rendering
────────────────────────────────────────────────────────────────── */

function updateNodeUI() {
  const running  = state.nodeRunning;
  const starting = state.nodeStarting;
  const stopping = state.nodeStopping;

  // Status dot
  dom.statusDot.className = `status-dot ${running ? 'running' : 'stopped'}`;

  // Status label
  if (starting)     dom.statusLabel.textContent = 'Starting…';
  else if (stopping) dom.statusLabel.textContent = 'Stopping…';
  else if (running)  dom.statusLabel.textContent = 'Running';
  else               dom.statusLabel.textContent = 'Stopped';

  // Toggle button
  if (running) {
    dom.btnNodeToggle.className = 'btn-node stop';
    dom.btnNodeToggle.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <rect x="3" y="3" width="18" height="18" rx="2"/>
      </svg>
      Stop Node`;
  } else {
    dom.btnNodeToggle.className = 'btn-node start';
    dom.btnNodeToggle.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <polygon points="5,3 19,12 5,21"/>
      </svg>
      ${starting ? 'Starting…' : 'Start Node'}`;
  }

  // Peer ID panel
  if (running && state.peerId) {
    dom.peerIdContainer.classList.remove('hidden');
    const short = state.peerId.length > 22
      ? state.peerId.slice(0, 10) + '…' + state.peerId.slice(-8)
      : state.peerId;
    dom.peerIdDisplay.textContent = short;
    dom.peerIdDisplay.title = `${state.peerId} (click to copy)`;
  } else {
    dom.peerIdContainer.classList.add('hidden');
  }

  // Network info sidebar section
  if (running) {
    dom.networkInfo.classList.add('visible');
    dom.statModel.textContent = state.modelId;
  } else {
    dom.networkInfo.classList.remove('visible');
    dom.statPeers.textContent   = '—';
    dom.statBalance.textContent = '0.00';
    dom.statPid.textContent     = '—';
  }

  // Chat input area
  const chatEnabled = running && !state.streaming;
  dom.chatInput.disabled = !chatEnabled;
  dom.btnSend.disabled   = !chatEnabled;
  dom.chatModelBadge.textContent = state.modelId || '—';

  if (running) {
    dom.nodeWarning.classList.add('hidden');
    dom.chatHint.textContent = 'Shift+Enter for newline';
  } else {
    dom.nodeWarning.classList.remove('hidden');
    dom.chatHint.textContent = 'Start the node to enable chat';
  }
}

/* ──────────────────────────────────────────────────────────────────
   Balance fetch
────────────────────────────────────────────────────────────────── */

async function fetchBalance() {
  if (!state.peerId) return;
  try {
    const res = await fetch(
      `${NODE_API_BASE}/v1/account/balance?peer_id=${encodeURIComponent(state.peerId)}`,
      { signal: AbortSignal.timeout(3000) }
    );
    if (res.ok) {
      const data    = await res.json();
      const hydra   = data.hydra_balance ?? data.balance ?? 0;
      const credits = data.barter_credits ?? 0;
      if (dom.statBalance) dom.statBalance.textContent = Number(hydra).toFixed(2);
      // Update the earnings widget
      updateEarningsWidget(hydra, credits);
    }
  } catch (_) {
    // Node might not have this endpoint yet — silently show 0.00
    if (dom.statBalance) dom.statBalance.textContent = '0.00';
    updateEarningsWidget(0, 0);
  }
}

/* ──────────────────────────────────────────────────────────────────
   Network status (sidebar summary + full panel)
────────────────────────────────────────────────────────────────── */

async function fetchNetworkSummary() {
  try {
    const res = await fetch(`${NODE_API_BASE}/v1/network/status`, {
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) return;
    const data = await res.json();
    const peers = data.peers ?? [];
    dom.statPeers.textContent = peers.length;
  } catch (_) {
    dom.statPeers.textContent = '—';
  }
}

async function fetchNetworkFull() {
  // Show loading state in cards
  [dom.ncTotalPeers, dom.ncModels, dom.ncRegion, dom.ncAvgRep].forEach(el => {
    el.textContent = '…';
  });
  dom.peerTableBody.innerHTML = `
    <tr><td colspan="4" class="empty-table">
      <div class="network-loading"><div class="spinner"></div>Fetching network status…</div>
    </td></tr>`;

  try {
    const res = await fetch(`${NODE_API_BASE}/v1/network/status`, {
      signal: AbortSignal.timeout(8000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    renderNetworkPanel(data);
  } catch (err) {
    dom.ncTotalPeers.textContent = '—';
    dom.ncModels.textContent     = '—';
    dom.ncRegion.textContent     = '—';
    dom.ncAvgRep.textContent     = '—';
    dom.peerTableBody.innerHTML  = `
      <tr><td colspan="4" class="empty-table">
        ${state.nodeRunning
          ? `Could not fetch network status: ${err.message}`
          : 'Start the node to view network peers.'}
      </td></tr>`;
  }
}

function renderNetworkPanel(data) {
  const peers = Array.isArray(data.peers) ? data.peers : [];

  // Compute summary stats
  const models  = [...new Set(peers.map(p => p.model).filter(Boolean))];
  const regions = [...new Set(peers.map(p => p.region).filter(Boolean))];
  const avgRep  = peers.length
    ? (peers.reduce((s, p) => s + (Number(p.reputation) || 0), 0) / peers.length).toFixed(1)
    : '—';

  dom.ncTotalPeers.textContent = peers.length || '0';
  dom.ncModels.textContent     = models.length  || '0';
  dom.ncRegion.textContent     = regions.length || '0';
  dom.ncAvgRep.textContent     = avgRep;

  if (peers.length === 0) {
    dom.peerTableBody.innerHTML = `
      <tr><td colspan="4" class="empty-table">No peers found on the network.</td></tr>`;
    return;
  }

  dom.peerTableBody.innerHTML = peers.map(peer => {
    const pid  = peer.peer_id  || '—';
    const pidShort = pid.length > 18 ? pid.slice(0, 8) + '…' + pid.slice(-6) : pid;
    const model    = peer.model      || '—';
    const region   = peer.region     || '—';
    const rep      = Number(peer.reputation) || 0;
    const repPct   = Math.min(100, Math.max(0, rep)).toFixed(0);
    return `
      <tr>
        <td title="${escHtml(pid)}">${escHtml(pidShort)}</td>
        <td><span class="tag">${escHtml(model)}</span></td>
        <td>${escHtml(region)}</td>
        <td>
          <div class="reputation-bar-wrap">
            <div class="reputation-bar">
              <div class="reputation-fill" style="width:${repPct}%"></div>
            </div>
            <span class="reputation-val">${repPct}</span>
          </div>
        </td>
      </tr>`;
  }).join('');

  // Also render the SVG peer map
  renderPeerMap(peers);
}

/* ──────────────────────────────────────────────────────────────────
   Chat — message management
────────────────────────────────────────────────────────────────── */

function addMessage(role, content) {
  // Hide empty state on first message
  if (dom.chatEmpty) dom.chatEmpty.style.display = 'none';

  const ts  = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const msg = { role, content, timestamp: ts, id: `msg-${Date.now()}-${Math.random()}` };
  state.messages.push(msg);

  const el = document.createElement('div');
  el.className = `message ${role}`;
  el.id = msg.id;
  el.innerHTML = `
    <div class="msg-avatar">${role === 'user' ? 'U' : 'H'}</div>
    <div>
      <div class="msg-bubble">${formatContent(content)}</div>
      <div class="msg-time">${ts}</div>
    </div>`;
  dom.chatMessages.appendChild(el);
  scrollToBottom();
  return msg.id;
}

function appendToMessage(msgId, chunk) {
  const el = document.getElementById(msgId);
  if (!el) return;
  const bubble = el.querySelector('.msg-bubble');
  if (!bubble) return;
  // Append raw text; update formatted display
  const msgObj = state.messages.find(m => m.id === msgId);
  if (msgObj) {
    msgObj.content += chunk;
    bubble.innerHTML = formatContent(msgObj.content);
    bubble.classList.add('streaming');
  }
  scrollToBottom();
}

function finaliseMessage(msgId) {
  const el = document.getElementById(msgId);
  if (el) {
    const bubble = el.querySelector('.msg-bubble');
    if (bubble) bubble.classList.remove('streaming');
  }
}

function addErrorMessage(text) {
  if (dom.chatEmpty) dom.chatEmpty.style.display = 'none';
  const ts  = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const el  = document.createElement('div');
  el.className = 'message assistant';
  el.innerHTML = `
    <div class="msg-avatar">!</div>
    <div>
      <div class="msg-bubble error">${escHtml(text)}</div>
      <div class="msg-time">${ts}</div>
    </div>`;
  dom.chatMessages.appendChild(el);
  scrollToBottom();
}

function scrollToBottom() {
  dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

/* ──────────────────────────────────────────────────────────────────
   Chat — SSE streaming inference
────────────────────────────────────────────────────────────────── */

async function sendMessage() {
  if (!state.nodeRunning || state.streaming) return;
  const text = dom.chatInput.value.trim();
  if (!text) return;

  // Ensure we have a conversation session for history
  await ensureSession(text);

  dom.chatInput.value = '';
  autoResizeTextarea();
  addMessage('user', text);

  // Persist user message to SQLite history
  await persistMessage('user', text);

  const messages = state.messages
    .filter(m => m.role === 'user' || m.role === 'assistant')
    .map(m => ({ role: m.role, content: m.content }));

  state.streaming  = true;
  state.abortCtrl  = new AbortController();
  updateChatInputState();

  const assistantMsgId = addMessage('assistant', '');
  // Mark as streaming immediately
  const bubble = document.getElementById(assistantMsgId)?.querySelector('.msg-bubble');
  if (bubble) bubble.classList.add('streaming');

  try {
    const res = await fetch(`${NODE_API_BASE}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model:    state.modelId,
        messages: messages,
        stream:   true,
      }),
      signal: state.abortCtrl.signal,
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => `HTTP ${res.status}`);
      finaliseMessage(assistantMsgId);
      addErrorMessage(`Node returned error: ${res.status} — ${errText}`);
    } else {
      await readSSEStream(res.body, assistantMsgId);
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      // User stopped node mid-stream — silently finalise
      finaliseMessage(assistantMsgId);
    } else {
      finaliseMessage(assistantMsgId);
      addErrorMessage(`Network error: ${err.message}`);
    }
  } finally {
    state.streaming = false;
    state.abortCtrl = null;
    updateChatInputState();

    // Persist the assistant reply to history
    const lastMsg = state.messages[state.messages.length - 1];
    if (lastMsg && lastMsg.role === 'assistant' && lastMsg.content) {
      await persistMessage('assistant', lastMsg.content);
    }
    await loadHistorySidebar();
  }
}

async function readSSEStream(body, msgId) {
  const reader  = body.getReader();
  const decoder = new TextDecoder();
  let   buffer  = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop(); // keep incomplete line

    for (const line of lines) {
      if (!line.startsWith('data:')) continue;
      const raw = line.slice(5).trim();
      if (raw === '[DONE]') break;
      try {
        const json  = JSON.parse(raw);
        const delta = json.choices?.[0]?.delta?.content ?? '';
        if (delta) appendToMessage(msgId, delta);
      } catch (_) { /* malformed SSE chunk — skip */ }
    }
  }

  finaliseMessage(msgId);
}

function updateChatInputState() {
  const enabled = state.nodeRunning && !state.streaming;
  dom.chatInput.disabled = !enabled;
  dom.btnSend.disabled   = !enabled;
  if (state.streaming) {
    dom.chatHint.textContent = 'Generating…';
  } else if (state.nodeRunning) {
    dom.chatHint.textContent = 'Shift+Enter for newline';
  } else {
    dom.chatHint.textContent = 'Start the node to enable chat';
  }
}

/* ──────────────────────────────────────────────────────────────────
   Textarea auto-resize
────────────────────────────────────────────────────────────────── */

function autoResizeTextarea() {
  const ta = dom.chatInput;
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
}

/* ──────────────────────────────────────────────────────────────────
   Tab switching
────────────────────────────────────────────────────────────────── */

function switchTab(name) {
  dom.tabBtns.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === name);
  });
  dom.tabPanels.forEach(panel => {
    const id = panel.id.replace('-panel', '');
    panel.classList.toggle('active', id === name);
  });
  if (name === 'network') fetchNetworkFull();
}

/* ──────────────────────────────────────────────────────────────────
   Toast notifications
────────────────────────────────────────────────────────────────── */

function showToast(message, type = 'info', duration = 4000) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = message;
  dom.toastContainer.appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity 0.3s';
    el.style.opacity    = '0';
    setTimeout(() => el.remove(), 320);
  }, duration);
}

/* ──────────────────────────────────────────────────────────────────
   Utilities
────────────────────────────────────────────────────────────────── */

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Very lightweight Markdown-ish formatter:
 * - ```code blocks```
 * - `inline code`
 * - **bold**
 * - line breaks
 */
function formatContent(text) {
  if (!text) return '';
  let html = escHtml(text);
  // Fenced code blocks
  html = html.replace(/```([\s\S]*?)```/g, (_, code) =>
    `<pre><code>${code.trim()}</code></pre>`
  );
  // Inline code
  html = html.replace(/`([^`]+)`/g, (_, code) => `<code>${code}</code>`);
  // Bold
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  // Newlines
  html = html.replace(/\n/g, '<br>');
  return html;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/* ──────────────────────────────────────────────────────────────────
   Event wiring
────────────────────────────────────────────────────────────────── */

function wireEvents() {
  // Node toggle button
  dom.btnNodeToggle.addEventListener('click', () => {
    if (state.nodeRunning) stopNode();
    else startNode();
  });

  // Tab bar
  dom.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Chat send button
  dom.btnSend.addEventListener('click', sendMessage);

  // Chat textarea — Enter to send, Shift+Enter for newline
  dom.chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  dom.chatInput.addEventListener('input', autoResizeTextarea);

  // Network refresh button
  dom.btnRefreshNet.addEventListener('click', fetchNetworkFull);

  // Peer ID copy on click
  dom.peerIdDisplay.addEventListener('click', () => {
    navigator.clipboard.writeText(state.peerId).then(() => {
      showToast('Peer ID copied to clipboard', 'info', 2000);
    }).catch(() => {});
  });

  // Settings auto-save on change
  [dom.inputPeerId, dom.inputModelId, dom.inputDhtUrl].forEach(el => {
    el.addEventListener('change', saveSettings);
  });

  // ── 6.6 additions ──

  // KV Compaction toggle
  wireCompactionToggle();

  // Model browser
  const btnRefreshModels = $('btn-refresh-models');
  if (btnRefreshModels) btnRefreshModels.addEventListener('click', fetchModelCatalog);
  const modelFilter    = $('model-filter');
  const modelTagFilter = $('model-tag-filter');
  if (modelFilter)    modelFilter.addEventListener('input', () => renderModelGrid(_allModels));
  if (modelTagFilter) modelTagFilter.addEventListener('change', () => renderModelGrid(_allModels));
  document.querySelectorAll('.tab-btn[data-tab="models"]').forEach(btn => {
    btn.addEventListener('click', () => { if (!_allModels.length) fetchModelCatalog(); });
  });

  // Theme toggle
  const themeBtn = $('theme-toggle');
  if (themeBtn) themeBtn.addEventListener('click', toggleTheme);

  // Command palette
  wirePalette();

  // Conversation history sidebar
  wireHistorySidebar();
}

/* ──────────────────────────────────────────────────────────────────
   Bootstrap
────────────────────────────────────────────────────────────────── */

async function init() {
  // Boot new features first
  initTheme();
  loadCompactionMode();

  loadSettings();
  wireEvents();
  updateNodeUI();
  updateChips();

  // If the process was already running before the window loaded (e.g. reopened),
  // check and sync state.
  try {
    const running = await tauriInvoke('is_node_running');
    if (running) {
      // Read settings so state.peerId etc. are populated
      readSettings();
      state.nodeRunning = true;
      startPolling();
      updateNodeUI();
      updateChips();
    }
  } catch (_) { /* not in Tauri, or node not started */ }

  // Initialise SQLite conversation history
  await initDb();

  // Show onboarding wizard on first launch
  if (!localStorage.getItem('openhydra_onboarded')) {
    showOnboarding();
  }
}

document.addEventListener('DOMContentLoaded', init);

/* ──────────────────────────────────────────────────────────────────
   6.6 — Compaction toggle
────────────────────────────────────────────────────────────────── */

function loadCompactionMode() {
  state.compactionMode = localStorage.getItem('openhydra_compaction') || 'off';
  document.querySelectorAll('#compaction-toggle .tgl-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === state.compactionMode);
  });
}

function wireCompactionToggle() {
  document.querySelectorAll('#compaction-toggle .tgl-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      state.compactionMode = btn.dataset.mode;
      localStorage.setItem('openhydra_compaction', state.compactionMode);
      document.querySelectorAll('#compaction-toggle .tgl-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.mode === state.compactionMode);
      });
      updateChips();
    });
  });
}

/* ──────────────────────────────────────────────────────────────────
   6.6 — Status chips
────────────────────────────────────────────────────────────────── */

function updateChips() {
  const chips     = $('status-chips');
  const chipModel = $('chip-model');
  const chipSess  = $('chip-session');
  const chipComp  = $('chip-compact');
  if (!chips) return;
  if (!state.nodeRunning) {
    chips.classList.add('hidden');
    return;
  }
  chips.classList.remove('hidden');
  if (chipModel) chipModel.textContent = state.modelId || 'unknown';
  if (chipSess)  chipSess.textContent  = 'no session';
  if (chipComp) {
    if (state.compactionMode !== 'off') {
      chipComp.textContent = `⚡ compact: ${state.compactionMode}`;
      chipComp.classList.remove('hidden');
    } else {
      chipComp.classList.add('hidden');
    }
  }
}

/* ──────────────────────────────────────────────────────────────────
   6.6 — Model browser
────────────────────────────────────────────────────────────────── */

let _allModels = [];

function renderModelCard(m) {
  const online = (m.healthy_peers || 0) >= (m.required_replicas || 1);
  const tags   = (m.tags || []).map(t => `<span class="model-tag">${t}</span>`).join('');
  const vram   = m.min_vram_gb ? `${m.min_vram_gb}GB` : '?';
  const ctx    = m.context_length ? Math.round(m.context_length / 1000) + 'k' : '—';
  const quant  = m.recommended_quantization || 'fp32';
  const isSelected = m.id === state.modelId;
  return `
    <div class="model-card ${online ? 'online' : 'offline'}${isSelected ? ' selected' : ''}">
      <div class="model-card-header">
        <span class="model-card-name">${m.id}</span>
        <span class="model-badge ${online ? 'badge-green' : 'badge-gray'}">
          ${online ? '● online' : '○ offline'}
        </span>
      </div>
      <p class="model-card-desc">${m.description || ''}</p>
      <div class="model-card-meta">
        <span>VRAM: ${vram}</span>
        <span>Quant: ${quant}</span>
        <span>Ctx: ${ctx}</span>
        <span>Peers: ${m.required_replicas || 1}</span>
      </div>
      <div class="model-tags">${tags}</div>
      <button class="btn-secondary btn-sm" onclick="selectModel('${m.id}')">
        ${isSelected ? '✓ Active' : 'Use this model'}
      </button>
    </div>`;
}

function selectModel(modelId) {
  state.modelId = modelId;
  dom.inputModelId.value = modelId;
  saveSettings();
  updateChips();
  renderModelGrid(_allModels);
  showToast(`Model set to ${modelId}`, 'success', 2000);
}

function renderModelGrid(models) {
  const grid = $('model-cards-grid');
  if (!grid) return;
  const filterText = ($('model-filter') || {}).value?.toLowerCase() || '';
  const filterTag  = ($('model-tag-filter') || {}).value || '';
  const filtered = models.filter(m => {
    if (filterText && !m.id.toLowerCase().includes(filterText) &&
        !(m.description || '').toLowerCase().includes(filterText)) return false;
    if (filterTag && !(m.tags || []).includes(filterTag)) return false;
    return true;
  });
  if (!filtered.length) {
    grid.innerHTML = '<p style="color:var(--text-muted);padding:1rem;">No models match the filter.</p>';
    return;
  }
  grid.innerHTML = filtered.map(renderModelCard).join('');
}

async function fetchModelCatalog() {
  try {
    const resp = await fetch(`${NODE_API_BASE}/v1/models`);
    if (!resp.ok) throw new Error(resp.status);
    const data = await resp.json();
    _allModels = Array.isArray(data) ? data : (data.data || []);
  } catch (_) {
    // Static fallback — small subset when node is not running
    _allModels = [
      { id: 'openhydra-qwen3.5-0.8b',    tags: ['chat','small','fast'],  description: 'Qwen 3.5 0.8B — single-peer.',  min_vram_gb: 2,  recommended_quantization: 'fp32', context_length: 32768,  required_replicas: 1, healthy_peers: 0 },
      { id: 'openhydra-llama3.2-1b',      tags: ['chat','small'],         description: 'Llama 3.2 1B.',                  min_vram_gb: 3,  recommended_quantization: 'fp32', context_length: 131072, required_replicas: 1, healthy_peers: 0 },
      { id: 'openhydra-qwen3-4b',         tags: ['chat','medium'],        description: 'Qwen3 4B — balanced.',           min_vram_gb: 9,  recommended_quantization: 'int4', context_length: 32768,  required_replicas: 1, healthy_peers: 0 },
      { id: 'openhydra-llama3.1-8b',      tags: ['chat','medium'],        description: 'Llama 3.1 8B Instruct.',         min_vram_gb: 16, recommended_quantization: 'int8', context_length: 131072, required_replicas: 2, healthy_peers: 0 },
      { id: 'openhydra-qwen2.5-coder-7b', tags: ['code','medium'],        description: 'Qwen 2.5 Coder 7B.',            min_vram_gb: 14, recommended_quantization: 'int8', context_length: 32768,  required_replicas: 2, healthy_peers: 0 },
    ];
  }
  renderModelGrid(_allModels);
}

/* ──────────────────────────────────────────────────────────────────
   6.6 — Onboarding wizard
────────────────────────────────────────────────────────────────── */

const _ONBOARD_MODELS = [
  'openhydra-qwen3.5-0.8b',
  'openhydra-llama3.2-1b',
  'openhydra-phi3-mini',
  'openhydra-smollm2-1.7b',
  'openhydra-qwen3-4b',
];

let _onboardSelectedModel = _ONBOARD_MODELS[0];

function showOnboarding() {
  const overlay = $('onboarding-overlay');
  if (!overlay) return;
  overlay.classList.remove('hidden');

  // Populate model list for step 2
  const list = $('onboard-model-list');
  if (list) {
    list.innerHTML = _ONBOARD_MODELS.map(m =>
      `<label class="onboard-model-item">
        <input type="radio" name="onboard-model" value="${m}" ${m === _onboardSelectedModel ? 'checked' : ''} />
        <span>${m}</span>
      </label>`
    ).join('');
    list.querySelectorAll('input[type=radio]').forEach(r => {
      r.addEventListener('change', () => { _onboardSelectedModel = r.value; });
    });
  }

  // Step navigation helpers
  const stepPanels = [null, $('onboard-step-1'), $('onboard-step-2'), $('onboard-step-3')];
  const stepDots   = document.querySelectorAll('.onboard-step-dot');
  function goTo(n) {
    stepPanels.forEach((s, i) => { if (s) s.classList.toggle('hidden', i !== n); });
    stepDots.forEach(d => d.classList.toggle('active', parseInt(d.dataset.step) <= n));
  }

  const elNext1  = $('onboard-next-1');
  const elNext2  = $('onboard-next-2');
  const elBack2  = $('onboard-back-2');
  const elBack3  = $('onboard-back-3');
  const elFinish = $('onboard-finish');
  const elAuto   = $('onboard-autogen');
  const elSkip   = $('onboard-skip');

  if (elNext1)  elNext1.onclick  = () => goTo(2);
  if (elNext2)  elNext2.onclick  = () => goTo(3);
  if (elBack2)  elBack2.onclick  = () => goTo(1);
  if (elBack3)  elBack3.onclick  = () => goTo(2);
  if (elAuto)   elAuto.onclick   = () => {
    const fld = $('onboard-peer-id');
    if (fld) fld.value = generatePeerId();
  };
  if (elSkip)   elSkip.onclick   = () => {
    localStorage.setItem('openhydra_onboarded', '1');
    overlay.classList.add('hidden');
  };
  if (elFinish) elFinish.onclick = () => {
    const peerFld = $('onboard-peer-id');
    const dhtFld  = $('onboard-dht-url');
    if (peerFld && peerFld.value.trim()) dom.inputPeerId.value = peerFld.value.trim();
    if (dhtFld  && dhtFld.value.trim())  dom.inputDhtUrl.value = dhtFld.value.trim();
    dom.inputModelId.value = _onboardSelectedModel;
    state.modelId          = _onboardSelectedModel;
    saveSettings();
    localStorage.setItem('openhydra_onboarded', '1');
    overlay.classList.add('hidden');
    startNode();
  };

  goTo(1);
}

// Extend state with compactionMode (initialised before loadCompactionMode)
state.compactionMode = 'off';

/* ──────────────────────────────────────────────────────────────────
   Feature: Dark / Light theme switch
────────────────────────────────────────────────────────────────── */

function initTheme() {
  const saved = localStorage.getItem('openhydra_theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
  const btn = $('theme-toggle');
  if (btn) btn.textContent = saved === 'dark' ? '🌙' : '☀️';
}

function toggleTheme() {
  const cur  = document.documentElement.getAttribute('data-theme') || 'light';
  const next = cur === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('openhydra_theme', next);
  const btn = $('theme-toggle');
  if (btn) btn.textContent = next === 'dark' ? '🌙' : '☀️';
}

/* ──────────────────────────────────────────────────────────────────
   Feature: Enhanced HYDRA earnings widget
────────────────────────────────────────────────────────────────── */

function updateEarningsWidget(hydraBalance, barterCredits) {
  const elHydra   = $('hydra-balance');
  const elCredits = $('barter-credits');
  const elBar     = $('earnings-bar');
  if (elHydra)   elHydra.textContent   = Number(hydraBalance || 0).toFixed(4) + ' HYDRA';
  if (elCredits) elCredits.textContent = Number(barterCredits || 0).toFixed(2);
  if (elBar) {
    const pct = Math.min(100, ((barterCredits || 0) / 100) * 100);
    elBar.style.width = pct + '%';
  }
}

/* ──────────────────────────────────────────────────────────────────
   Feature: ⌘K Command palette
────────────────────────────────────────────────────────────────── */

let _paletteIdx  = 0;
let _paletteOpen = false;
let _paletteMatches = [];

// Base command list — models are injected after fetchModelCatalog()
const PALETTE_COMMANDS = [
  { type: 'cmd',   label: 'Start Node',          action: () => startNode() },
  { type: 'cmd',   label: 'Stop Node',           action: () => stopNode() },
  { type: 'cmd',   label: 'New Chat',             action: () => newChatSession() },
  { type: 'cmd',   label: 'Toggle Theme',         action: () => toggleTheme() },
  { type: 'nav',   label: 'Go to Chat',           action: () => switchTab('chat') },
  { type: 'nav',   label: 'Go to Models',         action: () => switchTab('models') },
  { type: 'nav',   label: 'Go to Network',        action: () => switchTab('network') },
  { type: 'nav',   label: 'Go to About',          action: () => switchTab('about') },
  { type: 'cmd',   label: 'Refresh Network',      action: () => fetchNetworkFull() },
  { type: 'cmd',   label: 'Compaction: Off',      action: () => setCompaction('off') },
  { type: 'cmd',   label: 'Compaction: Auto',     action: () => setCompaction('auto') },
  { type: 'cmd',   label: 'Compaction: On',       action: () => setCompaction('on') },
];

function openPalette() {
  _paletteOpen = true;
  _paletteIdx = 0;
  const overlay = $('palette-overlay');
  const input   = $('palette-input');
  if (overlay) overlay.classList.remove('hidden');
  if (input)   { input.value = ''; input.focus(); }
  renderPaletteResults('');
}

function closePalette() {
  _paletteOpen = false;
  const overlay = $('palette-overlay');
  if (overlay) overlay.classList.add('hidden');
}

function buildPaletteItems() {
  const items = [...PALETTE_COMMANDS];
  // Inject models from the model browser catalog
  (_allModels || []).forEach(m => {
    items.push({ type: 'model', label: `Use ${m.id}`, action: () => selectModel(m.id) });
  });
  return items;
}

function renderPaletteResults(query) {
  const q    = query.trim().toLowerCase();
  const all  = buildPaletteItems();
  _paletteMatches = q
    ? all.filter(i => i.label.toLowerCase().includes(q))
    : all;
  _paletteMatches = _paletteMatches.slice(0, 9);

  const container = $('palette-results');
  if (!container) return;

  if (_paletteMatches.length === 0) {
    container.innerHTML = '<div style="padding:16px;text-align:center;color:var(--text-dim);font-size:13px;">No results</div>';
    return;
  }

  container.innerHTML = _paletteMatches.map((item, i) =>
    `<div class="palette-item${i === _paletteIdx ? ' selected' : ''}" data-idx="${i}">
       <span class="palette-type ${item.type}">${item.type}</span>
       <span class="palette-label">${escHtml(item.label)}</span>
     </div>`
  ).join('');

  // Wire mouse clicks
  container.querySelectorAll('.palette-item').forEach(el => {
    el.addEventListener('mousedown', e => {
      e.preventDefault();
      const idx = parseInt(el.dataset.idx);
      executePaletteItem(idx);
    });
  });
}

function executePaletteItem(idx) {
  const item = _paletteMatches[idx];
  if (!item) return;
  closePalette();
  item.action();
}

function setCompaction(mode) {
  state.compactionMode = mode;
  localStorage.setItem('openhydra_compaction', mode);
  document.querySelectorAll('#compaction-toggle .tgl-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  updateChips();
  showToast(`Compaction: ${mode}`, 'info', 1500);
}

function wirePalette() {
  const overlay = $('palette-overlay');
  const input   = $('palette-input');
  if (!overlay || !input) return;

  // Close on backdrop click
  overlay.addEventListener('mousedown', e => {
    if (e.target === overlay) closePalette();
  });

  // Filter on input
  input.addEventListener('input', () => {
    _paletteIdx = 0;
    renderPaletteResults(input.value);
  });

  // Keyboard navigation
  input.addEventListener('keydown', e => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      _paletteIdx = Math.min(_paletteIdx + 1, _paletteMatches.length - 1);
      renderPaletteResults(input.value);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      _paletteIdx = Math.max(_paletteIdx - 1, 0);
      renderPaletteResults(input.value);
    } else if (e.key === 'Enter') {
      e.preventDefault();
      executePaletteItem(_paletteIdx);
    } else if (e.key === 'Escape') {
      closePalette();
    }
  });

  // Global Cmd+K / Ctrl+K
  document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      if (_paletteOpen) closePalette();
      else openPalette();
    } else if (e.key === 'Escape' && _paletteOpen) {
      closePalette();
    }
  });
}

/* ──────────────────────────────────────────────────────────────────
   Feature: Conversation history sidebar (SQLite via Tauri plugin)
────────────────────────────────────────────────────────────────── */

let _db = null;
let _currentSessionId = null;
let _currentSessionTitle = null;

async function initDb() {
  if (!isTauri) return; // skip in browser preview
  try {
    // Dynamic import so the app still works without the plugin in browser mode
    const { default: Database } = await import('@tauri-apps/plugin-sql');
    _db = await Database.load('sqlite:history.db');
    await _db.execute(`
      CREATE TABLE IF NOT EXISTS sessions (
        id         TEXT PRIMARY KEY,
        title      TEXT NOT NULL,
        model_id   TEXT,
        created_at INTEGER NOT NULL
      );
      CREATE TABLE IF NOT EXISTS messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL,
        role        TEXT NOT NULL,
        content     TEXT NOT NULL,
        created_at  INTEGER NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
      );
    `);
    await loadHistorySidebar();
  } catch (err) {
    console.warn('[OpenHydra] SQLite history unavailable:', err);
  }
}

async function ensureSession(firstMessage) {
  if (_currentSessionId) return _currentSessionId;
  _currentSessionId    = `sess-${Date.now()}-${Math.random().toString(36).slice(2,7)}`;
  _currentSessionTitle = firstMessage.slice(0, 40) + (firstMessage.length > 40 ? '…' : '');
  if (_db) {
    await _db.execute(
      'INSERT INTO sessions (id, title, model_id, created_at) VALUES (?, ?, ?, ?)',
      [_currentSessionId, _currentSessionTitle, state.modelId, Date.now()]
    );
  }
  return _currentSessionId;
}

async function persistMessage(role, content) {
  if (!_db || !_currentSessionId) return;
  try {
    await _db.execute(
      'INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)',
      [_currentSessionId, role, content, Date.now()]
    );
  } catch (err) {
    console.warn('[OpenHydra] Failed to persist message:', err);
  }
}

async function loadHistorySidebar(filterQuery = '') {
  const container = $('history-list');
  if (!container) return;

  if (!_db) {
    container.innerHTML = '<div class="history-empty">History unavailable</div>';
    return;
  }

  try {
    const q = filterQuery.trim().toLowerCase();
    const sessions = await _db.select(
      'SELECT * FROM sessions ORDER BY created_at DESC LIMIT 60'
    );
    const filtered = q
      ? sessions.filter(s => s.title.toLowerCase().includes(q))
      : sessions;

    if (filtered.length === 0) {
      container.innerHTML = '<div class="history-empty">No conversations yet</div>';
      return;
    }

    container.innerHTML = filtered.map(s =>
      `<div class="history-item${s.id === _currentSessionId ? ' active' : ''}" data-id="${escHtml(s.id)}">
         <span class="history-title">${escHtml(s.title)}</span>
         <span class="history-date">${timeAgo(s.created_at)}</span>
       </div>`
    ).join('');

    container.querySelectorAll('.history-item').forEach(el => {
      el.addEventListener('click', () => restoreSession(el.dataset.id));
    });
  } catch (err) {
    container.innerHTML = '<div class="history-empty">Error loading history</div>';
  }
}

async function restoreSession(sessionId) {
  if (!_db) return;
  try {
    const messages = await _db.select(
      'SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC',
      [sessionId]
    );
    // Clear current chat
    state.messages = [];
    dom.chatMessages.innerHTML = '';
    if (dom.chatEmpty) dom.chatEmpty.style.display = 'none';

    // Re-render messages
    messages.forEach(m => {
      addMessage(m.role, m.content);
    });

    _currentSessionId = sessionId;
    switchTab('chat');
    await loadHistorySidebar();
  } catch (err) {
    showToast('Could not restore session', 'error');
  }
}

function newChatSession() {
  state.messages = [];
  _currentSessionId    = null;
  _currentSessionTitle = null;
  dom.chatMessages.innerHTML = '';
  if (dom.chatEmpty) dom.chatEmpty.style.display = '';
  switchTab('chat');
  loadHistorySidebar();
}

function timeAgo(ts) {
  const diff = Date.now() - ts;
  const mins  = Math.floor(diff / 60000);
  if (mins < 1)   return 'just now';
  if (mins < 60)  return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days  = Math.floor(hours / 24);
  return `${days}d ago`;
}

function wireHistorySidebar() {
  const btnNew   = $('btn-new-chat');
  const searchEl = $('history-search');
  if (btnNew)   btnNew.addEventListener('click', newChatSession);
  if (searchEl) searchEl.addEventListener('input', () => loadHistorySidebar(searchEl.value));
}

/* ──────────────────────────────────────────────────────────────────
   Feature: SVG peer map
────────────────────────────────────────────────────────────────── */

// Map region label → approximate SVG pixel coords (viewBox 800×380)
// Equirectangular projection: x = (lon+180)/360 * 800, y = (90-lat)/180 * 380
const REGION_COORDS = {
  'us-east':      { x: 175, y: 138 },   // Newark ~74W, 41N
  'us-west':      { x:  90, y: 135 },   // Fremont ~122W, 37N
  'us-central':   { x: 145, y: 140 },   // Dallas ~97W, 33N
  'eu-central':   { x: 398, y:  83 },   // Frankfurt ~8E, 50N
  'eu-west':      { x: 370, y:  82 },   // London ~0W, 51N
  'eu-north':     { x: 402, y:  58 },   // Stockholm ~18E, 59N
  'ap-south':     { x: 573, y: 155 },   // Singapore ~103E, 1N
  'ap-southeast': { x: 620, y: 175 },   // Sydney ~151E, -34N mapped
  'ap-north':     { x: 640, y: 103 },   // Tokyo ~139E, 35N
  'ap-west':      { x: 530, y: 140 },   // Mumbai ~72E, 19N
  'ca-central':   { x: 158, y: 118 },   // Toronto ~79W, 44N
  'sa-east':      { x: 198, y: 270 },   // São Paulo ~46W, -23N
};

function latencyColor(ms) {
  if (ms === null || ms === undefined || ms < 0) return '#888888';
  if (ms < 100)  return '#4caf50';
  if (ms < 300)  return '#ffb300';
  return '#e05252';
}

function guessRegion(peer) {
  // Try to infer region from peer_id, dht_url, or region field
  const src = ((peer.region || '') + (peer.dht_url || '') + (peer.peer_id || '')).toLowerCase();
  for (const key of Object.keys(REGION_COORDS)) {
    if (src.includes(key.replace('-', ''))) return key;
  }
  if (src.includes('bootstrap-us')) return 'us-east';
  if (src.includes('bootstrap-eu')) return 'eu-central';
  if (src.includes('bootstrap-ap')) return 'ap-south';
  return null;
}

function renderPeerMap(peers) {
  const g       = $('peer-dots');
  const counter = $('map-peer-count');
  if (!g) return;

  if (counter) counter.textContent = `${peers.length} peer${peers.length !== 1 ? 's' : ''}`;

  if (peers.length === 0) {
    g.innerHTML = '<text x="400" y="195" text-anchor="middle" fill="#555" font-size="12">No peers connected</text>';
    return;
  }

  // Group peers by region to jitter overlapping dots
  const regionCounts = {};
  g.innerHTML = peers.map(peer => {
    const region = peer.region || guessRegion(peer);
    const base   = REGION_COORDS[region] || { x: 400, y: 190 };

    // Count how many we've placed in this region already
    regionCounts[region] = (regionCounts[region] || 0) + 1;
    const jitter = regionCounts[region] - 1;
    const angle  = (jitter * 137.5) * Math.PI / 180; // golden angle
    const radius = Math.min(jitter * 7, 25);
    const x = base.x + Math.cos(angle) * radius;
    const y = base.y + Math.sin(angle) * radius;

    const col  = latencyColor(peer.latency_ms);
    const lat  = peer.latency_ms != null ? `${peer.latency_ms}ms` : '?ms';
    const pid  = (peer.peer_id || '—').slice(0, 12);
    const model = peer.model || '—';

    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6"
              fill="${col}" opacity="0.85"
              stroke="rgba(0,0,0,0.35)" stroke-width="1">
              <title>${pid} • ${model} • ${lat} • ${region || 'unknown'}</title>
            </circle>
            <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="10"
              fill="${col}" opacity="0.15" class="peer-pulse"/>`;
  }).join('');
}

/* ──────────────────────────────────────────────────────────────────
   NOTE: All feature logic (theme, earnings, palette, history,
   peer map) is integrated directly into the original functions
   above. No monkey-patching wrappers needed.
────────────────────────────────────────────────────────────────── */
