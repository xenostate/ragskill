(function () {
  "use strict";

  // ── Config from script tag ─────────────────────────────────────────────
  const scriptTag = document.currentScript;
  const SITE_ID = scriptTag?.getAttribute("data-site-id") || "1";
  const API_URL = scriptTag?.getAttribute("data-api") || window.location.origin;
  const TITLE = scriptTag?.getAttribute("data-title") || "Ask a question";
  const COLOR = scriptTag?.getAttribute("data-color") || "#2563eb";
  const POSITION = scriptTag?.getAttribute("data-position") || "right";
  const DEFAULT_PLACEHOLDER = "Type your question...";
  const titleLocked = Boolean(scriptTag?.hasAttribute("data-title"));

  // ── Session ────────────────────────────────────────────────────────────
  const SESSION_KEY = `wr_session_${SITE_ID}`;
  let sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = "s_" + Math.random().toString(36).slice(2, 12);
    localStorage.setItem(SESSION_KEY, sessionId);
  }

  const SESSION_TS_KEY = `wr_session_ts_${SITE_ID}`;
  const SESSION_MAX_AGE = 24 * 60 * 60 * 1000;
  const storedTs = parseInt(localStorage.getItem(SESSION_TS_KEY) || "0", 10);
  if (Date.now() - storedTs > SESSION_MAX_AGE) {
    sessionId = "s_" + Math.random().toString(36).slice(2, 12);
    localStorage.setItem(SESSION_KEY, sessionId);
    localStorage.setItem(SESSION_TS_KEY, String(Date.now()));
  } else if (!localStorage.getItem(SESSION_TS_KEY)) {
    localStorage.setItem(SESSION_TS_KEY, String(Date.now()));
  }

  // ── Assistant config state ─────────────────────────────────────────────
  let assistantConfig = {
    display: {},
    greeting: { enabled: false, message: "", show_once: true, delay_ms: 0 },
    starters: [],
    forms: [],
  };
  let assistantConfigLoaded = false;
  let initialContentScheduled = false;
  let greetingRendered = false;
  let startersRendered = false;
  let isOpen = false;

  // ── Page-view beacon (fire-and-forget, never blocks the widget) ───────
  try {
    fetch(`${API_URL}/api/track`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        site_id: parseInt(SITE_ID, 10),
        session_id: sessionId,
        referer: window.location.href,
      }),
      keepalive: true,
    }).catch(() => {});
  } catch (e) {}

  // ── Create shadow DOM container ────────────────────────────────────────
  const host = document.createElement("div");
  host.id = "web-rag-widget";
  document.body.appendChild(host);
  const shadow = host.attachShadow({ mode: "closed" });

  // ── Styles ─────────────────────────────────────────────────────────────
  const style = document.createElement("style");
  style.textContent = `
    * { box-sizing: border-box; margin: 0; padding: 0; }

    .wr-bubble {
      position: fixed;
      bottom: 24px;
      ${POSITION}: 24px;
      width: 56px;
      height: 56px;
      border-radius: 50%;
      background: ${COLOR};
      color: #fff;
      border: none;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(0,0,0,0.25);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 999999;
      transition: transform 0.2s;
    }
    .wr-bubble:hover { transform: scale(1.08); }
    .wr-bubble svg { width: 28px; height: 28px; }

    .wr-panel {
      position: fixed;
      bottom: 92px;
      ${POSITION}: 24px;
      width: 380px;
      max-width: calc(100vw - 48px);
      height: 520px;
      max-height: calc(100vh - 120px);
      background: #fff;
      border-radius: 16px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.18);
      display: none;
      flex-direction: column;
      overflow: hidden;
      z-index: 999998;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px;
      color: #1a1a1a;
    }
    .wr-panel.open { display: flex; }

    .wr-header {
      padding: 16px;
      background: ${COLOR};
      color: #fff;
      font-weight: 600;
      font-size: 15px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .wr-close {
      background: none;
      border: none;
      color: #fff;
      cursor: pointer;
      font-size: 20px;
      line-height: 1;
      opacity: 0.8;
    }
    .wr-close:hover { opacity: 1; }

    .wr-messages {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .wr-msg {
      max-width: 88%;
      padding: 10px 14px;
      border-radius: 12px;
      line-height: 1.5;
      word-wrap: break-word;
    }
    .wr-msg.user {
      align-self: flex-end;
      background: ${COLOR};
      color: #fff;
      border-bottom-right-radius: 4px;
    }
    .wr-msg.bot {
      align-self: flex-start;
      background: #f0f0f0;
      color: #1a1a1a;
      border-bottom-left-radius: 4px;
    }
    .wr-msg.bot a {
      color: ${COLOR};
      text-decoration: underline;
    }

    .wr-card {
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-width: 250px;
    }
    .wr-card-title {
      font-size: 13px;
      font-weight: 600;
    }
    .wr-card-text {
      font-size: 13px;
      color: #4b5563;
      line-height: 1.5;
    }
    .wr-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .wr-chip {
      padding: 8px 12px;
      border: 1px solid rgba(37, 99, 235, 0.2);
      border-radius: 999px;
      background: #fff;
      color: ${COLOR};
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
      transition: all 0.2s;
    }
    .wr-chip:hover {
      background: ${COLOR};
      color: #fff;
      border-color: ${COLOR};
    }

    .wr-form {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .wr-field {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .wr-field label {
      font-size: 12px;
      color: #4b5563;
      font-weight: 600;
    }
    .wr-field input,
    .wr-field textarea,
    .wr-field select {
      width: 100%;
      border: 1px solid #d1d5db;
      border-radius: 10px;
      padding: 9px 10px;
      font: inherit;
      outline: none;
      background: #fff;
      color: #111827;
    }
    .wr-field textarea {
      min-height: 76px;
      resize: vertical;
    }
    .wr-field input:focus,
    .wr-field textarea:focus,
    .wr-field select:focus {
      border-color: ${COLOR};
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
    }
    .wr-error-text {
      font-size: 11px;
      color: #b91c1c;
    }
    .wr-inline-status {
      font-size: 12px;
      color: #4b5563;
    }
    .wr-submit {
      border: none;
      background: ${COLOR};
      color: #fff;
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    }
    .wr-submit:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }

    .wr-typing {
      align-self: flex-start;
      padding: 10px 14px;
      background: #f0f0f0;
      border-radius: 12px;
      display: none;
    }
    .wr-typing.active { display: block; }
    .wr-typing span {
      display: inline-block;
      width: 6px;
      height: 6px;
      background: #999;
      border-radius: 50%;
      margin: 0 2px;
      animation: wr-bounce 1.2s infinite;
    }
    .wr-typing span:nth-child(2) { animation-delay: 0.2s; }
    .wr-typing span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes wr-bounce {
      0%, 60%, 100% { transform: translateY(0); }
      30% { transform: translateY(-6px); }
    }

    .wr-sources {
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid #ddd;
      font-size: 12px;
      color: #666;
    }
    .wr-sources a {
      color: ${COLOR};
      text-decoration: none;
      display: block;
      margin-top: 2px;
    }
    .wr-sources a:hover { text-decoration: underline; }

    .wr-input-row {
      padding: 12px;
      border-top: 1px solid #eee;
      display: flex;
      gap: 8px;
    }
    .wr-input {
      flex: 1;
      padding: 10px 14px;
      border: 1px solid #ddd;
      border-radius: 24px;
      outline: none;
      font-size: 14px;
      font-family: inherit;
    }
    .wr-input:focus { border-color: ${COLOR}; }
    .wr-send {
      width: 40px;
      height: 40px;
      border-radius: 50%;
      border: none;
      background: ${COLOR};
      color: #fff;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .wr-send:disabled { opacity: 0.5; cursor: not-allowed; }
    .wr-send svg { width: 18px; height: 18px; }

    .wr-confidence {
      font-size: 11px;
      padding: 2px 8px;
      border-radius: 8px;
      display: inline-block;
      margin-bottom: 4px;
    }
    .wr-confidence.high { background: #dcfce7; color: #166534; }
    .wr-confidence.medium { background: #fef9c3; color: #854d0e; }
    .wr-confidence.low { background: #fecaca; color: #991b1b; }

    @media (max-width: 560px) {
      .wr-panel {
        width: calc(100vw - 24px);
        max-width: calc(100vw - 24px);
        ${POSITION}: 12px;
        bottom: 82px;
        height: min(70vh, 520px);
      }
      .wr-bubble {
        ${POSITION}: 12px;
        bottom: 16px;
      }
      .wr-msg { max-width: 94%; }
    }
  `;
  shadow.appendChild(style);

  // ── HTML ───────────────────────────────────────────────────────────────
  const bubble = document.createElement("button");
  bubble.className = "wr-bubble";
  bubble.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>
  </svg>`;
  shadow.appendChild(bubble);

  const panel = document.createElement("div");
  panel.className = "wr-panel";
  panel.innerHTML = `
    <div class="wr-header">
      <span class="wr-header-title">${escapeHtml(TITLE)}</span>
      <button class="wr-close">&times;</button>
    </div>
    <div class="wr-messages">
      <div class="wr-typing"><span></span><span></span><span></span></div>
    </div>
    <div class="wr-input-row">
      <input class="wr-input" placeholder="${escapeHtml(DEFAULT_PLACEHOLDER)}" />
      <button class="wr-send">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
        </svg>
      </button>
    </div>
  `;
  shadow.appendChild(panel);

  const headerTitle = panel.querySelector(".wr-header-title");
  const messages = panel.querySelector(".wr-messages");
  const typing = panel.querySelector(".wr-typing");
  const input = panel.querySelector(".wr-input");
  const sendBtn = panel.querySelector(".wr-send");
  const closeBtn = panel.querySelector(".wr-close");

  // ── Utilities ──────────────────────────────────────────────────────────
  function escapeHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function getGreetingStorageKey() {
    return `wr_greeting_shown_${SITE_ID}`;
  }

  function scrollToBottom() {
    messages.scrollTop = messages.scrollHeight;
  }

  function safeSourceUrl(url) {
    try {
      const parsed = new URL(url, window.location.origin);
      if (parsed.protocol === "http:" || parsed.protocol === "https:") {
        return parsed.href;
      }
    } catch (e) {}
    return "#";
  }

  function renderBotHtml(text, sources, confidence) {
    let html = "";

    if (confidence) {
      html += `<span class="wr-confidence ${escapeHtml(confidence)}">${escapeHtml(confidence)}</span><br>`;
    }

    html += escapeHtml(text)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, url) => {
        const href = safeSourceUrl(url);
        return href === "#"
          ? escapeHtml(label)
          : `<a href="${href}" target="_blank" rel="noopener">${escapeHtml(label)}</a>`;
      })
      .replace(/\n/g, "<br>");

    if (sources && sources.length > 0) {
      html += `<div class="wr-sources"><strong>Sources:</strong>`;
      sources.forEach((source, index) => {
        html += `<a href="${safeSourceUrl(source.url)}" target="_blank" rel="noopener">[${index + 1}] ${escapeHtml(source.title)}</a>`;
      });
      html += `</div>`;
    }

    return html;
  }

  function addMessage(text, type, sources, confidence) {
    const msg = document.createElement("div");
    msg.className = `wr-msg ${type}`;

    if (type === "bot") {
      msg.innerHTML = renderBotHtml(text, sources, confidence);
    } else {
      msg.textContent = text;
    }

    messages.insertBefore(msg, typing);
    scrollToBottom();
    return msg;
  }

  function addBotCard(buildCard) {
    const msg = document.createElement("div");
    msg.className = "wr-msg bot";
    const card = document.createElement("div");
    card.className = "wr-card";
    buildCard(card, msg);
    msg.appendChild(card);
    messages.insertBefore(msg, typing);
    scrollToBottom();
    return { msg, card };
  }

  function getConfiguredForm(formId) {
    return (assistantConfig.forms || []).find((form) => form.id === formId) || null;
  }

  function applyDisplayConfig() {
    const display = assistantConfig.display || {};
    if (!titleLocked && display.title) {
      headerTitle.textContent = display.title;
    }
    input.placeholder = display.input_placeholder || DEFAULT_PLACEHOLDER;
  }

  // ── Assistant config loading ───────────────────────────────────────────
  async function loadAssistantConfig() {
    try {
      const resp = await fetch(`${API_URL}/api/widget/config/${encodeURIComponent(SITE_ID)}`);
      if (!resp.ok) return;
      const data = await resp.json();
      if (data && data.assistant) {
        assistantConfig = data.assistant;
      }
    } catch (e) {
      console.warn("web-rag widget config load failed:", e);
    } finally {
      assistantConfigLoaded = true;
      applyDisplayConfig();
      if (isOpen) {
        scheduleInitialContent();
      }
    }
  }

  // ── Greeting + starters ────────────────────────────────────────────────
  function renderGreetingIfNeeded() {
    const greeting = assistantConfig.greeting || {};
    const storageKey = getGreetingStorageKey();
    const alreadyShown = localStorage.getItem(storageKey) === "1";
    if (!greeting.enabled || !greeting.message || greetingRendered) {
      return false;
    }
    if (greeting.show_once && alreadyShown) {
      greetingRendered = true;
      return false;
    }

    addMessage(greeting.message, "bot");
    greetingRendered = true;
    if (greeting.show_once) {
      localStorage.setItem(storageKey, "1");
    }
    return true;
  }

  function renderStartersIfNeeded() {
    const starters = assistantConfig.starters || [];
    if (startersRendered || starters.length === 0) {
      return;
    }

    addBotCard((card) => {
      const title = document.createElement("div");
      title.className = "wr-card-title";
      title.textContent = "Quick actions";
      card.appendChild(title);

      const actions = document.createElement("div");
      actions.className = "wr-actions";

      starters.forEach((starter) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "wr-chip";
        button.textContent = starter.label || "Action";
        button.addEventListener("click", () => {
          if (starter.action === "open_form") {
            openForm(starter.form_id);
            return;
          }
          const query = (starter.message || starter.label || "").trim();
          if (query) {
            send(query, starter.label || query);
          }
        });
        actions.appendChild(button);
      });

      card.appendChild(actions);
    });

    startersRendered = true;
  }

  function scheduleInitialContent() {
    if (!isOpen || initialContentScheduled || !assistantConfigLoaded) {
      return;
    }
    initialContentScheduled = true;

    const greeting = assistantConfig.greeting || {};
    const delay = greeting.enabled && greeting.message
      ? Math.max(0, Math.min(parseInt(greeting.delay_ms || 0, 10), 10000))
      : 0;

    window.setTimeout(() => {
      if (!isOpen) return;
      renderGreetingIfNeeded();
      renderStartersIfNeeded();
    }, delay);
  }

  // ── Form workflows ─────────────────────────────────────────────────────
  function createInputForField(field) {
    if (field.type === "textarea") {
      return document.createElement("textarea");
    }
    if (field.type === "select") {
      const select = document.createElement("select");
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = field.placeholder || `Select ${field.label}`;
      select.appendChild(placeholder);
      (field.options || []).forEach((option) => {
        const opt = document.createElement("option");
        opt.value = option;
        opt.textContent = option;
        select.appendChild(opt);
      });
      return select;
    }
    const inputEl = document.createElement("input");
    inputEl.type = field.type || "text";
    return inputEl;
  }

  function openForm(formId) {
    const formDef = getConfiguredForm(formId);
    if (!formDef) {
      addMessage("Sorry, this form is not available right now.", "bot");
      return;
    }

    addBotCard((card) => {
      const title = document.createElement("div");
      title.className = "wr-card-title";
      title.textContent = formDef.title || "Form";
      card.appendChild(title);

      if (formDef.description) {
        const desc = document.createElement("div");
        desc.className = "wr-card-text";
        desc.textContent = formDef.description;
        card.appendChild(desc);
      }

      const formEl = document.createElement("form");
      formEl.className = "wr-form";
      const fieldRefs = {};

      (formDef.fields || []).forEach((field) => {
        const wrap = document.createElement("div");
        wrap.className = "wr-field";

        const label = document.createElement("label");
        label.textContent = `${field.label}${field.required ? " *" : ""}`;
        wrap.appendChild(label);

        const inputEl = createInputForField(field);
        inputEl.name = field.name;
        inputEl.placeholder = field.placeholder || "";
        if (field.required) {
          inputEl.required = true;
        }
        wrap.appendChild(inputEl);

        const error = document.createElement("div");
        error.className = "wr-error-text";
        wrap.appendChild(error);

        fieldRefs[field.name] = { field, inputEl, error };
        formEl.appendChild(wrap);
      });

      const status = document.createElement("div");
      status.className = "wr-inline-status";
      formEl.appendChild(status);

      const submitBtn = document.createElement("button");
      submitBtn.type = "submit";
      submitBtn.className = "wr-submit";
      submitBtn.textContent = formDef.submit_label || "Submit";
      formEl.appendChild(submitBtn);

      formEl.addEventListener("submit", async (event) => {
        event.preventDefault();
        Object.values(fieldRefs).forEach((ref) => {
          ref.error.textContent = "";
        });
        status.textContent = "";
        submitBtn.disabled = true;

        const values = {};
        Object.keys(fieldRefs).forEach((name) => {
          values[name] = fieldRefs[name].inputEl.value || "";
        });

        try {
          const resp = await fetch(`${API_URL}/api/widget/forms/submit`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              site_id: parseInt(SITE_ID, 10),
              form_id: formDef.id,
              values,
              session_id: sessionId,
              page_url: window.location.href,
            }),
          });
          const data = await resp.json();
          if (!resp.ok) {
            const errors = data.errors || {};
            Object.keys(errors).forEach((name) => {
              if (fieldRefs[name]) {
                fieldRefs[name].error.textContent = errors[name];
              }
            });
            status.textContent = data.error || "Please check the form and try again.";
            return;
          }

          addMessage(data.message || formDef.success_message || "Thanks. Your request has been sent.", "bot");
          formEl.remove();
          return;
        } catch (err) {
          status.textContent = "Sorry, something went wrong. Please try again.";
          console.error("web-rag form submit error:", err);
        } finally {
          submitBtn.disabled = false;
        }
      });

      card.appendChild(formEl);
    });
  }

  // ── Chat sending ───────────────────────────────────────────────────────
  async function send(queryOverride, visibleText) {
    const query = String(queryOverride || input.value || "").trim();
    if (!query) return;

    input.value = "";
    input.disabled = true;
    sendBtn.disabled = true;
    addMessage((visibleText || query).trim(), "user");
    typing.classList.add("active");
    scrollToBottom();

    try {
      const resp = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          site_id: parseInt(SITE_ID, 10),
          query,
          session_id: sessionId,
          origin_domain: window.location.hostname,
        }),
      });

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || "Request failed");
      }
      addMessage(data.answer, "bot", data.sources, data.confidence);
    } catch (err) {
      addMessage("Sorry, something went wrong. Please try again.", "bot");
      console.error("web-rag widget error:", err);
    } finally {
      typing.classList.remove("active");
      sendBtn.disabled = false;
      input.disabled = false;
      input.focus();
      scrollToBottom();
    }
  }

  // ── Actions ────────────────────────────────────────────────────────────
  function toggle() {
    isOpen = !isOpen;
    panel.classList.toggle("open", isOpen);
    if (isOpen) {
      scheduleInitialContent();
      input.focus();
      scrollToBottom();
    }
  }

  bubble.addEventListener("click", toggle);
  closeBtn.addEventListener("click", toggle);
  sendBtn.addEventListener("click", () => send());
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });

  loadAssistantConfig();
})();
