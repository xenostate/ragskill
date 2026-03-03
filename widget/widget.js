(function () {
  "use strict";

  // ── Config from script tag ─────────────────────────────────────────────
  const scriptTag = document.currentScript;
  const SITE_ID = scriptTag?.getAttribute("data-site-id") || "1";
  const API_URL =
    scriptTag?.getAttribute("data-api") || window.location.origin;
  const TITLE = scriptTag?.getAttribute("data-title") || "Ask a question";
  const COLOR = scriptTag?.getAttribute("data-color") || "#2563eb";
  const POSITION = scriptTag?.getAttribute("data-position") || "right";

  // ── Session ────────────────────────────────────────────────────────────
  const SESSION_KEY = `wr_session_${SITE_ID}`;
  let sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = "s_" + Math.random().toString(36).slice(2, 12);
    localStorage.setItem(SESSION_KEY, sessionId);
  }

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
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
      max-width: 85%;
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
      <span>${TITLE}</span>
      <button class="wr-close">&times;</button>
    </div>
    <div class="wr-messages">
      <div class="wr-typing"><span></span><span></span><span></span></div>
    </div>
    <div class="wr-input-row">
      <input class="wr-input" placeholder="Type your question..." />
      <button class="wr-send">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
        </svg>
      </button>
    </div>
  `;
  shadow.appendChild(panel);

  const messages = panel.querySelector(".wr-messages");
  const typing = panel.querySelector(".wr-typing");
  const input = panel.querySelector(".wr-input");
  const sendBtn = panel.querySelector(".wr-send");
  const closeBtn = panel.querySelector(".wr-close");

  // ── Actions ────────────────────────────────────────────────────────────
  let isOpen = false;

  function toggle() {
    isOpen = !isOpen;
    panel.classList.toggle("open", isOpen);
  }

  bubble.addEventListener("click", toggle);
  closeBtn.addEventListener("click", toggle);

  function addMessage(text, type, sources, confidence) {
    const msg = document.createElement("div");
    msg.className = `wr-msg ${type}`;

    if (type === "bot") {
      let html = "";

      if (confidence) {
        html += `<span class="wr-confidence ${confidence}">${confidence}</span><br>`;
      }

      // Basic markdown: **bold**, [links](url), newlines
      html += text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
        .replace(/\n/g, "<br>");

      if (sources && sources.length > 0) {
        html += `<div class="wr-sources"><strong>Sources:</strong>`;
        sources.forEach((s, i) => {
          html += `<a href="${s.url}" target="_blank">[${i + 1}] ${s.title}</a>`;
        });
        html += `</div>`;
      }

      msg.innerHTML = html;
    } else {
      msg.textContent = text;
    }

    messages.insertBefore(msg, typing);
    messages.scrollTop = messages.scrollHeight;
  }

  async function send() {
    const query = input.value.trim();
    if (!query) return;

    input.value = "";
    sendBtn.disabled = true;
    addMessage(query, "user");
    typing.classList.add("active");

    try {
      const resp = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          site_id: parseInt(SITE_ID),
          query: query,
          session_id: sessionId,
        }),
      });

      const data = await resp.json();
      addMessage(data.answer, "bot", data.sources, data.confidence);
    } catch (err) {
      addMessage("Sorry, something went wrong. Please try again.", "bot");
      console.error("web-rag widget error:", err);
    } finally {
      typing.classList.remove("active");
      sendBtn.disabled = false;
      input.focus();
    }
  }

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
})();
