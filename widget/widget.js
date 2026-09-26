(function () {
  "use strict";

  // ── Config from script tag ─────────────────────────────────────────────
  const scriptTag = document.currentScript;
  const SITE_ID = scriptTag?.getAttribute("data-site-id") || "1";
  const API_URL = scriptTag?.getAttribute("data-api") || window.location.origin;
  const TITLE = scriptTag?.getAttribute("data-title") || "Ask a question";
  const COLOR = normalizeHexColor(scriptTag?.getAttribute("data-color"), "#2F4BE5");
  const SECONDARY_COLOR = normalizeHexColor(scriptTag?.getAttribute("data-secondary-color"), "#EEF1FF");
  const EMBED_PRESET = normalizePreset(scriptTag?.getAttribute("data-preset"));
  const POSITION = normalizePosition(scriptTag?.getAttribute("data-position"));
  const BUBBLE_SIZE = clampNumber(scriptTag?.getAttribute("data-bubble-size"), 56, 44, 96);
  const PANEL_WIDTH = clampNumber(scriptTag?.getAttribute("data-panel-width"), 380, 320, 520);
  const PANEL_HEIGHT = clampNumber(scriptTag?.getAttribute("data-panel-height"), 520, 420, 760);
  const BUBBLE_SHAPE = normalizeBubbleShape(scriptTag?.getAttribute("data-bubble-shape"));
  const PANEL_RADIUS = clampNumber(scriptTag?.getAttribute("data-panel-radius"), 4, 0, 32);
  const ICON_NAME = normalizeIconName(scriptTag?.getAttribute("data-icon"));
  const FONT_FAMILY = sanitizeFontFamily(scriptTag?.getAttribute("data-font-family"));
  const FONT_URL = sanitizeFontUrl(scriptTag?.getAttribute("data-font-url"));
  const MOUNT_SELECTOR = scriptTag?.getAttribute("data-container") || "";
  let mountTarget = null;
  if (MOUNT_SELECTOR) {
    try {
      mountTarget = document.querySelector(MOUNT_SELECTOR);
    } catch (e) {
      console.warn("web-rag widget invalid data-container selector:", MOUNT_SELECTOR);
    }
  }
  const INLINE_MODE = Boolean(mountTarget);
  const INLINE_START_CLOSED = INLINE_MODE && scriptTag?.getAttribute("data-inline-start-closed") === "true";
  const INLINE_HIDE_HEADER = INLINE_MODE && scriptTag?.getAttribute("data-inline-hide-header") === "true";
  const INLINE_MIN_HEIGHT = clampNumber(scriptTag?.getAttribute("data-inline-min-height"), 420, 240, 760);
  const DARK_THEME = scriptTag?.getAttribute("data-theme") === "dark";
  const HIDE_BUBBLE = INLINE_MODE || scriptTag?.getAttribute("data-hide-bubble") === "true";
  const AUTO_OPEN = scriptTag?.getAttribute("data-auto-open") === "true";
  const PREVIEW_OPEN = scriptTag?.getAttribute("data-preview-open") === "true";
  const PREVIEW_RESET_GREETING = scriptTag?.getAttribute("data-preview-reset-greeting") === "true";
  const PREVIEW_ADMIN_TOKEN = scriptTag?.getAttribute("data-preview-admin-token") || "";
  const DEBUG_PANEL = scriptTag?.getAttribute("data-debug-panel") === "true";
  const DEFAULT_PLACEHOLDER = "Type your question...";
  const titleLocked = Boolean(scriptTag?.hasAttribute("data-title"));
  const colorLocked = Boolean(scriptTag?.hasAttribute("data-color"));
  const secondaryColorLocked = Boolean(scriptTag?.hasAttribute("data-secondary-color"));
  const presetLocked = Boolean(scriptTag?.hasAttribute("data-preset"));
  const positionLocked = Boolean(scriptTag?.hasAttribute("data-position"));
  const iconLocked = Boolean(scriptTag?.hasAttribute("data-icon"));

  // ── Session ────────────────────────────────────────────────────────────
  const SESSION_KEY = `wr_session_${SITE_ID}`;
  const LANGUAGE_KEY = `wr_lang_${SITE_ID}`;
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
    appearance: {
      preset: "professional",
      brand_color: COLOR,
      secondary_color: SECONDARY_COLOR,
      logo_url: "",
      logo_alt: "",
      launcher_icon: ICON_NAME,
      launcher_position: POSITION,
      mobile_fullscreen: true,
    },
    display: {},
    greeting: { enabled: false, message: "", show_once: true, delay_ms: 0 },
    contact: { enabled: false },
    feedback: { enabled: true },
    sources: { mode: "compact", label: "Sources" },
    starters: [],
    forms: [],
  };
  let assistantConfigLoaded = false;
  let initialContentScheduled = false;
  let greetingRendered = false;
  let startersRendered = false;
  let isOpen = INLINE_MODE && !INLINE_START_CLOSED;
  let inlineMounted = INLINE_MODE && !INLINE_START_CLOSED;
  let selectedLanguage = "";
  let debugPanel = null;
  let debugLines = null;
  let lastFocusedElement = null;
  let responseSequence = 0;
  let formSequence = 0;

  function clampNumber(value, fallback, min, max) {
    const num = parseInt(value || "", 10);
    if (!Number.isFinite(num)) {
      return fallback;
    }
    return Math.min(max, Math.max(min, num));
  }

  function normalizeHexColor(value, fallback) {
    const color = String(value || "").trim();
    if (/^#[0-9a-f]{6}$/i.test(color)) return color.toUpperCase();
    if (/^#[0-9a-f]{3}$/i.test(color)) {
      return ("#" + color.slice(1).split("").map((char) => char + char).join("")).toUpperCase();
    }
    return fallback;
  }

  function normalizePreset(value) {
    const preset = String(value || "").trim().toLowerCase();
    return ["professional", "friendly", "minimal"].includes(preset) ? preset : "professional";
  }

  function normalizeBubbleShape(value) {
    const shape = String(value || "").trim().toLowerCase();
    if (shape === "rounded-square" || shape === "rounded_square") {
      return "rounded-square";
    }
    if (shape === "pill") {
      return "pill";
    }
    return "circle";
  }

  function normalizeIconName(value) {
    const icon = String(value || "").trim().toLowerCase();
    if (["chat", "message", "sparkles", "question", "book", "cap"].includes(icon)) {
      return icon;
    }
    return "chat";
  }

  function normalizePosition(value) {
    const pos = String(value || "").trim().toLowerCase();
    if (pos === "left" || pos === "center") {
      return pos;
    }
    return "right";
  }

  function sanitizeFontFamily(value) {
    const text = String(value || "").trim();
    if (!text) return "";
    if (!/^[a-zA-Z0-9,'" _-]+(?:\s*,\s*[a-zA-Z0-9,'" _-]+)*$/.test(text)) {
      return "";
    }
    return text;
  }

  function sanitizeFontUrl(value) {
    const text = String(value || "").trim();
    if (!text) return "";
    try {
      const parsed = new URL(text, window.location.href);
      if (parsed.protocol === "https:" || parsed.protocol === "http:") {
        return parsed.href;
      }
    } catch (e) {}
    return "";
  }

  function getBubbleBorderRadius() {
    if (BUBBLE_SHAPE === "rounded-square") {
      return `${Math.round(BUBBLE_SIZE * 0.32)}px`;
    }
    if (BUBBLE_SHAPE === "pill") {
      return `${Math.round(BUBBLE_SIZE * 0.42)}px`;
    }
    return "50%";
  }

  function getBubblePositionCss() {
    if (POSITION === "left") {
      return "left: 24px;";
    }
    if (POSITION === "center") {
      return "left: 50%; transform: translateX(-50%);";
    }
    return "right: 24px;";
  }

  function getPanelPositionCss() {
    if (POSITION === "left") {
      return `bottom: ${BUBBLE_SIZE + 36}px; left: 24px;`;
    }
    if (POSITION === "center") {
      return "top: 50%; left: 50%; transform: translate(-50%, -50%);";
    }
    return `bottom: ${BUBBLE_SIZE + 36}px; right: 24px;`;
  }

  function getBubbleIconMarkup(iconName = ICON_NAME) {
    const icons = {
      chat: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
      </svg>`,
      message: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 5h16v10H8l-4 4z"/>
      </svg>`,
      sparkles: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 3l1.9 4.8L19 9.7l-4.2 2.4L13 17l-1.8-4.9L7 9.7l5.1-1.9z"/>
        <path d="M5 3v3"/>
        <path d="M3.5 4.5h3"/>
        <path d="M19 16v5"/>
        <path d="M16.5 18.5h5"/>
      </svg>`,
      question: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M9.1 9a3 3 0 1 1 5.8 1c0 2-3 2.5-3 4"/>
        <path d="M12 17h.01"/>
        <circle cx="12" cy="12" r="9"/>
      </svg>`,
      book: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
        <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
      </svg>`,
      cap: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M2 10l10-5 10 5-10 5-10-5z"/>
        <path d="M6 12v5c0 1.7 2.7 3 6 3s6-1.3 6-3v-5"/>
      </svg>`
    };
    return icons[iconName] || icons.chat;
  }

  function setupDebugPanel() {
    if (!DEBUG_PANEL || debugPanel) return;
    debugPanel = document.createElement("div");
    debugPanel.style.cssText = [
      "position:fixed",
      "top:12px",
      "left:12px",
      "width:min(360px, calc(100vw - 24px))",
      "max-height:50vh",
      "overflow:auto",
      "padding:12px",
      "border-radius:12px",
      "background:rgba(17,24,39,0.94)",
      "color:#e5e7eb",
      "font:12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace",
      "box-shadow:0 12px 32px rgba(0,0,0,0.28)",
      "z-index:1000000",
      "white-space:pre-wrap"
    ].join(";");
    const title = document.createElement("div");
    title.textContent = "Widget Debug";
    title.style.cssText = "font-weight:700;color:#fff;margin-bottom:8px;";
    debugLines = document.createElement("div");
    debugPanel.appendChild(title);
    debugPanel.appendChild(debugLines);
    document.body.appendChild(debugPanel);
  }

  function debugLog(message, data) {
    if (!DEBUG_PANEL) return;
    setupDebugPanel();
    const line = document.createElement("div");
    const stamp = new Date().toLocaleTimeString([], { hour12: false });
    let text = `[${stamp}] ${message}`;
    if (data !== undefined) {
      try {
        text += ` ${typeof data === "string" ? data : JSON.stringify(data)}`;
      } catch (e) {
        text += ` ${String(data)}`;
      }
    }
    line.textContent = text;
    debugLines.appendChild(line);
    debugPanel.scrollTop = debugPanel.scrollHeight;
    try {
      window.parent.postMessage({ source: "wr-widget-debug", text }, "*");
    } catch (e) {}
  }

  function requestHeaders(includeJson) {
    const headers = {};
    if (includeJson) {
      headers["Content-Type"] = "application/json";
    }
    if (PREVIEW_OPEN) {
      headers["X-Widget-Preview"] = "true";
      if (PREVIEW_ADMIN_TOKEN) {
        headers["X-Admin-Token"] = PREVIEW_ADMIN_TOKEN;
      }
    }
    return headers;
  }

  debugLog("widget bootstrap", {
    siteId: SITE_ID,
    apiUrl: API_URL,
    preview: PREVIEW_OPEN,
    debug: DEBUG_PANEL
  });

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
  host.dataset.position = POSITION;
  host.dataset.preset = "professional";
  host.dataset.mobileFullscreen = INLINE_MODE ? "false" : "true";
  if (INLINE_MODE) host.dataset.inline = "true";
  if (INLINE_MODE) {
    host.style.cssText = `display:block;flex:1;width:100%;min-width:0;min-height:${INLINE_MIN_HEIGHT}px;`;
    if (inlineMounted) {
      mountTarget.replaceChildren(host);
    }
  } else {
    document.body.appendChild(host);
  }
  const shadow = host.attachShadow({ mode: "closed" });

  if (FONT_URL) {
    const fontLink = document.createElement("link");
    fontLink.rel = "stylesheet";
    fontLink.href = FONT_URL;
    shadow.appendChild(fontLink);
  }

  // ── Styles ─────────────────────────────────────────────────────────────
  const style = document.createElement("style");
  style.textContent = `
    * { box-sizing: border-box; margin: 0; padding: 0; }

    .wr-bubble {
      position: fixed;
      bottom: 24px;
      ${getBubblePositionCss()}
      width: ${BUBBLE_SIZE}px;
      height: ${BUBBLE_SIZE}px;
      border-radius: ${getBubbleBorderRadius()};
      background: ${COLOR};
      color: #fff;
      border: 1px solid #181817;
      cursor: pointer;
      box-shadow: 4px 4px 0 #181817;
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 999999;
      transition: transform 0.2s;
      ${HIDE_BUBBLE ? "display:none;" : ""}
    }
    .wr-bubble:hover { ${POSITION === "center" ? "transform: translateX(-50%) scale(1.08);" : "transform: scale(1.08);"} }
    .wr-bubble svg {
      width: ${Math.round(BUBBLE_SIZE * 0.5)}px;
      height: ${Math.round(BUBBLE_SIZE * 0.5)}px;
    }

    .wr-panel {
      position: fixed;
      ${getPanelPositionCss()}
      width: ${PANEL_WIDTH}px;
      max-width: calc(100vw - 48px);
      height: ${PANEL_HEIGHT}px;
      max-height: calc(100vh - 120px);
      background: #f0ede5;
      border: 1px solid #181817;
      border-radius: ${PANEL_RADIUS}px;
      box-shadow: 8px 8px 0 #181817;
      display: none;
      flex-direction: column;
      overflow: hidden;
      z-index: 999998;
      font-family: ${FONT_FAMILY || '"DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'};
      font-size: 14px;
      color: #1a1a1a;
    }
    .wr-panel.open { display: flex; }

    .wr-header {
      padding: 18px;
      background: ${COLOR};
      color: #fff;
      border-bottom: 1px solid #181817;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }
    .wr-header-main {
      min-width: 0;
      flex: 1;
    }
    .wr-header-title {
      font-weight: 700;
      font-size: 15px;
      line-height: 1.2;
      letter-spacing: -0.02em;
    }
    .wr-lang-switch {
      display: flex;
      gap: 6px;
      margin-top: 10px;
      flex-wrap: wrap;
    }
    .wr-lang-btn {
      border: 1px solid rgba(255,255,255,0.35);
      background: rgba(255,255,255,0.12);
      color: #fff;
      border-radius: 0;
      padding: 4px 8px;
      font: inherit;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s;
    }
    .wr-lang-btn.active {
      background: #fff;
      color: ${COLOR};
      border-color: #fff;
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
      background: #f0ede5;
    }

    .wr-msg {
      max-width: 88%;
      padding: 10px 14px;
      border-radius: 0;
      line-height: 1.5;
      word-wrap: break-word;
    }
    .wr-msg.user {
      align-self: flex-end;
      background: ${COLOR};
      color: #fff;
      border: 1px solid #181817;
    }
    .wr-msg.bot {
      align-self: flex-start;
      background: #dedacf;
      color: #1a1a1a;
      border: 1px solid #aaa69b;
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
      border-radius: 0;
      background: #f7f4ed;
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
      border-radius: 0;
      padding: 9px 10px;
      font: inherit;
      outline: none;
      background: #f7f4ed;
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
      border-radius: 0;
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
      background: #dedacf;
      border: 1px solid #aaa69b;
      border-radius: 0;
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
      border-top: 1px solid #aaa69b;
      display: flex;
      gap: 8px;
      background: #f7f4ed;
    }
    .wr-input {
      flex: 1;
      padding: 10px 14px;
      border: 1px solid #aaa69b;
      border-radius: 0;
      outline: none;
      font-size: 14px;
      font-family: inherit;
    }
    .wr-input:focus { border-color: ${COLOR}; }
    .wr-send {
      width: 40px;
      height: 40px;
      border-radius: 0;
      border: 1px solid #181817;
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
      border-radius: 0;
      display: inline-block;
      margin-bottom: 4px;
    }
    .wr-confidence.high { background: rgba(47,75,229,.1); color: #243cc5; }
    .wr-confidence.medium { background: #dedacf; color: #514f4a; }
    .wr-confidence.low { background: #fecaca; color: #991b1b; }

    @media (max-width: 560px) {
      .wr-panel {
        width: calc(100vw - 24px);
        max-width: calc(100vw - 24px);
        ${POSITION === "left" ? "left: 12px;" : POSITION === "center" ? "left: 50%; transform: translate(-50%, -50%);" : "right: 12px;"}
        ${POSITION === "center" ? "top: 50%;" : `bottom: ${Math.max(BUBBLE_SIZE + 26, 82)}px;`}
        height: min(70vh, ${PANEL_HEIGHT}px);
      }
      .wr-bubble {
        ${POSITION === "left" ? "left: 12px;" : POSITION === "center" ? "left: 50%; transform: translateX(-50%);" : "right: 12px;"}
        bottom: 16px;
      }
      .wr-msg { max-width: 94%; }
    }

    ${INLINE_MODE ? `
      .wr-bubble, .wr-close${INLINE_HIDE_HEADER ? ", .wr-header" : ""} { display: none; }
      .wr-panel,
      .wr-panel.open {
        position: relative;
        inset: auto;
        width: 100%;
        max-width: none;
        height: 100%;
        min-height: ${INLINE_MIN_HEIGHT}px;
        max-height: none;
        border-radius: 0;
        box-shadow: none;
        display: flex;
        transform: none;
      }
      @media (max-width: 560px) {
        .wr-panel,
        .wr-panel.open {
          position: relative;
          inset: auto;
          width: 100%;
          max-width: none;
          height: 100%;
          min-height: ${INLINE_MIN_HEIGHT}px;
          transform: none;
        }
      }
    ` : ""}

    ${DARK_THEME ? `
      .wr-panel { background: transparent; color: #f2f5ef; }
      .wr-messages { background: #1c242b; }
      .wr-msg.bot { background: #27313a; color: #f2f5ef; }
      .wr-input-row { background: #1c242b; border-top-color: #3a4650; }
      .wr-input {
        background: #111920;
        border-color: #58636a;
        color: #f2f5ef;
      }
      .wr-input::placeholder { color: #798681; }
      .wr-send { color: #101419; }
      .wr-card-text, .wr-inline-status { color: #b9c4bf; }
      .wr-chip { background: #202831; border-color: #45534a; }
    ` : ""}
  `;
  shadow.appendChild(style);

  // The public design surface is intentionally small: three presets plus two
  // brand colors. Legacy embed attributes still work as safe fallbacks.
  const designStyle = document.createElement("style");
  designStyle.textContent = `
    :host {
      --wr-brand: ${COLOR};
      --wr-secondary: ${SECONDARY_COLOR};
      --wr-on-brand: #fff;
      --wr-accent-ink: ${COLOR};
      --wr-ink: #17181c;
      --wr-muted: #667085;
      --wr-surface: #fff;
      --wr-canvas: #f7f8fb;
      --wr-line: #e5e7eb;
      --wr-panel-radius: 18px;
      --wr-card-radius: 14px;
      --wr-control-radius: 11px;
      --wr-shadow: 0 24px 70px rgba(16, 24, 40, .20), 0 4px 14px rgba(16, 24, 40, .08);
    }
    button, input, textarea, select { font-family: inherit; }
    button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible, a:focus-visible, summary:focus-visible {
      outline: 3px solid color-mix(in srgb, var(--wr-brand) 42%, white);
      outline-offset: 2px;
    }
    .wr-sr-only {
      position: absolute !important;
      width: 1px !important;
      height: 1px !important;
      padding: 0 !important;
      margin: -1px !important;
      overflow: hidden !important;
      clip: rect(0, 0, 0, 0) !important;
      white-space: nowrap !important;
      border: 0 !important;
    }
    .wr-bubble {
      background: var(--wr-brand);
      color: var(--wr-on-brand);
      border: 0;
      border-radius: 50%;
      box-shadow: 0 10px 30px color-mix(in srgb, var(--wr-brand) 32%, transparent);
      transition: transform .18s ease, box-shadow .18s ease;
    }
    .wr-bubble:hover { box-shadow: 0 14px 34px color-mix(in srgb, var(--wr-brand) 42%, transparent); }
    :host([data-position="left"]) .wr-bubble { left: 24px !important; right: auto !important; transform: none; }
    :host([data-position="right"]) .wr-bubble { right: 24px !important; left: auto !important; transform: none; }
    :host([data-position="left"]) .wr-panel { left: 24px !important; right: auto !important; transform: none; }
    :host([data-position="right"]) .wr-panel { right: 24px !important; left: auto !important; transform: none; }
    :host([data-position="center"]) .wr-bubble { left: 50% !important; right: auto !important; transform: translateX(-50%); }
    :host([data-position="center"]) .wr-bubble:hover { transform: translateX(-50%) scale(1.08); }
    :host([data-position="center"]) .wr-panel { top: 50% !important; bottom: auto !important; left: 50% !important; right: auto !important; transform: translate(-50%, -50%); }
    .wr-panel {
      background: var(--wr-surface);
      color: var(--wr-ink);
      border: 1px solid var(--wr-line);
      border-radius: var(--wr-panel-radius);
      box-shadow: var(--wr-shadow);
    }
    .wr-header {
      position: relative;
      padding: 17px 16px;
      background: var(--wr-brand);
      color: var(--wr-on-brand);
      border: 0;
      align-items: center;
    }
    .wr-header-brand { display: flex; align-items: center; gap: 11px; min-width: 0; }
    .wr-logo {
      width: 40px;
      height: 40px;
      flex: 0 0 40px;
      display: grid;
      place-items: center;
      overflow: hidden;
      border-radius: 12px;
      background: rgba(255,255,255,.18);
      color: var(--wr-on-brand);
      font-weight: 800;
      font-size: 15px;
      border: 1px solid rgba(255,255,255,.24);
    }
    .wr-logo img { width: 100%; height: 100%; object-fit: cover; }
    .wr-header-main { min-width: 0; }
    .wr-header-title { font-size: 15px; font-weight: 720; letter-spacing: -.015em; }
    .wr-header-subtitle { margin-top: 3px; font-size: 11px; line-height: 1.3; opacity: .82; }
    .wr-close {
      width: 34px;
      height: 34px;
      display: grid;
      place-items: center;
      flex: 0 0 34px;
      border-radius: 9px;
      color: var(--wr-on-brand);
      opacity: .86;
    }
    .wr-close:hover { background: rgba(255,255,255,.14); }
    .wr-lang-switch { margin-top: 8px; }
    .wr-lang-btn { border-radius: 7px; }
    .wr-lang-btn.active { color: var(--wr-accent-ink); }
    .wr-contact {
      display: none;
      align-items: center;
      gap: 10px;
      padding: 10px 14px;
      background: color-mix(in srgb, var(--wr-secondary) 60%, white);
      border-bottom: 1px solid var(--wr-line);
    }
    .wr-contact.visible { display: flex; }
    .wr-contact-avatar {
      width: 34px;
      height: 34px;
      flex: 0 0 34px;
      overflow: hidden;
      display: grid;
      place-items: center;
      border-radius: 50%;
      background: var(--wr-brand);
      color: var(--wr-on-brand);
      font-size: 12px;
      font-weight: 800;
    }
    .wr-contact-avatar img { width: 100%; height: 100%; object-fit: cover; }
    .wr-contact-copy { min-width: 0; flex: 1; }
    .wr-contact-name { font-size: 12px; font-weight: 720; color: var(--wr-ink); }
    .wr-contact-role { margin-top: 1px; font-size: 11px; color: var(--wr-muted); }
    .wr-contact-action {
      min-height: 32px;
      padding: 0 10px;
      border: 1px solid color-mix(in srgb, var(--wr-brand) 28%, transparent);
      border-radius: 9px;
      background: var(--wr-surface);
      color: var(--wr-accent-ink);
      font-size: 11px;
      font-weight: 750;
      cursor: pointer;
    }
    .wr-messages {
      padding: 18px 16px;
      gap: 14px;
      background: color-mix(in srgb, var(--wr-secondary) 20%, white);
      overscroll-behavior: contain;
    }
    .wr-msg { max-width: 88%; padding: 11px 13px; border-radius: var(--wr-card-radius); line-height: 1.55; }
    .wr-msg.user {
      background: var(--wr-brand);
      color: var(--wr-on-brand);
      border: 0;
      border-bottom-right-radius: 5px;
    }
    .wr-msg.bot {
      background: var(--wr-surface);
      color: var(--wr-ink);
      border: 1px solid var(--wr-line);
      border-bottom-left-radius: 5px;
      box-shadow: 0 3px 10px rgba(16,24,40,.04);
    }
    .wr-msg.bot a, .wr-sources a { color: var(--wr-accent-ink); }
    .wr-card-title { color: var(--wr-ink); font-weight: 720; }
    .wr-card-text, .wr-inline-status { color: var(--wr-muted); }
    .wr-actions { gap: 7px; }
    .wr-chip {
      border: 1px solid color-mix(in srgb, var(--wr-brand) 24%, var(--wr-line));
      border-radius: var(--wr-control-radius);
      background: color-mix(in srgb, var(--wr-secondary) 52%, white);
      color: var(--wr-accent-ink);
    }
    .wr-chip:hover { background: var(--wr-brand); color: var(--wr-on-brand); border-color: var(--wr-brand); }
    .wr-field input, .wr-field textarea, .wr-field select {
      border: 1px solid var(--wr-line);
      border-radius: var(--wr-control-radius);
      background: var(--wr-surface);
      color: var(--wr-ink);
    }
    .wr-field input:focus, .wr-field textarea:focus, .wr-field select:focus {
      border-color: var(--wr-brand);
      box-shadow: 0 0 0 3px color-mix(in srgb, var(--wr-brand) 14%, transparent);
    }
    .wr-submit {
      min-height: 40px;
      border-radius: var(--wr-control-radius);
      background: var(--wr-brand);
      color: var(--wr-on-brand);
    }
    .wr-typing { background: var(--wr-surface); border: 1px solid var(--wr-line); border-radius: var(--wr-card-radius); }
    .wr-sources { margin-top: 10px; padding-top: 9px; border-top-color: var(--wr-line); color: var(--wr-muted); }
    .wr-sources summary { cursor: pointer; color: var(--wr-muted); font-size: 11px; font-weight: 700; }
    .wr-sources-list { display: flex; flex-direction: column; gap: 5px; margin-top: 7px; }
    .wr-sources a { display: flex; align-items: flex-start; gap: 6px; margin: 0; text-decoration: none; }
    .wr-sources a:hover { text-decoration: underline; }
    .wr-feedback { display: flex; align-items: center; gap: 5px; margin-top: 9px; color: var(--wr-muted); font-size: 11px; }
    .wr-feedback-prompt { margin-right: 2px; }
    .wr-feedback button {
      width: 29px;
      height: 29px;
      display: grid;
      place-items: center;
      border: 1px solid var(--wr-line);
      border-radius: 8px;
      background: var(--wr-surface);
      color: var(--wr-muted);
      cursor: pointer;
    }
    .wr-feedback button:hover, .wr-feedback button.selected { color: var(--wr-accent-ink); border-color: var(--wr-brand); background: var(--wr-secondary); }
    .wr-feedback svg { width: 14px; height: 14px; }
    .wr-input-row { padding: 11px 12px; border-top: 1px solid var(--wr-line); background: var(--wr-surface); }
    .wr-input { min-width: 0; border: 1px solid var(--wr-line); border-radius: var(--wr-control-radius); background: var(--wr-canvas); color: var(--wr-ink); }
    .wr-input:focus { border-color: var(--wr-brand); }
    .wr-send { border: 0; border-radius: var(--wr-control-radius); background: var(--wr-brand); color: var(--wr-on-brand); }

    .wr-panel[data-preset="friendly"] {
      --wr-panel-radius: 26px;
      --wr-card-radius: 18px;
      --wr-control-radius: 999px;
      --wr-shadow: 0 26px 80px rgba(56, 35, 93, .22);
    }
    .wr-panel[data-preset="friendly"] .wr-header { padding: 19px 17px; background: linear-gradient(135deg, var(--wr-brand), color-mix(in srgb, var(--wr-brand) 72%, #8b5cf6)); }
    .wr-panel[data-preset="friendly"] .wr-logo { border-radius: 50%; }
    .wr-panel[data-preset="friendly"] .wr-msg.user { border-bottom-right-radius: 7px; }
    .wr-panel[data-preset="friendly"] .wr-msg.bot { border-bottom-left-radius: 7px; }
    .wr-panel[data-preset="friendly"] .wr-feedback button { border-radius: 50%; }

    .wr-panel[data-preset="minimal"] {
      --wr-panel-radius: 5px;
      --wr-card-radius: 4px;
      --wr-control-radius: 3px;
      --wr-shadow: 7px 7px 0 var(--wr-ink);
      border-color: var(--wr-ink);
    }
    .wr-panel[data-preset="minimal"] .wr-header { border-bottom: 1px solid var(--wr-ink); }
    .wr-panel[data-preset="minimal"] .wr-logo { border-radius: 2px; }
    .wr-panel[data-preset="minimal"] .wr-msg.bot,
    .wr-panel[data-preset="minimal"] .wr-msg.user,
    .wr-panel[data-preset="minimal"] .wr-input-row,
    .wr-panel[data-preset="minimal"] .wr-contact { border-color: color-mix(in srgb, var(--wr-ink) 45%, transparent); }
    :host([data-preset="minimal"]) .wr-bubble { border: 1px solid var(--wr-ink); border-radius: 5px; box-shadow: 4px 4px 0 var(--wr-ink); }

    @media (max-width: 560px) {
      .wr-bubble.panel-open { display: none !important; }
      :host([data-position="left"]) .wr-bubble { left: 14px !important; }
      :host([data-position="right"]) .wr-bubble { right: 14px !important; }
      :host([data-mobile-fullscreen="false"]) .wr-panel {
        width: calc(100vw - 24px) !important;
        max-width: calc(100vw - 24px) !important;
        height: min(76vh, ${PANEL_HEIGHT}px) !important;
        top: auto !important;
        bottom: ${Math.max(BUBBLE_SIZE + 26, 82)}px !important;
      }
      :host([data-mobile-fullscreen="false"][data-position="left"]) .wr-panel { left: 12px !important; }
      :host([data-mobile-fullscreen="false"][data-position="right"]) .wr-panel { right: 12px !important; }
      :host([data-mobile-fullscreen="true"]) .wr-panel {
        position: fixed !important;
        inset: 0 !important;
        width: 100vw !important;
        max-width: none !important;
        height: 100vh !important;
        height: 100dvh !important;
        max-height: none !important;
        border: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        transform: none !important;
      }
      :host([data-mobile-fullscreen="true"]) .wr-header { padding-top: max(17px, env(safe-area-inset-top)); }
      :host([data-mobile-fullscreen="true"]) .wr-input-row { padding-bottom: max(11px, env(safe-area-inset-bottom)); }
    }
    @media (prefers-reduced-motion: reduce) {
      .wr-bubble, .wr-lang-btn, .wr-chip, .wr-typing span { transition: none !important; animation: none !important; }
    }
  `;
  shadow.appendChild(designStyle);

  // ── HTML ───────────────────────────────────────────────────────────────
  const bubble = document.createElement("button");
  bubble.type = "button";
  bubble.className = "wr-bubble";
  bubble.innerHTML = getBubbleIconMarkup();
  bubble.setAttribute("aria-label", "Open chat assistant");
  bubble.setAttribute("aria-expanded", "false");
  bubble.setAttribute("aria-controls", `wr-panel-${SITE_ID}`);
  shadow.appendChild(bubble);

  const panel = document.createElement("div");
  panel.className = "wr-panel";
  panel.id = `wr-panel-${SITE_ID}`;
  panel.setAttribute("role", "dialog");
  panel.setAttribute("aria-modal", INLINE_MODE ? "false" : "true");
  panel.setAttribute("aria-labelledby", `wr-title-${SITE_ID}`);
  panel.setAttribute("tabindex", "-1");
  panel.innerHTML = `
    <div class="wr-header">
      <div class="wr-header-brand">
        <div class="wr-logo"><span aria-hidden="true">AI</span></div>
        <div class="wr-header-main">
          <div class="wr-header-title" id="wr-title-${SITE_ID}">${escapeHtml(TITLE)}</div>
          <div class="wr-header-subtitle"></div>
          <div class="wr-lang-switch" style="display:none" aria-label="Language"></div>
        </div>
      </div>
      <button type="button" class="wr-close" aria-label="Close chat assistant">
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>
      </button>
    </div>
    <div class="wr-contact">
      <div class="wr-contact-avatar" aria-hidden="true"><span>?</span></div>
      <div class="wr-contact-copy">
        <div class="wr-contact-name"></div>
        <div class="wr-contact-role"></div>
      </div>
      <button type="button" class="wr-contact-action"></button>
    </div>
    <div class="wr-messages" role="log" aria-live="polite" aria-relevant="additions text" aria-label="Chat messages">
      <div class="wr-typing" role="status" aria-label="Assistant is typing"><span></span><span></span><span></span></div>
    </div>
    <div class="wr-input-row">
      <label class="wr-sr-only" for="wr-input-${SITE_ID}">Message</label>
      <input id="wr-input-${SITE_ID}" class="wr-input" aria-label="Message" autocomplete="off" placeholder="${escapeHtml(DEFAULT_PLACEHOLDER)}" />
      <button type="button" class="wr-send" aria-label="Send message">
        <svg viewBox="0 0 24 24" fill="currentColor">
          <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
        </svg>
      </button>
    </div>
    <div class="wr-live wr-sr-only" aria-live="polite" aria-atomic="true"></div>
  `;
  shadow.appendChild(panel);

  const headerTitle = panel.querySelector(".wr-header-title");
  const headerSubtitle = panel.querySelector(".wr-header-subtitle");
  const logo = panel.querySelector(".wr-logo");
  const langSwitch = panel.querySelector(".wr-lang-switch");
  const contactBar = panel.querySelector(".wr-contact");
  const contactAvatar = panel.querySelector(".wr-contact-avatar");
  const contactName = panel.querySelector(".wr-contact-name");
  const contactRole = panel.querySelector(".wr-contact-role");
  const contactAction = panel.querySelector(".wr-contact-action");
  const messages = panel.querySelector(".wr-messages");
  const typing = panel.querySelector(".wr-typing");
  const input = panel.querySelector(".wr-input");
  const sendBtn = panel.querySelector(".wr-send");
  const closeBtn = panel.querySelector(".wr-close");
  const liveRegion = panel.querySelector(".wr-live");
  panel.setAttribute("aria-hidden", String(!isOpen));
  bubble.setAttribute("aria-expanded", String(isOpen));
  if (HIDE_BUBBLE) {
    closeBtn.style.display = "none";
  }

  // ── Utilities ──────────────────────────────────────────────────────────
  function escapeHtml(text) {
    return String(text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function resolveText(value, lang, fallback = "ru") {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const normalizedLang = String(lang || "").toLowerCase();
      const normalizedFallback = String(fallback || "").toLowerCase();
      if (normalizedLang && typeof value[normalizedLang] === "string" && value[normalizedLang].trim()) {
        return value[normalizedLang].trim();
      }
      if (normalizedFallback && typeof value[normalizedFallback] === "string" && value[normalizedFallback].trim()) {
        return value[normalizedFallback].trim();
      }
      const first = Object.values(value).find((item) => typeof item === "string" && item.trim());
      return first ? first.trim() : "";
    }
    return String(value || "");
  }

  function getUiLanguage() {
    const pageLanguage = String(document.documentElement.lang || navigator.language || "en").toLowerCase().split(/[-_]/)[0];
    return selectedLanguage || getDefaultLanguage() || pageLanguage || "en";
  }

  function getGreetingStorageKey() {
    return `wr_greeting_shown_${SITE_ID}`;
  }

  function getAvailableLanguageCodes() {
    return (assistantConfig.language_switch?.options || []).map((item) => item.code);
  }

  function getDefaultLanguage() {
    const switchCfg = assistantConfig.language_switch || {};
    const codes = new Set(getAvailableLanguageCodes());
    if (switchCfg.default && codes.has(switchCfg.default)) {
      return switchCfg.default;
    }
    return "";
  }

  function applySelectedLanguage() {
    const stored = localStorage.getItem(LANGUAGE_KEY) || "";
    const codes = new Set(getAvailableLanguageCodes());
    if (stored && codes.has(stored)) {
      selectedLanguage = stored;
      return;
    }
    selectedLanguage = getDefaultLanguage();
    if (selectedLanguage) {
      localStorage.setItem(LANGUAGE_KEY, selectedLanguage);
    } else {
      localStorage.removeItem(LANGUAGE_KEY);
    }
  }

  function scrollToBottom() {
    messages.scrollTop = messages.scrollHeight;
  }

  function announce(text) {
    liveRegion.textContent = "";
    window.setTimeout(() => {
      liveRegion.textContent = String(text || "");
    }, 30);
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

  function safeAssetUrl(url) {
    const safe = safeSourceUrl(url);
    return safe === "#" ? "" : safe;
  }

  function readableTextColor(hex) {
    const color = normalizeHexColor(hex, "#2F4BE5").slice(1);
    const channels = [0, 2, 4].map((index) => {
      const value = parseInt(color.slice(index, index + 2), 16) / 255;
      return value <= 0.03928 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
    });
    const luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
    return luminance > 0.48 ? "#17181C" : "#FFFFFF";
  }

  function accessibleAccentColor(hex) {
    const normalized = normalizeHexColor(hex, "#2F4BE5");
    let channels = [1, 3, 5].map((index) => parseInt(normalized.slice(index, index + 2), 16));
    const luminance = (rgb) => {
      const linear = rgb.map((value) => {
        const channel = value / 255;
        return channel <= 0.03928 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4);
      });
      return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
    };
    while ((1.05 / (luminance(channels) + 0.05)) < 4.5) {
      channels = channels.map((value) => Math.max(0, Math.round(value * 0.86)));
    }
    return `#${channels.map((value) => value.toString(16).padStart(2, "0")).join("")}`;
  }

  function initials(value, fallback = "AI") {
    const words = String(value || "").trim().split(/\s+/).filter((word) => /[a-z0-9]/i.test(word[0] || ""));
    if (!words.length) return fallback;
    return words.slice(0, 2).map((word) => word[0]).join("").toUpperCase();
  }

  function renderAvatar(container, imageUrl, alt, fallbackText) {
    container.replaceChildren();
    const safeUrl = safeAssetUrl(imageUrl);
    if (safeUrl) {
      const image = document.createElement("img");
      image.src = safeUrl;
      image.alt = String(alt || "");
      image.addEventListener("error", () => {
        container.replaceChildren();
      const fallback = document.createElement("span");
      fallback.textContent = fallbackText;
      fallback.setAttribute("aria-hidden", "true");
        container.appendChild(fallback);
      }, { once: true });
      container.appendChild(image);
      return;
    }
    const fallback = document.createElement("span");
    fallback.textContent = fallbackText;
    fallback.setAttribute("aria-hidden", "true");
    container.appendChild(fallback);
  }

  function applyAppearanceConfig() {
    const appearance = assistantConfig.appearance || {};
    const preset = presetLocked ? EMBED_PRESET : normalizePreset(appearance.preset);
    const brand = colorLocked ? COLOR : normalizeHexColor(appearance.brand_color, COLOR);
    const secondary = secondaryColorLocked ? SECONDARY_COLOR : normalizeHexColor(appearance.secondary_color, SECONDARY_COLOR);
    const configuredPosition = ["left", "right"].includes(appearance.launcher_position)
      ? appearance.launcher_position
      : POSITION === "left" ? "left" : "right";
    const position = positionLocked ? POSITION : configuredPosition;
    const icon = iconLocked ? ICON_NAME : normalizeIconName(appearance.launcher_icon || ICON_NAME);
    const title = resolveText(assistantConfig.display?.title, getUiLanguage()) || TITLE;

    host.dataset.position = position;
    host.dataset.preset = preset;
    host.dataset.mobileFullscreen = INLINE_MODE ? "false" : String(appearance.mobile_fullscreen !== false);
    host.style.setProperty("--wr-brand", brand);
    host.style.setProperty("--wr-secondary", secondary);
    host.style.setProperty("--wr-on-brand", readableTextColor(brand));
    host.style.setProperty("--wr-accent-ink", accessibleAccentColor(brand));
    panel.dataset.preset = preset;
    bubble.innerHTML = getBubbleIconMarkup(icon);
    renderAvatar(
      logo,
      appearance.logo_url,
      resolveText(appearance.logo_alt, getUiLanguage()) || `${title} logo`,
      initials(title)
    );
  }

  function renderContact() {
    const contact = assistantConfig.contact || {};
    const name = resolveText(contact.name, getUiLanguage());
    const role = resolveText(contact.role, getUiLanguage());
    const label = resolveText(contact.action_label, getUiLanguage()) || (contact.action === "open_form" ? "Contact" : "WhatsApp");
    const hasWhatsApp = contact.action === "whatsapp" && String(contact.whatsapp_number || "").replace(/\D/g, "");
    const hasForm = contact.action === "open_form" && getConfiguredForm(contact.form_id);

    if (!contact.enabled || (!hasWhatsApp && !hasForm)) {
      contactBar.classList.remove("visible");
      return;
    }

    contactName.textContent = name || resolveText({ ru: "Связаться с нами", en: "Talk to our team", ko: "담당자에게 문의" }, getUiLanguage());
    contactRole.textContent = role;
    contactRole.style.display = role ? "block" : "none";
    contactAction.textContent = label;
    contactAction.setAttribute("aria-label", label);
    renderAvatar(contactAvatar, contact.avatar_url, name, initials(name, "?"));
    contactBar.classList.add("visible");

    contactAction.onclick = () => {
      if (contact.action === "open_form") {
        openForm(contact.form_id);
        return;
      }
      const phone = String(contact.whatsapp_number || "").replace(/\D/g, "");
      const message = resolveText(contact.prefilled_message, getUiLanguage());
      const url = `https://wa.me/${phone}${message ? `?text=${encodeURIComponent(message)}` : ""}`;
      window.open(url, "_blank", "noopener,noreferrer");
    };
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

    const sourceConfig = assistantConfig.sources || {};
    const sourceMode = sourceConfig.mode || "compact";
    const safeSources = (Array.isArray(sources) ? sources : []).filter((source) => safeSourceUrl(source?.url) !== "#");
    if (sourceMode !== "hidden" && safeSources.length > 0) {
      const label = resolveText(sourceConfig.label, getUiLanguage()) || "Sources";
      const open = sourceMode === "expanded" ? " open" : "";
      html += `<details class="wr-sources"${open}><summary>${escapeHtml(label)} · ${safeSources.length}</summary><div class="wr-sources-list">`;
      safeSources.forEach((source, index) => {
        const sourceTitle = source.title || (() => {
          try { return new URL(source.url).hostname; } catch (e) { return `Source ${index + 1}`; }
        })();
        html += `<a href="${safeSourceUrl(source.url)}" target="_blank" rel="noopener noreferrer"><span aria-hidden="true">↗</span><span>${escapeHtml(sourceTitle)}</span></a>`;
      });
      html += `</div></details>`;
    }

    return html;
  }

  function appendFeedback(message, messageId) {
    const feedback = assistantConfig.feedback || {};
    if (!feedback.enabled) return;

    const wrap = document.createElement("div");
    wrap.className = "wr-feedback";
    const prompt = document.createElement("span");
    prompt.className = "wr-feedback-prompt";
    prompt.textContent = resolveText(feedback.prompt, getUiLanguage()) || "Was this helpful?";
    wrap.appendChild(prompt);

    const icon = (direction) => direction === "up"
      ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M7 10v11H3V10h4Zm0 9h10.2a2 2 0 0 0 2-1.6l1.3-7A2 2 0 0 0 18.5 8H14l.6-3A2.4 2.4 0 0 0 12.2 2L7 10Z"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M7 14V3H3v11h4Zm0-9h10.2a2 2 0 0 1 2 1.6l1.3 7a2 2 0 0 1-2 2.4H14l.6 3a2.4 2.4 0 0 1-2.4 3L7 14Z"/></svg>`;

    ["up", "down"].forEach((rating) => {
      const button = document.createElement("button");
      button.type = "button";
      button.setAttribute("aria-label", rating === "up" ? "Helpful" : "Not helpful");
      button.setAttribute("aria-pressed", "false");
      button.innerHTML = icon(rating);
      button.addEventListener("click", async () => {
        wrap.querySelectorAll("button").forEach((item) => {
          const selected = item === button;
          item.classList.toggle("selected", selected);
          item.setAttribute("aria-pressed", String(selected));
          item.disabled = true;
        });
        const thanks = resolveText(feedback.thanks_message, getUiLanguage()) || "Thanks for your feedback.";
        prompt.textContent = thanks;
        announce(thanks);
        try {
          await fetch(`${API_URL}/api/widget/feedback`, {
            method: "POST",
            headers: requestHeaders(true),
            body: JSON.stringify({
              site_id: parseInt(SITE_ID, 10),
              session_id: sessionId,
              rating,
              message_id: messageId,
              page_url: window.location.href,
            }),
          });
        } catch (e) {}
      }, { once: true });
      wrap.appendChild(button);
    });
    message.appendChild(wrap);
  }

  function addMessage(text, type, sources, confidence, options = {}) {
    const msg = document.createElement("div");
    msg.className = `wr-msg ${type}`;

    if (type === "bot") {
      msg.innerHTML = renderBotHtml(resolveText(text, getUiLanguage()), sources, confidence);
      if (options.feedback) {
        responseSequence += 1;
        appendFeedback(msg, `response_${responseSequence}`);
      }
    } else {
      msg.textContent = resolveText(text, getUiLanguage());
    }

    messages.insertBefore(msg, typing);
    scrollToBottom();
    if (type === "bot" && options.announce !== false) {
      announce(`Assistant: ${resolveText(text, getUiLanguage())}`);
    }
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

  function getActionType(action) {
    return action?.type || action?.action || "send_message";
  }

  function getActionLabel(action) {
    const directLabel = resolveText(action?.label, getUiLanguage());
    if (directLabel) {
      return directLabel;
    }
    if (getActionType(action) === "open_form" && action?.form_id) {
      const formDef = getConfiguredForm(action.form_id);
      return resolveText(formDef?.title, getUiLanguage()) || "Open form";
    }
    return resolveText(action?.message, getUiLanguage()) || "Action";
  }

  function invokeAction(action) {
    const actionType = getActionType(action);
    const actionLabel = getActionLabel(action);
    const actionMessage = resolveText(action?.message, getUiLanguage()) || actionLabel;

    if (actionType === "open_form") {
      openForm(action.form_id);
      return;
    }

    const query = String(actionMessage || actionLabel || "").trim();
    if (query) {
      send(query, actionLabel || query);
    }
  }

  function createActionButton(action) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "wr-chip";
    button.textContent = getActionLabel(action);
    button.addEventListener("click", () => invokeAction(action));
    return button;
  }

  function applyDisplayConfig() {
    const display = assistantConfig.display || {};
    const resolvedTitle = resolveText(display.title, getUiLanguage());
    if (!titleLocked && resolvedTitle) {
      headerTitle.textContent = resolvedTitle;
    }
    const subtitle = resolveText(display.subtitle, getUiLanguage());
    headerSubtitle.textContent = subtitle;
    headerSubtitle.style.display = subtitle ? "block" : "none";
    const placeholder = resolveText(display.input_placeholder, getUiLanguage()) || DEFAULT_PLACEHOLDER;
    input.placeholder = placeholder;
    input.setAttribute("aria-label", placeholder);
    applyAppearanceConfig();
    renderContact();
    renderLanguageSwitch();
  }

  function renderLanguageSwitch() {
    const switchCfg = assistantConfig.language_switch || {};
    const options = switchCfg.options || [];
    if (!switchCfg.enabled || options.length < 2) {
      langSwitch.style.display = "none";
      langSwitch.innerHTML = "";
      return;
    }

    applySelectedLanguage();
    langSwitch.style.display = "flex";
    langSwitch.innerHTML = "";
    options.forEach((option) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `wr-lang-btn${selectedLanguage === option.code ? " active" : ""}`;
      btn.textContent = resolveText(option.label, option.code) || option.code.toUpperCase();
      btn.setAttribute("aria-pressed", String(selectedLanguage === option.code));
      btn.setAttribute("aria-label", `Use ${btn.textContent}`);
      btn.addEventListener("click", () => {
        selectedLanguage = option.code;
        localStorage.setItem(LANGUAGE_KEY, selectedLanguage);
        renderLanguageSwitch();
        applyDisplayConfig();
        debugLog("language switched", { selectedLanguage });
      });
      langSwitch.appendChild(btn);
    });
  }

  // ── Assistant config loading ───────────────────────────────────────────
  async function loadAssistantConfig() {
    debugLog("config fetch start");
    try {
      const resp = await fetch(`${API_URL}/api/widget/config/${encodeURIComponent(SITE_ID)}`, {
        headers: requestHeaders(false),
      });
      debugLog("config fetch response", { status: resp.status, ok: resp.ok });
      if (!resp.ok) {
        const text = await resp.text();
        debugLog("config fetch failed body", text.slice(0, 500));
        return;
      }
      const data = await resp.json();
      if (data && data.assistant) {
        assistantConfig = data.assistant;
        debugLog("config loaded", {
          title: resolveText(assistantConfig.display?.title, getUiLanguage()),
          greetingEnabled: !!assistantConfig.greeting?.enabled,
          greetingLength: resolveText(assistantConfig.greeting?.message, getUiLanguage()).length,
          starters: (assistantConfig.starters || []).length,
          forms: (assistantConfig.forms || []).length,
          languageSwitch: assistantConfig.language_switch || {}
        });
      }
    } catch (e) {
      console.warn("web-rag widget config load failed:", e);
      debugLog("config fetch exception", e.message || String(e));
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
    if (PREVIEW_RESET_GREETING) {
      try {
        localStorage.removeItem(storageKey);
      } catch (e) {}
    }
    const alreadyShown = localStorage.getItem(storageKey) === "1";
    debugLog("greeting check", {
      enabled: !!greeting.enabled,
      hasMessage: !!resolveText(greeting.message, getUiLanguage()),
      alreadyShown,
      greetingRendered
    });
    if (!greeting.enabled || !resolveText(greeting.message, getUiLanguage()) || greetingRendered) {
      return false;
    }
    if (greeting.show_once && alreadyShown) {
      greetingRendered = true;
      return false;
    }

    addMessage(resolveText(greeting.message, getUiLanguage()), "bot");
    greetingRendered = true;
    debugLog("greeting rendered");
    if (greeting.show_once) {
      localStorage.setItem(storageKey, "1");
    }
    return true;
  }

  function renderStartersIfNeeded() {
    const starters = assistantConfig.starters || [];
    debugLog("starters check", {
      count: starters.length,
      startersRendered
    });
    if (startersRendered || starters.length === 0) {
      return;
    }

    addBotCard((card) => {
      const title = document.createElement("div");
      title.className = "wr-card-title";
      title.textContent = resolveText({
        ru: "Быстрые действия",
        en: "Quick actions"
      }, getUiLanguage());
      card.appendChild(title);

      const actions = document.createElement("div");
      actions.className = "wr-actions";

      starters.forEach((starter) => {
        actions.appendChild(createActionButton(starter));
      });

      card.appendChild(actions);
    });

    startersRendered = true;
    debugLog("starters rendered", starters.map((starter) => ({
      id: starter.id,
      action: getActionType(starter),
      form_id: starter.form_id || ""
    })));
  }

  function renderResponseActions(actions) {
    if (!Array.isArray(actions) || actions.length === 0) {
      debugLog("response actions skipped", { count: 0 });
      return;
    }

    addBotCard((card) => {
      const title = document.createElement("div");
      title.className = "wr-card-title";
      title.textContent = resolveText({
        ru: "Подходящие действия",
        en: "Suggested actions",
        ko: "추천 작업"
      }, getUiLanguage());
      card.appendChild(title);

      const actionsWrap = document.createElement("div");
      actionsWrap.className = "wr-actions";
      actions.forEach((action) => {
        actionsWrap.appendChild(createActionButton(action));
      });
      card.appendChild(actionsWrap);
    });

    debugLog("response actions rendered", actions.map((action) => ({
      id: action.id || "",
      type: getActionType(action),
      form_id: action.form_id || ""
    })));
  }

  function scheduleInitialContent() {
    debugLog("schedule initial content", {
      isOpen,
      initialContentScheduled,
      assistantConfigLoaded
    });
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
      debugLog("initial content timer fired");
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
      placeholder.textContent = resolveText(field.placeholder, getUiLanguage()) || `Select ${resolveText(field.label, getUiLanguage())}`;
      select.appendChild(placeholder);
      (field.options || []).forEach((option) => {
        const opt = document.createElement("option");
        if (option && typeof option === "object" && !Array.isArray(option)) {
          opt.value = option.value || resolveText(option.label, getUiLanguage());
          opt.textContent = resolveText(option.label, getUiLanguage()) || opt.value;
        } else {
          opt.value = option;
          opt.textContent = resolveText(option, getUiLanguage());
        }
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
    debugLog("open form", {
      formId,
      found: !!formDef
    });
    if (!formDef) {
      addMessage("Sorry, this form is not available right now.", "bot");
      return;
    }

    formSequence += 1;
    const instanceId = `wr-form-${formSequence}`;
    const result = addBotCard((card) => {
      const title = document.createElement("div");
      title.className = "wr-card-title";
      title.id = `${instanceId}-title`;
      title.textContent = resolveText(formDef.title, getUiLanguage()) || "Form";
      card.appendChild(title);

      if (resolveText(formDef.description, getUiLanguage())) {
        const desc = document.createElement("div");
        desc.className = "wr-card-text";
        desc.textContent = resolveText(formDef.description, getUiLanguage());
        card.appendChild(desc);
      }

      const formEl = document.createElement("form");
      formEl.className = "wr-form";
      formEl.setAttribute("aria-labelledby", title.id);
      const fieldRefs = {};

      (formDef.fields || []).forEach((field) => {
        const wrap = document.createElement("div");
        wrap.className = "wr-field";

        const label = document.createElement("label");
        const fieldId = `${instanceId}-${field.name}`;
        const errorId = `${fieldId}-error`;
        label.htmlFor = fieldId;
        label.textContent = `${resolveText(field.label, getUiLanguage())}${field.required ? " *" : ""}`;
        wrap.appendChild(label);

        const inputEl = createInputForField(field);
        inputEl.id = fieldId;
        inputEl.name = field.name;
        inputEl.placeholder = resolveText(field.placeholder, getUiLanguage()) || "";
        inputEl.setAttribute("aria-describedby", errorId);
        if (field.required) {
          inputEl.required = true;
        }
        wrap.appendChild(inputEl);

        const error = document.createElement("div");
        error.className = "wr-error-text";
        error.id = errorId;
        error.setAttribute("aria-live", "polite");
        wrap.appendChild(error);

        fieldRefs[field.name] = { field, inputEl, error };
        formEl.appendChild(wrap);
      });

      const status = document.createElement("div");
      status.className = "wr-inline-status";
      status.setAttribute("role", "status");
      formEl.appendChild(status);

      const submitBtn = document.createElement("button");
      submitBtn.type = "submit";
      submitBtn.className = "wr-submit";
      submitBtn.textContent = resolveText(formDef.submit_label, getUiLanguage()) || "Submit";
      formEl.appendChild(submitBtn);

      formEl.addEventListener("submit", async (event) => {
        event.preventDefault();
        Object.values(fieldRefs).forEach((ref) => {
          ref.error.textContent = "";
          ref.inputEl.removeAttribute("aria-invalid");
        });
        status.textContent = "";
        submitBtn.disabled = true;

        const values = {};
        Object.keys(fieldRefs).forEach((name) => {
          values[name] = fieldRefs[name].inputEl.value || "";
        });

        try {
          debugLog("form submit start", { formId: formDef.id, fields: Object.keys(values) });
          const resp = await fetch(`${API_URL}/api/widget/forms/submit`, {
            method: "POST",
            headers: requestHeaders(true),
            body: JSON.stringify({
              site_id: parseInt(SITE_ID, 10),
              form_id: formDef.id,
              values,
              session_id: sessionId,
              page_url: window.location.href,
              response_language: selectedLanguage || undefined,
            }),
          });
          const data = await resp.json();
          debugLog("form submit response", { status: resp.status, ok: resp.ok, body: data });
          if (!resp.ok) {
            const errors = data.errors || {};
            Object.keys(errors).forEach((name) => {
              if (fieldRefs[name]) {
                fieldRefs[name].error.textContent = errors[name];
                fieldRefs[name].inputEl.setAttribute("aria-invalid", "true");
              }
            });
            status.textContent = data.error || "Please check the form and try again.";
            const firstInvalid = Object.keys(errors).map((name) => fieldRefs[name]?.inputEl).find(Boolean);
            if (firstInvalid) firstInvalid.focus();
            return;
          }

          addMessage(data.message || resolveText(formDef.success_message, getUiLanguage()) || "Thanks. Your request has been sent.", "bot");
          formEl.remove();
          return;
        } catch (err) {
          status.textContent = "Sorry, something went wrong. Please try again.";
          console.error("web-rag form submit error:", err);
          debugLog("form submit exception", err.message || String(err));
        } finally {
          submitBtn.disabled = false;
        }
      });

      card.appendChild(formEl);
    });
    window.setTimeout(() => result.card.querySelector("input, textarea, select")?.focus(), 0);
  }

  // ── Chat sending ───────────────────────────────────────────────────────
  async function send(queryOverride, visibleText) {
    const query = String(queryOverride || input.value || "").trim();
    if (!query) return;
    debugLog("chat send", { query, visibleText: visibleText || "" });

    input.value = "";
    input.disabled = true;
    sendBtn.disabled = true;
    addMessage((visibleText || query).trim(), "user");
    typing.classList.add("active");
    announce("Assistant is typing");
    scrollToBottom();

    try {
      const resp = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: requestHeaders(true),
        body: JSON.stringify({
          site_id: parseInt(SITE_ID, 10),
          query,
          session_id: sessionId,
          origin_domain: window.location.hostname,
          response_language: selectedLanguage || undefined,
        }),
      });

      const data = await resp.json();
      debugLog("chat response", { status: resp.status, ok: resp.ok, body: data });
      if (!resp.ok) {
        throw new Error(data.error || "Request failed");
      }
      addMessage(data.answer, "bot", data.sources, data.confidence, { feedback: true });
      renderResponseActions(data.actions || []);
    } catch (err) {
      addMessage("Sorry, something went wrong. Please try again.", "bot");
      console.error("web-rag widget error:", err);
      debugLog("chat exception", err.message || String(err));
    } finally {
      typing.classList.remove("active");
      sendBtn.disabled = false;
      input.disabled = false;
      input.focus();
      scrollToBottom();
    }
  }

  // ── Actions ────────────────────────────────────────────────────────────
  function mountInlineHost() {
    if (!INLINE_MODE || inlineMounted) return;
    mountTarget.replaceChildren(host);
    inlineMounted = true;
  }

  function setOpen(nextOpen, options = {}) {
    const shouldOpen = Boolean(nextOpen);
    if (shouldOpen === isOpen && !options.force) return;
    if (shouldOpen && !isOpen) {
      lastFocusedElement = shadow.activeElement || document.activeElement;
    }
    isOpen = shouldOpen;
    if (isOpen) {
      mountInlineHost();
    }
    panel.classList.toggle("open", isOpen);
    bubble.classList.toggle("panel-open", isOpen);
    panel.setAttribute("aria-hidden", String(!isOpen));
    bubble.setAttribute("aria-expanded", String(isOpen));
    bubble.setAttribute("aria-label", isOpen ? "Close chat assistant" : "Open chat assistant");
    debugLog("toggle", { isOpen });
    if (isOpen) {
      scheduleInitialContent();
      window.setTimeout(() => input.focus(), 0);
      scrollToBottom();
      announce("Chat assistant opened");
    } else {
      announce("Chat assistant closed");
      if (!options.skipFocusReturn) {
        if (!HIDE_BUBBLE) bubble.focus();
        else if (lastFocusedElement && typeof lastFocusedElement.focus === "function") lastFocusedElement.focus();
      }
    }
  }

  function toggle() {
    setOpen(!isOpen);
  }

  if (!HIDE_BUBBLE) {
    bubble.addEventListener("click", toggle);
  }
  window.addEventListener("web-rag:open", () => {
    if (!isOpen) {
      toggle();
    } else {
      input.focus();
      scrollToBottom();
    }
  });
  closeBtn.addEventListener("click", toggle);
  panel.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && isOpen && (!HIDE_BUBBLE || INLINE_START_CLOSED)) {
      event.preventDefault();
      setOpen(false);
      return;
    }
    if (event.key !== "Tab" || INLINE_MODE) return;
    const focusable = Array.from(panel.querySelectorAll(
      'button:not([disabled]), a[href], input:not([disabled]), textarea:not([disabled]), select:not([disabled]), summary, [tabindex]:not([tabindex="-1"])'
    )).filter((element) => element.getClientRects().length > 0);
    if (!focusable.length) {
      event.preventDefault();
      panel.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && shadow.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && shadow.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });
  sendBtn.addEventListener("click", () => send());
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });

  loadAssistantConfig();

  if (PREVIEW_OPEN || AUTO_OPEN) {
    window.setTimeout(() => {
      if (!isOpen) {
        toggle();
      }
    }, 150);
  }
})();
