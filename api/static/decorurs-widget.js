/*!
 * DecorUrs Visual Search Widget
 * -----------------------------
 * A small floating chat icon (bottom-right) that opens a slide-in panel
 * where a shopper can either type/speak a description ("a round marble
 * coffee table") or upload a photo, and get back the closest-matching
 * products from the catalog via the CLIP + Qdrant search API.
 *
 * Zero dependencies, no build step -- safe to drop into any storefront
 * (Shopify theme.liquid, WordPress footer, Wix custom code, etc.) via a
 * single <script> tag:
 *
 *   <script src="https://YOUR-API-DOMAIN/widget/decorurs-widget.js"
 *           data-api-url="https://YOUR-API-DOMAIN" defer></script>
 *
 * The API domain is read from data-api-url on this same <script> tag (or
 * from window.DecorUrsVisualSearchConfig.apiUrl, if you prefer to set it
 * separately). See api/static/README.md for full deployment notes.
 */
(function () {
  "use strict";

  // Guard against the script being included twice on the same page.
  if (window.__decorursVisualSearchLoaded) return;
  window.__decorursVisualSearchLoaded = true;

  var CURRENT_SCRIPT =
    document.currentScript ||
    (function () {
      var scripts = document.getElementsByTagName("script");
      return scripts[scripts.length - 1];
    })();

  var CONFIG = window.DecorUrsVisualSearchConfig || {};
  var API_BASE = (CURRENT_SCRIPT && CURRENT_SCRIPT.getAttribute("data-api-url")) || CONFIG.apiUrl;

  if (!API_BASE) {
    console.error(
      '[DecorUrs Visual Search] No API URL configured. Add data-api-url="https://your-api-domain" ' +
        "to the widget's <script> tag."
    );
    return;
  }
  API_BASE = API_BASE.replace(/\/+$/, "");

  var MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB, matches the API's limit
  var ALLOWED_TYPES = ["image/jpeg", "image/png"];
  var DEFAULT_PLACEHOLDER = "Describe it, or attach / speak a photo…";
  var CONTACT_EMAIL = (CONFIG.contactEmail || "alok@trustic.ca").trim();
  var CONTACT_PHONE = (CONFIG.contactPhone || "780-604-5390").trim();

  /* ---------------------------------------------------------------- */
  /* Styles                                                            */
  /* ---------------------------------------------------------------- */
  // Same palette as the standalone search microsite (warm travertine /
  // walnut / stone tones) so the widget looks native to the DecorUrs
  // brand. Override any of these from your own site CSS if you want a
  // different look -- they're plain custom properties on #dvs-root.
  var STYLE = document.createElement("style");
  STYLE.setAttribute("data-decorurs-widget", "");
  STYLE.textContent =
    '@import url("https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500&family=Inter:wght@400;500;600&display=swap");' +
    "#dvs-root{" +
    "--dvs-bg:#ede8e0;--dvs-surface:#ffffff;--dvs-ink:#2a2620;--dvs-muted:#756f63;" +
    "--dvs-accent:#46554c;--dvs-accent-soft:#dde3dc;--dvs-border:#dcd5c8;--dvs-error:#8c3b2e;" +
    "--dvs-font-display:'Fraunces',serif;--dvs-font-body:'Inter',system-ui,sans-serif;" +
    "all:initial;font-family:var(--dvs-font-body);}" +
    "#dvs-root *{box-sizing:border-box;font-family:var(--dvs-font-body);}" +
    "#dvs-root .dvs-fab{" +
    "position:fixed;right:22px;bottom:22px;width:56px;height:56px;border-radius:50%;" +
    "background:var(--dvs-ink);color:#fff;border:none;cursor:pointer;" +
    "display:flex;align-items:center;justify-content:center;" +
    "box-shadow:0 6px 20px rgba(42,38,32,.28);z-index:2147483000;" +
    "transition:transform .18s ease, box-shadow .18s ease;padding:0;}" +
    "#dvs-root .dvs-fab:hover{transform:translateY(-2px);box-shadow:0 10px 26px rgba(42,38,32,.34);}" +
    "#dvs-root .dvs-fab:active{transform:translateY(0);}" +
    "#dvs-root .dvs-fab svg{width:24px;height:24px;}" +
    "#dvs-root .dvs-fab-dot{position:absolute;top:6px;right:6px;width:9px;height:9px;border-radius:50%;" +
    "background:#c96a4f;border:2px solid var(--dvs-ink);}" +
    "#dvs-root .dvs-panel{" +
    "position:fixed;top:0;right:0;height:100%;width:392px;max-width:100vw;background:var(--dvs-bg);" +
    "box-shadow:-8px 0 32px rgba(42,38,32,.22);z-index:2147483001;" +
    "display:flex;flex-direction:column;transform:translateX(100%);" +
    "transition:transform .28s cubic-bezier(.32,.72,0,1);}" +
    "#dvs-root .dvs-panel.dvs-open{transform:translateX(0);}" +
    "@media (max-width:480px){#dvs-root .dvs-panel{width:100vw;}}" +
    "#dvs-root .dvs-header{" +
    "padding:18px 18px 14px;background:var(--dvs-ink);color:#fff;" +
    "display:flex;align-items:flex-start;justify-content:space-between;flex-shrink:0;}" +
    "#dvs-root .dvs-header-title{font-family:var(--dvs-font-display);font-weight:500;font-size:1.15rem;margin:0 0 2px;}" +
    "#dvs-root .dvs-header-sub{font-size:.78rem;color:#d8d3c8;margin:0;letter-spacing:.02em;}" +
    "#dvs-root .dvs-close{background:none;border:none;color:#fff;cursor:pointer;padding:4px;" +
    "opacity:.8;border-radius:6px;flex-shrink:0;margin-left:8px;}" +
    "#dvs-root .dvs-close:hover{opacity:1;background:rgba(255,255,255,.1);}" +
    "#dvs-root .dvs-messages{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px;}" +
    "#dvs-root .dvs-msg{max-width:88%;font-size:.88rem;line-height:1.5;}" +
    "#dvs-root .dvs-msg-bot{align-self:flex-start;}" +
    "#dvs-root .dvs-msg-user{align-self:flex-end;}" +
    "#dvs-root .dvs-bubble{padding:10px 13px;border-radius:14px;background:var(--dvs-surface);" +
    "color:var(--dvs-ink);border:1px solid var(--dvs-border);}" +
    "#dvs-root .dvs-msg-user .dvs-bubble{background:var(--dvs-ink);color:#fff;border-color:var(--dvs-ink);border-bottom-right-radius:4px;}" +
    "#dvs-root .dvs-msg-bot .dvs-bubble{border-bottom-left-radius:4px;}" +
    "#dvs-root .dvs-msg-user .dvs-thumb{max-width:160px;max-height:160px;border-radius:12px;display:block;border:1px solid var(--dvs-border);}" +
    "#dvs-root .dvs-typing{display:inline-flex;gap:4px;padding:4px 0;}" +
    "#dvs-root .dvs-typing span{width:6px;height:6px;border-radius:50%;background:var(--dvs-muted);" +
    "animation:dvs-bounce 1.2s infinite ease-in-out;}" +
    "#dvs-root .dvs-typing span:nth-child(2){animation-delay:.15s;}" +
    "#dvs-root .dvs-typing span:nth-child(3){animation-delay:.3s;}" +
    "@keyframes dvs-bounce{0%,60%,100%{transform:translateY(0);opacity:.5;}30%{transform:translateY(-4px);opacity:1;}}" +
    "#dvs-root .dvs-error .dvs-bubble{color:var(--dvs-error);border-color:#e3b6ab;background:#fbf1ee;}" +
    "#dvs-root .dvs-detected{font-size:.72rem;color:var(--dvs-muted);margin:2px 0 8px 2px;}" +
    "#dvs-root .dvs-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:2px;}" +
    "#dvs-root .dvs-card{background:var(--dvs-surface);border:1px solid var(--dvs-border);border-radius:10px;" +
    "overflow:hidden;text-decoration:none;color:var(--dvs-ink);display:block;transition:box-shadow .15s ease;}" +
    "#dvs-root .dvs-card:hover{box-shadow:0 4px 14px rgba(42,38,32,.16);}" +
    "#dvs-root .dvs-card-imgwrap{aspect-ratio:1/1;background:var(--dvs-bg);overflow:hidden;}" +
    "#dvs-root .dvs-card-imgwrap img{width:100%;height:100%;object-fit:cover;display:block;}" +
    "#dvs-root .dvs-card-body{padding:8px 9px 10px;}" +
    "#dvs-root .dvs-card-name{font-size:.74rem;font-weight:600;margin:0 0 3px;" +
    "display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;line-height:1.3;}" +
    "#dvs-root .dvs-card-material{font-size:.66rem;color:var(--dvs-muted);margin:0 0 5px;text-transform:capitalize;}" +
    "#dvs-root .dvs-card-meta{display:flex;align-items:center;justify-content:space-between;gap:4px;}" +
    "#dvs-root .dvs-card-price{font-size:.75rem;font-weight:600;margin:0;}" +
    "#dvs-root .dvs-card-match{font-size:.62rem;color:var(--dvs-accent);background:var(--dvs-accent-soft);" +
    "padding:2px 6px;border-radius:20px;font-weight:600;white-space:nowrap;}" +
    "#dvs-root .dvs-contact a{color:var(--dvs-accent);font-weight:600;}" +
    "#dvs-root .dvs-composer{flex-shrink:0;border-top:1px solid var(--dvs-border);background:var(--dvs-surface);" +
    "padding:10px 10px calc(10px + env(safe-area-inset-bottom));}" +
    "#dvs-root .dvs-preview-row{display:none;align-items:center;gap:8px;padding:0 2px 8px;}" +
    "#dvs-root .dvs-preview-row.dvs-show{display:flex;}" +
    "#dvs-root .dvs-preview-row img{width:38px;height:38px;object-fit:cover;border-radius:8px;border:1px solid var(--dvs-border);}" +
    "#dvs-root .dvs-preview-row span{font-size:.75rem;color:var(--dvs-muted);flex:1;}" +
    "#dvs-root .dvs-preview-remove{background:none;border:none;color:var(--dvs-muted);cursor:pointer;font-size:1rem;line-height:1;padding:4px;}" +
    "#dvs-root .dvs-input-row{display:flex;align-items:flex-end;gap:6px;background:var(--dvs-bg);" +
    "border:1px solid var(--dvs-border);border-radius:20px;padding:6px 6px 6px 14px;}" +
    "#dvs-root .dvs-textarea{flex:1;border:none;background:transparent;resize:none;outline:none;" +
    "font-size:.85rem;line-height:1.4;color:var(--dvs-ink);max-height:80px;padding:6px 0;font-family:var(--dvs-font-body);}" +
    "#dvs-root .dvs-textarea::placeholder{color:var(--dvs-muted);}" +
    "#dvs-root .dvs-icon-btn{width:32px;height:32px;border-radius:50%;border:none;background:transparent;" +
    "color:var(--dvs-muted);cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;" +
    "transition:background .15s ease,color .15s ease;padding:0;}" +
    "#dvs-root .dvs-icon-btn:hover{background:var(--dvs-accent-soft);color:var(--dvs-accent);}" +
    "#dvs-root .dvs-icon-btn svg{width:18px;height:18px;}" +
    "#dvs-root .dvs-mic-active{background:var(--dvs-error);color:#fff;animation:dvs-pulse 1.4s infinite;}" +
    "@keyframes dvs-pulse{0%{box-shadow:0 0 0 0 rgba(140,59,46,.4);}70%{box-shadow:0 0 0 8px rgba(140,59,46,0);}100%{box-shadow:0 0 0 0 rgba(140,59,46,0);}}" +
    "#dvs-root .dvs-send-btn{background:var(--dvs-ink);color:#fff;}" +
    "#dvs-root .dvs-send-btn:hover{background:var(--dvs-ink);opacity:.85;}" +
    "#dvs-root .dvs-send-btn:disabled{opacity:.35;cursor:not-allowed;}" +
    "#dvs-root .dvs-hidden-input{display:none;}" +
    "#dvs-root .dvs-dragover{outline:2px dashed var(--dvs-accent);outline-offset:-8px;}" +
    "@media (prefers-reduced-motion: reduce){#dvs-root *{animation-duration:.01ms !important;transition-duration:.01ms !important;}}";
  document.head.appendChild(STYLE);

  /* ---------------------------------------------------------------- */
  /* DOM scaffold                                                      */
  /* ---------------------------------------------------------------- */
  var root = document.createElement("div");
  root.id = "dvs-root";
  root.innerHTML =
    '<button type="button" class="dvs-fab" aria-label="Open visual search" aria-expanded="false">' +
    '  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8a2 2 0 0 1 2-2h1.2a1 1 0 0 0 .86-.5l.9-1.5a1 1 0 0 1 .86-.5h4.36a1 1 0 0 1 .86.5l.9 1.5a1 1 0 0 0 .86.5H18a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8Z"/><circle cx="12" cy="13" r="3.3"/></svg>' +
    '  <span class="dvs-fab-dot"></span>' +
    "</button>" +
    '<div class="dvs-panel" role="dialog" aria-modal="true" aria-label="DecorUrs visual search">' +
    '  <div class="dvs-header">' +
    "    <div>" +
    '      <p class="dvs-header-title">Visual Search</p>' +
    '      <p class="dvs-header-sub">Describe it, or show us a photo</p>' +
    "    </div>" +
    '    <button type="button" class="dvs-close" aria-label="Close">' +
    '      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>' +
    "    </button>" +
    "  </div>" +
    '  <div class="dvs-messages" id="dvs-messages"></div>' +
    '  <div class="dvs-composer">' +
    '    <div class="dvs-preview-row" id="dvs-preview-row">' +
    '      <img id="dvs-preview-img" alt="" />' +
    '      <span id="dvs-preview-name"></span>' +
    '      <button type="button" class="dvs-preview-remove" id="dvs-preview-remove" aria-label="Remove photo">&times;</button>' +
    "    </div>" +
    '    <div class="dvs-input-row">' +
    '      <button type="button" class="dvs-icon-btn" id="dvs-attach-btn" aria-label="Attach a photo">' +
    '        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05 12.25 20.24a5.5 5.5 0 0 1-7.78-7.78l9.19-9.19a3.5 3.5 0 0 1 4.95 4.95l-9.2 9.19a1.5 1.5 0 0 1-2.12-2.12l8.49-8.48"/></svg>' +
    "      </button>" +
    '      <textarea class="dvs-textarea" id="dvs-text-input" rows="1" placeholder="' +
    DEFAULT_PLACEHOLDER +
    '"></textarea>' +
    '      <button type="button" class="dvs-icon-btn" id="dvs-mic-btn" aria-label="Search by voice">' +
    '        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Z"/><path d="M19 11a7 7 0 0 1-14 0M12 18v3"/></svg>' +
    "      </button>" +
    '      <button type="button" class="dvs-icon-btn dvs-send-btn" id="dvs-send-btn" aria-label="Send">' +
    '        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m3 11 18-8-8 18-2-8-8-2Z"/></svg>' +
    "      </button>" +
    '      <input type="file" id="dvs-file-input" class="dvs-hidden-input" accept="image/jpeg,image/png" />' +
    "    </div>" +
    "  </div>" +
    "</div>";
  document.body.appendChild(root);

  var fab = root.querySelector(".dvs-fab");
  var panel = root.querySelector(".dvs-panel");
  var closeBtn = root.querySelector(".dvs-close");
  var messagesEl = root.querySelector("#dvs-messages");
  var textInput = root.querySelector("#dvs-text-input");
  var sendBtn = root.querySelector("#dvs-send-btn");
  var micBtn = root.querySelector("#dvs-mic-btn");
  var attachBtn = root.querySelector("#dvs-attach-btn");
  var fileInput = root.querySelector("#dvs-file-input");
  var previewRow = root.querySelector("#dvs-preview-row");
  var previewImg = root.querySelector("#dvs-preview-img");
  var previewName = root.querySelector("#dvs-preview-name");
  var previewRemove = root.querySelector("#dvs-preview-remove");

  var pendingFile = null;
  var isOpen = false;
  var isBusy = false;

  /* ---------------------------------------------------------------- */
  /* Panel open/close                                                  */
  /* ---------------------------------------------------------------- */
  function openPanel() {
    isOpen = true;
    panel.classList.add("dvs-open");
    fab.setAttribute("aria-expanded", "true");
    if (!messagesEl.childElementCount) {
      addBotMessage(
        "Hi! Tell me what you're looking for &mdash; e.g. <em>&ldquo;a round marble coffee table&rdquo;</em> &mdash; or tap " +
          "the paperclip to upload a photo, and I'll match it against the catalog."
      );
    }
    setTimeout(function () {
      textInput.focus();
    }, 300);
  }
  function closePanel() {
    isOpen = false;
    panel.classList.remove("dvs-open");
    fab.setAttribute("aria-expanded", "false");
  }
  fab.addEventListener("click", function () {
    isOpen ? closePanel() : openPanel();
  });
  closeBtn.addEventListener("click", closePanel);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && isOpen) closePanel();
  });

  /* ---------------------------------------------------------------- */
  /* Message rendering                                                 */
  /* ---------------------------------------------------------------- */
  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addUserTextMessage(text) {
    var wrap = document.createElement("div");
    wrap.className = "dvs-msg dvs-msg-user";
    var bubble = document.createElement("div");
    bubble.className = "dvs-bubble";
    bubble.textContent = text;
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollToBottom();
  }

  function addUserImageMessage(previewUrl) {
    var wrap = document.createElement("div");
    wrap.className = "dvs-msg dvs-msg-user";
    var img = document.createElement("img");
    img.src = previewUrl;
    img.alt = "Your uploaded photo";
    img.className = "dvs-thumb";
    wrap.appendChild(img);
    messagesEl.appendChild(wrap);
    scrollToBottom();
  }

  function addBotMessage(html) {
    var wrap = document.createElement("div");
    wrap.className = "dvs-msg dvs-msg-bot";
    var bubble = document.createElement("div");
    bubble.className = "dvs-bubble";
    bubble.innerHTML = html;
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function addTyping() {
    var wrap = document.createElement("div");
    wrap.className = "dvs-msg dvs-msg-bot";
    wrap.innerHTML =
      '<div class="dvs-bubble"><span class="dvs-typing"><span></span><span></span><span></span></span></div>';
    messagesEl.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function escapeHtml(str) {
    var div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function renderResults(data, typingEl) {
    if (!data.results || data.results.length === 0) {
      typingEl.className = "dvs-msg dvs-msg-bot";
      var detected = "";
      if (data.query_material && data.query_shape) {
        detected =
          '<p class="dvs-detected">Detected: ' +
          escapeHtml(data.query_shape) +
          ", " +
          escapeHtml(data.query_material) +
          "</p>";
      }
      typingEl.innerHTML =
        '<div class="dvs-bubble">' +
        "No matching pieces in the catalog yet." +
        detected +
        '<p class="dvs-contact" style="margin:8px 0 0;">Want something custom made? Email ' +
        '<a href="mailto:' +
        CONTACT_EMAIL +
        '">' +
        CONTACT_EMAIL +
        '</a> or call <a href="tel:' +
        CONTACT_PHONE.replace(/[^\d+]/g, "") +
        '">' +
        CONTACT_PHONE +
        "</a>.</p></div>";
      scrollToBottom();
      return;
    }

    var detectedHtml = "";
    if (data.query_material && data.query_shape) {
      detectedHtml =
        '<p class="dvs-detected">Closest matches &mdash; detected as ' +
        escapeHtml(data.query_shape) +
        ", " +
        escapeHtml(data.query_material) +
        "</p>";
    } else {
      detectedHtml = '<p class="dvs-detected">Closest matches</p>';
    }

    var cardsHtml = data.results
      .map(function (p) {
        return (
          '<a class="dvs-card" href="' +
          escapeHtml(p.product_url) +
          '" target="_blank" rel="noopener noreferrer">' +
          '<div class="dvs-card-imgwrap"><img src="' +
          escapeHtml(p.image_url) +
          '" alt="' +
          escapeHtml(p.name) +
          '" loading="lazy"/></div>' +
          '<div class="dvs-card-body">' +
          '<p class="dvs-card-name">' +
          escapeHtml(p.name) +
          "</p>" +
          '<p class="dvs-card-material">' +
          escapeHtml(p.shape) +
          ", " +
          escapeHtml(p.material) +
          "</p>" +
          '<div class="dvs-card-meta">' +
          '<p class="dvs-card-price">$' +
          escapeHtml(p.price) +
          " CAD</p>" +
          '<span class="dvs-card-match">' +
          Math.round(p.score * 100) +
          "% match</span>" +
          "</div></div></a>"
        );
      })
      .join("");

    typingEl.className = "dvs-msg dvs-msg-bot";
    typingEl.style.maxWidth = "100%";
    typingEl.innerHTML =
      '<div class="dvs-bubble" style="width:100%;">' + detectedHtml + '<div class="dvs-grid">' + cardsHtml + "</div></div>";
    scrollToBottom();
  }

  function renderError(message, typingEl) {
    typingEl.className = "dvs-msg dvs-msg-bot dvs-error";
    typingEl.innerHTML = '<div class="dvs-bubble">' + escapeHtml(message) + "</div>";
    scrollToBottom();
  }

  /* ---------------------------------------------------------------- */
  /* Composer state                                                    */
  /* ---------------------------------------------------------------- */
  function setBusy(busy) {
    isBusy = busy;
    sendBtn.disabled = busy;
    textInput.disabled = busy;
    attachBtn.disabled = busy;
  }

  function autoResize() {
    textInput.style.height = "auto";
    textInput.style.height = Math.min(textInput.scrollHeight, 80) + "px";
  }
  textInput.addEventListener("input", autoResize);
  textInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  /* ---------------------------------------------------------------- */
  /* Image attach / drag-drop                                          */
  /* ---------------------------------------------------------------- */
  function validateFile(file) {
    if (ALLOWED_TYPES.indexOf(file.type) === -1) {
      return "That file didn't make it through. Use a JPG or PNG image.";
    }
    if (file.size > MAX_FILE_SIZE) {
      return "That file is too large. Keep it under 10MB.";
    }
    return null;
  }

  function setPendingFile(file) {
    var err = validateFile(file);
    if (err) {
      addBotMessage(escapeHtml(err));
      return;
    }
    pendingFile = file;
    var url = URL.createObjectURL(file);
    previewImg.src = url;
    previewName.textContent = file.name;
    previewRow.classList.add("dvs-show");
    textInput.focus();
  }

  attachBtn.addEventListener("click", function () {
    fileInput.click();
  });
  fileInput.addEventListener("change", function (e) {
    var file = e.target.files && e.target.files[0];
    if (file) setPendingFile(file);
    fileInput.value = "";
  });
  previewRemove.addEventListener("click", function () {
    pendingFile = null;
    previewRow.classList.remove("dvs-show");
  });

  ["dragenter", "dragover"].forEach(function (evt) {
    panel.addEventListener(evt, function (e) {
      e.preventDefault();
      panel.classList.add("dvs-dragover");
    });
  });
  ["dragleave", "drop"].forEach(function (evt) {
    panel.addEventListener(evt, function (e) {
      e.preventDefault();
      panel.classList.remove("dvs-dragover");
    });
  });
  panel.addEventListener("drop", function (e) {
    var file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) setPendingFile(file);
  });

  /* ---------------------------------------------------------------- */
  /* Speech-to-text ("verbal prompts")                                 */
  /* ---------------------------------------------------------------- */
  var SpeechRecognitionImpl = window.SpeechRecognition || window.webkitSpeechRecognition;
  var recognition = null;
  var isListening = false;

  if (SpeechRecognitionImpl) {
    recognition = new SpeechRecognitionImpl();
    recognition.lang = (CONFIG.speechLang || "en-US");
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;

    recognition.onresult = function (event) {
      var transcript = "";
      for (var i = event.resultIndex; i < event.results.length; i++) {
        transcript += event.results[i][0].transcript;
      }
      textInput.value = transcript;
      autoResize();
      var last = event.results[event.results.length - 1];
      if (last.isFinal) {
        stopListening();
        if (transcript.trim()) handleSend();
      }
    };
    recognition.onerror = function () {
      stopListening();
    };
    recognition.onend = function () {
      isListening = false;
      micBtn.classList.remove("dvs-mic-active");
      textInput.placeholder = DEFAULT_PLACEHOLDER;
    };
  } else {
    micBtn.style.display = "none";
  }

  function startListening() {
    if (!recognition || isListening || isBusy) return;
    isListening = true;
    micBtn.classList.add("dvs-mic-active");
    textInput.placeholder = "Listening…";
    try {
      recognition.start();
    } catch (e) {
      /* already started -- ignore */
    }
  }
  function stopListening() {
    if (!recognition) return;
    isListening = false;
    micBtn.classList.remove("dvs-mic-active");
    try {
      recognition.stop();
    } catch (e) {}
  }
  micBtn.addEventListener("click", function () {
    isListening ? stopListening() : startListening();
  });

  /* ---------------------------------------------------------------- */
  /* Search calls                                                      */
  /* ---------------------------------------------------------------- */
  function handleSend() {
    if (isBusy) return;
    if (pendingFile) {
      var file = pendingFile;
      pendingFile = null;
      var url = URL.createObjectURL(file);
      previewRow.classList.remove("dvs-show");
      textInput.value = "";
      autoResize();
      searchByImage(file, url);
      return;
    }
    var text = textInput.value.trim();
    if (!text) return;
    textInput.value = "";
    autoResize();
    searchByText(text);
  }
  sendBtn.addEventListener("click", handleSend);

  function searchByText(query) {
    addUserTextMessage(query);
    var typingEl = addTyping();
    setBusy(true);
    fetch(API_BASE + "/search/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query }),
    })
      .then(function (res) {
        return res.json().then(function (body) {
          if (!res.ok) throw new Error(body.detail || "Search failed. Try again.");
          return body;
        });
      })
      .then(function (data) {
        renderResults(data, typingEl);
      })
      .catch(function (err) {
        renderError(err.message || "Something went wrong. Try again.", typingEl);
      })
      .finally(function () {
        setBusy(false);
      });
  }

  function searchByImage(file, previewUrl) {
    addUserImageMessage(previewUrl);
    var typingEl = addTyping();
    setBusy(true);
    var formData = new FormData();
    formData.append("file", file);
    fetch(API_BASE + "/search", { method: "POST", body: formData })
      .then(function (res) {
        return res.json().then(function (body) {
          if (!res.ok) throw new Error(body.detail || "Search failed. Try again.");
          return body;
        });
      })
      .then(function (data) {
        renderResults(data, typingEl);
      })
      .catch(function (err) {
        renderError(err.message || "Something went wrong. Try again.", typingEl);
      })
      .finally(function () {
        setBusy(false);
      });
  }
})();
