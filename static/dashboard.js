(() => {
  "use strict";

  const els = {
    cameraStatus: document.querySelector("#camera-status .dot"),
    cameraLabel: document.querySelector("#camera-status .label"),
    ollamaStatus: document.querySelector("#ollama-status .dot"),
    ollamaLabel: document.querySelector("#ollama-status .label"),
    videoFeed: document.getElementById("video-feed"),
    videoOverlay: document.getElementById("video-overlay"),
    eventsList: document.getElementById("events-list"),
    chatMessages: document.getElementById("chat-messages"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
    chatSend: document.getElementById("chat-send"),
    quickPrompts: document.querySelectorAll(".quick-prompts button"),
  };

  const history = [];

  // ---------- Utilidades DOM ----------

  function setPill(dotEl, labelEl, state, text) {
    dotEl.dataset.state = state;
    if (text) labelEl.textContent = text;
  }

  function addMessage(role, content, { error = false } = {}) {
    const wrapper = document.createElement("div");
    wrapper.className = `chat-msg ${role}${error ? " error" : ""}`;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = content;
    wrapper.appendChild(bubble);
    els.chatMessages.appendChild(wrapper);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    return bubble;
  }

  function addTypingBubble() {
    const wrapper = document.createElement("div");
    wrapper.className = "chat-msg assistant";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
    wrapper.appendChild(bubble);
    els.chatMessages.appendChild(wrapper);
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
    return bubble;
  }

  // ---------- Polling ----------

  async function refreshCameraStatus() {
    try {
      const r = await fetch("/camera/status");
      if (!r.ok) throw new Error("status " + r.status);
      const data = await r.json();
      if (data.online && data.connected && data.has_live_frame) {
        setPill(els.cameraStatus, els.cameraLabel, "ok", "Camera online");
        els.videoOverlay.hidden = true;
      } else if (data.online) {
        setPill(els.cameraStatus, els.cameraLabel, "warn", "Reconectando...");
        els.videoOverlay.hidden = false;
      } else {
        setPill(els.cameraStatus, els.cameraLabel, "error", "Camera offline");
        els.videoOverlay.hidden = false;
      }
    } catch {
      setPill(els.cameraStatus, els.cameraLabel, "error", "Camera offline");
      els.videoOverlay.hidden = false;
    }
  }

  async function refreshHealth() {
    try {
      const r = await fetch("/health");
      const data = await r.json();
      if (data.ollama_available) {
        setPill(els.ollamaStatus, els.ollamaLabel, "ok", "Agente online");
      } else {
        setPill(els.ollamaStatus, els.ollamaLabel, "error", "Agente offline");
      }
    } catch {
      setPill(els.ollamaStatus, els.ollamaLabel, "error", "Agente offline");
    }
  }

  async function refreshEvents() {
    try {
      const r = await fetch("/events");
      if (!r.ok) return;
      const data = await r.json();
      renderEvents(data);
    } catch {
      /* ignora, proxima volta tenta de novo */
    }
  }

  function renderEvents(events) {
    if (!events || events.length === 0) {
      els.eventsList.innerHTML = '<li class="events-empty">Nenhuma deteccao registrada ainda.</li>';
      return;
    }
    els.eventsList.innerHTML = events
      .map(
        (e) => `
          <li>
            <span class="label">${escapeHtml(e.label)}</span>
            <span class="conf">${Number(e.confidence).toFixed(2)}</span>
            <span class="ts">${escapeHtml(e.timestamp || "")}</span>
          </li>`
      )
      .join("");
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  async function refreshWeather() {
    try {
      const r = await fetch("/weather");
      if (!r.ok) return;
      const data = await r.json();
      if (!data.enabled || !data.available) {
        document.getElementById("weather-card").hidden = true;
        return;
      }
      document.getElementById("weather-card").hidden = false;
      document.getElementById("weather-temp").textContent = `${data.temperature_c.toFixed(1)} °C`;
      document.getElementById("weather-humidity").textContent = `${data.humidity_pct}%`;
      document.getElementById("weather-precip").textContent =
        data.precipitation_mm != null ? `${data.precipitation_mm.toFixed(1)} mm/h` : "—";
      document.getElementById("weather-wind").textContent =
        data.wind_kmh != null ? `${data.wind_kmh.toFixed(1)} km/h` : "—";
      document.getElementById("weather-condition").textContent = data.condition_label || "—";
      document.getElementById("weather-stale").hidden = !data.is_stale;
    } catch {
      /* silencioso, proxima volta tenta de novo */
    }
  }

  // ---------- Chat ----------

  async function sendMessage(question) {
    const text = question.trim();
    if (!text) return;

    addMessage("user", text);
    history.push({ role: "user", content: text });
    els.chatInput.value = "";
    els.chatSend.disabled = true;

    const bubble = addTypingBubble();
    let accumulated = "";

    try {
      const res = await fetch("/chat?stream=1", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/x-ndjson",
        },
        body: JSON.stringify({ question: text, history }),
      });

      if (!res.ok || !res.body) {
        const detail = await safeJson(res);
        throw new Error(detail?.detail || `Erro ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, idx).trim();
          buffer = buffer.slice(idx + 1);
          if (!line) continue;
          let obj;
          try { obj = JSON.parse(line); } catch { continue; }
          if (obj.error) {
            throw new Error(obj.error);
          }
          if (obj.chunk) {
            accumulated += obj.chunk;
            bubble.textContent = accumulated;
            els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
          }
          if (obj.done) break;
        }
      }

      if (!accumulated) {
        bubble.textContent = "(resposta vazia do agente)";
      } else {
        history.push({ role: "assistant", content: accumulated });
      }
    } catch (err) {
      bubble.parentElement.classList.add("error");
      bubble.textContent = `Erro: ${err.message || err}`;
    } finally {
      els.chatSend.disabled = false;
      els.chatInput.focus();
    }
  }

  async function safeJson(res) {
    try { return await res.json(); } catch { return null; }
  }

  // ---------- Eventos UI ----------

  els.chatForm.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(els.chatInput.value);
  });

  els.quickPrompts.forEach((btn) => {
    btn.addEventListener("click", () => sendMessage(btn.dataset.prompt || ""));
  });

  els.videoFeed.addEventListener("error", () => {
    els.videoOverlay.hidden = false;
  });
  els.videoFeed.addEventListener("load", () => {
    els.videoOverlay.hidden = true;
  });

  // ---------- Boot ----------

  refreshCameraStatus();
  refreshHealth();
  refreshEvents();
  refreshWeather();
  setInterval(refreshCameraStatus, 5000);
  setInterval(refreshEvents, 5000);
  setInterval(refreshHealth, 15000);
  setInterval(refreshWeather, 5 * 60 * 1000);  // 5 minutos
})();
