// Wait for full page load before doing anything
window.addEventListener('load', function() {

  // ── ELEMENTS ──
  const chatBox  = document.getElementById("chat-box");
  const handBtn  = document.getElementById("hand-btn");
  const camFloat = document.getElementById("cam-float");
  const video    = document.getElementById("video");
  const canvas   = document.getElementById("canvas");
  const ctx      = canvas.getContext("2d");

  // ── STATE ──
  let cameraStarted = false;
  let cameraOpen    = false;
  let currentText   = "";
  let buffer        = [];
  let lastSmooth    = "";
  let lastAddedTime = 0;
  let autoAiSign    = "";
  let autoAiCount   = 0;
  let frameCount    = 0;
  const AUTO_FRAMES = 15;

  // ── THEME ──
  window.toggleTheme = function() {
    const html = document.documentElement;
    const dark = html.getAttribute("data-theme") === "dark";
    html.setAttribute("data-theme", dark ? "light" : "dark");
    document.getElementById("theme-btn").innerText = dark ? "☀️" : "🌙";
  };

  // ── NEW CHAT ──
  window.newChat = function() {
    chatBox.innerHTML =
      '<div class="welcome" id="welcome"><h1>What\'s on your mind today?</h1></div>';
    currentText = "";
  };

  // ── TOGGLE CAMERA ──
  window.toggleCamera = function() {
    cameraOpen = !cameraOpen;
    camFloat.classList.toggle("open", cameraOpen);
    handBtn.classList.toggle("active", cameraOpen);
    if (cameraOpen && !cameraStarted) {
      startCamera();
      cameraStarted = true;
    }
  };

  function resizeCanvas() {
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
  }

  function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        video.srcObject = stream;
        video.onloadedmetadata = resizeCanvas;
      })
      .catch(err => console.error("Camera denied:", err));

    try {
      const cam = new Camera(video, {
        onFrame: async () => {
          try { await hands.send({ image: video }); } catch(e) {}
        },
        width: 640, height: 480
      });
      cam.start();
    } catch(e) {
      console.error("Camera util error:", e);
    }
  }

  // ── MEDIAPIPE ──
  const hands = new Hands({
    locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
  });
  hands.setOptions({
    maxNumHands: 1, modelComplexity: 1,
    minDetectionConfidence: 0.7, minTrackingConfidence: 0.7
  });

  function normalize(lm) {
    let c = lm.map(p => [p.x, p.y, p.z]);
    const w = c[0];
    c = c.map(p => [p[0]-w[0], p[1]-w[1], p[2]-w[2]]);
    const mx = Math.max(...c.flat().map(Math.abs));
    if (mx) c = c.map(p => p.map(v => v/mx));
    return c.flat();
  }

  hands.onResults(async results => {
    resizeCanvas();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!results.multiHandLandmarks || !results.multiHandLandmarks.length) {
      autoAiSign = ""; autoAiCount = 0;
      return;
    }

    const lm = results.multiHandLandmarks[0];

    // Draw skeleton
    if (Hands.HAND_CONNECTIONS) {
      ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 1.5;
      for (const [s, e] of Hands.HAND_CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(lm[s].x*canvas.width, lm[s].y*canvas.height);
        ctx.lineTo(lm[e].x*canvas.width, lm[e].y*canvas.height);
        ctx.stroke();
      }
    }
    ctx.fillStyle = "#22c55e";
    for (const p of lm) {
      ctx.beginPath();
      ctx.arc(p.x*canvas.width, p.y*canvas.height, 4, 0, 2*Math.PI);
      ctx.fill();
    }

    // Every 3rd frame only
    if (++frameCount % 3 !== 0) return;

    try {
      const res  = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: normalize(lm) })
      });
      const data = await res.json();
      if (!data.prediction) return;

      const pred = data.prediction;
      const conf = data.confidence;

      // Smoothing
      buffer.push(pred);
      if (buffer.length > 7) buffer.shift();
      const cnt = {};
      buffer.forEach(l => cnt[l] = (cnt[l]||0)+1);
      const smooth = Object.keys(cnt).reduce((a,b) => cnt[a]>cnt[b]?a:b);

      // Build text on hold
      if (conf > 0.75) {
        if (smooth !== lastSmooth) {
          lastSmooth = smooth; lastAddedTime = Date.now();
        } else if (Date.now() - lastAddedTime > 600) {
          currentText += smooth;
          lastAddedTime = Date.now() + 9999;
        }
        // Auto send after 15 consistent frames
        if (conf > 0.85) {
          if (smooth === autoAiSign) {
            autoAiCount++;
            if (autoAiCount >= AUTO_FRAMES) {
              autoAiCount = 0; autoAiSign = "";
              autoSend(smooth);
            }
          } else { autoAiSign = smooth; autoAiCount = 1; }
        }
      } else {
        lastSmooth = ""; autoAiSign = ""; autoAiCount = 0;
      }
    } catch(e) { /* silent */ }
  });

  // ── TTS ──
  function speak(t) {
    if (!t || !t.trim()) return;
    try {
      speechSynthesis.cancel();
      speechSynthesis.speak(new SpeechSynthesisUtterance(t));
    } catch(e) {}
  }

  // ── AUTO SEND ──
  async function autoSend(sign) {
    document.getElementById("welcome")?.remove();
    addMsg("user", "[Sign: " + sign + "]");
    const typing = addTyping();
    try {
      const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "The person signed: '" + sign + "'" })
      });
      const d = await r.json();
      typing.remove();
      addMsg("bot", d.reply || "No response");
      speak(d.reply || "");
    } catch {
      typing.remove();
      addMsg("bot", "❌ Could not reach AI.");
    }
  }

  // ── SEND CHAT ──
  window.sendChat = async function() {
    const el  = document.getElementById("chat-input");
    const msg = el.value.trim() || currentText.trim();
    if (!msg) return;
    el.value = ""; currentText = "";

    document.getElementById("welcome")?.remove();
    addMsg("user", msg);
    const typing = addTyping();

    try {
      const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg })
      });
      const d = await r.json();
      typing.remove();
      addMsg("bot", d.reply || "Error");
      speak(d.reply || "");
    } catch {
      typing.remove();
      addMsg("bot", "❌ Could not reach AI.");
    }
  };

  function addMsg(type, text) {
    const d = document.createElement("div");
    d.className = "message " + type;
    d.innerText = text;
    chatBox.appendChild(d);
    chatBox.scrollTop = chatBox.scrollHeight;
    return d;
  }

  function addTyping() {
    const d = document.createElement("div");
    d.className = "message typing";
    d.innerText = "SignAI is thinking...";
    chatBox.appendChild(d);
    chatBox.scrollTop = chatBox.scrollHeight;
    return d;
  }

}); // end window.load