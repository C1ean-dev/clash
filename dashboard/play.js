"use strict";

(() => {
  const el = id => document.getElementById(`play-${id}`);
  const board = el("board");

  // Audio effects synthesizer using Web Audio API
  let audioCtx = null;
  let soundEnabled = true;

  function initAudio() {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) audioCtx = new AudioContextClass();
    }
    if (audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume();
    }
  }

  function playTone(freq, duration, type = "sine", gainVal = 0.15) {
    if (!soundEnabled || !audioCtx) return;
    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
      gain.gain.setValueAtTime(gainVal, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + duration);
    } catch (_) {}
  }

  function playSelectSound() {
    initAudio();
    playTone(520, 0.08, "triangle", 0.12);
    setTimeout(() => playTone(680, 0.1, "sine", 0.15), 50);
  }

  function playDeploySound() {
    initAudio();
    if (!soundEnabled || !audioCtx) return;
    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(260, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(60, audioCtx.currentTime + 0.22);
      gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.22);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.22);
    } catch (_) {}
  }

  function playErrorSound() {
    initAudio();
    playTone(180, 0.15, "sawtooth", 0.15);
  }

  function playFanfareSound(win) {
    initAudio();
    if (!soundEnabled || !audioCtx) return;
    const notes = win ? [523.25, 659.25, 783.99, 1046.50] : [440, 370, 311, 220];
    notes.forEach((freq, idx) => {
      setTimeout(() => playTone(freq, 0.35, "triangle", 0.2), idx * 160);
    });
  }

  // State management
  let activeSessionId = null;
  let currentState = null;
  let previousState = null;
  let lastStateTime = 0;
  let pollTimer = null;
  let animFrameId = null;

  let selectedSlot = null; // 0, 1, 2, 3 or null
  let hoveredTile = null;  // { x, y, legal: bool }
  let ripples = [];        // local placement visual effects [{x, y, radius, alpha, color}]

  const tones = ["#8bceff", "#ffa7ab"]; // Blue (human), Red (AI)

  // Load available models and populates select dropdown
  async function loadModels() {
    try {
      const res = await fetch("/api/play/models");
      if (!res.ok) return;
      const data = await res.json();
      const picker = el("ai-model");
      picker.innerHTML = "";

      const ckptGroup = document.createElement("optgroup");
      ckptGroup.label = "🧠 Modelos Treinados (PPO / BC)";

      const botGroup = document.createElement("optgroup");
      botGroup.label = "🤖 Bots Táticos de Regras";

      let tacticalPpoFound = false;

      (data.models || []).forEach(m => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.label;
        if (m.kind === "checkpoint") {
          ckptGroup.appendChild(opt);
          if (m.id.includes("tactical/xbow_30/ppo.pt") && !tacticalPpoFound) {
            opt.selected = true;
            tacticalPpoFound = true;
          }
        } else {
          botGroup.appendChild(opt);
        }
      });

      if (ckptGroup.children.length) picker.appendChild(ckptGroup);
      if (botGroup.children.length) picker.appendChild(botGroup);

      if (!tacticalPpoFound && picker.options.length) {
        picker.selectedIndex = 0;
      }
    } catch (err) {
      console.warn("Falha ao carregar modelos:", err);
    }
  }

  // Start new battle
  async function startBattle(event) {
    if (event) event.preventDefault();
    initAudio();

    const humanDeck = el("human-deck").value;
    const modelId = el("ai-model").value;
    const speed = parseFloat(el("speed").value) || 1.0;

    const startBtn = el("start-btn");
    startBtn.disabled = true;
    startBtn.textContent = "Preparando Arena...";

    try {
      const res = await fetch("/api/play/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          human_deck: humanDeck,
          model_id: modelId,
          speed: speed
        })
      });

      if (!res.ok) {
        const err = await res.json();
        alert(err.error || "Erro ao iniciar partida.");
        startBtn.disabled = false;
        startBtn.textContent = "⚔️ Iniciar Batalha";
        return;
      }

      const data = await res.json();
      activeSessionId = data.session_id;
      currentState = data.state;
      previousState = data.state;
      lastStateTime = performance.now();

      // Switch views
      el("setup").hidden = true;
      el("arena-area").hidden = false;
      el("end-modal").hidden = true;
      el("badge").textContent = "Em Batalha";
      el("badge").style.borderColor = "var(--bc)";
      el("badge").style.color = "var(--bc)";
      el("live-speed").value = String(speed);

      selectedSlot = null;
      hoveredTile = null;
      ripples = [];

      startPolling();
      if (!animFrameId) animFrameId = requestAnimationFrame(renderLoop);
    } catch (err) {
      console.error(err);
      alert("Falha de conexão com o servidor do jogo.");
    } finally {
      startBtn.disabled = false;
      startBtn.textContent = "⚔️ Iniciar Batalha";
    }
  }

  // State Polling
  function startPolling() {
    stopPolling();
    pollTimer = setInterval(pollState, 140);
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  async function pollState() {
    if (!activeSessionId) return;
    try {
      const res = await fetch("/api/play/state");
      if (!res.ok) return;
      const data = await res.json();
      if (!data.active || !data.state) {
        activeSessionId = null;
        stopPolling();
        return;
      }

      previousState = currentState || data.state;
      currentState = data.state;
      lastStateTime = performance.now();

      updateHud(currentState);

      if (currentState.status === "finished") {
        stopPolling();
        showEndGame(currentState);
      }
    } catch (err) {
      console.warn("Erro ao buscar estado:", err);
    }
  }

  // HUD and UI Updates
  function updateHud(state) {
    if (!state) return;

    // Clock and phase
    el("clock").textContent = state.remaining || state.clock;
    el("clock-phase").textContent = state.overtime ? "⚡ Morte Súbita! (Elixir 2×)" : "Tempo Normal";
    el("clock-phase").style.color = state.overtime ? "var(--warning)" : "var(--muted)";

    // Opening overlay (first ~4.5 seconds before deploy becomes legal)
    const isOpening = state.ms < 4500;
    el("opening-overlay").hidden = !isOpening;

    // Human side
    const human = state.human;
    el("human-name").textContent = `${human.deck_label || "Você"}`;
    el("human-crowns").textContent = `${human.crowns} 👑`;
    el("human-towers").textContent = `Torres: ${human.tower_hp.map(v => Math.max(0, v)).join(" · ")} PV`;

    // AI side
    const ai = state.ai;
    el("ai-name").textContent = `${ai.model_label || "IA"}`;
    el("ai-crowns").textContent = `👑 ${ai.crowns}`;
    el("ai-towers").textContent = `Torres: ${ai.tower_hp.map(v => Math.max(0, v)).join(" · ")} PV`;

    // Elixir bar
    const elixir = Math.min(10, Math.max(0, human.elixir));
    el("elixir-val").textContent = `${elixir.toFixed(1)} / 10 💧`;
    el("elixir-fill").style.width = `${(elixir / 10) * 100}%`;

    // Render Cards in Hand
    renderHand(human.hand, elixir);

    // Render Next Card
    const next = human.next_card;
    if (next && next.name !== "—") {
      el("next-card").innerHTML = `
        <span class="cost-drop">${next.cost} 💧</span>
        <span class="card-name">${next.pt_name || next.name}</span>
      `;
    } else {
      el("next-card").textContent = "—";
    }

    // Moves Feed
    renderMoves(state.moves || []);
  }

  function renderHand(hand, currentElixir) {
    const container = el("hand");
    container.innerHTML = "";

    hand.forEach((card, slot) => {
      const cardEl = document.createElement("button");
      cardEl.type = "button";
      cardEl.className = "battle-card";
      const isSelected = (selectedSlot === slot);
      const canAfford = card.can_afford && (currentElixir >= card.cost);

      if (isSelected) cardEl.classList.add("selected");
      if (!canAfford) cardEl.classList.add("disabled");

      cardEl.innerHTML = `
        <div class="card-top">
          <span class="card-hotkey">[${slot + 1}]</span>
          <span class="card-cost">${card.cost} 💧</span>
        </div>
        <div class="card-title">${card.pt_name || card.name}</div>
        <div class="card-status-pill">${isSelected ? "Selecionada" : (canAfford ? "Pronta" : "Falta Elixir")}</div>
      `;

      cardEl.addEventListener("click", () => {
        if (!canAfford) {
          playErrorSound();
          return;
        }
        if (selectedSlot === slot) {
          selectedSlot = null; // deselect
        } else {
          selectedSlot = slot;
          playSelectSound();
        }
        renderHand(hand, currentElixir);
      });

      container.appendChild(cardEl);
    });
  }

  function renderMoves(moves) {
    const list = el("moves-list");
    if (!moves.length) {
      list.innerHTML = "<li>Aguardando primeira jogada...</li>";
      return;
    }

    const items = moves.slice(-6).map(m => {
      const isHuman = (m.seat === 0);
      const who = isHuman ? "Você" : "IA";
      const teamClass = isHuman ? "human-move" : "ai-move";
      const timeStr = `${Math.floor(m.ms / 60000)}:${String(Math.floor((m.ms % 60000) / 1000)).padStart(2, "0")}`;
      let detail = m.kind === "card"
        ? `${m.pt_card || m.card} (${Math.floor(m.x)}, ${Math.floor(m.y)})`
        : (m.kind === "ability" ? "Habilidade ativada" : "Esperar");

      return `<li class="${teamClass}"><span><strong>${who}:</strong> ${detail}</span><time>${timeStr}</time></li>`;
    });

    list.innerHTML = items.join("");
  }

  // End Game Modal
  function showEndGame(state) {
    el("end-modal").hidden = false;
    const isWin = (state.result === "win");
    const isLoss = (state.result === "loss");

    playFanfareSound(isWin);

    el("result-title").textContent = isWin ? "🏆 VITÓRIA!" : (isLoss ? "💀 DERROTA!" : "🤝 EMPATE!");
    el("result-title").style.color = isWin ? "var(--bc)" : (isLoss ? "#ff7c82" : "var(--warning)");
    el("result-detail").textContent = isWin
      ? "Parabéns! Sua estratégia superou a rede neural da IA!"
      : (isLoss ? "A IA prevaleceu desta vez. Ajuste sua tática e tente novamente!" : "Uma batalha equilibrada até o último segundo!");

    el("result-stats").innerHTML = `
      <div class="end-stat-box"><span>Suas Coroas</span><strong>${state.human.crowns} 👑</strong></div>
      <div class="end-stat-box"><span>Coroas da IA</span><strong>${state.ai.crowns} 👑</strong></div>
      <div class="end-stat-box"><span>Duração</span><strong>${state.clock}</strong></div>
    `;

    el("badge").textContent = isWin ? "Vitória!" : (isLoss ? "Derrota" : "Empate");
    el("badge").style.borderColor = isWin ? "var(--bc)" : "#ff7c82";
    el("badge").style.color = isWin ? "var(--bc)" : "#ff7c82";
  }

  // Canvas Drawing & Animation Loop
  function renderLoop(now) {
    animFrameId = requestAnimationFrame(renderLoop);
    if (!currentState || el("arena-area").hidden) return;

    drawBoard(now);
  }

  function drawBoard(now) {
    if (!board) return;
    const { width, height } = (currentState.arena || { width: 18, height: 32 });
    const viewWidth = board.clientWidth;
    const viewHeight = board.clientHeight;
    const pixelRatio = window.devicePixelRatio || 1;

    if (!viewWidth || !viewHeight) return;

    if (board.width !== Math.round(viewWidth * pixelRatio) || board.height !== Math.round(viewHeight * pixelRatio)) {
      board.width = Math.round(viewWidth * pixelRatio);
      board.height = Math.round(viewHeight * pixelRatio);
    }

    const ctx = board.getContext("2d");
    ctx.setTransform(board.width / width, 0, 0, board.height / height, 0, 0);

    // 1. Arena Background
    // Bottom (Human side)
    ctx.fillStyle = "#2d493a";
    ctx.fillRect(0, height / 2, width, height / 2);

    // Top (AI side)
    ctx.fillStyle = "#324e41";
    ctx.fillRect(0, 0, width, height / 2);

    // Grid lines
    ctx.fillStyle = "#abc8a510";
    for (let x = 1; x < width; x++) ctx.fillRect(x, 0, 0.045, height);
    for (let y = 1; y < height; y++) ctx.fillRect(0, y, width, 0.045);

    // River
    ctx.fillStyle = "#244756";
    ctx.fillRect(0, height / 2 - 1, width, 2);

    // Bridges
    const bridges = [2.5, width - 5.5];
    ctx.fillStyle = "#b39464";
    for (const left of bridges) {
      ctx.fillRect(left, height / 2 - 1.15, 3, 2.3);
      ctx.fillStyle = "#705942";
      for (let row = -1; row <= 1; row += 0.4) {
        ctx.fillRect(left, height / 2 + row, 3, 0.06);
      }
      ctx.fillStyle = "#b39464";
    }

    // River banks lines
    ctx.strokeStyle = "#466c75";
    ctx.lineWidth = 0.08;
    ctx.beginPath();
    ctx.moveTo(0, height / 2 - 1);
    ctx.lineTo(width, height / 2 - 1);
    ctx.moveTo(0, height / 2 + 1);
    ctx.lineTo(width, height / 2 + 1);
    ctx.stroke();

    // 2. Legal Placement Overlay when a card is selected
    if (selectedSlot !== null && currentState.human && currentState.human.hand) {
      const card = currentState.human.hand[selectedSlot];
      if (card && card.legal_tiles) {
        ctx.fillStyle = "#36e39818";
        for (const [lx, ly] of card.legal_tiles) {
          const sy = height - 1 - ly;
          ctx.fillRect(lx + 0.05, sy + 0.05, 0.9, 0.9);
        }
      }
    }

    // 3. Interpolate and Draw Pieces (troops, buildings, towers)
    const prevMap = new Map((previousState?.pieces || []).map(p => [p[0], p]));
    const dt = Math.max(1, Math.min(500, now - lastStateTime));
    const ratio = Math.min(1.0, dt / 500);

    const pieces = (currentState.pieces || []).map(p => {
      const prev = prevMap.get(p[0]);
      if (!prev) return p;
      const copy = p.slice();
      copy[3] = prev[3] + (p[3] - prev[3]) * ratio;
      copy[4] = prev[4] + (p[4] - prev[4]) * ratio;
      return copy;
    });

    const positions = new Map(pieces.map(p => [p[0], p]));

    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (const [, seat, kind, x, y, hp, maxHp, target, unit] of pieces) {
      const screenY = height - y;
      const tower = kind >= 2;
      const isKing = kind === 2;
      const radius = isKing ? 1.25 : (tower ? 1.05 : (kind === 1 ? 0.72 : 0.42));

      // X-Bow targeting laser beam
      if (/x.?bow/i.test(unit) && positions.has(target)) {
        const enemy = positions.get(target);
        ctx.beginPath();
        ctx.moveTo(x, screenY);
        ctx.lineTo(enemy[3], height - enemy[4]);
        ctx.strokeStyle = tones[seat];
        ctx.lineWidth = 0.12;
        ctx.stroke();
      }

      // Base unit circle / square
      ctx.fillStyle = tones[seat];
      ctx.strokeStyle = "#101920";
      ctx.lineWidth = 0.14;

      ctx.beginPath();
      if (kind >= 1) {
        ctx.rect(x - radius, screenY - radius, radius * 2, radius * 2);
      } else {
        ctx.arc(x, screenY, radius, 0, Math.PI * 2);
      }
      ctx.fill();
      ctx.stroke();

      // Unit Label / Initials
      ctx.fillStyle = "#111b24";
      ctx.font = `bold ${tower ? 0.58 : 0.4}px system-ui`;
      let text = "";
      if (isKing) {
        text = "👑 REI";
      } else if (kind === 3) {
        text = "TORRE";
      } else {
        text = unit.split(/[ -]/).filter(Boolean).map(w => w[0]).join("").slice(0, 2).toUpperCase();
      }
      ctx.fillText(text, x, screenY);

      // Hitpoints Bar
      const barW = radius * 2.15;
      const barH = 0.18;
      const barY = screenY - radius - 0.32;
      ctx.fillStyle = "#121b22";
      ctx.fillRect(x - barW / 2, barY, barW, barH);
      ctx.fillStyle = tones[seat];
      const fillW = barW * Math.min(1.0, Math.max(0, hp / Math.max(1, maxHp)));
      ctx.fillRect(x - barW / 2, barY, fillW, barH);
    }

    // 4. Placement Ripples
    for (let i = ripples.length - 1; i >= 0; i--) {
      const rip = ripples[i];
      rip.radius += 0.12;
      rip.alpha -= 0.04;
      if (rip.alpha <= 0) {
        ripples.splice(i, 1);
        continue;
      }
      ctx.strokeStyle = `rgba(139, 206, 255, ${rip.alpha})`;
      ctx.lineWidth = 0.12;
      ctx.beginPath();
      ctx.arc(rip.x, height - rip.y, rip.radius, 0, Math.PI * 2);
      ctx.stroke();
    }

    // 5. Hover Reticle when card is selected
    if (selectedSlot !== null && hoveredTile && currentState.human) {
      const card = currentState.human.hand[selectedSlot];
      const isLegal = hoveredTile.legal;
      const hx = hoveredTile.x + 0.5;
      const hy = height - (hoveredTile.y + 0.5);

      ctx.strokeStyle = isLegal ? "#36e398" : "#ff6b72";
      ctx.fillStyle = isLegal ? "#36e39833" : "#ff6b7233";
      ctx.lineWidth = 0.12;

      ctx.beginPath();
      ctx.arc(hx, hy, 0.95, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Label preview
      ctx.fillStyle = "#101720f0";
      ctx.fillRect(hx - 2.5, hy - 1.8, 5.0, 0.9);
      ctx.strokeStyle = isLegal ? "#36e398" : "#ff6b72";
      ctx.strokeRect(hx - 2.5, hy - 1.8, 5.0, 0.9);

      ctx.fillStyle = isLegal ? "#e0ffea" : "#ffd4d6";
      ctx.font = "bold 0.42px system-ui";
      const reticleText = isLegal
        ? `${card.pt_name || card.name} (${card.cost} 💧)`
        : "⊘ Área Inválida";
      ctx.fillText(reticleText, hx, hy - 1.35);
    }
  }

  // Mouse & Touch Interaction on Canvas
  function getCanvasTile(evt) {
    if (!board) return null;
    const rect = board.getBoundingClientRect();
    const mouseX = evt.clientX - rect.left;
    const mouseY = evt.clientY - rect.top;

    const tileX = Math.floor((mouseX / rect.width) * 18);
    const tileY = Math.floor((1 - (mouseY / rect.height)) * 32);

    const clampedX = Math.max(0, Math.min(17, tileX));
    const clampedY = Math.max(0, Math.min(31, tileY));

    let legal = false;
    if (selectedSlot !== null && currentState && currentState.human) {
      const card = currentState.human.hand[selectedSlot];
      if (card && card.legal_tiles) {
        legal = card.legal_tiles.some(([lx, ly]) => lx === clampedX && ly === clampedY);
      }
    }

    return { x: clampedX, y: clampedY, legal };
  }

  board.addEventListener("mousemove", evt => {
    if (selectedSlot === null) {
      hoveredTile = null;
      return;
    }
    hoveredTile = getCanvasTile(evt);
  });

  board.addEventListener("mouseleave", () => {
    hoveredTile = null;
  });

  board.addEventListener("click", async evt => {
    if (selectedSlot === null || !currentState) return;

    const tile = getCanvasTile(evt);
    if (!tile) return;

    const card = currentState.human.hand[selectedSlot];
    if (!card) return;

    if (!tile.legal) {
      playErrorSound();
      return;
    }

    // Play local deployment effect and sound
    playDeploySound();
    ripples.push({ x: tile.x + 0.5, y: tile.y + 0.5, radius: 0.3, alpha: 0.9 });

    const slotToPlay = selectedSlot;
    selectedSlot = null; // deselect immediately for responsiveness

    try {
      const res = await fetch("/api/play/act", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          slot: slotToPlay,
          x: tile.x,
          y: tile.y
        })
      });

      const data = await res.json();
      if (!data.ok) {
        console.warn("Deploy recusado:", data.error);
        playErrorSound();
      }
    } catch (err) {
      console.error("Falha ao enviar jogada:", err);
    }
  });

  // Action Buttons
  el("pause-btn").addEventListener("click", async () => {
    if (!activeSessionId) return;
    try {
      const res = await fetch("/api/play/pause", { method: "POST" });
      const data = await res.json();
      el("pause-btn").textContent = data.paused ? "▶️ Retomar" : "⏸️ Pausar";
    } catch (err) {
      console.warn(err);
    }
  });

  el("live-speed").addEventListener("change", async evt => {
    if (!activeSessionId) return;
    try {
      await fetch("/api/play/speed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speed: parseFloat(evt.target.value) })
      });
    } catch (err) {
      console.warn(err);
    }
  });

  el("sound-btn").addEventListener("click", () => {
    soundEnabled = !soundEnabled;
    el("sound-btn").textContent = soundEnabled ? "🔊 Som" : "🔇 Mudo";
    if (soundEnabled) playSelectSound();
  });

  el("surrender-btn").addEventListener("click", async () => {
    if (!confirm("Deseja realmente sair da partida?")) return;
    stopPolling();
    try {
      await fetch("/api/play/stop", { method: "POST" });
    } catch (_) {}
    activeSessionId = null;
    el("arena-area").hidden = true;
    el("setup").hidden = false;
    el("badge").textContent = "Pronto para jogar";
    el("badge").style.borderColor = "var(--line)";
    el("badge").style.color = "var(--muted)";
  });

  el("rematch-btn").addEventListener("click", () => {
    el("end-modal").hidden = true;
    startBattle();
  });

  el("new-match-btn").addEventListener("click", () => {
    el("end-modal").hidden = true;
    el("arena-area").hidden = true;
    el("setup").hidden = false;
    el("badge").textContent = "Pronto para jogar";
    el("badge").style.borderColor = "var(--line)";
    el("badge").style.color = "var(--muted)";
  });

  // Keyboard Shortcuts
  window.addEventListener("keydown", evt => {
    if (el("arena-area").hidden || !currentState) return;

    // Keys 1, 2, 3, 4 select card
    if (["1", "2", "3", "4"].includes(evt.key)) {
      const slot = parseInt(evt.key, 10) - 1;
      const hand = currentState.human?.hand;
      if (hand && hand[slot]) {
        const card = hand[slot];
        if (card.can_afford) {
          selectedSlot = (selectedSlot === slot ? null : slot);
          playSelectSound();
          renderHand(hand, currentState.human.elixir);
        } else {
          playErrorSound();
        }
      }
    } else if (evt.code === "Space") {
      evt.preventDefault();
      el("pause-btn").click();
    } else if (evt.key === "Escape") {
      selectedSlot = null;
      if (currentState.human) {
        renderHand(currentState.human.hand, currentState.human.elixir);
      }
    }
  });

  // Form submission
  el("form").addEventListener("submit", startBattle);

  // Initialize
  loadModels();
})();
