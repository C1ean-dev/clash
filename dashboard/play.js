"use strict";

(() => {
  const el = id => document.getElementById("play-" + id);
  const board = el("board");
  const {colors, reducedMotion, skin, portraitPosition, spellName, spellCast, castEnds, castVisible,
    createRenderer} = window.battleArt;
  const {ctx, begin, ring, stroke, drawPieces, paintSpellArea, drawCastArea, drawCast} = createRenderer(board);
  const clock = s => Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");

  let state, selected = null, cursor = null, drag = null, pending = null;
  let legal = [], movesSeen = new Set(), effects = [], spellEffects = [];
  let pollTimer = 0, frame = 0, lastDraw = 0, failures = 0;
  let audio, sound = true, pausing = false, feedbackTimer = 0;
  let motionStart = 0, motionDuration = 1, motionOrigins = new Map(), spellFrameMs = 0;
  const opening = () => state.opening_ms ?? Math.max(0, 4500 - state.ms);
  const boardKey = (x, y, width = state.arena.width) => y * width + x;

  function text(node, value) {
    if (node.textContent !== value) node.textContent = value;
  }

  function hint() {
    if (state?.status === "finished") return "";
    if (state?.status === "paused") return "Retome a partida para jogar.";
    if (pending) return "Posicionando " + pending.name + "…";
    if (selected !== null) {
      const card = state.human.hand[selected];
      const tip = {Log: "aponte para ver a faixa do rolamento.", Fireball: "aponte para ver a área de impacto."};
      return card.pt_name + " · " + (tip[card.name] || "escolha uma casa destacada.");
    }
    return "Arraste uma carta ou selecione com 1–4 e escolha a casa.";
  }

  function message(value, error = false) {
    clearTimeout(feedbackTimer);
    const node = el("feedback");
    node.toggleAttribute("data-error", error);
    text(node, value);
    if (error) feedbackTimer = setTimeout(() => message(hint()), 3200);
  }

  function initAudio() {
    const Audio = window.AudioContext || window.webkitAudioContext;
    if (sound && !audio && Audio) audio = new Audio();
    if (sound && audio?.state === "suspended") audio.resume().catch(() => {});
  }

  function tone(from, to, duration) {
    if (!sound || audio?.state !== "running") return;
    const oscillator = audio.createOscillator();
    const gain = audio.createGain();
    const now = audio.currentTime;
    oscillator.type = "triangle";
    oscillator.frequency.setValueAtTime(from, now);
    oscillator.frequency.exponentialRampToValueAtTime(to, now + duration);
    gain.gain.setValueAtTime(.07, now);
    gain.gain.exponentialRampToValueAtTime(.001, now + duration);
    oscillator.connect(gain).connect(audio.destination);
    oscillator.start(now);
    oscillator.stop(now + duration);
  }

  async function api(route, payload) {
    const options = payload ? {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)
    } : {cache: "no-store"};
    const response = await fetch("/api/play/" + route, options);
    const data = await response.json();
    if (!response.ok || data.ok === false) throw new Error(data.error || "Falha na comunicação com a arena.");
    return data;
  }

  async function loadModels() {
    try {
      const {models} = await api("models");
      const groups = Object.fromEntries(["checkpoint", "scripted"].map(kind => [kind, document.createElement("optgroup")]));
      groups.checkpoint.label = "Modelos treinados";
      groups.scripted.label = "Bots táticos";
      for (const model of models) groups[model.kind]?.append(new Option(model.label, model.id));
      const options = Object.values(groups).filter(group => group.children.length);
      if (!options.length) return;
      el("ai-model").replaceChildren(...options);
      const preferred = models.find(model => model.id === "ckpt:tactical/xbow_30/ppo.pt");
      if (preferred) el("ai-model").value = preferred.id;
    } catch {
      text(el("setup-message"), "Lista de modelos indisponível. O bot tático continua na seleção.");
    }
  }

  async function start(event) {
    event?.preventDefault();
    const button = el("start-btn");
    if (button.disabled) return;
    initAudio();
    button.disabled = el("rematch-btn").disabled = true;
    text(button, "Preparando…");
    text(el("setup-message"), "");
    el("setup-message").removeAttribute("data-error");
    try {
      const data = await api("start", {
        human_deck: el("human-deck").value, model_id: el("ai-model").value,
        speed: Number(el(state ? "live-speed" : "speed").value)
      });
      clearTimeout(pollTimer);
      state = selected = cursor = drag = pending = null;
      movesSeen = new Set();
      effects = [];
      spellEffects = [];
      failures = 0;
      el("setup").hidden = el("end-modal").hidden = true;
      el("arena-area").hidden = false;
      el("live-speed").value = String(data.state.speed);
      update(data.state, true);
      message(hint());
      el("stage").scrollIntoView({block: "start"});
      board.focus({preventScroll: true});
      if (state.status !== "finished") pollTimer = setTimeout(() => poll(data.state.id), 90);
      if (!frame) frame = requestAnimationFrame(render);
    } catch (error) {
      const node = el(state ? "result-detail" : "setup-message");
      node.setAttribute("data-error", "");
      text(node, error.message);
    } finally {
      button.disabled = el("rematch-btn").disabled = false;
      text(button, "Iniciar batalha");
    }
  }

  async function poll(id) {
    if (state?.id !== id || state.status === "finished") return;
    try {
      const data = await api("state");
      if (state?.id !== id) return;
      if (!data.active || data.state.id !== id) {
        update({...state, status: "finished", result: "interrupted"});
        return;
      }
      if (failures >= 2) message(hint());
      failures = 0;
      update(data.state);
    } catch {
      if (state?.id === id && ++failures === 2) message("Reconectando à partida…", true);
    } finally {
      if (state?.id === id && state.status !== "finished") {
        pollTimer = setTimeout(() => poll(id), failures ? 500 : 90);
      }
    }
  }

  function update(next, initial = false) {
    const before = state;
    if (before && next.seq < before.seq) return;
    const now = performance.now();
    if (!before || before.ms !== next.ms) {
      motionOrigins = new Map((before?.pieces || []).map(piece => [piece[0], position(piece, now)]));
      motionDuration = Math.min(360, Math.max(80, (next.ms - (before?.ms ?? next.ms)) / next.speed));
      motionStart = now;
      spellFrameMs = Math.min(500, Math.max(0, next.ms - (before?.ms ?? next.ms)));
      legal = next.human.hand.map(card =>
        new Set(card.legal_tiles.map(([x, y]) => boardKey(x, y, next.arena.width))));
    }
    state = next;
    if (selected !== null && before?.human.hand[selected].id !== next.human.hand[selected].id) {
      selected = cursor = null;
      message(hint());
    }
    const moveKey = move => move.id ?? move.seat + ":" + move.ms;
    const newMoves = next.moves.filter(move => !movesSeen.has(moveKey(move)));
    movesSeen = new Set(next.moves.map(moveKey));
    if (!initial) for (const move of newMoves) {
      if (move.kind !== "card") continue;
      const cast = spellCast(move, state);
      if (cast) spellEffects.push(cast);
      else effects.push({...move, started: now});
      if (!move.seat) tone(265, 85, .2);
    }
    if (newMoves.length || initial) showMoves(next.moves);
    if (pending) {
      const played = newMoves.some(move => move.seat === 0 && move.slot === pending.slot && move.ms >= pending.ms);
      const refused = next.command?.id === pending.commandId && next.command?.status === "refused";
      if (played || refused || next.status === "finished" || now - pending.started > 1800) {
        pending = null;
        message(refused ? "A posição deixou de estar disponível." : hint(), refused);
      }
    }
    hud();
    if (next.status === "finished" && before?.status !== "finished") showResult();
  }

  function makeToken(node, nameVisible) {
    const parts = ["card-cost", "card-emblem", "card-title"].map(className =>
      Object.assign(document.createElement("span"), {className}));
    node.append(...parts.slice(0, nameVisible ? 3 : 2));
    return {node, cost: parts[0], emblem: parts[1], name: parts[2], id: null};
  }

  const nextToken = makeToken(el("next-card"), false);
  const tokens = Array.from({length: 4}, (_, slot) => {
    const node = Object.assign(document.createElement("button"), {type: "button", className: "battle-card"});
    node.setAttribute("aria-keyshortcuts", String(slot + 1));
    const token = makeToken(node, true);
    const key = Object.assign(document.createElement("span"), {className: "card-hotkey", textContent: slot + 1});
    key.setAttribute("aria-hidden", "true");
    node.append(key);
    el("hand").append(node);
    node.addEventListener("pointerdown", event => beginDrag(event, slot));
    node.addEventListener("pointermove", moveDrag);
    node.addEventListener("pointerup", endDrag);
    node.addEventListener("pointercancel", cancelDrag);
    node.addEventListener("lostpointercapture", cancelDrag);
    node.addEventListener("click", event => { if (event.detail === 0) choose(slot, true); });
    return token;
  });

  function updateCard(token, card) {
    const name = card.pt_name || card.name;
    if (token.id !== card.id || token.name.textContent !== name) {
      const [glyph, tint, portrait] = skin(card.name);
      token.node.toggleAttribute("data-spent", token.id !== null && !token.node.hasAttribute("data-spent"));
      token.id = card.id;
      token.node.style.setProperty("--card-tone", tint);
      text(token.name, name);
      text(token.emblem, glyph);
      token.emblem.toggleAttribute("data-portrait", portrait !== undefined);
      if (portrait !== undefined) token.emblem.style.backgroundPosition = portraitPosition(portrait);
    }
    text(token.cost, card.cost < 0 ? "–" : String(card.cost));
    const title = name + " · " + token.cost.textContent + " de elixir";
    if (token.node.title !== title) {
      token.node.title = title;
      token.node.setAttribute("aria-label", title);
    }
  }

  function selection() {
    tokens.forEach((token, slot) => {
      token.node.setAttribute("aria-pressed", String(slot === selected));
      token.node.setAttribute("aria-disabled", String(state.status !== "running" || !!pending || !legal[slot]?.size));
    });
    board.toggleAttribute("data-aiming", selected !== null);
  }

  function hud() {
    const {human, ai, status} = state;
    const rate = state.elixir_rate ?? (state.overtime || state.ms >= 120000 ? 2 : 1);
    const remaining = state.remaining_ms == null && state.overtime
      ? clock(Math.max(0, 300 - Math.floor(state.ms / 1000))) : state.remaining;
    text(el("clock"), remaining || state.clock);
    text(el("clock-phase"), [state.overtime && "Prorrogação", rate > 1 && rate + "× elixir"].filter(Boolean).join(" · "));
    text(el("human-name"), human.deck_label.split(" (")[0]);
    text(el("ai-name"), ai.model_label);
    text(el("human-crowns"), String(human.crowns));
    text(el("ai-crowns"), String(ai.crowns));
    text(el("badge"), status === "paused" ? "Pausada" : status === "finished" ? "Encerrada" : "Em batalha");
    text(el("pause-btn"), status === "paused" ? "Retomar" : "Pausar");
    el("pause-btn").disabled = pausing || status === "finished";
    el("live-speed").disabled = status === "finished";
    text(el("elixir-val"), human.elixir.toFixed(1).replace(".", ",") + " / 10");
    el("elixir-fill").style.transform = "scaleX(" + Math.min(1, human.elixir / 10) + ")";
    human.hand.forEach((card, slot) => updateCard(tokens[slot], card));
    updateCard(nextToken, human.next_card);
    selection();
    el("opening-overlay").hidden = status === "finished" || (!opening() && status !== "paused");
    text(el("opening-overlay"), status === "paused" ? "Partida pausada" : "Abertura · " + Math.ceil(opening() / 1000) + " s");
  }

  function showMoves(moves) {
    const rows = moves.slice(-5).reverse().map(move => {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.className = move.seat ? "ai-move" : "human-move";
      name.textContent = (move.seat ? "IA" : "Você") + " · " + (move.pt_card || move.card);
      const time = document.createElement("time");
      time.textContent = clock(Math.floor(move.ms / 1000));
      li.append(name, time);
      return li;
    });
    if (!rows.length) rows.push(Object.assign(document.createElement("li"), {textContent: "Aguardando a primeira jogada."}));
    el("moves-list").replaceChildren(...rows);
  }

  function showResult() {
    const names = {win: "Vitória", loss: "Derrota", draw: "Empate", interrupted: "Partida interrompida"};
    text(el("result-title"), names[state.result]);
    text(el("result-detail"), "Você " + state.human.crowns + " × " + state.ai.crowns + " IA · " + state.clock);
    el("end-modal").hidden = false;
    el("rematch-btn").focus({preventScroll: true});
    if (state.result !== "interrupted") tone(400, state.result === "win" ? 710 : 210, .32);
  }

  function choose(slot, toggle = false) {
    if (!state || state.status === "finished") return false;
    const card = state.human.hand[slot];
    if (state.status !== "running" || pending || !legal[slot]?.size) {
      const reasons = [
        [state.status === "paused", "Retome a partida para jogar."],
        [pending, "Aguardando a jogada anterior."],
        [opening(), "Aguarde o fim da abertura."],
        [!card.can_afford && card.cost >= 0, "Você precisa de " + card.cost + " de elixir."]
      ];
      message(reasons.find(([condition]) => condition)?.[1] || "Essa carta está indisponível.", true);
      return false;
    }
    initAudio();
    if (selected !== slot) tone(390, 540, .08);
    selected = toggle && selected === slot ? null : slot;
    selection();
    message(hint());
    return true;
  }

  function tileAt(event) {
    const rect = board.getBoundingClientRect();
    const {width, height} = state.arena;
    const x = Math.floor((event.clientX - rect.left) * width / rect.width);
    const row = Math.floor((event.clientY - rect.top) * height / rect.height);
    return x >= 0 && x < width && row >= 0 && row < height ? {x, y: height - 1 - row} : null;
  }

  const isLegal = tile => tile && selected !== null && legal[selected]?.has(boardKey(tile.x, tile.y));

  async function deploy(tile) {
    if (!state || selected === null || pending || state.status !== "running") return;
    if (!isLegal(tile)) {
      message("Escolha uma casa destacada na arena.", true);
      return;
    }
    const card = state.human.hand[selected];
    const action = {...tile, slot: selected, ms: state.ms, started: performance.now(),
      name: card.pt_name, card: card.name};
    pending = action;
    selected = cursor = null;
    selection();
    message(hint());
    try {
      const result = await api("act", {slot: action.slot, ...tile, card_id: card.id, session_id: state.id});
      action.commandId = result.command_id;
    } catch (error) {
      if (pending !== action) return;
      pending = null;
      if (state.human.hand[action.slot].id === card.id) selected = action.slot;
      selection();
      message(error.message, true);
    }
  }

  function beginDrag(event, slot) {
    if (event.button !== 0 || !event.isPrimary) return;
    const same = selected === slot;
    if (!choose(slot)) return;
    drag = {id: event.pointerId, x: event.clientX, y: event.clientY, same, moved: false};
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function moveDrag(event) {
    if (drag?.id !== event.pointerId) return;
    if (Math.hypot(event.clientX - drag.x, event.clientY - drag.y) > 7) drag.moved = true;
    if (drag.moved) cursor = tileAt(event);
  }

  function endDrag(event) {
    if (drag?.id !== event.pointerId) return;
    const {moved, same} = drag;
    drag = null;
    if (moved) {
      const tile = tileAt(event);
      if (tile) deploy(tile);
    } else if (same && selected !== null) choose(selected, true);
  }

  function cancelDrag(event) {
    if (drag?.id === event.pointerId) drag = cursor = null;
  }

  board.addEventListener("pointermove", event => { if (state) cursor = tileAt(event); });
  board.addEventListener("pointerleave", () => { if (!drag) cursor = null; });
  board.addEventListener("click", event => { if (state) deploy(tileAt(event)); });
  board.addEventListener("keydown", event => {
    if (selected === null) return;
    if (event.key === "Enter" && cursor) deploy(cursor);
    const delta = {ArrowUp: [0, 1], ArrowDown: [0, -1], ArrowLeft: [-1, 0], ArrowRight: [1, 0]}[event.key];
    if (!delta) return;
    event.preventDefault();
    const {width, height} = state.arena;
    const tile = cursor || {x: Math.floor(width / 2), y: Math.floor(height / 3)};
    cursor = {x: Math.max(0, Math.min(width - 1, tile.x + delta[0])),
      y: Math.max(0, Math.min(height - 1, tile.y + delta[1]))};
  });

  window.addEventListener("keydown", event => {
    if (!state || state.status === "finished" || event.repeat || event.ctrlKey || event.metaKey ||
        event.target.closest?.("input, select, textarea, [contenteditable]")) return;
    if (/^[1-4]$/.test(event.key)) {
      event.preventDefault();
      if (choose(Number(event.key) - 1, true)) board.focus({preventScroll: true});
    } else if (event.code === "Space" && !event.target.closest?.("button")) {
      event.preventDefault();
      el("pause-btn").click();
    } else if (event.key === "Escape") {
      selected = cursor = drag = null;
      selection();
      message(hint());
    }
  });

  el("pause-btn").addEventListener("click", async () => {
    if (!state || pausing) return;
    const id = state.id;
    pausing = el("pause-btn").disabled = true;
    try {
      const {paused} = await api("pause", {session_id: id, paused: state.status !== "paused"});
      if (state?.id === id && state.status !== "finished") {
        update({...state, status: paused ? "paused" : "running"});
        message(hint());
      }
    } catch (error) {
      message(error.message, true);
    } finally {
      pausing = false;
      el("pause-btn").disabled = state?.status === "finished";
    }
  });

  el("live-speed").addEventListener("change", async event => {
    if (!state) return;
    try {
      await api("speed", {session_id: state.id, speed: Number(event.target.value)});
    } catch (error) {
      if (state) event.target.value = String(state.speed);
      message(error.message, true);
    }
  });

  el("sound-btn").addEventListener("click", () => {
    sound = !sound;
    el("sound-btn").setAttribute("aria-pressed", String(sound));
    text(el("sound-btn"), sound ? "Som" : "Sem som");
    initAudio();
    tone(390, 510, .08);
  });

  if (document.fullscreenEnabled) {
    el("fullscreen-btn").hidden = false;
    el("fullscreen-btn").addEventListener("click", async () => {
      try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await el("arena-area").requestFullscreen();
      } catch {
        message("Tela cheia indisponível neste navegador.", true);
      }
    });
    document.addEventListener("fullscreenchange", () => {
      text(el("fullscreen-btn"), document.fullscreenElement ? "Sair da tela cheia" : "Tela cheia");
    });
  }

  function backToSetup() {
    state = null;
    clearTimeout(pollTimer);
    cancelAnimationFrame(frame);
    frame = 0;
    if (document.fullscreenElement === el("arena-area")) document.exitFullscreen().catch(() => {});
    el("arena-area").hidden = true;
    el("setup").hidden = false;
    text(el("badge"), "Pronto");
    el("start-btn").focus();
  }

  el("surrender-btn").addEventListener("click", async () => {
    if (!state) return;
    if (state.status === "finished") { backToSetup(); return; }
    const button = el("surrender-btn");
    button.disabled = true;
    try {
      await api("stop", {session_id: state.id});
      backToSetup();
    } catch (error) {
      message(error.message, true);
    } finally {
      button.disabled = false;
    }
  });
  el("rematch-btn").addEventListener("click", start);
  el("new-match-btn").addEventListener("click", backToSetup);
  el("form").addEventListener("submit", start);
  loadModels();

  function position(piece, now) {
    const from = motionOrigins.get(piece[0]);
    if (!from || reducedMotion) return [piece[3], piece[4]];
    const progress = Math.min(1, Math.max(0, (now - motionStart) / motionDuration));
    return [from[0] + progress * (piece[3] - from[0]), from[1] + progress * (piece[4] - from[1])];
  }

  function spellClock(now) {
    if (state.status !== "running") return state.ms;
    const elapsed = (now - motionStart) * state.speed;
    return state.ms - spellFrameMs + Math.min(spellFrameMs, Math.max(0, elapsed));
  }

  function draw(now) {
    const {width, height} = state.arena;
    if (!begin(state.arena)) return;
    const aimedCard = pending?.card ?? (selected === null ? null : state.human.hand[selected].name);
    const spellPreview = spellName(aimedCard);
    if (selected !== null && state.status === "running") {
      ctx.fillStyle = spellPreview ? "#a2f2de17" : "#a2f2de44";
      for (const tile of legal[selected]) {
        ctx.fillRect(tile % width + .04, height - 1 - Math.floor(tile / width) + .04, .92, .92);
      }
    }
    const target = pending || (selected !== null && cursor);
    if (target && spellPreview && (pending || isLegal(target))) {
      paintSpellArea(spellPreview, target.x + .5, target.y + .5, height, 1, .95, true);
    }
    const visualMs = spellClock(now);
    spellEffects = spellEffects.filter(cast => visualMs < castEnds(cast));
    const activeCasts = spellEffects.filter(cast => castVisible(cast, visualMs));
    for (const cast of activeCasts) drawCastArea(cast, visualMs, height);
    drawPieces(state.pieces, height, piece => position(piece, now));
    for (const [team, x, y, aimX, aimY, splash] of state.shots || []) {
      const angle = Math.atan2(y - aimY, aimX - x);
      stroke(x - Math.cos(angle) * .42, height - y - Math.sin(angle) * .42,
        x, height - y, colors[team], splash ? .16 : .09);
      ring(x, height - y, splash ? .16 : .10, "#fff1ca", true);
    }
    for (const [team, x, y, , card] of state.spells || []) {
      if (spellName(card)) continue;
      ctx.lineWidth = .13;
      ring(x, height - y, .45, colors[team]);
      ring(x, height - y, .16, "#fff1ca", true);
    }
    for (const cast of activeCasts) drawCast(cast, visualMs, height);
    effects = effects.filter(effect => now - effect.started < 560);
    for (const effect of effects) {
      const ratio = (now - effect.started) / 560;
      ctx.globalAlpha = 1 - ratio;
      ctx.lineWidth = .13;
      const radius = (effect.card_kind === "SPELL" ? .9 : .45) + ratio * (reducedMotion ? 0 : 1.15);
      ring(effect.x, height - effect.y, radius, colors[effect.seat]);
    }
    ctx.globalAlpha = 1;
    if (target) {
      const color = pending ? "#f6d299" : isLegal(target) ? "#edffed" : "#ff9c83";
      ctx.lineWidth = .13;
      ring(target.x + .5, height - target.y - .5, .51, color);
    }
  }

  function render(now) {
    frame = requestAnimationFrame(render);
    if (!state || now - lastDraw < 1000 / 35) return;
    lastDraw = now;
    draw(now);
  }
})();
