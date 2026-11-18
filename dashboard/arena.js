"use strict";

(() => {
  const el = id => document.getElementById(`arena-${id}`);
  const board = el("board");
  const video = el("replay-video");
  const recordings = new Map();
  const tabs = [...document.querySelectorAll("[data-view-stage]")];
  const tones = ["#8bceff", "#ffa7ab"];
  const rivals = { bc: "Imitação", other_bc: "Imitação do outro deck", scripted: "X-Bow programado",
    hog_push: "Corredor", rg_patient: "Gigante Real", random: "Aleatório" };
  const cards = { "X-Bow": "X-Bow", Fireball: "Bola de Fogo", Knight: "Cavaleiro", Archers: "Arqueiras",
    Skeletons: "Esqueletos", "The Log": "O Tronco", "Ice Wizard": "Mago de Gelo",
    "Electro Spirit": "Espírito Elétrico", Rocket: "Foguete", Tornado: "Tornado",
    "Hog Rider": "Corredor", "Royal Giant": "Gigante Real" };
  const replayErrors = { "ability unavailable or ambiguous": "habilidade indisponível ou ambígua",
    "card not in hand": "a carta registrada não estava na mão",
    "play refused by engine": "jogada recusada pelo simulador",
    "simulator ended before replay": "a simulação terminou antes da última jogada",
    "simulator refused replay setup": "configuração do replay recusada pelo simulador" };
  const name = card => cards[card] || card;
  const safe = value => String(value ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const percent = new Intl.NumberFormat("pt-BR", { style: "percent", maximumFractionDigits: 2 });
  const decimal = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 3 });
  let context = { deck: "xbow_30", run: "", active: false, phase: "replays" };
  let stage = "replays";
  let follow = true;
  let pinned = "";
  let generation = 0;
  let clip = null;
  let frames = [];
  let index = 0;
  let playing = false;
  let anchor = 0;
  let position = 0;
  let demo = false;
  let requesting = false;
  let compareOpen = false;
  let videoTag = "";
  let playRequest = null;
  let videoBlocked = false;

  function clock(ms) {
    const seconds = Math.floor(ms / 1000);
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  }

  function rival(name) {
    return rivals[name] || name?.replace(/^snapshot_/, "Versão anterior ") || "Rival";
  }

  function label(header) {
    if (header.replay) return `Replay ${header.replay}`;
    const parts = [];
    if (header.update) parts.push(`Atualização ${header.update}`);
    if (header.model) parts.push(header.model === "bc" ? "Imitação" : "PPO");
    if (header.number) parts.push(`Partida ${header.number}`);
    return [...parts, rival(header.rival)].join(" · ");
  }

  function team(seat) {
    if (stage === "replays" || stage === "bc") return seat === clip.header.seat ? "Jogador do deck" : "Adversário";
    return seat === clip.header.seat ? "IA" : rival(clip.header.rival);
  }

  function actionText(action) {
    if (action.kind === "wait") return "Esperar";
    if (action.kind === "ability") return "Ativar habilidade";
    return `${name(action.card)} · casa ${Math.floor(action.x) + 1}, ${Math.floor(action.y) + 1}`;
  }

  function resultText(result) {
    if (!result) return "";
    switch (result.status) {
      case "accepted": return `Replay aceito · ${result.frames} quadros de decisão extraídos`;
      case "rejected": return `Replay descartado · ${replayErrors[result.reason] || result.reason}`;
      case "win": case "loss": case "draw": {
        const outcomes = { win: "Vitória da IA", loss: "Derrota da IA", draw: "Empate" };
        return `${outcomes[result.status]} · ${result.locks} conexões de X-Bow · recompensa ${decimal.format(result.reward)}`;
      }
      case "interrupted": return "Gravação interrompida";
      default: return "";
    }
  }

  function renderFrame() {
    const frame = frames[index];
    if (!frame) return;
    el("clock").textContent = clock(frame.ms);
    el("seek").max = frames.length - 1;
    el("seek").value = index;
    el("seek").setAttribute("aria-valuetext", frame.comparison?.epoch
      ? `Época ${frame.comparison.epoch}, lote ${frame.comparison.batch}` : clock(frame.ms));
    el("play").textContent = playing ? "Pausar" : (index === frames.length - 1 ? "Rever" : "Reproduzir");
    el("play").hidden = stage === "bc";
    el("speed").hidden = stage === "bc";
    el("teams").innerHTML = frame.sides.map((side, seat) =>
      `<div class="arena-team" style="--team-color:${tones[seat]}">
        <div class="arena-team-head"><strong>${safe(team(seat))}</strong><span>${decimal.format(side.elixir)}/10 elixir</span></div>
        <small>Torres · ${side.hp.map(value => Math.max(0, value)).join(" / ")} PV</small>
        <small>Mão · ${side.hand.map(card => safe(name(card))).join(" · ")}</small>
      </div>`).reverse().join("");
    const comparison = frame.comparison;
    el("decision").hidden = !comparison;
    el("moves-box").hidden = Boolean(comparison);
    if (comparison) {
      const title = comparison.epoch ? `Época ${comparison.epoch} · lote ${comparison.batch}/${comparison.batches}` : "Decisão no replay";
      el("decision").innerHTML = `<h3>${title}</h3>
        <p class="human">Jogador · ${safe(actionText(comparison.human))}</p>
        <p class="prediction">Modelo · ${safe(actionText(comparison.model))}</p>
        <small>${comparison.exact ? "Mesma decisão" : "Decisão diferente"} · ${percent.format(comparison.play)} de chance de jogar</small>
        <small>${percent.format(comparison.target)} de chance da jogada registrada${comparison.loss != null
          ? ` · perda do lote ${decimal.format(comparison.loss)}` : ""}</small>`;
    } else {
      const moves = [];
      for (let cursor = index; cursor >= 0 && moves.length < 6; cursor -= 1) {
        for (const move of frames[cursor].moves) moves.push({ ...move, ms: frames[cursor].ms });
      }
      el("moves").innerHTML = moves.slice(0, 6).map(move =>
        `<li class="arena-move" style="--team-color:${tones[move.seat]}"><span>
        <strong>${safe(team(move.seat))}</strong> · ${safe(actionText(move))}</span><time>${clock(move.ms)}</time></li>`).join("");
      if (!moves.length) el("moves").textContent = "Aguardando a primeira jogada.";
    }
    const result = index === frames.length - 1 ? resultText(clip.result) : "";
    el("result").textContent = result;
    el("result").hidden = !result;
  }

  function draw(ratio = 0) {
    const frame = frames[index];
    if (!frame) return;
    const { width, height } = clip.header;
    const viewWidth = board.clientWidth, viewHeight = board.clientHeight, scale = window.devicePixelRatio || 1;
    if (!viewWidth || !viewHeight) return;
    if (board.width !== Math.round(viewWidth * scale) || board.height !== Math.round(viewHeight * scale)) {
      board.width = Math.round(viewWidth * scale);
      board.height = Math.round(viewHeight * scale);
    }
    const ctx = board.getContext("2d");
    ctx.setTransform(board.width / width, 0, 0, board.height / height, 0, 0);
    ctx.fillStyle = "#304d3d";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "#344f43";
    ctx.fillRect(0, 0, width, height / 2);
    ctx.fillStyle = "#abc8a50d";
    for (let x = 1; x < width; x++) ctx.fillRect(x, 0, .045, height);
    for (let y = 1; y < height; y++) ctx.fillRect(0, y, width, .045);
    ctx.fillStyle = "#244756";
    ctx.fillRect(0, height / 2 - 1, width, 2);
    ctx.fillStyle = "#b39464";
    for (const left of [2.5, width - 5.5]) {
      ctx.fillRect(left, height / 2 - 1.15, 3, 2.3);
      ctx.fillStyle = "#705942";
      for (let row = -1; row <= 1; row += .4) ctx.fillRect(left, height / 2 + row, 3, .06);
      ctx.fillStyle = "#b39464";
    }
    const next = ratio > 0 ? new Map(frames[index + 1]?.pieces.map(piece => [piece[0], piece])) : new Map();
    const pieces = frame.pieces.map(piece => {
      const later = next.get(piece[0]);
      if (!later) return piece;
      const point = piece.slice();
      point[3] += (later[3] - point[3]) * ratio;
      point[4] += (later[4] - point[4]) * ratio;
      return point;
    });
    const positions = new Map(pieces.map(piece => [piece[0], piece]));
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const [, seat, kind, x, y, hp, maxHp, target, unit] of pieces) {
      const screenY = height - y;
      const tower = kind >= 2;
      const radius = tower ? 1.08 : kind === 1 ? .7 : .38;
      if (/x.?bow/i.test(unit) && positions.has(target)) {
        const enemy = positions.get(target);
        ctx.beginPath();
        ctx.moveTo(x, screenY);
        ctx.lineTo(enemy[3], height - enemy[4]);
        ctx.strokeStyle = tones[seat];
        ctx.lineWidth = .085;
        ctx.stroke();
      }
      ctx.fillStyle = tones[seat];
      ctx.strokeStyle = "#122027";
      ctx.lineWidth = .12;
      ctx.beginPath();
      if (kind >= 1) ctx.rect(x - radius, screenY - radius, radius * 2, radius * 2);
      else ctx.arc(x, screenY, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "#15232c";
      ctx.font = `bold ${tower ? .58 : .39}px system-ui`;
      const text = kind === 2 ? "REI" : kind === 3 ? "T" : unit.split(/[ -]/).filter(Boolean).map(word => word[0]).join("").slice(0, 2).toUpperCase();
      ctx.fillText(text, x, screenY);
      const bar = radius * 2.1;
      ctx.fillRect(x - bar / 2, screenY - radius - .28, bar, .15);
      ctx.fillStyle = tones[seat];
      ctx.fillRect(x - bar / 2, screenY - radius - .28, bar * Math.min(1, hp / Math.max(1, maxHp)), .15);
    }
    function mark(move, color, markLabel) {
      if (move.kind !== "card") return;
      const y = height - move.y;
      ctx.strokeStyle = color;
      ctx.lineWidth = .18;
      ctx.strokeRect(move.x - .62, y - .62, 1.24, 1.24);
      ctx.fillStyle = "#14232b";
      ctx.fillRect(move.x - 1.25, y + .64, 2.5, .8);
      ctx.fillStyle = color;
      ctx.font = "bold .5px system-ui";
      ctx.fillText(markLabel, move.x, y + 1.04);
    }
    if (frame.comparison) {
      mark(frame.comparison.human, "#ffe194", "JOGADOR");
      if (!frame.comparison.exact) mark(frame.comparison.model, "#d2b4ff", "MODELO");
    } else {
      for (const move of frame.moves) mark(move, tones[move.seat], move.seat === clip.header.seat ? (stage === "replays" ? "JOGADOR" : "IA") : "RIVAL");
    }
  }

  function showIndex(next) {
    index = next;
    renderFrame();
    draw();
  }

  function renderComparison() {
    const tag = stage === "replays" && frames.length ? clip.header.replay : "";
    const visible = Boolean(tag && compareOpen);
    el("compare").hidden = !tag;
    el("compare").setAttribute("aria-expanded", String(visible));
    el("compare").textContent = visible ? "Fechar comparação" : "Comparar com vídeo";
    el("comparison").hidden = !visible;
    el("reconstruction-title").hidden = !visible;
    el("content").classList.toggle("comparing", visible);
    const activeTag = visible ? tag : "";
    const recording = recordings.get(activeTag);
    if (activeTag !== videoTag || recording?.url !== (video.getAttribute("src") || undefined)) {
      videoTag = activeTag;
      video.pause();
      el("align-video").disabled = true;
      el("video-status").textContent = "";
      videoBlocked = false;
      if (recording) {
        video.src = recording.url;
      } else {
        video.removeAttribute("src");
        video.load();
      }
    }
    video.hidden = !recording;
    el("video-placeholder").hidden = Boolean(recording);
  }

  function syncVideo(ms, force = false) {
    const recording = recordings.get(videoTag);
    if (!recording || video.readyState < 1 || !Number.isFinite(video.duration)) return;
    const target = ms / 1000 + recording.offset;
    const current = Math.max(0, Math.min(video.duration, target));
    const speed = Number(el("speed").value);
    if (force || Math.abs(video.currentTime - current) > Math.max(.3, speed / 12)) {
      video.currentTime = current;
    }
    if (!playing || target < 0 || target >= video.duration) {
      video.pause();
      return;
    }
    if (video.playbackRate !== speed) video.playbackRate = speed;
    if (video.paused && !playRequest && !videoBlocked) {
      const requestedTag = videoTag;
      playRequest = video.play().catch(() => {
        if (requestedTag === videoTag && playing) {
          videoBlocked = true;
          el("video-status").textContent = "O navegador não conseguiu iniciar o vídeo junto com o replay.";
        }
      }).finally(() => { playRequest = null; });
    }
  }

  function animate(now) {
    if (!playing) return;
    const speed = context.active && !demo && !pinned
      ? Math.max(8, (frames.at(-1).ms - position) / 950) : Number(el("speed").value);
    position += (now - anchor) * speed;
    anchor = now;
    let cursor = index;
    while (cursor < frames.length - 1 && frames[cursor + 1].ms <= position) cursor += 1;
    if (cursor === frames.length - 1) {
      playing = false;
      showIndex(cursor);
      syncVideo(frames[cursor].ms, true);
      return;
    }
    if (index !== cursor) showIndex(cursor);
    const time = frames[cursor].ms;
    draw(Math.max(0, (position - time) / Math.max(1, frames[cursor + 1].ms - time)));
    syncVideo(position);
    requestAnimationFrame(animate);
  }

  function play() {
    if (playing) {
      playing = false;
      if (videoTag) video.pause();
      renderFrame();
      return;
    }
    if (frames.length < 2 || stage === "bc") return;
    if (index === frames.length - 1) index = 0;
    position = frames[index].ms;
    anchor = performance.now();
    playing = true;
    videoBlocked = false;
    syncVideo(position, true);
    renderFrame();
    requestAnimationFrame(animate);
  }

  function updateButtons() {
    for (const tab of tabs) tab.setAttribute("aria-pressed", String(tab.dataset.viewStage === stage));
    el("follow").disabled = follow && !pinned;
  }

  function clear() {
    generation += 1;
    pinned = "";
    clip = null;
    frames = [];
    index = 0;
    playing = false;
    demo = false;
    compareOpen = false;
    el("content").hidden = true;
    el("empty").hidden = false;
    el("empty").textContent = "Carregando os dados da arena…";
    renderComparison();
    updateButtons();
  }

  async function poll() {
    if (requesting) return;
    requesting = true;
    const version = generation;
    const params = new URLSearchParams({ deck: context.deck, run: context.run, stage, clip: pinned,
      known: clip?.header.id || "", after: frames.at(-1)?.seq ?? -1 });
    try {
      const response = await fetch(`/api/arena?${params}`, { signal: AbortSignal.timeout(4000) });
      if (!response.ok) throw new Error();
      const data = await response.json();
      if (version !== generation) return;
      const { clip: incoming, preview } = data;
      if (incoming || !frames.length) demo = preview;
      const source = demo ? "Demonstração com dados locais" :
        context.active ? "Partidas e decisões capturadas durante o treino" : "Gravações do experimento";
      el("source").textContent = source;
      // A recording already open in the browser stays playable after rotation on disk.
      if (!incoming && !frames.length) {
        el("empty").textContent = "Aguardando o primeiro quadro desta etapa.";
        return;
      }
      if (incoming) {
        const changed = !clip || clip.header.id !== incoming.header.id;
        if (changed) {
          playing = false;
          frames = [];
          index = 0;
        }
        clip = incoming;
        const previousCount = frames.length;
        frames.push(...incoming.frames);
        const live = context.active && !demo && !pinned;
        const selectedIndex = stage === "bc" && !pinned ? frames.length - 1 : index;
        if (frames.length && !playing) showIndex(Math.max(0, selectedIndex));
        if (changed && demo && frames.length > 1) play();
        if (live && stage !== "bc" && !playing && index < frames.length - 1 && frames.length > previousCount) play();
      }
      el("content").hidden = !frames.length;
      el("empty").hidden = Boolean(frames.length);
      renderComparison();
      if (!frames.length) return;
      if (!playing) draw();
      el("context").textContent = stage === "bc" ?
        (demo ? "Checkpoint salvo aplicado ao mesmo quadro do replay." :
          "O modelo responde ao mesmo exemplo de treino enquanto os lotes avançam.") : label(clip.header);
      const choices = [...data.clips];
      if (!choices.some(item => item.id === clip.header.id)) choices.push(clip.header);
      const key = choices.map(item => item.id).join(" ");
      if (el("list").dataset.items !== key) {
        el("list").replaceChildren(...choices.map(item => new Option(label(item), item.id)));
        el("list").dataset.items = key;
      }
      el("list").value = clip.header.id;
      el("list-label").hidden = choices.length < 2;
    } catch {
      if (version === generation && !frames.length) el("empty").textContent = "A arena está aguardando conexão com o painel.";
    } finally {
      requesting = false;
      if (version !== generation) void poll();
    }
  }

  function applyStage(next, tracking) {
    if (stage === next && follow === tracking && !pinned) return;
    stage = next;
    follow = tracking;
    clear();
    void poll();
  }

  window.trainingArena = {
    update(selection) {
      const changed = context.run !== selection.run || context.deck !== selection.deck;
      context = selection;
      const next = follow && (!pinned || changed) ? context.phase : stage;
      if (changed || next !== stage) {
        stage = next;
        clear();
        void poll();
      }
    },
  };
  for (const tab of tabs) tab.addEventListener("click", () => applyStage(tab.dataset.viewStage, false));
  el("follow").addEventListener("click", () => applyStage(context.phase, true));
  el("compare").addEventListener("click", () => {
    compareOpen = !compareOpen;
    if (compareOpen) {
      pinned = clip.header.id;
      playing = false;
      el("speed").value = "1";
      updateButtons();
      renderFrame();
    }
    renderComparison();
    draw();
  });
  el("video-file").addEventListener("change", event => {
    const file = event.target.files[0];
    event.target.value = "";
    if (!file || !videoTag) return;
    playing = false;
    renderFrame();
    const previous = recordings.get(videoTag);
    if (previous) URL.revokeObjectURL(previous.url);
    recordings.set(videoTag, { url: URL.createObjectURL(file), name: file.name, offset: 0 });
    renderComparison();
  });
  video.addEventListener("loadedmetadata", () => {
    if (!videoTag || !frames[index]) return;
    el("align-video").disabled = !Number.isFinite(video.duration);
    el("video-status").textContent = recordings.get(videoTag).name;
    syncVideo(frames[index].ms, true);
  });
  video.addEventListener("error", () => {
    if (!videoTag) return;
    el("align-video").disabled = true;
    el("video-status").textContent = "O navegador não conseguiu abrir este arquivo. Tente MP4 ou WebM.";
  });
  el("align-video").addEventListener("click", () => {
    const recording = recordings.get(videoTag);
    if (!recording || !frames[index] || video.readyState < 1) return;
    playing = false;
    video.pause();
    recording.offset = video.currentTime - frames[index].ms / 1000;
    el("video-status").textContent = `Vídeo alinhado com o replay em ${clock(frames[index].ms)}.`;
    renderFrame();
  });
  el("play").addEventListener("click", () => {
    pinned = clip.header.id;
    updateButtons();
    play();
  });
  el("seek").addEventListener("input", event => {
    playing = false;
    pinned = clip.header.id;
    updateButtons();
    showIndex(Number(event.target.value));
    syncVideo(frames[index].ms, true);
  });
  el("list").addEventListener("change", event => {
    const pick = event.target.value;
    clear();
    pinned = pick;
    updateButtons();
    void poll();
  });
  new ResizeObserver(() => draw()).observe(board);
  updateButtons();
  void poll();
  setInterval(poll, 1150);
})();
