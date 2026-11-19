"use strict";

(() => {
  // The replay viewer and the playable battle share the arena artwork and spell drawing.
  const colors = ["#6ed4e1", "#f89078"];
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  // Tile-based geometry and speed from the bundled RoyaleSim card table and tick rate.
  const spellRules = Object.freeze({
    Fireball: {radius: 2.5, speed: 12, tickMs: 50, fadeMs: 480},
    Log: {halfWidth: 1.95, halfDepth: .6, range: 10.1, throwDistance: 3,
      throwMs: 450, speed: 4, tickMs: 50, fadeMs: 260},
  });
  const skins = {
    xbow: ["➶", "#e7bd88", 0], tesla: ["ϟ", "#80deeb", 1], knight: ["⚔", "#d2d7e8", 2],
    archers: ["➵", "#edb4dc", 3], skeletons: ["◇", "#ece1bb", 4], fireball: ["✹", "#ffad76", 5],
    log: ["≋", "#dfb572", 6], electrospirit: ["✦", "#a5d8ff", 7], icespirits: ["❄", "#a8e6f3", 8],
    icewizard: ["✧", "#9dcff1", 9], rocket: ["▲", "#ffab9e", 10], tornado: ["◎", "#c3b7ec", 11],
    hogrider: ["➤", "#efae85", 12], musketeer: ["⊙", "#c9aef6", 13], cannon: ["◉", "#d2bc91", 14],
    icegolemite: ["◆", "#bce3ef", 15], royalgiant: ["♜", "#e5cb79", 16], hunter: ["⋔", "#b9dca2", 17],
    fisherman: ["⚓", "#accfd6", 18], ghost: ["◌", "#d9e7ee", 19], lightning: ["ϟ", "#e0beff", 20],
  };
  const columns = 5;
  const portraits = new Image();
  const portraitAsset = getComputedStyle(document.documentElement).getPropertyValue("--battle-card-portraits").trim();
  const portraitUrl = portraitAsset.match(/^url\("([^"]+)"\)$/)?.[1];
  if (portraitUrl) {
    portraits.addEventListener("load", () => { document.documentElement.dataset.cardPortraits = ""; }, {once: true});
    portraits.src = portraitUrl;
  }

  function cardKey(name) {
    const key = name.toLowerCase().replace(/_ev\d.*$/, "").replace(/[^a-z]/g, "");
    return ({thelog: "log", otronco: "log", boladefogo: "fireball",
      icegolem: "icegolemite", royalghost: "ghost",
      archer: "archers", skeleton: "skeletons", icespirit: "icespirits"})[key] || key;
  }

  function skin(name) {
    const key = cardKey(name);
    return skins[key] || skins[key.replace(/s$/, "")] || skins[key + "s"] ||
      [name[0] || "•", "#e8e6d6"];
  }


  function portraitPosition(portrait) {
    const step = 100 / (columns - 1);
    return (portrait % columns) * step + "% " + Math.floor(portrait / columns) * step + "%";
  }

  function spellName(name) {
    const key = cardKey(name || "");
    if (key === "log") return "Log";
    if (key === "fireball") return "Fireball";
    return null;
  }

  function spellCast(move, scene) {
    const card = spellName(move.card);
    if (!card || move.kind !== "card") return null;
    const rule = spellRules[card];
    const {x, y, seat, ms} = move;
    const cast = {card, rule, x, y, ms, direction: seat === 0 ? 1 : -1};
    if (card === "Log") {
      const rollMs = Math.ceil(rule.range / (rule.speed * rule.tickMs / 1000)) * rule.tickMs;
      return {...cast, duration: rule.throwMs + rollMs};
    }
    const king = scene.pieces.find(piece => piece[1] === seat && piece[2] === 2);
    const originX = king?.[3] ?? scene.arena.width / 2;
    const originY = king?.[4] ?? (seat === 0 ? 3 : scene.arena.height - 3);
    const distance = Math.hypot(x - originX, y - originY);
    const duration = Math.max(rule.tickMs, Math.ceil(distance * 1000 / (rule.speed * rule.tickMs)) * rule.tickMs);
    return {...cast, originX, originY, duration};
  }

  const castEnds = cast => cast.ms + cast.duration + cast.rule.fadeMs;
  const castVisible = (cast, time) => time >= cast.ms && time < castEnds(cast);

  function createRenderer(board, onPortraitsReady) {
    const ctx = board.getContext("2d");
    const ground = document.createElement("canvas");
    const groundCtx = ground.getContext("2d");
    let groundKey = "";

    function setupGround({width, height, river = [15, 17], bridges = [[2.5, 4.5], [13.5, 15.5]]}) {
      const key = [board.width, board.height, width, height, ...river, ...bridges.flat()].join(":");
      if (key === groundKey) return;
      groundKey = key;
      ground.width = board.width;
      ground.height = board.height;
      const paint = groundCtx;
      paint.setTransform(ground.width / width, 0, 0, ground.height / height, 0, 0);
      const terrain = paint.createLinearGradient(0, 0, 0, height);
      terrain.addColorStop(0, "#735144");
      terrain.addColorStop(.27, "#697059");
      terrain.addColorStop(.67, "#526f64");
      terrain.addColorStop(1, "#375e60");
      paint.fillStyle = terrain;
      paint.fillRect(0, 0, width, height);
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        if ((x + y) % 2 === 0) {
          paint.fillStyle = "#f0e5be0f";
          paint.fillRect(x, y, 1, 1);
        }
        const speck = (x * 7 + y * 13) % 11;
        if (speck < 4) {
          paint.fillStyle = "#232f325a";
          paint.fillRect(x + .15 + speck * .12, y + .57, .12, .045);
        }
      }

      const top = height - river[1];
      const waterHeight = river[1] - river[0];
      for (const [left, right] of bridges) {
        const center = (left + right) / 2;
        paint.fillStyle = "#e0bc892b";
        paint.fillRect(center - 1.5, 0, 3, height);
        paint.fillStyle = "#b1b69443";
        for (let y = .8; y < height; y += .82) paint.fillRect(center - 1.43, y, 2.86, .04);
      }
      paint.fillStyle = "#242c28";
      paint.fillRect(0, top - .16, width, waterHeight + .32);
      const water = paint.createLinearGradient(0, top, 0, top + waterHeight);
      water.addColorStop(0, "#1b5961");
      water.addColorStop(.58, "#237783");
      water.addColorStop(1, "#193e55");
      paint.fillStyle = water;
      paint.fillRect(0, top, width, waterHeight);
      paint.fillStyle = "#9bdddf56";
      for (let x = .5; x < width; x += .84) {
        const y = top + .23 + (Math.round(x * 110) % 8) * .16;
        paint.fillRect(x, y, .62, .045);
      }
      for (const [left, right] of bridges) {
        paint.fillStyle = "#3e2f2b";
        paint.fillRect(left - .1, top - .17, right - left + .2, waterHeight + .34);
        for (let y = top - .12; y < top + waterHeight; y += .28) {
          paint.fillStyle = "#b4946d";
          paint.fillRect(left, y, right - left, .24);
          paint.fillStyle = "#dac19b";
          paint.fillRect(left, y, right - left, .045);
        }
        paint.fillStyle = "#76624e";
        paint.fillRect(left, top - .17, .12, waterHeight + .34);
        paint.fillRect(right - .12, top - .17, .12, waterHeight + .34);
      }
      paint.fillStyle = "#27323a";
      paint.fillRect(0, 0, .22, height);
      paint.fillRect(width - .22, 0, .22, height);
      paint.fillStyle = "#b7a482";
      for (let y = .23; y < height; y += .62) {
        paint.fillRect(.04, y, .17, .055);
        paint.fillRect(width - .2, y, .17, .055);
      }
    }

    function ring(x, y, radius, color, fill = false) {
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      if (fill) {
        ctx.fillStyle = color;
        ctx.fill();
      } else {
        ctx.strokeStyle = color;
        ctx.stroke();
      }
    }

    function drawPortrait(portrait, x, y, radius, color) {
      const tileWidth = portraits.naturalWidth / columns;
      const tileHeight = portraits.naturalHeight / columns;
      const left = (portrait % columns) * tileWidth;
      const top = Math.floor(portrait / columns) * tileHeight;
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.clip();
      ctx.drawImage(portraits, left, top, tileWidth, tileHeight,
        x - radius, y - radius, radius * 2, radius * 2);
      ctx.restore();
      ctx.lineWidth = .12;
      ring(x, y, radius, color);
    }

    function drawPiece({piece, x, y}) {
      const [, seat, kind, , , hp, maxHp, , name] = piece;
      const tower = kind >= 2;
      const king = kind === 2;
      const [glyph, ink, portrait] = tower ? [king ? "♜" : "◆", colors[seat]] : skin(name);
      const hasPortrait = portrait !== undefined && portraits.naturalWidth > 0;
      const radius = king ? 1.06 : tower ? .88 : kind === 1 ? .67 : hasPortrait ? .54 : .43;
      const color = colors[seat];
      ctx.fillStyle = "#11212c88";
      ctx.beginPath();
      ctx.ellipse(x, y + radius * .43, radius * 1.18, radius * .63, 0, 0, Math.PI * 2);
      ctx.fill();
      if (hasPortrait) {
        drawPortrait(portrait, x, y, radius, color);
      } else if (kind >= 1) {
        const top = y - radius;
        ctx.fillStyle = "#273943";
        ctx.fillRect(x - radius, top + radius * .24, radius * 2, radius * 1.62);
        ctx.fillStyle = color;
        ctx.fillRect(x - radius, top + radius * .32, radius * 2, radius * .28);
        ctx.fillStyle = "#cdd8cb";
        ctx.fillRect(x - radius, top + radius * .14, radius * 2, radius * .18);
        for (const step of [-1, -.18, .64]) ctx.fillRect(x + radius * step, top - .06, radius * .36, radius * .23);
        ctx.fillStyle = "#13212c";
        ctx.fillRect(x - radius * .48, y + radius * .12, radius * .96, radius * .67);
      } else {
        ctx.lineWidth = .13;
        ring(x, y, radius, "#162737", true);
        ring(x, y, radius, color);
      }
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      if (!hasPortrait) {
        ctx.font = tower ? "bold .72px system-ui" : "bold .56px system-ui";
        ctx.fillStyle = ink;
        ctx.fillText(glyph, x, y + (kind ? radius * .45 : 0));
      }
      if (tower || hp < maxHp) {
        const bar = radius * 2.1, above = y - radius - .31;
        ctx.fillStyle = "#12232c";
        ctx.fillRect(x - bar / 2, above, bar, .2);
        ctx.fillStyle = color;
        ctx.fillRect(x - bar / 2, above, bar * Math.max(0, Math.min(1, hp / maxHp)), .2);
        if (tower) {
          ctx.font = "bold .51px system-ui";
          ctx.lineWidth = .14;
          ctx.strokeStyle = "#12232c";
          ctx.strokeText(String(hp), x, above - .24);
          ctx.fillStyle = "#fff6e3";
          ctx.fillText(String(hp), x, above - .24);
        }
      }
    }

    function stroke(x, y, endX, endY, color, width) {
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(endX, endY);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke();
    }

    function paintSpellArea(card, x, y, height, direction, opacity, dashed, tint) {
      const rule = spellRules[card];
      const row = height - y;
      ctx.save();
      ctx.globalAlpha = opacity;
      const edge = tint || (card === "Log" ? "#ffe4a1" : "#ffc572");
      ctx.fillStyle = tint ? tint + "36" : card === "Log" ? "#ffd88b30" : "#ff8d3c38";
      ctx.strokeStyle = edge;
      ctx.lineWidth = .085;
      ctx.setLineDash(dashed ? [.23, .16] : []);
      ctx.beginPath();
      if (card === "Log") {
        const last = row - direction * rule.range;
        const top = Math.min(last, row);
        const bottom = Math.max(last, row);
        const {halfWidth, halfDepth} = rule;
        ctx.ellipse(x, top, halfWidth, halfDepth, 0, Math.PI, 2 * Math.PI);
        ctx.lineTo(x + halfWidth, bottom);
        ctx.ellipse(x, bottom, halfWidth, halfDepth, 0, 0, Math.PI);
        ctx.closePath();
      } else {
        ctx.arc(x, row, rule.radius, 0, 2 * Math.PI);
      }
      ctx.fill();
      ctx.stroke();
      if (card === "Log") {
        const tip = row - direction * (rule.range - .38);
        ctx.setLineDash([.32, .2]);
        stroke(x, row, x, tip, edge, .07);
        ctx.setLineDash([]);
        stroke(x - .32, tip + direction * .48, x, tip, edge, .10);
        stroke(x + .32, tip + direction * .48, x, tip, edge, .10);
      }
      ctx.restore();
    }

    function drawFireball(cast, age, height) {
      const {x, y, originX, originY, duration, rule} = cast;
      if (age >= duration) {
        const fade = (age - duration) / rule.fadeMs;
        ctx.save();
        ctx.globalAlpha = 1 - fade;
        ring(x, height - y, rule.radius, "#ff9b4255", true);
        ring(x, height - y, Math.max(.2, rule.radius * (.22 + .78 * fade)), "#ffe1a1");
        ring(x, height - y, .57 * (1 - fade), "#fff3be", true);
        ctx.restore();
        return;
      }
      const flight = age / duration;
      const lift = reducedMotion ? 0 : .9;
      const tail = reducedMotion ? 1 : 4;
      for (let i = tail; i >= 0; i--) {
        const t = Math.max(0, flight - i * .045);
        const px = originX + (x - originX) * t;
        const py = height - (originY + (y - originY) * t) - lift * Math.sin(Math.PI * t);
        const size = .36 - i * .058;
        ring(px, py, size * 1.54, "#ff642c62", true);
        ring(px, py, size, i ? "#ff9c43bb" : "#fff0ab", true);
      }
    }

    function drawRollingLog(cast, age, height) {
      const {x, y, direction, duration, rule} = cast;
      if (age >= duration) return;
      const airborne = age < rule.throwMs;
      const progress = Math.min(1, age / rule.throwMs);
      const travelled = Math.min(rule.range, Math.max(0, age - rule.throwMs) * rule.speed / 1000);
      const groundY = height - y + direction * (airborne ? rule.throwDistance * (1 - progress) : -travelled);
      const row = groundY - (airborne && !reducedMotion ? .65 * Math.sin(Math.PI * progress) : 0);
      ctx.save();
      ctx.beginPath();
      ctx.ellipse(x, groundY, rule.halfWidth, rule.halfDepth, 0, 0, 2 * Math.PI);
      ctx.fillStyle = airborne ? "#20151055" : "#ffe3ab48";
      ctx.fill();
      if (!airborne) {
        ctx.lineWidth = .095;
        ctx.strokeStyle = "#fff0b5";
        ctx.stroke();
      }
      const bark = ctx.createLinearGradient(0, row - .35, 0, row + .35);
      bark.addColorStop(0, "#d39b61");
      bark.addColorStop(.46, "#986036");
      bark.addColorStop(1, "#48271b");
      ctx.beginPath();
      ctx.roundRect(x - 1.72, row - .34, 3.44, .68, .28);
      ctx.fillStyle = bark;
      ctx.fill();
      ctx.lineWidth = .07;
      ctx.strokeStyle = "#57301c";
      ctx.stroke();
      ctx.save();
      ctx.clip();
      for (let i = 0; i < 3; i++) {
        const groove = row - .31 + ((i * .22 + direction * travelled + 12) % .62);
        stroke(x - 1.65, groove, x + 1.65, groove - .045, "#452b1bba", .055);
      }
      for (const offset of [-1.23, 1.23]) {
        ctx.fillStyle = "#d4b17d";
        ctx.fillRect(x + offset - .08, row - .34, .16, .68);
      }
      ctx.restore();
      for (const offset of [-1.62, 1.62]) {
        ctx.beginPath();
        ctx.ellipse(x + offset, row, .19, .26, 0, 0, 2 * Math.PI);
        ctx.fillStyle = "#e9bb80";
        ctx.fill();
        ctx.strokeStyle = "#714927";
        ctx.stroke();
      }
      ctx.restore();
    }


    function begin(scene) {
      const {width, height} = scene;
      const resolution = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.round(board.clientWidth * resolution);
      const h = Math.round(board.clientHeight * resolution);
      if (!w || !h) return false;
      if (board.width !== w || board.height !== h) {
        board.width = w;
        board.height = h;
      }
      setupGround(scene);
      ctx.setTransform(w / width, 0, 0, h / height, 0, 0);
      ctx.drawImage(ground, 0, 0, width, height);
      return true;
    }

    function drawPieces(pieces, height, locate) {
      const points = pieces.map(piece => {
        const [x, y] = locate(piece);
        return {piece, x, y: height - y};
      });
      const positions = new Map(points.map(entry => [entry.piece[0], entry]));
      for (const {piece, x, y} of points) {
        if (!/^x.?bow/i.test(piece[8])) continue;
        const target = positions.get(piece[7]);
        if (target) stroke(x, y, target.x, target.y, colors[piece[1]] + "aa", .08);
      }
      points.sort((a, b) => a.y - b.y).forEach(drawPiece);
    }

    function drawCastArea(cast, time, height) {
      const fade = Math.max(0, time - cast.ms - cast.duration) / cast.rule.fadeMs;
      paintSpellArea(cast.card, cast.x, cast.y, height, cast.direction, .60 * (1 - fade), false);
    }

    function drawCast(cast, time, height) {
      const age = Math.max(0, time - cast.ms);
      if (cast.card === "Log") drawRollingLog(cast, age, height);
      else drawFireball(cast, age, height);
    }

    function drawCardPreview(card, x, y, radius, color) {
      const [glyph, , portrait] = skin(card);
      ring(x, y, radius, "#162737", true);
      if (portrait !== undefined && portraits.naturalWidth > 0) {
        drawPortrait(portrait, x, y, radius, color);
      } else {
        ctx.lineWidth = .12;
        ring(x, y, radius, color);
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = "bold .60px system-ui";
        ctx.fillStyle = color;
        ctx.fillText(glyph, x, y);
      }
    }

    if (onPortraitsReady && !portraits.naturalWidth) {
      portraits.addEventListener("load", onPortraitsReady, {once: true});
    }
    return {ctx, begin, ring, stroke, drawPieces, paintSpellArea, drawCastArea, drawCast, drawCardPreview};
  }

  window.battleArt = {colors, reducedMotion, skin, cardKey, portraitPosition, spellName, spellCast,
    castEnds, castVisible, createRenderer};
})();

(() => {
  const el = id => document.getElementById(`arena-${id}`);
  const board = el("board");
  const {colors: tones, reducedMotion, skin, cardKey, portraitPosition, spellName, spellCast, castVisible,
    createRenderer} = window.battleArt;
  const {ctx, begin, ring, drawPieces, paintSpellArea, drawCastArea, drawCast, drawCardPreview} =
    createRenderer(board, () => draw());
  const video = el("replay-video");
  const recordings = new Map();
  const tabs = [...document.querySelectorAll("[data-view-stage]")];
  const rivals = { bc: "Imitação", other_bc: "Imitação do outro deck", scripted: "X-Bow programado",
    hog_push: "Corredor", rg_patient: "Gigante Real", random: "Aleatório" };
  const cards = {xbow: "X-Bow", tesla: "Tesla", knight: "Cavaleiro", archers: "Arqueiras",
    skeletons: "Esqueletos", fireball: "Bola de Fogo", log: "O Tronco", electrospirit: "Espírito Elétrico",
    icespirits: "Espírito de Gelo", icewizard: "Mago de Gelo", rocket: "Foguete", tornado: "Tornado",
    hogrider: "Corredor", musketeer: "Mosqueteira", cannon: "Canhão", icegolemite: "Golem de Gelo",
    royalgiant: "Gigante Real", hunter: "Caçador", fisherman: "Pescador", ghost: "Fantasma Real",
    lightning: "Relâmpago"};
  const replayErrors = { "ability unavailable or ambiguous": "habilidade indisponível ou ambígua",
    "card not in hand": "a carta registrada não estava na mão",
    "play refused by engine": "jogada recusada pelo simulador",
    "simulator ended before replay": "a simulação terminou antes da última jogada",
    "simulator refused replay setup": "configuração do replay recusada pelo simulador" };
  const name = card => cards[cardKey(card)] || card;
  const safe = value => String(value ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const percent = new Intl.NumberFormat("pt-BR", { style: "percent", maximumFractionDigits: 2 });
  const decimal = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 3 });
  const signedPoints = new Intl.NumberFormat("pt-BR", {maximumFractionDigits: 2, signDisplay: "exceptZero"});
  const comparisonStepMs = 1350;
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
  let displayedFrame = null;
  let previewFrame = 0;
  let previewProgress = 1;

  function clock(ms) {
    const seconds = Math.floor(ms / 1000);
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  }

  function rival(name) {
    return rivals[name] || name?.replace(/^snapshot_/, "Versão anterior ") || "Rival";
  }

  function label(header) {
    if (header.probe_card) return `${name(header.probe_card)} · replay ${header.replay.slice(0, 8)}`;
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

  function handCard(card) {
    const [glyph, tint, portrait] = skin(card);
    const image = portrait === undefined ? "" : ` data-portrait style="background-position:${portraitPosition(portrait)}"`;
    const label = safe(name(card));
    return `<span class="arena-hand-card" title="${label}" style="--card-tone:${tint}">
      <span class="arena-card-icon"${image} aria-hidden="true">${safe(glyph)}</span>
      <span class="arena-card-label">${label}</span>
    </span>`;
  }

  function probabilityRow(label, value, previous, teamClass) {
    const delta = Number.isFinite(previous) ? `${signedPoints.format((value - previous) * 100)} p.p. desde o registro anterior` : "";
    const fill = Math.max(0, Math.min(100, value * 100));
    return `<div class="arena-probability ${teamClass}">
      <div><span>${label}</span><strong>${percent.format(value)}</strong></div>
      <div class="arena-probability-track" aria-hidden="true"><i style="width:${fill}%"></i></div>
      ${delta ? `<small>${delta}</small>` : ""}
    </div>`;
  }

  function renderDecision(comparison) {
    const title = comparison.epoch ? `Época ${comparison.epoch} · lote ${comparison.batch}/${comparison.batches}`
      : "Decisão no replay";
    const previous = frames[index - 1]?.comparison;
    const model = comparison.model.kind === "wait" ? `Esperar · ${percent.format(1 - comparison.play)}`
      : actionText(comparison.model);
    const record = context.active && !demo && !pinned ? "Acompanhando os lotes" : `Registro ${index + 1} de ${frames.length}`;
    el("decision").innerHTML = `<div class="arena-decision-head"><h3>${title}</h3><small>${record}</small></div>
      <div class="arena-decision-choices">
        <div class="human"><small>Jogador</small><strong>${safe(actionText(comparison.human))}</strong></div>
        <div class="prediction"><small>Modelo</small><strong>${safe(model)}</strong></div>
      </div>
      ${comparison.alternative ? `<small>Se escolhesse jogar: ${safe(actionText(comparison.alternative))}</small>` : ""}
      <div class="arena-probabilities">
        ${probabilityRow("Chance de jogar", comparison.play, previous?.play, "prediction")}
        ${probabilityRow("Chance da jogada registrada", comparison.target, previous?.target, "human")}
      </div>
      <small>${comparison.exact ? "Mesma decisão" : "Decisão diferente"}${comparison.loss != null
        ? ` · perda do lote ${decimal.format(comparison.loss)}` : ""}</small>`;
  }

  function renderFrame() {
    const frame = frames[index];
    if (!frame) return;
    const comparison = frame.comparison;
    el("clock").textContent = comparison ? `Quadro ${clock(frame.ms)}` : clock(frame.ms);
    el("seek").max = frames.length - 1;
    el("seek").value = index;
    el("seek").setAttribute("aria-label", comparison ? "Registro de imitação" : "Posição na gravação");
    el("seek").setAttribute("aria-valuetext", frame.comparison?.epoch
      ? `Época ${frame.comparison.epoch}, lote ${frame.comparison.batch}` : clock(frame.ms));
    el("play").textContent = playing ? "Pausar" : stage === "bc" ? "Reproduzir histórico" :
      index === frames.length - 1 ? "Rever" : "Reproduzir";
    el("play").hidden = stage === "bc" && frames.length < 2;
    el("preview").hidden = !comparison;
    el("speed").hidden = stage === "bc";
    el("teams").innerHTML = frame.sides.map((side, seat) =>
      `<div class="arena-team" style="--team-color:${tones[seat]};--elixir:${Math.min(10, Math.max(0, side.elixir)) * 10}%">
        <div class="arena-team-head"><strong>${safe(team(seat))}</strong><span>${decimal.format(side.elixir)}/10 elixir</span></div>
        <div class="arena-elixir" aria-hidden="true"></div>
        <small>Torres · ${side.hp.map(value => Math.max(0, value)).join(" / ")} PV</small>
        <div class="arena-hand" role="group" aria-label="Mão de ${safe(team(seat))}">${side.hand.map(handCard).join("")}</div>
      </div>`).reverse().join("");
    el("decision").hidden = !comparison;
    el("moves-box").hidden = Boolean(comparison);
    if (comparison) {
      renderDecision(comparison);
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

  function replayCasts(time) {
    const casts = [];
    for (let cursor = index; cursor >= 0; cursor -= 1) {
      const frame = frames[cursor];
      // Both spell animations, including their fading areas, take less than four seconds.
      if (time - frame.ms >= 4000) break;
      for (const move of frame.moves) {
        const cast = spellCast({...move, ms: frame.ms}, {arena: clip.header, pieces: frame.pieces});
        if (cast && castVisible(cast, time)) casts.push(cast);
      }
    }
    return casts.reverse();
  }

  function draw(ratio = 0) {
    const frame = frames[index];
    if (!frame || !begin(clip.header)) return;
    const {height} = clip.header;
    const following = frames[index + 1];
    const time = frame.ms + (following ? (following.ms - frame.ms) * ratio : 0);
    const casts = frame.comparison ? [] : replayCasts(time);
    for (const cast of casts) drawCastArea(cast, time, height);

    function compareArea(move, color) {
      const card = spellName(move.card);
      if (move.kind === "card" && card) {
        paintSpellArea(card, move.x, move.y, height, clip.header.seat === 0 ? 1 : -1, .95, true, color);
      }
    }
    const comparison = frame.comparison;
    if (comparison) {
      compareArea(comparison.human, "#ffe194");
      if (!comparison.exact) compareArea(comparison.model, "#d2b4ff");
    }

    const next = new Map(ratio > 0 ? following?.pieces.map(piece => [piece[0], piece]) : []);
    drawPieces(frame.pieces, height, piece => {
      const later = next.get(piece[0]);
      return later
        ? [piece[3] + (later[3] - piece[3]) * ratio, piece[4] + (later[4] - piece[4]) * ratio]
        : [piece[3], piece[4]];
    });
    for (const cast of casts) drawCast(cast, time, height);

    function mark(move, color, markLabel) {
      if (move.kind !== "card") return;
      const y = height - move.y;
      ctx.strokeStyle = color;
      ctx.lineWidth = .18;
      ctx.strokeRect(move.x - .62, y - .62, 1.24, 1.24);
      ctx.fillStyle = "#14232b";
      const labelX = Math.max(1.25, Math.min(clip.header.width - 1.25, move.x));
      const labelY = y + 1.44 > height ? y - 1.44 : y + .64;
      ctx.fillRect(labelX - 1.25, labelY, 2.5, .8);
      ctx.fillStyle = color;
      ctx.font = "bold .5px system-ui";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(markLabel, labelX, labelY + .4);
    }
    if (comparison) {
      function previewAction(move, color, label) {
        if (move.kind !== "card") return;
        const cast = spellCast({...move, seat: clip.header.seat, ms: 0},
          {arena: clip.header, pieces: frame.pieces});
        const y = height - move.y;
        if (cast && previewProgress < 1) {
          drawCast(cast, previewProgress * (cast.duration + cast.rule.fadeMs * .55), height);
        } else {
          drawCardPreview(move.card, move.x, y, .32 + .27 * previewProgress, color);
          if (previewProgress < 1) {
            ctx.save();
            ctx.globalAlpha = 1 - previewProgress;
            ctx.lineWidth = .1;
            ring(move.x, y, .6 + .55 * previewProgress, color);
            ctx.restore();
          }
        }
        mark(move, color, label);
      }
      previewAction(comparison.human, "#ffe194", comparison.exact ? "AMBOS" : "JOGADOR");
      if (!comparison.exact) previewAction(comparison.model, "#d2b4ff", "MODELO");
    } else {
      for (const move of frame.moves) {
        if (spellName(move.card)) continue;
        const actor = move.seat === clip.header.seat ? (stage === "replays" ? "JOGADOR" : "IA") : "RIVAL";
        mark(move, tones[move.seat], actor);
      }
    }
  }

  function previewDecision() {
    cancelAnimationFrame(previewFrame);
    previewFrame = 0;
    previewProgress = 1;
    const frame = frames[index];
    if (!frame?.comparison || reducedMotion) {
      draw();
      return;
    }
    const started = performance.now();
    previewProgress = 0;
    function animatePreview(now) {
      previewFrame = 0;
      if (frames[index] !== frame) return;
      previewProgress = Math.max(0, Math.min(1, (now - started) / 1050));
      draw();
      if (previewProgress < 1) previewFrame = requestAnimationFrame(animatePreview);
    }
    previewFrame = requestAnimationFrame(animatePreview);
  }

  function showIndex(next) {
    index = next;
    renderFrame();
    if (frames[index] !== displayedFrame) {
      displayedFrame = frames[index];
      previewDecision();
    } else {
      draw();
    }
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
    if (stage === "bc") {
      position += (now - anchor) / comparisonStepMs;
      anchor = now;
      const cursor = Math.min(frames.length - 1, Math.floor(position));
      if (cursor === frames.length - 1) playing = false;
      if (index !== cursor || !playing) showIndex(cursor);
      if (playing) requestAnimationFrame(animate);
      return;
    }
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
    if (frames.length < 2) return;
    if (index === frames.length - 1) index = 0;
    position = stage === "bc" ? index : frames[index].ms;
    anchor = performance.now();
    playing = true;
    videoBlocked = false;
    syncVideo(position, true);
    showIndex(index);
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
    displayedFrame = null;
    cancelAnimationFrame(previewFrame);
    previewFrame = 0;
    previewProgress = 1;
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
        const selectedIndex = stage === "bc" && (!pinned || changed) ? frames.length - 1 : index;
        if (frames.length && !playing) showIndex(Math.max(0, selectedIndex));
        if (changed && demo && stage !== "bc" && frames.length > 1) play();
        if (live && stage !== "bc" && !playing && index < frames.length - 1 && frames.length > previousCount) play();
      }
      el("content").hidden = !frames.length;
      el("empty").hidden = Boolean(frames.length);
      renderComparison();
      if (!frames.length) return;
      if (!playing) draw();
      el("context").textContent = stage === "bc" ?
        (demo ? "Previsão de um checkpoint salvo neste quadro." :
          "O quadro do replay é fixo. O histórico mostra como a previsão muda a cada registro.") : label(clip.header);
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
  el("preview").addEventListener("click", previewDecision);
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
