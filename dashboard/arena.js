"use strict";

(() => {
  // The live training view and the playable battle share the arena artwork and spell drawing.
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
    graveyard: ["†", "#88c7bb", 21], poison: ["✹", "#a9d96f", 22], babydragon: ["◆", "#e59573", 23],
    tombstone: ["▰", "#c3b5a2", 24], barblog: ["◉", "#e0ac74", 25],
    megaminion: ["▼", "#a7add7", 26], angrybarbarians: ["⚔", "#e2a08d", 27], goblins: ["◆", "#94c790", 28],
    valkyrie: ["◇", "#e4af8c", 29], balloon: ["◉", "#d7a2bd", 30], infernotower: ["▲", "#f2b370", 31],
    bats: ["✦", "#c2a6db", 32], miner: ["◆", "#d3b18a", 33], mortar: ["▰", "#b8c19b", 34],
    witch: ["✧", "#a8c3b2", 35],
  };
  const columns = 6;
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
      archer: "archers", skeleton: "skeletons", icespirit: "icespirits",
      barbarianbarrel: "barblog", elitebarbarians: "angrybarbarians"})[key] || key;
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
  if (!el("live")) return;
  const {colors: tones} = window.battleArt;
  const trendCards = [...document.querySelectorAll("[data-arena-model]")];
  const trendWindow = 20;
  const rivals = {
    bc: "Imitação (BC)", other_bc: "BC outro deck", scripted: "X-Bow bot",
    hog_push: "Corredor 2.6", rg_patient: "Gigante Real", splashyard_defend: "Splashyard",
    log_bait_push: "Log Bait", golem_patient: "Golem Beatdown", pekka_push: "P.E.K.K.A BS",
    random: "Aleatório", hog_26: "Corredor 2.6", royal_giant: "Gigante Real",
    splashyard: "Splashyard", log_bait: "Log Bait", golem_beatdown: "Golem Beatdown",
    pekka_bs: "P.E.K.K.A BS",
  };
  const intFmt = new Intl.NumberFormat("pt-BR", {maximumFractionDigits: 0});
  const decFmt = new Intl.NumberFormat("pt-BR", {maximumFractionDigits: 2});
  const dec3Fmt = new Intl.NumberFormat("pt-BR", {maximumFractionDigits: 3});
  const pctFmt = new Intl.NumberFormat("pt-BR", {style: "percent", maximumFractionDigits: 1});
  const signedPoints = new Intl.NumberFormat("pt-BR", {maximumFractionDigits: 1, signDisplay: "exceptZero"});
  const signedInt = new Intl.NumberFormat("pt-BR", {maximumFractionDigits: 0, signDisplay: "exceptZero"});

  const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;",
  }[c]));

  function rivalLabel(name) {
    if (!name) return "Rival";
    if (rivals[name]) return rivals[name];
    if (name.startsWith("snapshot_")) return `Snapshot #${name.slice(9)}`;
    if (name.startsWith("dyn_patient_")) return `Descoberto · ${name.slice(12)}`;
    if (name.startsWith("dyn_push_")) return `Descoberto · ${name.slice(9)}`;
    if (name.startsWith("dyn_defend_")) return `Descoberto · ${name.slice(11)}`;
    if (name.startsWith("live_")) return `Deck real · ${name.slice(5)}`;
    if (name.startsWith("replay_")) return `Deck meta · ${name.slice(7)}`;
    return name;
  }

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "—";
    const s = Math.floor(seconds);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const rem = s % 60;
    if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m ${String(rem).padStart(2, "0")}s`;
    return `${m}m ${String(rem).padStart(2, "0")}s`;
  }

  function winSummary(rows, model) {
    const tally = rows.reduce((sum, row) => {
      const result = model === "rival" ? row.free : row;
      sum.games += result.games;
      sum.wins += result.wins;
      return sum;
    }, {games: 0, wins: 0});
    return {...tally, rate: tally.games ? tally.wins / tally.games : null};
  }

  function renderTrends(ctx) {
    const history = (ctx?.updates || []).filter(row => row.games > 0 && row.free?.games > 0).slice(-80)
      .flatMap(row => row.match_results ? row.match_results.map(result => ({
        update: row.update, games: 1, wins: Number(result === "win"),
        free: {update: row.free.update, games: 1, wins: Number(result === "loss")},
      })) : [row]).slice(-80);
    const recent = history.slice(-trendWindow);
    const previous = history.length >= trendWindow * 2 ? history.slice(-trendWindow * 2, -trendWindow) : [];
    el("live-sample").textContent = recent.length
      ? `Vitórias no treino conjunto · últimas ${recent.length} partidas${previous.length ? " e variação ante as 20 anteriores" : ""}`
      : "Aguardando partidas do treino conjunto";
    for (const card of trendCards) {
      const model = card.dataset.arenaModel;
      const current = winSummary(recent, model);
      const before = winSummary(previous, model);
      const delta = card.querySelector("[data-delta]");
      card.querySelector("[data-rate]").textContent = current.rate === null ? "—" : pctFmt.format(current.rate);
      delta.textContent = before.rate === null ? "" : `${signedPoints.format(100 * (current.rate - before.rate))} p.p.`;
      delta.dataset.direction = current.rate > before.rate ? "up" : current.rate < before.rate ? "down" : "flat";
      card.querySelector("[data-record]").textContent = current.games
        ? `${current.wins} de ${current.games} vitórias · atualizações até #${model === "rival" ? recent.at(-1).free.update : recent.at(-1).update}`
        : "Taxa de vitórias indisponível";
      const graph = card.querySelector("canvas"), paint = graph.getContext("2d");
      paint.clearRect(0, 0, graph.width, graph.height);
      if (history.length < 2) continue;
      paint.strokeStyle = "#2d3a49";
      paint.beginPath(); paint.moveTo(0, graph.height / 2); paint.lineTo(graph.width, graph.height / 2); paint.stroke();
      paint.strokeStyle = tones[model === "rival" ? 1 : 0];
      paint.lineWidth = 2;
      paint.beginPath();
      history.forEach((_, i) => {
        const rate = winSummary(history.slice(Math.max(0, i - trendWindow + 1), i + 1), model).rate;
        const x = 2 + i * (graph.width - 4) / (history.length - 1), y = 2 + (1 - rate) * (graph.height - 4);
        if (i === 0) paint.moveTo(x, y); else paint.lineTo(x, y);
      });
      paint.stroke();
    }
  }

  function renderTelemetry(ctx) {
    const updates = Array.isArray(ctx?.updates) ? ctx.updates : [];
    const progress = ctx?.progress || {};
    const job = ctx?.job || null;
    const active = Boolean(ctx?.active);
    const modelInfo = ctx?.modelInfo || {};
    const rec = ctx?.recommended || {};
    const last = updates.at(-1) || null;

    // Header & status summary
    const statusPhase = active
      ? (progress.event === "ppo_game" && progress.done >= progress.games && progress.games > 0
          ? "Otimizando pesos PPO (CPU Xeon)"
          : (ctx.detail || "Treino em execução"))
      : (updates.length ? `Última atualização #${last.update} concluída` : "Nenhum treino em andamento");
    el("source").textContent = active
      ? `Sessão ativa · ${job?.parameters?.coevolve ? "Co-evolução (X-Bow + Rival)" : "Self-Play PPO"}`
      : "Telemetria acumulada do experimento";
    el("live-status").textContent = statusPhase;

    // Live batch progress bar
    const batchBar = el("batch-bar");
    const hasBatchProg = active && Number.isFinite(progress.games) && progress.games > 0;
    batchBar.hidden = !hasBatchProg;
    if (hasBatchProg) {
      const done = Math.min(progress.games, Math.max(0, Number(progress.done || 0)));
      const optimizing = progress.event === "ppo_game" && done >= progress.games;
      el("batch-title").textContent = optimizing
        ? `Lote #${progress.update || "—"} · ${intFmt.format(done)}/${intFmt.format(progress.games)} jogos concluídos — Otimizando redes PPO…`
        : `Lote #${progress.update || "—"} · Simulando ${intFmt.format(done)} / ${intFmt.format(progress.games)} partidas na RX 6600`;
      const pct = progress.games > 0 ? Math.round((done / progress.games) * 100) : 0;
      el("batch-meta").textContent = `${pct}% · ${progress.rollout_workers || job?.parameters?.rollout_workers || 12} workers · batch ${job?.parameters?.batch_size || 512}`;
      const bar = el("batch-progress");
      bar.max = progress.games;
      bar.value = done;
    }

    // Aggregate totals
    const totalExpGames = updates.reduce((s, r) => s + Number(r.games || 0), 0);
    const totalExpWins = updates.reduce((s, r) => s + Number(r.wins || 0), 0);
    const totalExpLosses = updates.reduce((s, r) => s + Number(r.losses || 0), 0);
    const totalExpDraws = updates.reduce((s, r) => s + Number(r.draws || 0), 0);

    const sessionStartUpd = Number(progress.session_start || 0);
    const sessionUpdates = active && sessionStartUpd > 0
      ? updates.filter(r => Number(r.update || 0) > sessionStartUpd)
      : [];
    const completedInProgress = Number(progress.completed_games || 0);
    const sessionGamesFromUpdates = sessionUpdates.reduce((s, r) => s + Number(r.games || 0), 0);
    const sessionGames = active ? Math.max(completedInProgress, sessionGamesFromUpdates) : 0;

    // Speed & timing
    const simS = Number(last?.simulation_seconds || 0);
    const optS = Number(last?.optimization_seconds || 0);
    const totalBatchS = simS + optS;
    const lastGames = Number(last?.games || 0);
    const simGph = simS > 0 && lastGames > 0 ? (lastGames / simS) * 3600 : null;
    const effGph = totalBatchS > 0 && lastGames > 0 ? (lastGames / totalBatchS) * 3600 : null;
    const elapsedS = active && job?.started ? Math.max(0, (Date.now() - Date.parse(job.started)) / 1000) : null;
    const avgSessionGph = elapsedS && elapsedS > 15 && sessionGames > 0 ? (sessionGames / elapsedS) * 3600 : null;
    const displayGph = effGph ?? avgSessionGph;
    const gamesPerDay = displayGph ? displayGph * 24 : null;

    // Last batch combat & PPO metrics
    const lastWr = last && last.games > 0 ? (last.wins + 0.5 * (last.draws || 0)) / last.games : null;
    const netDmg = last && Number.isFinite(last.mean_tower_damage_dealt) && Number.isFinite(last.mean_tower_damage_taken)
      ? last.mean_tower_damage_dealt - last.mean_tower_damage_taken
      : null;
    const ppo = last?.ppo || {};
    const klVal = Number.isFinite(last?.kl_bc_to_ppo) ? last.kl_bc_to_ppo : null;
    const expVar = Number.isFinite(ppo.explained_variance) ? ppo.explained_variance : null;
    const entropy = Number.isFinite(ppo.entropy) ? ppo.entropy : null;
    const clipFrac = Number.isFinite(ppo.clip_fraction) ? ppo.clip_fraction : null;

    const pfspMap = modelInfo?.pfsp_wr && typeof modelInfo.pfsp_wr === "object" ? modelInfo.pfsp_wr : {};
    const pfspEntries = Object.entries(pfspMap)
      .filter(([, wr]) => Number.isFinite(wr))
      .sort((a, b) => a[1] - b[1]);
    const hardest = pfspEntries[0] || null;
    const easiest = pfspEntries.at(-1) || null;
    const lastRivals = last?.rivals && typeof last.rivals === "object" ? last.rivals : {};
    const rivalCountLast = Object.keys(lastRivals).length;

    const hasAnyData = active || updates.length > 0 || pfspEntries.length > 0;
    el("live-empty").hidden = hasAnyData;
    el("kpi-grid").hidden = !hasAnyData;
    el("detail-grid").hidden = !hasAnyData;
    if (!hasAnyData) {
      el("live-empty").textContent = "Nenhum dado de treino encontrado para este experimento. Inicie um treino na aba Treinamento.";
      return;
    }

    const klTone = klVal === null ? "" : (klVal <= 0.35 ? "good" : klVal <= 0.8 ? "warn" : "bad");
    const wrTone = lastWr === null ? "" : (lastWr >= 0.55 ? "good" : lastWr >= 0.42 ? "warn" : "bad");
    const evTone = expVar === null ? "" : (expVar >= 0.75 ? "good" : expVar >= 0.4 ? "warn" : "bad");

    const kpis = [
      {
        label: active ? "Partidas nesta sessão" : "Total de partidas",
        value: active ? intFmt.format(sessionGames) : intFmt.format(totalExpGames),
        tone: "accent",
        sub: active
          ? `Acumulado no experimento: ${intFmt.format(totalExpGames + Math.max(0, completedInProgress - sessionGamesFromUpdates))} · Update #${progress.update || last?.update || 0}`
          : `${updates.length} atualizações · ${intFmt.format(totalExpWins)}V-${intFmt.format(totalExpLosses)}D-${intFmt.format(totalExpDraws)}E`,
      },
      {
        label: "Velocidade real",
        value: displayGph ? `${intFmt.format(displayGph)} j/h` : "Calculando…",
        tone: displayGph && displayGph >= 3000 ? "good" : "",
        sub: simGph
          ? `Simulação: ${intFmt.format(simGph)} j/h · ~${intFmt.format(gamesPerDay)} jogos/dia`
          : (avgSessionGph ? `Média da sessão · ~${intFmt.format(gamesPerDay)} jogos/dia` : "Aguardando 1º lote completo"),
      },
      {
        label: "Ritmo & Tempo",
        value: active ? formatDuration(elapsedS) : (totalBatchS > 0 ? formatDuration(totalBatchS) : "—"),
        tone: "",
        sub: totalBatchS > 0
          ? `Último lote (${intFmt.format(lastGames)} j): sim ${decFmt.format(simS)}s · opt ${decFmt.format(optS)}s`
          : (job?.parameters ? `Lote: ${job.parameters.games} jogos · ${job.parameters.threads} threads` : "Sem tempo registrado"),
      },
      {
        label: "Desempenho (último lote)",
        value: lastWr !== null ? pctFmt.format(lastWr) : "—",
        tone: wrTone,
        sub: last
          ? `${last.wins}V-${last.losses}D-${last.draws || 0}E · Travas ${decFmt.format(last.mean_locks || 0)} · Dano ${netDmg !== null ? signedInt.format(netDmg) : "—"} · Leak ${decFmt.format(last.mean_elixir_leaked || 0)}`
          : "Aguardando resultados do lote",
      },
      {
        label: "Saúde PPO (Crítico & KL)",
        value: expVar !== null ? `EV ${pctFmt.format(expVar)}` : "—",
        tone: evTone || klTone,
        sub: last
          ? `KL ${klVal !== null ? dec3Fmt.format(klVal) : "—"} · Entropia ${entropy !== null ? decFmt.format(entropy) : "—"} · Clip ${clipFrac !== null ? pctFmt.format(clipFrac) : "—"}`
          : "Aguardando métricas PPO",
      },
      {
        label: "Diversidade & Motor",
        value: `${6 + Number(rec.discovered_decks || 0)} decks`,
        tone: "",
        sub: hardest
          ? `Mais difícil: ${rivalLabel(hardest[0])} (${pctFmt.format(hardest[1])})${rivalCountLast ? ` · ${rivalCountLast} no lote` : ""}`
          : `RX 6600 DirectML + Xeon · ${easiest ? rivalLabel(easiest[0]) : "119 cartas"}`,
      },
    ];

    el("kpi-grid").innerHTML = kpis.map(k => `
      <article class="arena-kpi" data-tone="${esc(k.tone)}">
        <span class="arena-kpi-label">${esc(k.label)}</span>
        <strong class="arena-kpi-value">${esc(k.value)}</strong>
        <small class="arena-kpi-sub">${esc(k.sub)}</small>
      </article>`).join("");

    // Recent completed batches table (newest first, up to 10)
    const recentUpdates = updates.slice(-10).reverse();
    el("batches-caption").textContent = updates.length
      ? `Mostrando ${recentUpdates.length} de ${updates.length} atualizações`
      : "Sem lotes concluídos";
    if (recentUpdates.length) {
      el("batches-table").innerHTML = `
        <table class="arena-table">
          <thead>
            <tr>
              <th>Lote</th>
              <th>Jogos</th>
              <th>Placar</th>
              <th>WR</th>
              <th>Travas</th>
              <th>Dano líq.</th>
              <th>KL</th>
              <th>Sim / Opt</th>
              <th>Jogos/h</th>
            </tr>
          </thead>
          <tbody>
            ${recentUpdates.map(row => {
              const g = Number(row.games || 0);
              const wr = g > 0 ? (row.wins + 0.5 * (row.draws || 0)) / g : null;
              const dmg = Number.isFinite(row.mean_tower_damage_dealt) && Number.isFinite(row.mean_tower_damage_taken)
                ? row.mean_tower_damage_dealt - row.mean_tower_damage_taken : null;
              const ss = Number(row.simulation_seconds || 0);
              const os = Number(row.optimization_seconds || 0);
              const gph = g > 0 && (ss + os) > 0 ? (g / (ss + os)) * 3600 : null;
              const wrCls = wr === null ? "" : (wr >= 0.55 ? "good" : wr >= 0.42 ? "warn" : "bad");
              return `
                <tr>
                  <td><strong>#${esc(row.update)}</strong></td>
                  <td>${intFmt.format(g)}</td>
                  <td>${row.wins}V-${row.losses}D${row.draws ? `-${row.draws}E` : ""}</td>
                  <td class="tone-${wrCls}">${wr !== null ? pctFmt.format(wr) : "—"}</td>
                  <td>${Number.isFinite(row.mean_locks) ? decFmt.format(row.mean_locks) : "—"}</td>
                  <td>${dmg !== null ? signedInt.format(dmg) : "—"}</td>
                  <td>${Number.isFinite(row.kl_bc_to_ppo) ? dec3Fmt.format(row.kl_bc_to_ppo) : "—"}</td>
                  <td>${ss > 0 ? `${Math.round(ss)}s / ${Math.round(os)}s` : "—"}</td>
                  <td>${gph ? intFmt.format(gph) : "—"}</td>
                </tr>`;
            }).join("")}
          </tbody>
        </table>`;
    } else {
      el("batches-table").innerHTML = `<p class="empty">Nenhum lote finalizado ainda.</p>`;
    }

    // PFSP rival deck win rates (hardest -> easiest)
    el("pfsp-caption").textContent = pfspEntries.length
      ? `${pfspEntries.length} rivais monitorados · foco automático nos mais difíceis`
      : "Sem histórico PFSP";
    if (pfspEntries.length) {
      el("pfsp-list").innerHTML = pfspEntries.slice(0, 14).map(([rKey, wr]) => {
        const pct = Math.max(0, Math.min(100, wr * 100));
        const tone = wr >= 0.65 ? "good" : wr >= 0.40 ? "warn" : "bad";
        const lastCount = lastRivals[rKey] || 0;
        return `
          <div class="arena-pfsp-row">
            <div class="arena-pfsp-meta">
              <span class="arena-pfsp-name">${esc(rivalLabel(rKey))}</span>
              <span class="arena-pfsp-right">
                ${lastCount ? `<small class="arena-pfsp-count">${lastCount} no lote</small>` : ""}
                <strong class="tone-${tone}">${pctFmt.format(wr)}</strong>
              </span>
            </div>
            <div class="arena-pfsp-track"><span class="arena-pfsp-fill tone-${tone}" style="width:${pct.toFixed(1)}%"></span></div>
          </div>`;
      }).join("");
    } else {
      el("pfsp-list").innerHTML = `<p class="empty">Aguardando dados de confrontos por deck.</p>`;
    }
  }

  window.trainingArena = {
    update(selection) {
      renderTrends(selection);
      renderTelemetry(selection);
    },
  };
})();
