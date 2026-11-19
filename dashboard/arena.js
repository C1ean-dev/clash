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
  const {colors: tones, cardKey, spellCast, castVisible, createRenderer} = window.battleArt;
  const liveTiles = new Map();
  const trendCards = [...document.querySelectorAll("[data-arena-model]")];
  const trendWindow = 20;
  const rivals = { bc: "Imitação", other_bc: "Imitação do outro deck", scripted: "X-Bow programado",
    hog_push: "Corredor", rg_patient: "Gigante Real", random: "Aleatório",
    hog_26: "Corredor 2.6", royal_giant: "Gigante Real", splashyard: "Splashyard" };
  const cards = {xbow: "X-Bow", tesla: "Tesla", knight: "Cavaleiro", archers: "Arqueiras",
    skeletons: "Esqueletos", fireball: "Bola de Fogo", log: "O Tronco", electrospirit: "Espírito Elétrico",
    icespirits: "Espírito de Gelo", icewizard: "Mago de Gelo", rocket: "Foguete", tornado: "Tornado",
    hogrider: "Corredor", musketeer: "Mosqueteira", cannon: "Canhão", icegolemite: "Golem de Gelo",
    royalgiant: "Gigante Real", hunter: "Caçador", fisherman: "Pescador", ghost: "Fantasma Real",
    lightning: "Relâmpago", graveyard: "Cemitério", poison: "Veneno", babydragon: "Bebê Dragão",
    tombstone: "Lápide", barblog: "Barril de Bárbaro", megaminion: "Megasservo",
    angrybarbarians: "Bárbaros de Elite", goblins: "Goblins", valkyrie: "Valquíria",
    balloon: "Balão", infernotower: "Torre Inferno", bats: "Morcegos", miner: "Mineiro",
    mortar: "Morteiro", witch: "Bruxa"};
  const name = card => cards[cardKey(card)] || card;
  const percent = new Intl.NumberFormat("pt-BR", { style: "percent", maximumFractionDigits: 2 });
  const decimal = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 3 });
  const signedPoints = new Intl.NumberFormat("pt-BR", {maximumFractionDigits: 2, signDisplay: "exceptZero"});
  let context = {};
  let generation = 0;
  let requesting = false;

  function clock(ms) {
    const seconds = Math.floor(ms / 1000);
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  }

  function rival(name) {
    return rivals[name] || name?.replace(/^snapshot_/, "Versão anterior ") || "Rival";
  }

  function winSummary(rows, model) {
    const tally = rows.reduce((sum, row) => {
      const result = model === "rival" ? row.free : row;
      sum.games += result.games;
      sum.wins += result.wins;
      return sum;
    }, { games: 0, wins: 0 });
    return { ...tally, rate: tally.games ? tally.wins / tally.games : null };
  }

  function renderTrends(live) {
    const history = (live?.updates || []).filter(row => row.games > 0 && row.free?.games > 0).slice(-80)
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
      card.querySelector("[data-rate]").textContent = current.rate === null ? "—" : percent.format(current.rate);
      delta.textContent = before.rate === null ? "" : `${signedPoints.format(100 * (current.rate - before.rate))} p.p.`;
      delta.dataset.direction = current.rate > before.rate ? "up" : current.rate < before.rate ? "down" : "flat";
      card.querySelector("[data-record]").textContent = current.games
        ? `${current.wins} de ${current.games} vitórias · atualizações até ${model === "rival" ? recent.at(-1).free.update : recent.at(-1).update}`
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

  function paintLiveBoard(tile) {
    const { header, frames: snapshots } = tile.snapshot;
    const frame = snapshots.at(-1);
    const renderer = tile.renderer;
    if (!frame || !renderer.begin(header)) return;
    const casts = snapshots.flatMap(snapshot => snapshot.moves.map(move => spellCast({ ...move, ms: snapshot.ms },
      { arena: header, pieces: snapshot.pieces }))).filter(cast => cast && castVisible(cast, frame.ms));
    for (const cast of casts) renderer.drawCastArea(cast, frame.ms, header.height);
    renderer.drawPieces(frame.pieces, header.height, piece => [piece[3], piece[4]]);
    for (const cast of casts) renderer.drawCast(cast, frame.ms, header.height);
  }

  function renderLive({active, boards}) {
    const ids = new Set(boards.map(board => board.header.id));
    for (const [id, tile] of liveTiles) {
      if (ids.has(id)) continue;
      tile.card.remove();
      liveTiles.delete(id);
    }
    for (const snapshot of boards) {
      const { header, frames: snapshots } = snapshot;
      let tile = liveTiles.get(header.id);
      if (!tile) {
        const card = document.createElement("article");
        card.className = "arena-live-tile";
        card.innerHTML = `<h4></h4><small></small><div class="arena-video"><canvas role="img"></canvas><time>0:00</time></div>
          <div class="arena-live-teams"><span></span><span></span></div>`;
        const canvas = card.querySelector("canvas");
        const title = label(header);
        card.querySelector("h4").textContent = title;
        canvas.setAttribute("aria-label", title);
        tile = { card, snapshot, renderer: createRenderer(canvas, () => paintLiveBoard(tile)) };
        liveTiles.set(header.id, tile);
        el("live-grid").append(card);
      }
      const oldFrame = tile.snapshot.frames.at(-1);
      const frame = snapshots.at(-1);
      const last = snapshots.findLast(item => item.moves.length)?.moves.at(-1);
      const prediction = frame?.comparison?.model;
      const current = prediction ? `Modelo · ${prediction.card ? name(prediction.card) : "Esperar"}`
        : last?.card ? `Ao vivo · ${name(last.card)}` : "Ao vivo";
      tile.card.querySelector("small").textContent = current;
      tile.card.querySelector("time").textContent = frame ? clock(frame.ms) : "0:00";
      tile.card.querySelectorAll(".arena-live-teams span").forEach((side, seat) => {
        const own = header.seat === seat;
        const teamName = header.joint ? (own ? "X-Bow 3.0" : "Rival Live") : (own ? "Modelo" : "Rival");
        side.textContent = frame ? `${teamName} · ${decimal.format(frame.sides[seat].elixir)}` : teamName;
        side.style.setProperty("--side-color", tones[seat]);
      });
      tile.snapshot = snapshot;
      if (frame?.seq !== oldFrame?.seq || !tile.painted) {
        paintLiveBoard(tile);
        tile.painted = Boolean(frame);
      }
    }
    const running = boards.length;
    el("source").textContent = active ? "Transmissão ao vivo" : "Nenhum treino em andamento";
    el("live-status").textContent = active ? `${context.detail || "Preparando o treino"}${running ? ` · ${running} ao vivo` : ""}` : "";
    el("live-empty").hidden = Boolean(boards.length);
    el("live-empty").textContent = active ? "Aguardando os tabuleiros desta etapa." : "Nenhum treino em andamento. Inicie um treino no painel para acompanhar as partidas.";
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

  async function poll() {
    if (requesting || !context.active) return;
    requesting = true;
    const version = generation;
    const {deck, run, phase} = context;
    const params = new URLSearchParams({stage: "live", deck, run, phase});
    try {
      const response = await fetch('/api/arena?' + params, {signal: AbortSignal.timeout(4000)});
      if (!response.ok) throw new Error();
      const data = await response.json();
      if (version === generation) renderLive(data);
    } catch {
      if (version === generation) {
        renderLive({active: true, boards: []});
        el("live-status").textContent = "Aguardando conexão com o painel";
      }
    } finally {
      requesting = false;
      if (version !== generation) void poll();
    }
  }

  window.trainingArena = {
    update(selection) {
      const changed = context.run !== selection.run || context.phase !== selection.phase ||
        context.active !== selection.active;
      context = selection;
      renderTrends(context);
      if (changed) {
        generation += 1;
        renderLive({active: context.active, boards: []});
        void poll();
      }
    },
  };
  new ResizeObserver(() => liveTiles.forEach(paintLiveBoard)).observe(el("live-grid"));
  setInterval(poll, 900);
})();
