"use strict";

const byId = id => document.getElementById(id);
const number = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 0 });
const decimal = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const percent = new Intl.NumberFormat("pt-BR", { style: "percent", maximumFractionDigits: 1 });
const clock = new Intl.DateTimeFormat("pt-BR", { timeStyle: "medium" });
const deckNames = { xbow_30: "X-Bow 3.0", icebow: "IceBow" };
const rivalNames = {
  bc: "Modelo de imitação do mesmo deck", other_bc: "Modelo de imitação do outro deck",
  scripted: "X-Bow programado", hog_push: "Corredor", rg_patient: "Gigante Real",
  random: "Rival aleatório",
};
const groupNames = {
  ".": "Inicial", calibrated: "Calibrado", tactical: "Posicionamento por carta",
  public: "Observação pública", "resume-check": "Teste de retomada",
};
const empty = message => `<p class="empty">${escapeText(message)}</p>`;
const ratio = row => row?.games > 0 && Number.isFinite(row.wins) ? row.wins / row.games : null;
const show = (value, format = number) => Number.isFinite(value) ? format.format(value) : "—";
const kl = value => Number.isFinite(value) ? value.toFixed(4).replace(".", ",") : "—";
let data = { runs: [], job: null };
let deck = "xbow_30";
let current = "";
let pending = "";
let revision = "";
let series = [];
let epochs = [];
let points = [];
let connected = false;
let booted = false;
let sending = false;
let lastJobId = "";
let chartJobId = "";
let actionVersion = 0;

function escapeText(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[character]));
}
function selected() {
  return data.runs.find(run => run.id === current);
}
function updateRuns() {
  const runs = data.runs.filter(run => run.deck === deck);
  if (pending) {
    if (runs.some(run => run.id === pending)) {
      current = pending;
      pending = "";
    } else if (!activeJob(data.job)) {
      pending = "";
    }
  }
  if (!pending && !runs.some(run => run.id === current)) current = runs[0]?.id || "";
  const picker = byId("run");
  const options = runs.map(run => [run.id, groupNames[run.group] || run.group]);
  if (pending) options.unshift([pending, "Preparando experimento…"]);
  const optionKey = JSON.stringify(options);
  if (picker.dataset.options !== optionKey) {
    picker.replaceChildren(...(options.length ?
      options.map(([value, label]) => new Option(label, value)) :
      [new Option("Sem experimentos", "")]));
    picker.dataset.options = optionKey;
  }
  picker.value = current;
  picker.disabled = options.length === 0;
  for (const button of document.querySelectorAll("[data-deck]")) {
    button.setAttribute("aria-pressed", String(button.dataset.deck === deck));
  }
}
function phase(run) {
  const info = run?.phase || {};
  const names = {
    started: "Início do processo", stopped: "Execução interrompida", failed: "Falha na execução",
    finished: "Execução concluída", replay_validation: "Validação em replays",
    replays: "Reconstrução de replays", bc_start: "Imitação", bc_batch: "Imitação", bc_epoch: "Imitação",
    ppo_start: "Self-play", ppo_game: "Self-play", ppo_update: "Self-play",
    evaluation_start: "Avaliação na arena", evaluation_pair: "Avaliação na arena",
  };
  let details = "";
  if (info.event === "ppo_game") {
    details = `Atualização ${info.update}/${info.total} · partida ${info.done}/${info.games}`;
  } else if (info.event === "bc_batch") {
    details = `Época ${info.epoch}/${info.total} · lote ${info.done}/${info.batches}`;
  } else if (Number.isFinite(info.done) && Number.isFinite(info.total)) {
    details = `${info.done}/${info.total} concluídos`;
  }
  if (info.event === "failed") details = info.message;
  return names[info.event] ?
    `<strong>${names[info.event]}</strong><span>${escapeText(details || (info.time ? clock.format(new Date(info.time)) : ""))}</span>` : "";
}
function rivalRows(evaluation) {
  const pairs = new Map();
  for (const [name, result] of Object.entries(evaluation || {})) {
    const match = /^(bc|ppo)_vs_(.+)$/.exec(name);
    if (!match || ratio(result) === null) continue;
    const [, model, rival] = match;
    if (!pairs.has(rival)) pairs.set(rival, { rival });
    pairs.get(rival)[model] = result;
  }
  return [...pairs.values()];
}
function overview(run, pairs) {
  const common = pairs.filter(row => row.bc && row.ppo && row.bc.games === row.ppo.games);
  const comparison = common.length > 0;
  const summary = model => {
    const items = (comparison ? common : pairs).map(row => row[model]).filter(Boolean);
    return items.reduce((sum, row) => ({ wins: sum.wins + row.wins, games: sum.games + row.games }),
      { wins: 0, games: 0 });
  };
  const bc = summary("bc"), ppo = summary("ppo");
  const side = (label, model, row) => `<div class="score-side"><span class="score-value ${model}">${show(ratio(row), percent)}</span>
    <small>${label}${row.games ? ` · ${row.wins}/${row.games}` : ""}</small></div>`;
  byId("score").innerHTML = side("Imitação", "bc", bc) + '<span class="score-separator" aria-hidden="true"></span>' + side("PPO", "ppo", ppo);
  const delta = comparison ? 100 * (ratio(ppo) - ratio(bc)) : null;
  byId("comparison-delta").innerHTML = delta === null ? "" :
    `<span class="delta ${delta > 0 ? "positive" : delta < 0 ? "negative" : ""}">${delta > 0 ? "+" : ""}${show(delta, decimal)} p.p.</span>`;
  byId("samples").textContent = comparison ?
    `${common.length} rivais em comum · ${bc.games} partidas por modelo` :
    (pairs.length ? "Comparação pareada indisponível para esta avaliação." : "Aguardando uma avaliação na arena.");
  const updates = run?.updates || [];
  const last = updates.at(-1);
  const completedGames = updates.reduce((sum, row) => sum + (row.games || 0), 0);
  const tiles = [
    ["Atualizações PPO", show(last?.update), "Última atualização registrada"],
    ["Partidas de treino", updates.length ? show(completedGames) : "—", "Nos registros de self-play"],
    ["Conexões na torre", show(last?.mean_locks, decimal), "Média por partida · última atualização"],
    ["Divergência KL", kl(last?.kl_bc_to_ppo), "Imitação → PPO · última atualização"],
  ];
  byId("metrics").innerHTML = tiles.map(([label, value, hint]) =>
    `<article class="metric"><span class="name">${label}</span><strong class="number">${value}</strong><small>${hint}</small></article>`).join("");
}
function table(headers, rows) {
  return `<div class="table-wrap"><table><thead><tr>${headers.map(h => `<th scope="col">${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(([heading, ...values]) => `<tr><th scope="row">${escapeText(heading)}</th>${values.map(value => `<td>${value}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}
function matchups(pairs) {
  const bar = (row, model) => row ? `<div class="bar-cell ${model}">${row.wins}/${row.games} <span class="muted">· ${show(ratio(row), percent)}</span>
    <div class="mini-track" aria-hidden="true"><span class="mini-bar" style="width:${Math.min(100, Math.max(0, ratio(row) * 100))}%"></span></div></div>` : "—";
  byId("matchups").innerHTML = pairs.length ? table(["Rival", "Imitação", "PPO"], pairs.map(row => [
    rivalNames[row.rival] || row.rival, bar(row.bc, "bc"), bar(row.ppo, "ppo"),
  ])) : empty("Aguardando os resultados contra os rivais.");
}
function validation(run) {
  const val = run?.validation || {};
  const lastEpoch = run?.epochs.at(-1);
  const bc = val.bc || lastEpoch;
  const ppo = val.ppo;
  const matches = run?.split.validation;
  byId("validation-note").textContent = bc ?
    `${Number.isFinite(matches) ? `${show(matches)} partidas reservadas · ` : ""}${show(bc.frames)} quadros · ${show(bc.card_plays)} jogadas de carta` :
    "Resultados nas partidas reservadas.";
  const rows = [
    ["Perda de validação (NLL)", show(bc?.nll, decimal), show(ppo?.nll, decimal), "Menor é melhor"],
    ["Escolha da carta", show(bc?.conditional_card_accuracy, percent), show(ppo?.conditional_card_accuracy, percent), "Quando a pessoa jogou"],
    ["Carta e casa exatas", show(bc?.conditional_card_and_tile_accuracy, percent), show(ppo?.conditional_card_and_tile_accuracy, percent), "Quando a pessoa jogou"],
    ["Probabilidade média de jogar", show(bc?.mean_play_probability, percent), show(ppo?.mean_play_probability, percent), `Humano: ${show(bc?.expert_play_rate, percent)}`],
  ];
  const proximity = (model, card) => {
    const row = model?.within_two_tiles_given_card?.[card];
    return row ? `${show(row.correct)}/${show(row.examples)}` : "—";
  };
  for (const [key, label] of [["x-bow", "X-Bow"], ["tesla", "Tesla"], ["one_elixir", "Cartas de 1 elixir"]]) {
    if (bc?.within_two_tiles_given_card?.[key] || ppo?.within_two_tiles_given_card?.[key]) {
      rows.push([`${label} · até 2 casas`, proximity(bc, key), proximity(ppo, key), "Com a carta indicada"]);
    }
  }
  byId("validation-result").innerHTML = bc ?
    table(["Métrica", "Imitação", "PPO", "Leitura"], rows) : empty("Aguardando os resultados da validação em replays.");
}
const charts = {
  win: { value: ratio, format: percent, label: "Taxa de vitórias" },
  locks: { value: row => row.mean_locks, format: decimal, label: "Conexões na torre" },
  reward: { value: row => row.mean_reward, format: decimal, label: "Recompensa média" },
  kl: { value: row => row.kl_bc_to_ppo, format: { format: kl }, label: "Divergência KL" },
  bc: { value: row => row.nll, format: decimal, label: "Perda de imitação (NLL)" },
};
function drawChart() {
  const canvas = byId("chart");
  const metric = byId("chart-metric").value;
  const spec = charts[metric];
  const isClone = metric === "bc";
  const rows = (isClone ? epochs : series).map(row => [isClone ? row.epoch : row.update, spec.value(row)])
    .filter(([, value]) => Number.isFinite(value));
  byId("chart-caption").textContent = isClone ? "Perda de validação por época de imitação" : "Resultados por atualização do PPO";
  byId("chart-empty").textContent = isClone ? "Aguardando a primeira época de imitação."
    : "Aguardando a primeira atualização de self-play.";
  byId("chart-empty").hidden = rows.length !== 0;
  byId("history").hidden = !rows.length;
  byId("history-table").innerHTML = isClone ?
    table(["Época", "Perda de validação", "Escolha da carta"], epochs.map(row =>
      [row.epoch, show(row.nll, decimal), show(row.conditional_card_accuracy, percent)])) :
    table(["Atualização", "Vitórias", "Jogos", "Conexões", "Recompensa", "KL"],
      series.map(row => [row.update, show(row.wins), show(row.games), show(row.mean_locks, decimal),
        show(row.mean_reward, decimal), kl(row.kl_bc_to_ppo)]));
  canvas.setAttribute("aria-label", `${spec.label}. Os valores estão em Ver dados do gráfico.`);
  byId("plot-tooltip").hidden = true;
  points = [];
  const width = canvas.clientWidth, height = canvas.clientHeight, scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  if (!rows.length || width < 130) return;
  const left = 57, right = width - 16, top = 13, bottom = height - 30;
  const values = rows.map(([, value]) => value);
  const min = metric === "win" ? 0 : Math.min(0, ...values);
  const max = metric === "win" ? 1 : Math.max(metric === "kl" ? 0.001 : 0.1, ...values) * 1.08;
  ctx.font = '11px "Segoe UI", sans-serif';
  const chartX = index => rows.length === 1 ? (left + right) / 2 : left + (right - left) * index / (rows.length - 1);
  const chartY = value => bottom - (bottom - top) * (value - min) / (max - min);
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  for (let tick = 0; tick <= 4; tick++) {
    const value = min + (max - min) * tick / 4;
    const y = chartY(value);
    ctx.strokeStyle = "#303b4c";
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    ctx.fillStyle = "#a5b4c8";
    ctx.fillText(show(value, spec.format), left - 8, y + 4);
  }
  points = rows.map(([update, value], index) => ({ update, value, x: chartX(index), y: chartY(value) }));
  const grad = ctx.createLinearGradient(0, top, 0, bottom);
  grad.addColorStop(0, "#a99aff46");
  grad.addColorStop(1, "#a99aff03");
  ctx.beginPath(); ctx.moveTo(points[0].x, bottom);
  for (const point of points) ctx.lineTo(point.x, point.y);
  ctx.lineTo(points.at(-1).x, bottom); ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
  ctx.beginPath();
  points.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
  ctx.lineWidth = 2; ctx.strokeStyle = "#b3a2fa"; ctx.stroke();
  const labelEvery = Math.ceil(points.length / Math.max(2, Math.floor((right - left) / 44)));
  ctx.textAlign = "center";
  points.forEach((point, index) => {
    ctx.fillStyle = "#b3a2fa"; ctx.beginPath(); ctx.arc(point.x, point.y, 3, 0, Math.PI * 2); ctx.fill();
    if (index % labelEvery === 0 || index === points.length - 1) {
      ctx.fillStyle = "#a5b4c8"; ctx.fillText(String(point.update), point.x, bottom + 17);
    }
  });
}
const activeJob = job => ["starting", "running", "stopping"].includes(job?.status);
const jobStatus = { starting: "Iniciando", running: "Em treinamento", stopping: "Parada solicitada",
  finished: "Concluído", stopped: "Parado", failed: "Falhou", interrupted: "Interrompido" };
const parameterKeys = ["matches", "epochs", "updates", "games", "evaluate"];
const replaySelection = { xbow_30: 48, icebow: 36 };

function executionMode() {
  return byId("training-mode").valueAsNumber === 1 ? "resume" : "new";
}

function syncSlider(slider, available = true) {
  const span = Number(slider.max) - Number(slider.min);
  const proportion = span > 0 ? (slider.valueAsNumber - Number(slider.min)) / span : 0;
  slider.style.setProperty("--slider-progress", `${proportion * 100}%`);
  const isMode = slider.id === "training-mode";
  const label = isMode ? (executionMode() === "resume" ? "Retomar PPO" : "Novo treino")
    : show(available ? slider.valueAsNumber : null);
  slider.labels[0].querySelector(".slider-value").textContent = label;
  if (isMode) slider.setAttribute("aria-valuetext", label);
}

function syncReplayLimit(resume) {
  const count = data.inventory?.counts?.[deck];
  const ready = count >= 2;
  const slider = byId("training-matches");
  slider.max = Math.max(2, count ?? 2);
  if (ready) {
    replaySelection[deck] = Math.min(replaySelection[deck], count);
    slider.value = replaySelection[deck];
  }
  slider.disabled = resume || !ready;
  byId("training-replay-limit").textContent = show(count);
  syncSlider(slider, ready);
  return ready;
}

function stageProgress(job, info) {
  const event = info.event || "";
  if (event.startsWith("replays")) {
    return ["replays", "Reconstrução de replays", `${info.done}/${info.total} replays · ${info.rejected || 0} descartados`, info.done, info.total];
  }
  if (event === "bc_batch") {
    return ["bc", "Imitação", `Época ${info.epoch}/${info.total} · lote ${info.done}/${info.batches} · perda ${show(info.loss, decimal)}`,
      info.epoch - 1 + info.done / info.batches, info.total];
  }
  if (event.startsWith("bc")) {
    return ["bc", "Imitação", `${info.done || 0}/${info.total} épocas concluídas`, info.done || 0, info.total];
  }
  if (event === "ppo_game") {
    return ["ppo", "Self-play", `Atualização ${info.update}/${info.total} · partida ${info.done}/${info.games}`,
      info.update - 1 + info.done / (info.games + 1), info.total];
  }
  if (event.startsWith("ppo")) {
    return ["ppo", "Self-play", `${info.done}/${info.total} atualizações concluídas`, info.done, info.total];
  }
  if (event.startsWith("evaluation")) {
    return ["evaluation", "Avaliação na arena", `${info.done}/${info.total} confrontos por modelo e rival`, info.done, info.total];
  }
  return [job.mode === "resume" ? "ppo" : "replays", "Preparando o treino", "Carregando o ambiente", null, null];
}

function renderTraining() {
  const run = selected();
  const job = data.job;
  const active = activeJob(job);
  const mode = byId("training-mode");
  const canResume = Boolean(run?.checkpoints.bc && run?.checkpoints.ppo);
  if (!canResume) mode.value = "0";
  mode.disabled = !canResume;
  const resume = executionMode() === "resume";
  for (const field of document.querySelectorAll("[data-new]")) {
    field.hidden = resume;
    field.querySelector("input").disabled = field.hidden;
  }
  syncSlider(mode);
  const canTrain = syncReplayLimit(resume);
  for (const key of parameterKeys.slice(1)) syncSlider(byId(`training-${key}`));
  byId("training-deck").textContent = `Deck selecionado · ${deckNames[active ? job.deck : deck]}`;
  byId("training-form").hidden = active;
  byId("training-settings").disabled = sending;
  byId("start-training").disabled = sending || !connected || (!resume && !canTrain);
  byId("start-training").textContent = sending ? "Iniciando…" :
    resume ? "Retomar treino" : "Iniciar treino";
  let hint = "O máximo conta os replays candidatos do deck. O simulador pode aceitar menos.";
  if (resume) {
    hint = `Continua o experimento ${groupNames[run.group] || run.group} a partir do último checkpoint.`;
  } else if (!canTrain) {
    const messages = {
      ready: "Este deck precisa de pelo menos 2 replays candidatos.",
      unavailable: "A contagem dos replays está indisponível. O portal vai tentar novamente.",
    };
    hint = messages[data.inventory?.status] || "Contando os replays candidatos do deck…";
  }
  byId("training-hint").textContent = hint;
  const badge = byId("job-badge");
  badge.textContent = jobStatus[job?.status] || "Pronto";
  badge.dataset.state = job?.status || "idle";
  byId("job-monitor").hidden = !job;
  if (!job) return;
  const tracked = data.runs.find(item => item.id === job.run_id);
  const progress = tracked?.progress;
  const duringJob = progress && Date.parse(progress.time) >= Date.parse(job.started);
  const [stage, title, detail, done, total] = stageProgress(job, duringJob ? progress : {});
  const stages = ["replays", "bc", "ppo", "evaluation"];
  for (const marker of byId("training-stages").children) {
    const key = marker.dataset.stage;
    marker.hidden = (job.mode === "resume" && ["replays", "bc"].includes(key)) ||
      (key === "evaluation" && job.parameters.evaluate === 0);
    const completed = job.status === "finished" || stages.indexOf(key) < stages.indexOf(stage);
    marker.classList.toggle("completed", completed);
    if (active && key === stage) marker.setAttribute("aria-current", "step");
    else marker.removeAttribute("aria-current");
  }
  byId("job-stage").textContent = active ? title : jobStatus[job.status];
  byId("job-detail").textContent = job.status === "stopping" ?
    "Aguardando o fim do lote ou da partida atual." : active ? detail : job.run_id;
  const bar = byId("job-progress");
  bar.hidden = !active && job.status !== "finished";
  if (job.status === "finished") { bar.max = 1; bar.value = 1; }
  else if (Number.isFinite(done) && total > 0) { bar.max = total; bar.value = done; }
  else { bar.removeAttribute("value"); bar.max = 1; }
  const stop = byId("stop-job");
  stop.hidden = !active;
  stop.disabled = sending || !connected || job.status === "stopping";
  stop.textContent = job.status === "stopping" ? "Parada solicitada" : "Parar treino";
  byId("view-job").hidden = !tracked || current === job.run_id;
  const log = byId("job-log");
  const lines = job.log?.join("\n") || "Aguardando o primeiro registro…";
  if (log.textContent !== lines) {
    const atBottom = log.scrollHeight - log.clientHeight - log.scrollTop < 24;
    log.textContent = lines;
    if (atBottom) log.scrollTop = log.scrollHeight;
  }
  if (lastJobId !== job.id) {
    byId("training-log").open = active || job.status === "failed";
    lastJobId = job.id;
  } else if (job.status === "failed") {
    byId("training-log").open = true;
  }
}

function follow(job) {
  deck = job.deck;
  pending = job.run_id;
  current = job.run_id;
  const trained = selected()?.updates.length;
  byId("chart-metric").value = job.mode === "resume" || trained ? "win" : "bc";
  chartJobId = trained ? job.id : "";
}

function trainingError(message = "") {
  const box = byId("training-error");
  box.hidden = !message;
  box.textContent = message;
}

async function jobAction(action, payload) {
  actionVersion += 1;
  sending = true;
  trainingError();
  renderTraining();
  try {
    const response = await fetch(`/api/job/${action}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "A solicitação não foi concluída.");
    data.job = result.job;
    if (action === "start") follow(result.job);
  } catch (error) {
    trainingError(error.message);
  } finally {
    actionVersion += 1;
    sending = false;
    render(true);
  }
}
function render(force = false) {
  updateRuns();
  const run = selected();
  renderTraining();
  const job = data.job;
  const watchingJob = job && job.run_id === current;
  const viewPhase = watchingJob ? (run?.progress.event || "") : "";
  const stage = ["replays", "bc", "ppo", "evaluation"].find(key => viewPhase.startsWith(key));
  window.trainingArena.update({
    deck, run: current, active: watchingJob && activeJob(job),
    phase: stage || (watchingJob && job.mode === "resume" ? "ppo" : "replays"),
  });
  const next = JSON.stringify([run, deck]);
  if (!force && revision === next) return;
  revision = next;
  const pairs = rivalRows(run?.evaluation);
  byId("status-panel").innerHTML = phase(run);
  overview(run, pairs);
  matchups(pairs);
  validation(run);
  series = run?.updates || [];
  epochs = run?.epochs || [];
  const metric = byId("chart-metric");
  metric.options[4].disabled = !epochs.length && Boolean(series.length);
  if (metric.options[4].disabled && metric.value === "bc") metric.value = "win";
  if (series.length && current === data.job?.run_id && chartJobId !== data.job.id) {
    if (metric.value === "bc") metric.value = "win";
    chartJobId = data.job.id;
  }
  drawChart();
}
async function refresh() {
  const seenAction = actionVersion;
  try {
    const response = await fetch("/api/status", { cache: "no-store", signal: AbortSignal.timeout(4000) });
    if (!response.ok) throw new Error("Painel indisponível");
    const incoming = await response.json();
    if (sending || seenAction !== actionVersion) incoming.job = data.job;
    data = incoming;
    connected = true;
    if (!booted && activeJob(data.job)) follow(data.job);
    booted = true;
    byId("sync").textContent = "Sincronizado";
    document.querySelector(".connection").classList.remove("offline");
    byId("dataset").textContent = `${show(data.parts)} partições locais`;
    byId("updated").textContent = `Atualizado às ${clock.format(new Date(data.refreshed))}`;
    render();
  } catch {
    connected = false;
    renderTraining();
    byId("sync").textContent = "Sem conexão";
    document.querySelector(".connection").classList.add("offline");
  } finally {
    setTimeout(refresh, 2500);
  }
}
byId("run").addEventListener("change", event => { current = event.target.value; pending = ""; render(true); });
document.querySelector(".deck-switch").addEventListener("click", event => {
  const next = event.target.closest("[data-deck]")?.dataset.deck;
  if (!next || next === deck) return;
  deck = next;
  pending = "";
  render(true);
});
byId("chart-metric").addEventListener("change", drawChart);
byId("training-settings").addEventListener("input", event => {
  const slider = event.target;
  if (slider.id === "training-mode") {
    renderTraining();
  } else {
    if (slider.id === "training-matches") replaySelection[deck] = slider.valueAsNumber;
    syncSlider(slider);
  }
});
byId("training-form").addEventListener("submit", event => {
  event.preventDefault();
  if (byId("start-training").disabled || activeJob(data.job)) return;
  const payload = { deck, mode: executionMode(), run_id: current };
  for (const key of parameterKeys) {
    payload[key] = byId(`training-${key}`).valueAsNumber;
  }
  jobAction("start", payload);
});
byId("stop-job").addEventListener("click", () => jobAction("stop", { id: data.job.id }));
byId("view-job").addEventListener("click", () => { follow(data.job); render(true); });
byId("chart").addEventListener("pointermove", event => {
  if (!points.length) return;
  const point = points.reduce((nearest, next) =>
    Math.abs(next.x - event.offsetX) < Math.abs(nearest.x - event.offsetX) ? next : nearest);
  const tooltip = byId("plot-tooltip");
  tooltip.hidden = false;
  const metric = byId("chart-metric").value;
  tooltip.textContent = `${metric === "bc" ? "Época" : "Atualização"} ${point.update} · ${show(point.value, charts[metric].format)}`;
  tooltip.style.left = `${Math.max(4, Math.min(point.x + 6, byId("plot").clientWidth - tooltip.offsetWidth - 4))}px`;
});
byId("chart").addEventListener("pointerleave", () => { byId("plot-tooltip").hidden = true; });
document.querySelector("nav").addEventListener("click", event => {
  const link = event.target.closest("a");
  if (!link) return;
  for (const existing of document.querySelectorAll("nav a")) existing.removeAttribute("aria-current");
  link.setAttribute("aria-current", "location");
});
new ResizeObserver(drawChart).observe(byId("plot"));
render();
refresh();
