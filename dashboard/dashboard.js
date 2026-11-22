"use strict";

const byId = id => document.getElementById(id);
const number = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 0 });
const decimal = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const decimal1 = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const percent = new Intl.NumberFormat("pt-BR", { style: "percent", maximumFractionDigits: 1 });
const clock = new Intl.DateTimeFormat("pt-BR", { timeStyle: "medium" });
const deckNames = { xbow_30: "X-Bow 3.0", icebow: "IceBow" };
const rivalNames = {
  bc: "Modelo de imitação (mesmo deck)", other_bc: "Modelo de imitação (outro deck)",
  scripted: "X-Bow programado", hog_push: "Corredor (bot)", rg_patient: "Gigante Real (bot)",
  random: "Rival aleatório", free: "IA de deck livre",
  hog_26: "Corredor 2.6", royal_giant: "Gigante Real", splashyard: "Splashyard",
  log_bait: "Log Bait", golem_beatdown: "Golem Beatdown", pekka_bs: "P.E.K.K.A Bridge Spam",
};
const groupNames = {
  ".": "Inicial", calibrated: "Calibrado", tactical: "Posicionamento por carta",
  public: "Observação pública", "resume-check": "Teste de retomada",
};

const PAGES = {
  visao: {
    title: "Visão geral",
    breadcrumb: "Visão geral",
    subtitle: "Resumo executivo de força na arena, saúde do PPO e qualidade do modelo.",
    toolbar: true,
  },
  evolucao: {
    title: "Evolução & Telemetria da IA",
    breadcrumb: "Evolução da IA",
    subtitle: "Curvas de aprendizado, diagnóstico do cérebro PPO, liga PFSP e validação em replays.",
    toolbar: true,
  },
  treino: {
    title: "Treinamento assistido",
    breadcrumb: "Treinamento",
    subtitle: "Configure visão, batch size, pool de processos no Xeon e acompanhe o treino.",
    toolbar: true,
  },
  arena: {
    title: "Telemetria do treino",
    breadcrumb: "Telemetria do treino",
    subtitle: "Partidas acumuladas, velocidade real (jogos/h), saúde do PPO e desempenho por deck rival.",
    toolbar: true,
  },
  jogar: {
    title: "Jogar contra a IA",
    breadcrumb: "Jogar contra a IA",
    subtitle: "Desafie os modelos treinados diretamente no tabuleiro interativo.",
    toolbar: false,
  },
  "ao-vivo": {
    title: "Jogo real & Aprendizado contínuo",
    breadcrumb: "Jogo real & Overlay",
    subtitle: "Overlay tático no MuMuPlayer / scrcpy e fine-tuning AWR com suas partidas reais.",
    toolbar: true,
  },
  videos: {
    title: "Vídeos do Royale TV",
    breadcrumb: "Vídeos Royale TV",
    subtitle: "Biblioteca AlphaClash para estudar partidas reais de X-Bow.",
    toolbar: false,
  },
};

const LEGACY_HASH_MAP = {
  "#rivais": { page: "evolucao", anchor: "rivais" },
  "#validacao": { page: "evolucao", anchor: "validacao" },
};

const empty = message => `<p class="empty">${escapeText(message)}</p>`;
const ratio = row => row?.games > 0 && Number.isFinite(row.wins) ? row.wins / row.games : null;
const show = (value, format = number) => Number.isFinite(value) ? format.format(value) : "—";
const kl = value => Number.isFinite(value) ? value.toFixed(4).replace(".", ",") : "—";
const signed = (value, format = decimal1) => {
  if (!Number.isFinite(value)) return "—";
  return `${value > 0 ? "+" : ""}${format.format(value)}`;
};

function rivalLabel(name) {
  if (!name) return "—";
  if (rivalNames[name]) return rivalNames[name];
  if (name.startsWith("snapshot_")) return `Snapshot #${name.slice(9)}`;
  return name;
}

let data = { runs: [], job: null, live_matches: null };
let deck = "xbow_30";
let current = "";
let pending = "";
let revision = "";
let activePage = "visao";
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

/* ==================== ROUTER DE PÁGINAS ==================== */
function resolveRoute(rawHash) {
  const hash = rawHash || "#visao";
  if (hash === "#cartas-decks") return { redirect: "/cartas-decks" };
  if (LEGACY_HASH_MAP[hash]) return LEGACY_HASH_MAP[hash];
  const key = hash.replace(/^#/, "");
  if (PAGES[key]) return { page: key, anchor: null };
  return { page: "visao", anchor: null };
}

function applyRoute() {
  const route = resolveRoute(location.hash);
  if (route.redirect) {
    location.replace(route.redirect);
    return;
  }
  activePage = route.page;
  const meta = PAGES[activePage];
  byId("page-title").textContent = meta.title;
  byId("page-subtitle").textContent = meta.subtitle;
  byId("breadcrumb-page").textContent = meta.breadcrumb;
  byId("global-toolbar").hidden = !meta.toolbar;
  byId("page-actions").hidden = activePage === "jogar";
  document.title = `${meta.breadcrumb} · X-Bow Control`;

  for (const view of document.querySelectorAll(".page-view")) {
    view.hidden = view.dataset.page !== activePage;
  }
  for (const link of document.querySelectorAll(".sidebar nav a")) {
    if (link.dataset.nav === activePage) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  renderGlobalJobBar();

  if (route.anchor) {
    requestAnimationFrame(() => {
      byId(route.anchor)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  } else {
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  requestAnimationFrame(() => {
    if (activePage === "evolucao") drawChart();
    if (activePage === "ao-vivo") refreshLiveStatus();
    window.dispatchEvent(new Event("resize"));
  });
}

/* ==================== SELEÇÃO DE EXPERIMENTO ==================== */
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

/* ==================== FASE & PROGRESSO ==================== */
function simultaneousGames(info) {
  return info.parallel_games > 1 ? ` · até ${info.parallel_games} simultâneas` : "";
}

function ppoProgress(info, total = info.total) {
  const title = info.joint ? `X-Bow 3.0 × ${rivalLabel(info.rival)}` : "Self-play";
  if (info.continuous) {
    const done = info.completed_games ?? (info.event === "ppo_game" ? info.update - 1 : info.done) - info.session_start;
    let detail = `${show(done)} partidas nesta sessão`;
    if (info.event === "ppo_match") detail += info.parallel_games > 1
      ? ` · Simulando ${info.parallel_games} em paralelo` : " · Simulando a próxima";
    if (info.event === "ppo_game") detail += " · Atualizando as duas IAs";
    return ["ppo", title, detail, done, null];
  }
  const checkpoint = info.total - total;
  const update = info.update - checkpoint;
  const done = info.done - checkpoint;
  const unit = info.joint ? "Partida" : "Atualização";
  const position = `${unit} ${update}/${total}`;
  if (info.event === "ppo_match") {
    const games = `${info.games} ${info.games === 1 ? "partida" : "partidas"}`;
    const simulation = info.joint ? "Jogando" : `Simulando ${games}${simultaneousGames(info)}`;
    return ["ppo", title, `${position} · ${simulation}`, done, total];
  }
  if (info.event === "ppo_game") {
    let detail = `${info.done}/${info.games} partidas simuladas${simultaneousGames(info)}`;
    if (info.joint) detail = "Atualizando as duas IAs";
    else if (info.done === info.games) detail = "Otimizando o PPO";
    return ["ppo", title, `${position} · ${detail}`, update - 1 + info.done / (info.games + 1), total];
  }
  const completed = info.joint ? "partidas concluídas" : "atualizações concluídas";
  const origin = info.event === "ppo_start" && checkpoint > 0 ? ` · checkpoint ${checkpoint}` : "";
  return ["ppo", title, `${done}/${total} ${completed}${origin}`, done, total];
}

function phase(run) {
  const info = run?.phase || {};
  const names = {
    started: "Início do processo", stopped: "Execução interrompida", failed: "Falha na execução",
    finished: "Execução concluída", replay_validation: "Validação em replays",
    replays: "Reconstrução de replays", bc_start: "Imitação", bc_batch: "Imitação", bc_epoch: "Imitação",
    ppo_start: "Self-play", ppo_match: "Self-play", ppo_game: "Self-play", ppo_update: "Self-play",
    evaluation_start: "Avaliação na arena", evaluation_pair: "Avaliação na arena",
  };
  let details = "";
  if (info.event?.startsWith("ppo")) {
    const job = data.job;
    const tracked = activeJob(job) && job.run_id === run.id && Date.parse(info.time) >= Date.parse(job.started);
    details = ppoProgress(info, tracked ? job.parameters.updates : info.total)[2];
  } else if (info.event === "bc_batch") {
    details = `Época ${info.epoch}/${info.total} · lote ${info.done}/${info.batches}`;
  } else if (Number.isFinite(info.done) && Number.isFinite(info.total)) {
    details = `${info.done}/${info.total} concluídos${simultaneousGames(info)}`;
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

/* ==================== BADGES DA ARQUITETURA ==================== */
function renderModelBadges(run) {
  const box = byId("model-badges");
  if (!run) { box.innerHTML = ""; return; }
  const info = run.model_info || {};
  const last = run.updates?.at(-1);
  const pills = [];
  if (info.architecture) {
    const archTone = info.architecture >= 3 ? "ppo" : "";
    pills.push([`Arquitetura`, `v${info.architecture}${info.architecture >= 3 ? " ResNet" : ""}`, archTone]);
  }
  if (info.obs_profile) {
    const ch = info.channels ? ` · ${info.channels}ch` : "";
    const tone = info.obs_profile === "live" ? "good" : info.obs_profile === "v3" ? "ppo" : "";
    pills.push([`Visão`, `${info.obs_profile}${ch}`, tone]);
  }
  const rewardProfile = last?.reward_profile || info.reward_profile;
  if (rewardProfile) {
    pills.push([`Recompensa`, rewardProfile === "potential" ? "Potential-Based" : rewardProfile, rewardProfile === "potential" ? "good" : ""]);
  }
  if (info.mirror_augmentation) pills.push([`Aumento`, `Espelhamento 2×`, "good"]);
  if (info.competitive_only) pills.push([`Replays`, `Filtro competitivo`, ""]);
  if (last?.rollout_workers > 1) pills.push([`Pool CPU`, `${last.rollout_workers} workers`, ""]);
  if (info.live_finetune) {
    const count = info.live_finetune.matches || info.live_finetune.frames;
    pills.push([`Jogo Real`, count ? `Fine-tune AWR (${show(count)})` : "Fine-tune AWR", "good"]);
  }
  box.innerHTML = pills.map(([k, v, tone]) =>
    `<span class="pill"${tone ? ` data-tone="${tone}"` : ""}>${escapeText(k)} <strong>${escapeText(v)}</strong></span>`
  ).join("");
}

/* ==================== VISÃO GERAL (8 KPIS + RESUMOS) ==================== */
function netDamage(row) {
  if (!row || !Number.isFinite(row.mean_tower_damage_dealt) || !Number.isFinite(row.mean_tower_damage_taken)) return null;
  return row.mean_tower_damage_dealt - row.mean_tower_damage_taken;
}

function throughputPerMin(row) {
  if (!row || !row.games) return null;
  const totalSec = (row.simulation_seconds || 0) + (row.optimization_seconds || 0);
  if (totalSec <= 0) return null;
  return row.games / (totalSec / 60);
}

function ppoHealthItems(last) {
  const ppo = last?.ppo;
  if (!ppo) return [];
  const items = [];
  const ev = ppo.explained_variance;
  if (Number.isFinite(ev)) {
    const health = ev >= 0.4 ? "good" : ev >= 0.15 ? "warn" : "bad";
    const status = ev >= 0.4 ? "Saudável" : ev >= 0.15 ? "Calibrando" : "Crítico fraco";
    items.push({
      name: "Explained Variance (Crítico)",
      value: show(ev, decimal),
      health, status,
      desc: ev >= 0.4
        ? "O crítico prevê bem o retorno futuro das jogadas; vantagens GAE confiáveis."
        : ev >= 0.15
          ? "Crítico ainda se adaptando ao valor das posições. Tende a subir com mais updates."
          : "Alta variância nos retornos. Considere aumentar partidas/update ou manter warmup.",
    });
  }
  const clip = ppo.clip_fraction;
  if (Number.isFinite(clip)) {
    const health = clip >= 0.03 && clip <= 0.22 ? "good" : clip > 0.32 ? "bad" : "warn";
    const status = health === "good" ? "Ideal" : clip > 0.32 ? "Passo alto" : clip < 0.03 ? "Conservador" : "Atenção";
    items.push({
      name: "Clip Fraction (PPO)",
      value: show(clip, percent),
      health, status,
      desc: health === "good"
        ? "Faixa ideal (3%–22%): a política evolui em passos firmes sem colapsar."
        : clip > 0.32
          ? "Muitos gradientes sendo cortados; passos agressivos no PPO."
          : "Mudanças muito pequenas por passo; política quase estática.",
    });
  }
  const akl = ppo.approx_kl;
  if (Number.isFinite(akl)) {
    const health = akl <= 0.025 ? "good" : akl <= 0.045 ? "warn" : "bad";
    const status = health === "good" ? "Estável" : health === "warn" ? "Elevado" : "Early-stop";
    items.push({
      name: "Approx KL (por passo)",
      value: kl(akl),
      health, status,
      desc: `Épocas completadas: ${show(ppo.epochs_completed)}. Alvo <= 0,03 para evitar saltos bruscos.`,
    });
  }
  const ent = ppo.entropy;
  if (Number.isFinite(ent)) {
    const health = ent >= 0.25 ? "good" : ent >= 0.12 ? "warn" : "bad";
    const status = health === "good" ? "Explorando" : health === "warn" ? "Afunilando" : "Colapso";
    items.push({
      name: "Entropia da Política",
      value: show(ent, decimal),
      health, status,
      desc: `Entropia de espera (wait): ${show(ppo.wait_entropy, decimal)}. Mede diversidade de jogadas.`,
    });
  }
  if (Number.isFinite(last.kl_bc_to_ppo) || Number.isFinite(ppo.kl_coef)) {
    const dir = ppo.kl_direction === "reverse" ? "Reverse KL" : ppo.kl_direction === "forward" ? "Forward KL" : "KL";
    items.push({
      name: `Âncora ${dir} (BC → PPO)`,
      value: kl(last.kl_bc_to_ppo),
      health: "neutral",
      status: Number.isFinite(ppo.kl_coef) ? `coef ${ppo.kl_coef.toFixed(3).replace(".", ",")}` : dir,
      desc: ppo.kl_direction === "reverse"
        ? "Reverse KL com decaimento permite superar a imitação sem esquecer o deck."
        : "Regularização mantém a política próxima aos replays humanos.",
    });
  }
  const gpm = throughputPerMin(last);
  if (Number.isFinite(gpm)) {
    const sim = last.simulation_seconds || 0;
    const opt = last.optimization_seconds || 0;
    const simShare = (sim + opt) > 0 ? sim / (sim + opt) : 0;
    items.push({
      name: "Eficiência no Xeon",
      value: `${show(gpm, decimal1)} /min`,
      health: gpm >= 10 ? "good" : "neutral",
      status: `${last.rollout_workers || 1} workers`,
      desc: `Simulação: ${show(sim, decimal1)}s (${show(simShare, percent)}) · Otimização: ${show(opt, decimal1)}s.`,
    });
  }
  return items;
}

function overview(run, pairs) {
  renderModelBadges(run);
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

  // Rolling win rate (last 5 vs first 5)
  const avgWin = slice => {
    const g = slice.reduce((s, r) => s + (r.games || 0), 0);
    const w = slice.reduce((s, r) => s + (r.wins || 0), 0);
    return g > 0 ? w / g : null;
  };
  const recentWr = updates.length ? avgWin(updates.slice(-5)) : null;
  const earlyWr = updates.length >= 6 ? avgWin(updates.slice(0, 5)) : null;
  const wrDelta = recentWr !== null && earlyWr !== null ? 100 * (recentWr - earlyWr) : null;
  const wrTag = wrDelta !== null
    ? [`${signed(wrDelta, decimal1)} p.p.`, wrDelta >= 0 ? "good" : "warn"]
    : (last ? [`Upd #${last.update}`, ""] : null);

  const dmgNet = netDamage(last);
  const dmgTag = dmgNet !== null ? [dmgNet >= 0 ? "Vantagem" : "Desvantagem", dmgNet >= 0 ? "good" : "warn"] : null;

  const leak = last?.mean_elixir_leaked;
  const leakTag = Number.isFinite(leak)
    ? [leak <= 1.0 ? "Limpo" : leak <= 2.5 ? "Moderado" : "Alto", leak <= 1.0 ? "good" : leak <= 2.5 ? "warn" : "bad"]
    : null;

  const ev = last?.ppo?.explained_variance;
  const evTag = Number.isFinite(ev)
    ? [ev >= 0.4 ? "Saudável" : ev >= 0.15 ? "Calibrando" : "Fraco", ev >= 0.4 ? "good" : ev >= 0.15 ? "warn" : "bad"]
    : null;

  const bcMetrics = run?.validation?.bc || run?.epochs?.at(-1);
  const bcHint = bcMetrics
    ? `NLL ${show(bcMetrics.nll, decimal)}${Number.isFinite(bcMetrics.value_mae) ? ` · Value MAE ${show(bcMetrics.value_mae, decimal)}` : ""}`
    : "Aguardando imitação (BC)";

  const gpm = throughputPerMin(last);
  const perfHint = last && Number.isFinite(last.simulation_seconds)
    ? `sim ${show(last.simulation_seconds, decimal1)}s · opt ${show(last.optimization_seconds, decimal1)}s`
    : `${show(completedGames)} partidas totais`;

  const klTag = last?.ppo?.kl_direction
    ? [`${last.ppo.kl_direction === "reverse" ? "Rev" : "Fwd"} · ${last.ppo.kl_coef?.toFixed(3) ?? ""}`, ""]
    : null;

  const tiles = [
    ["Win Rate (ult. 5)", show(recentWr, percent), `${show(completedGames)} partidas de self-play`, wrTag],
    ["Saldo de Dano (Torres)", dmgNet !== null ? signed(dmgNet, number) : "—",
      dmgNet !== null ? `Feito ${show(last.mean_tower_damage_dealt)} · Sofrido ${show(last.mean_tower_damage_taken)}` : "Disponível em treinos recentes",
      dmgTag],
    ["Conexões X-Bow", show(last?.mean_locks, decimal), "Média de travadas na torre · último upd", null],
    ["Elixir Vazado", Number.isFinite(leak) ? show(leak, decimal) : show(last?.play_rate, percent),
      Number.isFinite(leak) ? "Média de elixir desperdiçado por partida" : "Taxa de jogadas por passo (legado)", leakTag],
    ["Explained Variance", show(ev, decimal), "Qualidade da função de valor (crítico)", evTag],
    ["Divergência KL", kl(last?.kl_bc_to_ppo), "Distância da política BC → PPO", klTag],
    ["Precisão BC (Carta)", show(bcMetrics?.conditional_card_accuracy, percent), bcHint, null],
    ["Velocidade no Xeon", gpm !== null ? `${show(gpm, decimal1)}/m` : show(last?.update),
      gpm !== null ? perfHint : "Atualizações PPO registradas",
      last?.rollout_workers ? [`${last.rollout_workers}w`, "good"] : null],
  ];

  byId("metrics").innerHTML = tiles.map(([label, value, hint, tag]) => `
    <article class="metric">
      <div class="metric-head">
        <span class="name">${label}</span>
        ${tag ? `<span class="metric-tag ${tag[1]}">${escapeText(tag[0])}</span>` : ""}
      </div>
      <strong class="number">${value}</strong>
      <small>${escapeText(hint)}</small>
    </article>`).join("");

  // Quick PPO Health summary on Visão Geral
  const health = ppoHealthItems(last);
  byId("overview-health").innerHTML = health.length
    ? `<div class="overview-mini-list">${health.slice(0, 4).map(item => `
        <div class="overview-mini-row">
          <span>${escapeText(item.name)}</span>
          <span style="display:flex;align-items:center;gap:8px">
            <strong>${escapeText(item.value)}</strong>
            <span class="metric-tag ${item.health}">${escapeText(item.status)}</span>
          </span>
        </div>`).join("")}</div>`
    : empty("Telemetria interna do PPO aparecerá a partir da próxima atualização com o motor v3.");

  // Quick Matchup summary on Visão Geral
  const pfspWr = run?.model_info?.pfsp_wr;
  let matchupSummaryRows = [];
  if (pairs.length) {
    matchupSummaryRows = pairs
      .filter(r => r.ppo || r.bc)
      .map(r => ({
        name: rivalLabel(r.rival),
        wr: ratio(r.ppo || r.bc),
        tag: r.ppo ? `${r.ppo.wins}/${r.ppo.games}` : `BC ${r.bc.wins}/${r.bc.games}`,
      }))
      .sort((a, b) => (b.wr ?? -1) - (a.wr ?? -1));
  } else if (pfspWr && Object.keys(pfspWr).length) {
    matchupSummaryRows = Object.entries(pfspWr)
      .map(([k, v]) => ({ name: rivalLabel(k), wr: v, tag: "EMA PFSP" }))
      .sort((a, b) => b.wr - a.wr);
  }
  byId("overview-matchups").innerHTML = matchupSummaryRows.length
    ? `<div class="overview-mini-list">${matchupSummaryRows.slice(0, 5).map(row => {
        const cls = row.wr >= 0.55 ? "good" : row.wr < 0.42 ? "warn" : "";
        return `<div class="overview-mini-row">
          <span>${escapeText(row.name)}</span>
          <span style="display:flex;align-items:center;gap:8px">
            <small class="muted">${escapeText(row.tag)}</small>
            <span class="metric-tag ${cls}">${show(row.wr, percent)}</span>
          </span>
        </div>`;
      }).join("")}</div>`
    : empty("Aguardando avaliação na arena ou partidas da liga PFSP.");
}

/* ==================== DIAGNÓSTICO PPO & PFSP ==================== */
function renderPPODiagnostics(run) {
  const last = run?.updates?.at(-1);
  const items = ppoHealthItems(last);
  byId("ppo-diagnostics-note").textContent = last?.ppo
    ? `Atualização #${last.update} · ${show(last.games)} partidas · ${last.ppo.kl_direction === "reverse" ? "Reverse KL" : "Forward KL"}`
    : "Indicadores internos de estabilidade, exploração e qualidade do crítico.";
  byId("ppo-diagnostics").innerHTML = items.length
    ? items.map(item => `
      <article class="diag-card" data-health="${item.health}">
        <div class="diag-top">
          <span class="diag-name">${escapeText(item.name)}</span>
          <span class="metric-tag ${item.health}">${escapeText(item.status)}</span>
        </div>
        <strong class="diag-val">${escapeText(item.value)}</strong>
        <p>${escapeText(item.desc)}</p>
      </article>`).join("")
    : empty("Execute ou retome um treino PPO com o motor atualizado para ver a telemetria completa do cérebro.");
}

function renderPFSPBoard(run) {
  const pfspWr = run?.model_info?.pfsp_wr;
  const lastRivals = run?.updates?.at(-1)?.rivals || {};
  const entries = pfspWr ? Object.entries(pfspWr) : [];
  if (!entries.length) {
    byId("pfsp-board").innerHTML = empty("Os pesos da liga PFSP aparecerão após a primeira atualização PPO.");
    return;
  }
  const floor = 0.15;
  const maxShare = 0.28;
  const rows = entries.map(([name, wr]) => ({
    name,
    wr,
    raw: (1 - wr) * (1 - wr) + floor,
    recent: lastRivals[name] || 0,
  }));
  const totalRaw = rows.reduce((s, r) => s + r.raw, 0) || 1;
  rows.forEach(r => { r.prob = r.raw / totalRaw; r.capped = false; });
  if (rows.length >= 4 && maxShare > 1 / rows.length + 1e-6) {
    for (let iter = 0; iter < rows.length; iter++) {
      const over = rows.filter(r => !r.capped && r.prob > maxShare);
      if (!over.length) break;
      over.forEach(r => { r.prob = maxShare; r.capped = true; });
      const free = rows.filter(r => !r.capped);
      if (!free.length) break;
      const rem = Math.max(0, 1 - rows.filter(r => r.capped).reduce((s, r) => s + r.prob, 0));
      const freeSum = free.reduce((s, r) => s + r.prob, 0);
      free.forEach(r => { r.prob = freeSum > 0 ? r.prob * (rem / freeSum) : rem / free.length; });
    }
  }
  rows.sort((a, b) => b.prob - a.prob);

  byId("pfsp-note").textContent = `${rows.length} oponentes rastreados · foco automático nos matchups mais difíceis.`;
  byId("pfsp-board").innerHTML = `<div class="pfsp-list">${rows.map(r => `
    <div class="pfsp-row">
      <div class="pfsp-head">
        <strong>${escapeText(rivalLabel(r.name))}</strong>
        <span class="pfsp-meta">
          <span>WR EMA <strong style="color:var(--bc)">${show(r.wr, percent)}</strong></span>
          <span>Foco <strong style="color:var(--ppo)">${show(r.prob, percent)}</strong></span>
          ${r.recent ? `<span>+${r.recent} último upd</span>` : ""}
        </span>
      </div>
      <div class="pfsp-bars">
        <div class="pfsp-track" title="Taxa de vitórias EMA"><span class="pfsp-fill-wr" style="width:${Math.min(100, Math.max(0, r.wr * 100))}%"></span></div>
        <div class="pfsp-track" title="Prioridade de amostragem PFSP"><span class="pfsp-fill-prio" style="width:${Math.min(100, Math.max(0, r.prob * 100 * 2.2))}%"></span></div>
      </div>
    </div>`).join("")}</div>`;
}

/* ==================== TABELAS & VALIDAÇÃO ==================== */
function table(headers, rows) {
  return `<div class="table-wrap"><table><thead><tr>${headers.map(h => `<th scope="col">${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(([heading, ...values]) => `<tr><th scope="row">${escapeText(heading)}</th>${values.map(value => `<td>${value}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function matchups(pairs) {
  const bar = (row, model) => row ? `<div class="bar-cell ${model}">${row.wins}/${row.games} <span class="muted">· ${show(ratio(row), percent)}</span>
    <div class="mini-track" aria-hidden="true"><span class="mini-bar" style="width:${Math.min(100, Math.max(0, ratio(row) * 100))}%"></span></div></div>` : "—";
  byId("matchups").innerHTML = pairs.length ? table(["Rival", "Imitação", "PPO"], pairs.map(row => [
    rivalLabel(row.rival), bar(row.bc, "bc"), bar(row.ppo, "ppo"),
  ])) : empty("Aguardando os resultados da avaliação pareada contra os rivais.");
}

function validation(run) {
  const val = run?.validation || {};
  const lastEpoch = run?.epochs?.at(-1);
  const bc = val.bc || lastEpoch;
  const ppo = val.ppo;
  const matches = run?.split?.validation;
  byId("validation-note").textContent = bc ?
    `${Number.isFinite(matches) ? `${show(matches)} partidas reservadas · ` : ""}${show(bc.frames)} quadros · ${show(bc.card_plays)} jogadas de carta` :
    "Resultados nas partidas reservadas (holdout).";
  const rows = [
    ["Perda de validação (NLL)", show(bc?.nll, decimal), show(ppo?.nll, decimal), "Menor é melhor"],
    ["Escolha da carta", show(bc?.conditional_card_accuracy, percent), show(ppo?.conditional_card_accuracy, percent), "Quando a pessoa jogou"],
    ["Carta e casa exatas", show(bc?.conditional_card_and_tile_accuracy, percent), show(ppo?.conditional_card_and_tile_accuracy, percent), "Quando a pessoa jogou"],
    ["Probabilidade média de jogar", show(bc?.mean_play_probability, percent), show(ppo?.mean_play_probability, percent), `Humano: ${show(bc?.expert_play_rate, percent)}`],
  ];
  if (Number.isFinite(bc?.value_mae) || Number.isFinite(ppo?.value_mae)) {
    rows.push(["Erro do crítico (Value MAE)", show(bc?.value_mae, decimal), show(ppo?.value_mae, decimal), "Pré-treino auxiliar de valor · menor é melhor"]);
  }
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

/* ==================== GRÁFICO DE EVOLUÇÃO (11 MÉTRICAS + MÉDIA MÓVEL) ==================== */
const charts = {
  win: { value: ratio, format: percent, label: "Taxa de vitórias", clamp01: true },
  net_damage: { value: netDamage, format: number, label: "Saldo de dano nas torres" },
  locks: { value: row => row.mean_locks, format: decimal, label: "Conexões de X-Bow na torre" },
  leak: { value: row => row.mean_elixir_leaked, format: decimal, label: "Elixir vazado por partida" },
  reward: { value: row => row.mean_reward, format: decimal, label: "Recompensa média" },
  explained_var: { value: row => row.ppo?.explained_variance, format: decimal, label: "Explained Variance (Crítico)" },
  kl: { value: row => row.kl_bc_to_ppo, format: { format: kl }, label: "Divergência KL (BC → PPO)" },
  entropy: { value: row => row.ppo?.entropy, format: decimal, label: "Entropia da política" },
  clip_frac: { value: row => row.ppo?.clip_fraction, format: percent, label: "Clip Fraction PPO" },
  bc: { value: row => row.nll, format: decimal, label: "Perda de imitação (NLL)", isBC: true },
  bc_acc: { value: row => row.conditional_card_accuracy, format: percent, label: "Acurácia de carta (BC)", isBC: true, clamp01: true },
};

function movingAverage(values, windowSize = 5) {
  return values.map((_, idx) => {
    const start = Math.max(0, idx - windowSize + 1);
    const slice = values.slice(start, idx + 1);
    return slice.reduce((s, v) => s + v, 0) / slice.length;
  });
}

function drawChart() {
  const canvas = byId("chart");
  const metric = byId("chart-metric").value;
  const spec = charts[metric] || charts.win;
  const isClone = Boolean(spec.isBC);
  const rawList = isClone ? epochs : series;
  const rows = rawList.map(row => [isClone ? row.epoch : row.update, spec.value(row)])
    .filter(([, value]) => Number.isFinite(value));
  byId("chart-caption").textContent = isClone
    ? `${spec.label} por época de imitação (BC)`
    : `${spec.label} por atualização do PPO`;
  byId("chart-empty").textContent = isClone
    ? "Aguardando a primeira época de imitação."
    : (series.length ? "Esta métrica não foi registrada neste experimento (use motor v3)." : "Aguardando a primeira atualização de self-play.");
  byId("chart-empty").hidden = rows.length !== 0;
  byId("history").hidden = !rawList.length;

  byId("history-table").innerHTML = isClone
    ? table(["Época", "NLL (Val)", "Carta (%)", "Carta+Casa", "Value MAE", "LR"],
        epochs.map(row => [
          row.epoch, show(row.nll, decimal), show(row.conditional_card_accuracy, percent),
          show(row.conditional_card_and_tile_accuracy, percent), show(row.value_mae, decimal),
          Number.isFinite(row.lr) ? row.lr.toExponential(1) : "—",
        ]))
    : table(["Upd", "Vitórias", "WR", "Dano líq.", "Conexões", "Leak", "EV", "Clip", "Entropia", "KL", "Tempo"],
        series.map(row => {
          const nd = netDamage(row);
          const totalSec = Number.isFinite(row.simulation_seconds)
            ? (row.simulation_seconds + (row.optimization_seconds || 0)) : null;
          return [
            row.update,
            `${show(row.wins)}/${show(row.games)}`,
            show(ratio(row), percent),
            nd !== null ? signed(nd, number) : "—",
            show(row.mean_locks, decimal),
            show(row.mean_elixir_leaked, decimal),
            show(row.ppo?.explained_variance, decimal),
            show(row.ppo?.clip_fraction, percent),
            show(row.ppo?.entropy, decimal),
            kl(row.kl_bc_to_ppo),
            totalSec !== null ? `${show(totalSec, decimal1)}s` : "—",
          ];
        }));

  canvas.setAttribute("aria-label", `${spec.label}. Os valores estão na tabela abaixo.`);
  byId("plot-tooltip").hidden = true;
  points = [];
  const width = canvas.clientWidth, height = canvas.clientHeight, scale = window.devicePixelRatio || 1;
  if (width < 130 || height < 60) return;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  if (!rows.length) return;

  const left = 62, right = width - 16, top = 14, bottom = height - 30;
  const values = rows.map(([, value]) => value);
  const smooth = byId("chart-smooth")?.checked && values.length >= 3;
  const maValues = smooth ? movingAverage(values, 5) : [];

  let min = spec.clamp01 ? 0 : Math.min(0, ...values);
  let max = spec.clamp01 ? 1 : Math.max(metric === "kl" ? 0.001 : 0.05, ...values);
  if (!spec.clamp01) {
    const pad = Math.max(Math.abs(max - min) * 0.1, metric === "kl" ? 0.0005 : 0.05);
    if (min < 0) min -= pad;
    max += pad;
  }
  if (max <= min) max = min + 1;

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
  // Zero baseline for signed metrics (net_damage, reward, explained_var)
  if (min < 0 && max > 0) {
    const y0 = chartY(0);
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#5b6d8a";
    ctx.beginPath(); ctx.moveTo(left, y0); ctx.lineTo(right, y0); ctx.stroke();
    ctx.restore();
  }

  points = rows.map(([update, value], index) => ({
    update, value, ma: smooth ? maValues[index] : null,
    x: chartX(index), y: chartY(value),
  }));

  const grad = ctx.createLinearGradient(0, top, 0, bottom);
  grad.addColorStop(0, "#a99aff40");
  grad.addColorStop(1, "#a99aff02");
  ctx.beginPath(); ctx.moveTo(points[0].x, bottom);
  for (const point of points) ctx.lineTo(point.x, point.y);
  ctx.lineTo(points.at(-1).x, bottom); ctx.closePath(); ctx.fillStyle = grad; ctx.fill();

  // Raw series line
  ctx.beginPath();
  points.forEach((point, index) => index ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y));
  ctx.lineWidth = smooth ? 1.6 : 2.1;
  ctx.strokeStyle = smooth ? "#b3a2fa99" : "#b3a2fa";
  ctx.stroke();

  // 5-point moving average trend overlay
  if (smooth) {
    ctx.beginPath();
    points.forEach((point, index) => {
      const y = chartY(point.ma);
      if (index) ctx.lineTo(point.x, y); else ctx.moveTo(point.x, y);
    });
    ctx.lineWidth = 2.4;
    ctx.strokeStyle = "#71dbc7";
    ctx.stroke();
  }

  const labelEvery = Math.ceil(points.length / Math.max(2, Math.floor((right - left) / 44)));
  ctx.textAlign = "center";
  points.forEach((point, index) => {
    ctx.fillStyle = "#b3a2fa";
    ctx.beginPath(); ctx.arc(point.x, point.y, smooth ? 2.5 : 3, 0, Math.PI * 2); ctx.fill();
    if (index % labelEvery === 0 || index === points.length - 1) {
      ctx.fillStyle = "#a5b4c8"; ctx.fillText(String(point.update), point.x, bottom + 17);
    }
  });
}

/* ==================== TREINAMENTO ASSISTIDO & BARRA GLOBAL ==================== */
const activeJob = job => ["starting", "running", "stopping"].includes(job?.status);
const jobStatus = {
  starting: "Iniciando", running: "Em treinamento", stopping: "Parada solicitada",
  finished: "Concluído", stopped: "Parado", failed: "Falhou", interrupted: "Interrompido",
};
const parameterKeys = ["matches", "epochs", "updates", "games", "evaluate", "threads", "rollout_workers"];
const replaySelection = { xbow_30: 120, icebow: 60 };

function executionMode() {
  return byId("training-mode").valueAsNumber === 1 ? "resume" : "new";
}

function syncSlider(slider, available = true) {
  if (!slider) return;
  const span = Number(slider.max) - Number(slider.min);
  const proportion = span > 0 ? (slider.valueAsNumber - Number(slider.min)) / span : 0;
  slider.style.setProperty("--slider-progress", `${proportion * 100}%`);
  const isMode = slider.id === "training-mode";
  const label = isMode ? (executionMode() === "resume" ? "Retomar PPO" : "Novo treino")
    : show(available ? slider.valueAsNumber : null);
  const valueNode = slider.labels?.[0]?.querySelector(".slider-value");
  if (valueNode) valueNode.textContent = label;
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
  byId("training-max-replays").disabled = resume || !ready;
  byId("training-replay-limit").textContent = show(count);
  syncSlider(slider, ready);
  return ready;
}

function stageProgress(job, info) {
  const event = info.event || "";
  if (job.mode === "recommended") {
    if (event === "rec_vision") return ["replays", "1/4 · Visão YOLO", info.detail || "Treinando detector YOLO", info.done, info.total];
    if (event === "rec_meta") return ["bc", "2/4 · Meta ao vivo", info.detail || "Analisando partidas reais", info.done, info.total];
    if (event === "rec_hybrid") return ["ppo", "3/4 · AWR + PPO (6 decks)", info.detail || "Reforçando política no RoyaleSim", info.done, info.total];
    if (event === "rec_gate" || event === "rec_done") return ["evaluation", "4/4 · Gate A/B & Promoção", info.detail || "Validando promoção de ppo_live.pt", info.done, info.total];
    return ["replays", "Pipeline recomendada", "Inicializando visão e checkpoints…", null, null];
  }
  if (event.startsWith("ppo")) return ppoProgress(info, job.parameters.updates);
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
  if (event.startsWith("evaluation")) {
    const detail = info.joint ? "partidas entre os dois modelos" : "confrontos por modelo e rival";
    return ["evaluation", "Avaliação na arena", `${info.done}/${info.total} ${detail}${simultaneousGames(info)}`, info.done, info.total];
  }
  return [job.mode === "resume" ? "ppo" : "replays", "Preparando o treino", "Carregando o ambiente", null, null];
}

function renderGlobalJobBar() {
  const bar = byId("global-job-bar");
  const job = data.job;
  const running = activeJob(job);
  if (!running || activePage === "treino") {
    bar.hidden = true;
    return;
  }
  const tracked = data.runs.find(item => item.id === job.run_id);
  const progress = tracked?.progress;
  const duringJob = progress && Date.parse(progress.time) >= Date.parse(job.started);
  const [, title, detail, done, total] = stageProgress(job, duringJob ? progress : {});
  byId("global-job-title").textContent = `${title} · ${deckNames[job.deck] || job.deck}`;
  byId("global-job-detail").textContent = detail;
  const p = byId("global-job-progress");
  if (job.parameters?.continuous) {
    p.hidden = true;
  } else {
    p.hidden = false;
    if (Number.isFinite(done) && total > 0) { p.max = total; p.value = done; }
    else { p.removeAttribute("value"); p.max = 1; }
  }
  bar.hidden = false;
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
  const coevolve = byId("training-coevolve");
  coevolve.disabled = data.joint_training !== "shared_match" || !resume || deck !== "xbow_30";
  byId("training-coevolve-option").hidden = coevolve.disabled;
  const joint = !coevolve.disabled && coevolve.checked;
  const continuousInput = byId("training-continuous");
  continuousInput.disabled = !joint;
  byId("training-continuous-option").hidden = !joint;
  const continuous = joint && continuousInput.checked;

  const maxThreads = data.system?.max_threads || 27;
  const maxWorkers = data.system?.max_workers || 12;

  const threadsSlider = byId("training-threads");
  if (threadsSlider) {
    threadsSlider.max = maxThreads;
    byId("training-threads-limit").textContent = maxThreads;
    if (threadsSlider.valueAsNumber > maxThreads) threadsSlider.value = maxThreads;
  }
  const workersSlider = byId("training-rollout-workers");
  if (workersSlider) {
    workersSlider.max = maxWorkers;
    byId("training-rollout-workers-limit").textContent = maxWorkers;
    if (workersSlider.valueAsNumber > maxWorkers) workersSlider.value = maxWorkers;
  }
  const contGamesSlider = byId("training-continuous-games");
  if (contGamesSlider) {
    contGamesSlider.max = 256;
    byId("training-continuous-games-limit").textContent = "256 (GPU RX 6600)";
    if (contGamesSlider.valueAsNumber > 256) contGamesSlider.value = 64;
  }
  byId("training-continuous-games-option").hidden = !continuous;
  syncSlider(contGamesSlider);

  byId("training-updates-title").textContent = joint ? "Lotes de co-evolução" : "Atualizações PPO";
  byId("training-games").closest("label").hidden = continuous;
  byId("training-updates").closest("label").hidden = continuous;
  byId("training-evaluate").closest("label").hidden = continuous;

  for (const field of document.querySelectorAll("[data-new]")) {
    field.hidden = resume;
    for (const ctrl of field.querySelectorAll("input, select, button")) {
      ctrl.disabled = field.hidden;
    }
  }
  byId("training-obs-badge").textContent = byId("training-obs-profile").value;
  byId("training-batch-badge").textContent = byId("training-batch-size").value;

  syncSlider(mode);
  const canTrain = syncReplayLimit(resume);
  for (const key of parameterKeys.slice(1)) {
    syncSlider(byId(`training-${key.replace("_", "-")}`));
  }

  byId("training-deck").textContent = active && job.mode === "recommended"
    ? `Em treino · Pipeline recomendada (${deckNames[job.deck] || job.deck})`
    : active && job.parameters.coevolve
      ? "Em treino · X-Bow 3.0 e IA adversária (Co-evolução)"
      : `Deck selecionado · ${deckNames[active ? job.deck : deck]}`;

  const rec = data.recommended || {};
  const liveM = data.live_matches || {};
  const deckRuns = data.runs.filter(r => r.deck === deck && (r.checkpoints?.ppo || r.checkpoints?.bc));
  const recTarget = (run && run.deck === deck && (run.checkpoints?.ppo || run.checkpoints?.bc))
    ? run
    : deckRuns[0] || null;
  const recPills = byId("recommended-pills");
  if (recPills) {
    recPills.innerHTML = [
      `<span class="rec-pill">Visão ativa: <strong>${escapeText(rec.active_vision || "hp_bar_only")}</strong></span>`,
      `<span class="rec-pill">Quadros YOLO: <strong>${show(rec.vision_live_frames ?? 0)}</strong> (${show(rec.vision_sprites ?? 0)} sprites)</span>`,
      `<span class="rec-pill">Partidas reais: <strong>${show(liveM.count ?? 0)}</strong> (${show(liveM.wins ?? 0)}V-${show(liveM.losses ?? 0)}D)</span>`,
      `<span class="rec-pill">Decks rivais: <strong>6 + ${show(rec.discovered_decks ?? 0)}</strong> descobertos (${show(rec.simulator_cards ?? 119)} cartas)</span>`,
      `<span class="rec-pill">Experimento alvo: <strong>${escapeText(recTarget ? (groupNames[recTarget.group] || recTarget.group) : "nenhum")}</strong></span>`,
    ].join("");
  }
  const recBtn = byId("start-recommended");
  if (recBtn) {
    recBtn.disabled = sending || !connected || active || !recTarget;
    recBtn.textContent = active && job?.mode === "recommended"
      ? "⚡ Pipeline recomendada em execução…"
      : sending ? "Iniciando…" : "⚡ Iniciar pipeline recomendada";
  }
  const recHint = byId("recommended-hint");
  if (recHint) {
    recHint.textContent = !recTarget
      ? `Crie primeiro um experimento base para ${deckNames[deck]} em "Treinamento assistido".`
      : active
        ? "Já existe um treinamento em andamento. Acompanhe o progresso abaixo ou clique em Parar treino."
        : `Executa YOLO (sintético + ${rec.vision_live_frames ?? 0} quadros reais) → Meta ao vivo → AWR + PPO nos 6 decks → Gate A/B em ${recTarget.id}.`;
  }

  byId("training-form").hidden = active;
  byId("training-settings").disabled = sending;
  byId("start-training").disabled = sending || !connected || (!resume && !canTrain);
  byId("start-training").textContent = sending ? "Iniciando…" :
    resume ? "Retomar treino" : "Iniciar treino";

  let hint = "O botão Todos seleciona todos os replays disponíveis. O filtro competitivo e o espelhamento 2× são aplicados automaticamente.";
  if (continuous) {
    hint = "As duas IAs aprendem em 6 decks do meta e salvam ao fim de cada lote. Parar treino encerra com segurança após o lote atual.";
  } else if (resume) {
    hint = joint
      ? "Co-evolução contra 6 decks do meta (Corredor 2.6, Gigante Real, Splashyard, Log Bait, Golem, P.E.K.K.A) + bots âncora."
      : `Continua o experimento ${groupNames[run.group] || run.group} a partir do último checkpoint (com pool PFSP e Reverse KL).`;
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
  renderGlobalJobBar();
  if (!job) return;

  const tracked = data.runs.find(item => item.id === job.run_id);
  const progress = tracked?.progress;
  const duringJob = progress && Date.parse(progress.time) >= Date.parse(job.started);
  const [stage, title, detail, done, total] = stageProgress(job, duringJob ? progress : {});
  const stages = ["replays", "bc", "ppo", "evaluation"];
  const stageLabels = job.mode === "recommended"
    ? { replays: "Visão YOLO", bc: "Meta ao vivo", ppo: "AWR + PPO", evaluation: "Gate A/B" }
    : { replays: "Replays", bc: "Imitação", ppo: "Self-play", evaluation: "Avaliação" };
  for (const marker of byId("training-stages").children) {
    const key = marker.dataset.stage;
    marker.textContent = stageLabels[key] || key;
    marker.hidden = (job.mode === "resume" && ["replays", "bc"].includes(key)) ||
      (job.mode !== "recommended" && key === "evaluation" && job.parameters.evaluate === 0);
    const completed = job.status === "finished" || stages.indexOf(key) < stages.indexOf(stage);
    marker.classList.toggle("completed", completed);
    if (active && key === stage) marker.setAttribute("aria-current", "step");
    else marker.removeAttribute("aria-current");
  }
  byId("job-stage").textContent = active ? title : jobStatus[job.status];
  byId("job-detail").textContent = job.status === "stopping" ?
    (job.parameters.continuous ? "Terminando o lote atual e salvando o checkpoint."
      : "Aguardando o fim do lote ou da partida atual.") : active ? detail : job.run_id;
  const bar = byId("job-progress");
  bar.hidden = Boolean(job.parameters.continuous) || (!active && job.status !== "finished");
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

/* ==================== PÁGINA 6: JOGO REAL & OVERLAY ==================== */
const DECK_DEFAULT_ORDER = {
  xbow_30: "x-bow tesla archers knight skeletons electro-spirit fireball the-log",
  icebow: "x-bow tesla ice-wizard knight skeletons tornado rocket the-log",
};
let liveStatus = null;
let livePreviewKind = "deck";
let liveBusy = false;
let lastPreviewStamp = "";
let lastLiveDeckOrder = "";

function syncLiveModelSelect(fallbackRun) {
  const picker = byId("live-model-select");
  const candidates = data.runs
    .filter(r => r.deck === deck && (r.checkpoints?.ppo || r.checkpoints?.bc))
    .sort((a, b) => {
      const aLive = a.model_info?.obs_profile === "live" ? 0 : 1;
      const bLive = b.model_info?.obs_profile === "live" ? 0 : 1;
      return aLive - bLive || (b.updates?.length || 0) - (a.updates?.length || 0);
    });
  const options = candidates.map(r => {
    const prof = r.model_info?.obs_profile || "v3";
    const ch = r.model_info?.channels ? ` ${r.model_info.channels}ch` : "";
    const upd = r.updates?.length ? ` · ${r.updates.length} upd` : "";
    const tag = prof === "live" ? "★ LIVE" : prof.toUpperCase();
    return [r.id, `[${tag}${ch}] ${groupNames[r.group] || r.group}${upd}`];
  });
  const key = JSON.stringify(options);
  if (picker.dataset.options !== key) {
    const prev = picker.value;
    picker.replaceChildren(...(options.length
      ? options.map(([v, label]) => new Option(label, v))
      : [new Option("Sem modelos treinados para este deck", "")]));
    picker.dataset.options = key;
    if (options.some(([v]) => v === prev)) picker.value = prev;
    else if (options.length) picker.value = options[0][0];
    else if (fallbackRun) picker.value = fallbackRun.id;
  }
  picker.disabled = options.length === 0;
  if (lastLiveDeckOrder !== deck) {
    byId("live-cards-order").value = DECK_DEFAULT_ORDER[deck] || DECK_DEFAULT_ORDER.xbow_30;
    lastLiveDeckOrder = deck;
  }
}

function renderLiveStatus() {
  const st = liveStatus;
  const strip = byId("live-status-strip");
  if (!st) {
    strip.innerHTML = `<span class="pill">Captura <strong>verificando…</strong></span>`;
    return;
  }
  const winTxt = st.window_rect
    ? `Detectada (${st.window_rect[2]}×${st.window_rect[3]})`
    : "Não encontrada · abra o Clash Royale";
  const tmplOk = st.templates_saved >= st.templates_total;
  const ovRun = Boolean(st.overlay?.running);
  const lrRun = Boolean(st.learner?.running);
  const auRun = Boolean(st.auto_trainer?.running);
  const auSnap = st.auto_trainer?.snapshot || null;
  strip.innerHTML = [
    `<span class="pill" data-tone="${st.window_rect ? "good" : "warn"}">Janela do Jogo <strong>${escapeText(winTxt)}</strong></span>`,
    `<span class="pill" data-tone="${tmplOk ? "good" : "warn"}">Templates reais <strong>${st.templates_saved}/${st.templates_total}</strong></span>`,
    `<span class="pill" data-tone="${ovRun ? "good" : ""}">Overlay <strong>${ovRun ? `Ativo (${st.overlay.mode === "demo" ? "Demo" : "Ao vivo"} · PID ${st.overlay.pid})` : "Parado"}</strong></span>`,
    auRun ? `<span class="pill" data-tone="good">Treino Real 1v1 <strong>${escapeText(auSnap?.state || "Ativo")} · ${auSnap?.games_played ?? 0}/${auSnap?.max_games ?? 20}</strong></span>` : "",
    lrRun ? `<span class="pill" data-tone="ppo">Treino AWR <strong>Em andamento…</strong></span>` : "",
  ].join("");

  const badge = byId("live-overlay-badge");
  badge.textContent = ovRun ? `Overlay ativo · PID ${st.overlay.pid}` : "Overlay parado";
  badge.dataset.state = ovRun ? "running" : "idle";
  byId("live-start-btn").hidden = ovRun;
  byId("live-stop-btn").hidden = !ovRun;
  byId("live-learn-btn").disabled = liveBusy || lrRun;

  // Auto-trainer panel state & cursor permission prompt
  const auBadge = byId("auto-train-badge");
  const needsCursor = Boolean(!auRun && auSnap?.needs_cursor_permission);
  auBadge.textContent = auRun
    ? `Ativo · ${auSnap?.state || "Rodando"}`
    : needsCursor ? "Aguardando permissão de cursor"
    : auSnap?.status === "paused_unsafe_mode" ? "Pausado (Trava anti-troféus)"
    : auSnap?.status === "finished" ? "Sessão concluída"
    : "Parado";
  auBadge.dataset.state = auRun ? "running" : (needsCursor || auSnap?.status === "paused_unsafe_mode") ? "stopping" : "idle";
  byId("auto-start-btn").hidden = auRun;
  byId("auto-stop-btn").hidden = !auRun;
  byId("auto-cursor-banner").hidden = !needsCursor;
  if (auSnap) {
    const auTiles = [
      ["Estado", auSnap.state || "—", auSnap.dry_run ? "Simulação (dry-run)" : `Cliques: ${auSnap.click_mode || "background"}`],
      ["Partidas na sessão", `${show(auSnap.games_played)} / ${show(auSnap.max_games)}`, `Modo: ${auSnap.mode === "trainer_camp" ? "Campo de Treino" : "1x1 Clássico"}`],
      ["Placar (V / D / E)", `${show(auSnap.wins)} / ${show(auSnap.losses)} / ${show(auSnap.draws)}`, `AWR a cada ${show(auSnap.learn_every)} partidas`],
      ["Updates AWR (Hot-Reload)", show(auSnap.awr_updates), auSnap.last_action ? `Última jogada: ${auSnap.last_action}` : "Aguardando jogada"],
    ];
    byId("auto-session-strip").innerHTML = auTiles.map(([lbl, val, h]) => `
      <article class="metric">
        <span class="name">${escapeText(lbl)}</span>
        <strong class="number" style="font-size:1.35rem">${escapeText(val)}</strong>
        <small>${escapeText(h)}</small>
      </article>`).join("");
    byId("auto-session-msg").textContent = auSnap.message || "";
  } else {
    byId("auto-session-strip").innerHTML = "";
    byId("auto-session-msg").textContent = "";
  }

  // Update live logs (overlay + auto_trainer + learner)
  const lines = [];
  if (st.auto_trainer?.log?.length) {
    lines.push("─── Treino Real Autônomo (1v1 Clássico) ───", ...st.auto_trainer.log);
  }
  if (st.overlay?.log?.length) {
    if (lines.length) lines.push("─── Overlay Tático ───");
    lines.push(...st.overlay.log);
  }
  if (st.learner?.log?.length) {
    if (lines.length) lines.push("─── Aprendizado Contínuo (AWR) ───");
    lines.push(...st.learner.log);
  }
  const logNode = byId("live-log");
  if (lines.length) {
    const joined = lines.join("\n");
    if (logNode.textContent !== joined) {
      logNode.textContent = joined;
      logNode.scrollTop = logNode.scrollHeight;
    }
    if (ovRun || lrRun || auRun) byId("live-log-wrap").open = true;
  }

  // Update embedded capture preview image
  const mtime = st.previews?.[livePreviewKind] || 0;
  byId("live-tab-deck").setAttribute("aria-pressed", String(livePreviewKind === "deck"));
  byId("live-tab-arena").setAttribute("aria-pressed", String(livePreviewKind === "arena"));
  const img = byId("live-preview-img");
  const emptyNode = byId("live-preview-empty");
  if (mtime > 0) {
    const stamp = `${livePreviewKind}:${mtime}`;
    if (lastPreviewStamp !== stamp) {
      img.src = `/api/live/preview?kind=${livePreviewKind}&t=${mtime}`;
      lastPreviewStamp = stamp;
    }
    img.hidden = false;
    emptyNode.hidden = true;
  } else {
    img.hidden = true;
    emptyNode.hidden = false;
  }
}

async function refreshLiveStatus() {
  if (!connected && booted) return;
  try {
    const resp = await fetch(`/api/live/status?deck=${encodeURIComponent(deck)}`, {
      cache: "no-store", signal: AbortSignal.timeout(3500),
    });
    if (!resp.ok) return;
    liveStatus = await resp.json();
    renderLiveStatus();
  } catch {
    // ignore transient poll errors
  }
}

function showLiveFeedback(message, tone = "good") {
  const box = byId("live-action-feedback");
  box.hidden = !message;
  box.dataset.tone = tone;
  box.textContent = message;
}

async function runLiveAction(actionKind) {
  if (liveBusy) return;
  liveBusy = true;
  const buttons = [...document.querySelectorAll("[data-live-action]")];
  buttons.forEach(btn => { btn.disabled = true; });
  showLiveFeedback("Executando ação…", "");
  try {
    const payload = {
      action: actionKind,
      deck,
      run_id: byId("live-model-select").value || current,
      tower_level: byId("live-tower-level").value,
      deck_slot: byId("live-deck-slot").value,
      record: byId("live-opt-record").checked,
      ensure_deck: byId("live-opt-ensure-deck").checked,
      cards: byId("live-cards-order").value,
      auto_mode: byId("auto-mode").value,
      max_games: Number(byId("auto-max-games").value) || 20,
      learn_every: Number(byId("auto-learn-every").value) || 5,
      show_overlay: Boolean(byId("auto-show-overlay")?.checked ?? true),
      dry_run: byId("auto-dry-run").checked,
    };
    const resp = await fetch("/api/live/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const res = await resp.json();
    if (!resp.ok) throw new Error(res.error || "Não foi possível concluir a ação.");
    if (res.preview) {
      livePreviewKind = res.preview;
      lastPreviewStamp = "";
    }
    showLiveFeedback(res.message || "Concluído.", res.ok === false ? "warn" : "good");
    await refreshLiveStatus();
  } catch (err) {
    showLiveFeedback(err.message, "warn");
  } finally {
    liveBusy = false;
    buttons.forEach(btn => { btn.disabled = false; });
    renderLiveStatus();
  }
}

function renderLivePage(run) {
  syncLiveModelSelect(run);
  const chosenId = byId("live-model-select").value;
  const activeRun = data.runs.find(r => r.id === chosenId) || run;
  const info = activeRun?.model_info || {};
  const isLiveReady = info.obs_profile === "live";
  const ckptStage = activeRun?.checkpoints?.ppo ? "ppo.pt" : activeRun?.checkpoints?.bc ? "bc.pt" : "ppo.pt";
  const runRel = activeRun ? `runs\\${activeRun.id.replace(/\//g, "\\")}` : `runs\\live_v3\\${deck}`;
  const ckptPath = `${runRel}\\${ckptStage}`;
  const bcPath = `${runRel}\\bc.pt`;
  const liveOutPath = `${runRel}\\ppo_live.pt`;

  byId("live-readiness").innerHTML = `
    <div class="readiness-card" data-ready="${isLiveReady}">
      <div class="metric-head">
        <h3>${isLiveReady ? "Modelo pronto para o Clash Royale real" : "Perfil de visão incompatível com tela real"}</h3>
        <span class="metric-tag ${isLiveReady ? "good" : "warn"}">${escapeText(info.obs_profile || "sem modelo")}</span>
      </div>
      <p>${isLiveReady
        ? `O modelo selecionado usa o perfil <strong>live (34 canais)</strong>: recebe apenas o que a visão computacional enxerga (tropas, velocidades, elixir e cartas), sem depender de HP exato oculto.`
        : `O modelo selecionado usa perfil <strong>${escapeText(info.obs_profile || "v3")}</strong>. Para jogar no Google Play Games sem degradação Sim-to-Real, selecione um modelo <strong>★ LIVE</strong> acima ou treine um em <strong>Treinamento → Perfil de visão: live (34 canais)</strong>.`
      }</p>
      ${!isLiveReady ? `<div><a class="secondary" href="#treino">Treinar modelo com visão "live"</a></div>` : ""}
    </div>
    <div class="readiness-card">
      <h3>Como usar direto pelo site (1 vez + dia a dia)</h3>
      <ol class="readiness-steps">
        <li>Abra o Clash Royale (<strong>Google Play Games</strong>) na aba <strong>Decks</strong> com o X-Bow 3.0 montado e clique em <strong>Salvar 8 templates</strong> (só precisa 1 vez!).</li>
        <li>Antes de puxar partida, clique em <strong>Trocar deck (cliques)</strong> ou marque a opção automática e clique em <strong>Iniciar Overlay ao Vivo</strong>.</li>
        <li>Depois de acumular partidas gravadas, clique em <strong>Treinar com partidas reais (AWR)</strong>.</li>
      </ol>
    </div>`;

  const lm = data.live_matches || { count: 0, wins: 0, losses: 0, draws: 0, total_frames: 0, matches: [] };
  const wrLive = lm.count > 0 ? lm.wins / lm.count : null;
  const kpis = [
    ["Partidas gravadas", show(lm.count), "Arquivos JSONL em runs/live_matches"],
    ["Vitórias reais", show(lm.wins), wrLive !== null ? `Taxa de vitórias: ${show(wrLive, percent)}` : "Nenhuma partida registrada ainda"],
    ["Derrotas / Empates", `${show(lm.losses)} / ${show(lm.draws)}`, "Usadas no aprendizado ponderado AWR"],
    ["Quadros capturados", show(lm.total_frames), "Decisões gravadas para treino offline"],
  ];
  byId("live-kpis").innerHTML = kpis.map(([label, val, hint]) => `
    <article class="metric">
      <span class="name">${label}</span>
      <strong class="number">${val}</strong>
      <small>${escapeText(hint)}</small>
    </article>`).join("");

  const commands = [
    {
      title: "1. Salvar 8 templates reais (1 única vez na tela de Decks)",
      desc: "Recorta os 8 ícones reais da sua tela para reconhecimento >90%.",
      cmd: `.\\.venv\\Scripts\\python.exe xbow_screen_vision.py --deck ${deck} --save-deck-templates`,
    },
    {
      title: "2. Ir ao Deck e garantir deck correto por cliques",
      desc: "Navega para Coleção → Decks, valida os 8 slots e troca de slot salvo se necessário.",
      cmd: `.\\.venv\\Scripts\\python.exe xbow_screen_vision.py --deck ${deck} --ensure-deck`,
    },
    {
      title: "3. Jogar ao vivo no Google Play Games (Visão local + Overlay + Gravador)",
      desc: "Captura a janela do Clash Royale, exibe o alvo exato na tela e grava em runs\\live_matches.",
      cmd: `.\\.venv\\Scripts\\python.exe xbow_live_overlay.py --checkpoint ${ckptPath} --deck ${deck} --vision-local --ensure-deck --record-dir runs\\live_matches`,
    },
    {
      title: "4. Aprendizado Contínuo Offline (AWR + Reverse KL)",
      desc: "Atualiza o cérebro da IA usando as partidas reais gravadas, dando mais peso às vitórias.",
      cmd: `.\\.venv\\Scripts\\python.exe xbow_live_learn.py --checkpoint ${ckptPath} --reference ${bcPath} --matches-dir runs\\live_matches --output ${liveOutPath} --deck ${deck}`,
    },
  ];
  byId("live-commands").innerHTML = commands.map((item, idx) => `
    <div class="cmd-card">
      <div class="cmd-head">
        <div><strong>${escapeText(item.title)}</strong><small>${escapeText(item.desc)}</small></div>
        <button type="button" class="copy-btn" data-copy-idx="${idx}">Copiar comando</button>
      </div>
      <pre><code id="live-cmd-${idx}">${escapeText(item.cmd)}</code></pre>
    </div>`).join("");

  const resultLabels = { win: "Vitória", loss: "Derrota", draw: "Empate", unknown: "Em aberto" };
  byId("live-matches-table").innerHTML = lm.matches?.length
    ? table(["Arquivo", "Deck", "Resultado", "Quadros", "Início"],
        lm.matches.map(m => [
          m.file,
          deckNames[m.deck] || m.deck,
          `<span class="badge-result ${escapeText(m.result)}">${escapeText(resultLabels[m.result] || m.result)}</span>`,
          show(m.frames),
          m.started ? escapeText(m.started.replace("T", " ").slice(0, 19)) : "—",
        ]))
    : empty("Nenhuma partida real gravada ainda. Clique em Iniciar Overlay ao Vivo com a gravação marcada para começar.");
}

/* ==================== CONTROLE DE JOBS & RENDER ==================== */
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
  const running = activeJob(job);
  const liveRun = running ? data.runs.find(item => item.id === job.run_id) : run;
  const progress = liveRun?.progress;
  const liveProgress = running && Date.parse(progress?.time) >= Date.parse(job.started) ? progress : {};
  const viewPhase = liveProgress.event || "";
  const stage = ["replays", "bc", "ppo", "evaluation"].find(key => viewPhase.startsWith(key));
  window.trainingArena.update({
    deck: running ? job.deck : deck, run: running ? job.run_id : current, active: running,
    phase: stage || (job?.mode === "resume" ? "ppo" : "replays"),
    detail: running ? stageProgress(job, liveProgress)[2] : "",
    updates: liveRun?.updates || [],
    progress: liveProgress,
    job,
    modelInfo: liveRun?.model_info || null,
    recommended: data.recommended || null,
    system: data.system || null,
  });

  const next = JSON.stringify([run, deck, data.live_matches]);
  if (!force && revision === next) return;
  revision = next;
  const pairs = rivalRows(run?.evaluation);
  byId("status-panel").innerHTML = phase(run);
  overview(run, pairs);
  renderPPODiagnostics(run);
  renderPFSPBoard(run);
  matchups(pairs);
  validation(run);
  renderLivePage(run);

  series = run?.updates || [];
  epochs = run?.epochs || [];
  const metric = byId("chart-metric");
  for (const opt of metric.options) {
    if (charts[opt.value]?.isBC) {
      opt.disabled = !epochs.length && Boolean(series.length);
    }
  }
  if (charts[metric.value]?.isBC && !epochs.length && series.length) metric.value = "win";
  if (series.length && current === data.job?.run_id && chartJobId !== data.job.id) {
    if (charts[metric.value]?.isBC) metric.value = "win";
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
    if (activePage === "ao-vivo") refreshLiveStatus();
  } catch {
    connected = false;
    renderTraining();
    byId("sync").textContent = "Sem conexão";
    document.querySelector(".connection").classList.add("offline");
  } finally {
    setTimeout(refresh, 2500);
  }
}

/* ==================== EVENTOS ==================== */
byId("run").addEventListener("change", event => { current = event.target.value; pending = ""; render(true); });
document.querySelector(".deck-switch").addEventListener("click", event => {
  const next = event.target.closest("[data-deck]")?.dataset.deck;
  if (!next || next === deck) return;
  deck = next;
  pending = "";
  render(true);
});
byId("chart-metric").addEventListener("change", drawChart);
byId("chart-smooth").addEventListener("change", drawChart);

byId("training-settings").addEventListener("input", event => {
  const ctrl = event.target;
  if (ctrl.id === "training-obs-profile" || ctrl.id === "training-batch-size") {
    renderTraining();
    return;
  }
  if (ctrl.type !== "range") return;
  if (ctrl.id === "training-mode") {
    renderTraining();
  } else {
    if (ctrl.id === "training-matches") replaySelection[deck] = ctrl.valueAsNumber;
    syncSlider(ctrl);
  }
});

byId("training-max-replays").addEventListener("click", () => {
  const slider = byId("training-matches");
  if (slider.disabled) return;
  const maxVal = Number(slider.max);
  slider.value = maxVal;
  replaySelection[deck] = maxVal;
  syncSlider(slider, true);
});

byId("training-form").addEventListener("submit", event => {
  event.preventDefault();
  if (byId("start-training").disabled || activeJob(data.job)) return;
  const payload = {
    deck,
    mode: executionMode(),
    run_id: current,
    obs_profile: byId("training-obs-profile").value,
    batch_size: Number(byId("training-batch-size").value),
  };
  const coevolve = byId("training-coevolve");
  payload.coevolve = !coevolve.disabled && coevolve.checked;
  const continuous = byId("training-continuous");
  payload.continuous = !continuous.disabled && continuous.checked;
  if (payload.continuous && !data.continuous_parallel) {
    byId("training-hint").textContent = "Reinicie o servidor do painel para ativar as partidas simultâneas.";
    return;
  }
  for (const key of parameterKeys) {
    payload[key] = byId(`training-${key.replace("_", "-")}`).valueAsNumber;
  }
  if (payload.continuous) payload.games = byId("training-continuous-games").valueAsNumber;
  jobAction("start", payload);
});

byId("start-recommended")?.addEventListener("click", () => {
  if (byId("start-recommended").disabled || activeJob(data.job)) return;
  const maxThreads = data.system?.max_threads || 8;
  jobAction("start", {
    deck,
    mode: "recommended",
    run_id: current,
    train_vision: Boolean(byId("recommended-train-vision")?.checked),
    updates: 4,
    games: 10,
    evaluate: 2,
    threads: Math.min(8, maxThreads),
  });
});

byId("training-coevolve").addEventListener("change", renderTraining);
byId("training-continuous").addEventListener("change", renderTraining);
byId("stop-job").addEventListener("click", () => jobAction("stop", { id: data.job.id }));
byId("view-job").addEventListener("click", () => {
  follow(data.job);
  location.hash = "#visao";
  render(true);
});

byId("auto-train-panel").addEventListener("click", event => {
  const actionBtn = event.target.closest("[data-live-action]");
  if (actionBtn) runLiveAction(actionBtn.dataset.liveAction);
});
byId("live-control-panel").addEventListener("click", event => {
  const actionBtn = event.target.closest("[data-live-action]");
  if (actionBtn) {
    runLiveAction(actionBtn.dataset.liveAction);
    return;
  }
  const tabBtn = event.target.closest("[data-preview-kind]");
  if (tabBtn) {
    livePreviewKind = tabBtn.dataset.previewKind;
    lastPreviewStamp = "";
    renderLiveStatus();
  }
});
byId("live-model-select").addEventListener("change", () => renderLivePage(selected()));

byId("live-commands").addEventListener("click", async event => {
  const btn = event.target.closest("[data-copy-idx]");
  if (!btn) return;
  const text = byId(`live-cmd-${btn.dataset.copyIdx}`)?.textContent || "";
  try {
    await navigator.clipboard.writeText(text);
    btn.dataset.copied = "true";
    btn.textContent = "Copiado!";
    setTimeout(() => { btn.dataset.copied = "false"; btn.textContent = "Copiar comando"; }, 1800);
  } catch {
    btn.textContent = "Selecione manualmente";
  }
});

byId("chart").addEventListener("pointermove", event => {
  if (!points.length) return;
  const point = points.reduce((nearest, next) =>
    Math.abs(next.x - event.offsetX) < Math.abs(nearest.x - event.offsetX) ? next : nearest);
  const tooltip = byId("plot-tooltip");
  tooltip.hidden = false;
  const metric = byId("chart-metric").value;
  const spec = charts[metric] || charts.win;
  const maText = Number.isFinite(point.ma) ? ` · MM5: ${show(point.ma, spec.format)}` : "";
  tooltip.textContent = `${spec.isBC ? "Época" : "Atualização"} ${point.update} · ${show(point.value, spec.format)}${maText}`;
  tooltip.style.left = `${Math.max(4, Math.min(point.x + 6, byId("plot").clientWidth - tooltip.offsetWidth - 4))}px`;
});
byId("chart").addEventListener("pointerleave", () => { byId("plot-tooltip").hidden = true; });

window.addEventListener("hashchange", applyRoute);
applyRoute();
new ResizeObserver(() => { if (activePage === "evolucao") drawChart(); }).observe(byId("plot"));
render();
refresh();
