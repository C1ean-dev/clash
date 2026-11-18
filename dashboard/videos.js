"use strict";

(() => {
  const el = id => document.getElementById(`video-${id}`);
  const player = el("library-player");
  const format = new Intl.NumberFormat("pt-BR");
  const deckNames = { xbow_30: "X-Bow 3.0", icebow: "IceBow" };
  const arenas = { ranked: "Ranquiado", arena: "Arena" };
  let active = "";
  let request = null;
  let page = 1;
  let searchTimer;

  function arenaName(row) {
    const arena = row.arena || "";
    const name = arena.replace(/^(ranked|arena)_?/, prefix => `${arenas[prefix.replace("_", "")]} `);
    return name.replaceAll("_", " ").trim() || "Royale TV";
  }

  function play(row) {
    active = row.id;
    for (const button of el("rows").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(button.dataset.id === active));
    }
    player.hidden = false;
    player.src = row.video;
    el("instruction").hidden = true;
    el("playing").hidden = false;
    el("playing").textContent = row.speed ? `Gravação feita a ${row.speed}× · ${row.id}` : row.id;
    const link = el("source-link");
    link.href = row.source;
    link.hidden = false;
    player.play().catch(() => {});
  }

  async function load(requestedPage = 1) {
    request?.abort();
    const controller = new AbortController();
    request = controller;
    const deck = el("deck").value;
    const wins = el("wins");
    wins.disabled = !deck;
    const params = new URLSearchParams({ page: requestedPage, deck, q: el("search").value,
      wins: deck && wins.checked ? "1" : "0" });
    try {
      const response = await fetch(`/api/videos?${params}`, { signal: controller.signal });
      if (!response.ok) throw new Error();
      const data = await response.json();
      if (controller.signal.aborted) return;
      el("count").textContent = data.available ?
        `${format.format(data.found)} de ${format.format(data.total)} gravações · AlphaClash` :
        "Catálogo AlphaClash indisponível.";
      page = data.page;
      el("page").textContent = `${format.format(page)} de ${format.format(data.pages)}`;
      el("previous").disabled = page <= 1;
      el("next").disabled = page >= data.pages;
      el("rows").replaceChildren(...data.rows.map(row => {
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.id = row.id;
        button.setAttribute("aria-pressed", String(row.id === active));
        const title = document.createElement("span");
        title.textContent = row.decks.map(deck => deckNames[deck]).filter(Boolean).join(" × ") || arenaName(row);
        const detail = document.createElement("small");
        const score = Array.isArray(row.crowns) ? row.crowns.join("–") : "";
        detail.textContent = [row.id.slice(0, 8).replace(/^(\d{4})(\d{2})(\d{2})$/, "$3/$2/$1"), score]
          .filter(Boolean).join(" · ");
        button.append(title, detail);
        button.addEventListener("click", () => play(row));
        item.append(button);
        return item;
      }));
    } catch {
      if (!controller.signal.aborted) el("count").textContent = "Não foi possível carregar o catálogo.";
    }
  }

  player.addEventListener("error", () => {
    el("playing").textContent = "A reprodução está indisponível aqui. Você pode abrir o vídeo na fonte.";
  });
  el("deck").addEventListener("change", () => load());
  el("wins").addEventListener("change", () => load());
  el("search").addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(load, 200);
  });
  el("previous").addEventListener("click", () => load(page - 1));
  el("next").addEventListener("click", () => load(page + 1));
  load();
})();
