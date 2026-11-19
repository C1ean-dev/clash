"use strict";

(() => {
  const el = id => document.getElementById("catalog-" + id);
  const {skin, portraitPosition} = window.battleArt;
  const byName = new Map();
  const decks = new Map();
  const deckButtons = new Map();
  const configured = new Set();
  const numbers = new Intl.NumberFormat("pt-BR", {maximumFractionDigits: 1});
  const kinds = {TROOP: "Tropa", BUILDING: "Construção", SPELL: "Feitiço"};
  const fold = text => text.normalize("NFD").replace(/\p{Diacritic}/gu, "").toLowerCase();

  function element(tag, className, text = "") {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = text;
    return node;
  }

  function illustration(card) {
    const [glyph, tint, portrait] = skin(card.name);
    const node = element("span", "catalog-card-image", glyph);
    node.style.setProperty("--card-tone", tint);
    if (portrait !== undefined) {
      node.setAttribute("data-portrait", "");
      node.style.backgroundPosition = portraitPosition(portrait);
    }
    node.setAttribute("aria-hidden", "true");
    return node;
  }

  function deckTile(deck) {
    const cards = deck.cards.map(name => byName.get(name));
    const item = element("li");
    const button = element("button", "catalog-deck");
    button.type = "button";
    button.setAttribute("aria-pressed", "false");
    deckButtons.set(deck.id, button);
    const heading = element("span", "catalog-deck-heading", deck.label);
    const average = cards.reduce((total, card) => total + card.elixir, 0) / cards.length;
    heading.append(element("small", "muted", numbers.format(average) + " elixir médio"));
    const thumbnails = element("span", "catalog-thumbnails");
    for (const card of cards) {
      const image = illustration(card);
      image.title = card.label;
      thumbnails.append(image);
    }
    button.append(heading, thumbnails);
    button.addEventListener("click", () => {
      const filter = el("deck-filter");
      filter.value = filter.value === deck.id ? "" : deck.id;
      render();
    });
    item.append(button);
    return item;
  }

  function cardTile(card) {
    const node = element("li", "catalog-card");
    const displayName = card.label;
    const kind = card.kind === "TROOP" && card.flying ? "Tropa aérea" : kinds[card.kind];
    node.append(illustration(card), element("strong", "", displayName),
      element("small", "muted", kind), element("span", "catalog-card-cost", card.elixir + " elixir"));
    return node;
  }

  function render() {
    const filter = el("deck-filter").value;
    const deck = decks.get(filter);
    const query = fold(el("search").value.trim());
    const cards = deck ? deck.cards.map(name => byName.get(name))
      : [...byName.values()].filter(card => filter === "all" || configured.has(card.name));
    const results = cards.filter(card => fold(card.label + " " + card.name).includes(query));
    for (const [id, button] of deckButtons) button.setAttribute("aria-pressed", String(id === filter));
    el("cards").replaceChildren(...results.map(cardTile));
    const play = el("play");
    play.hidden = !deck;
    if (deck) play.href = `/?deck=${encodeURIComponent(deck.id)}#jogar`;
    el("status").textContent = results.length
      ? numbers.format(results.length) + (results.length === 1 ? " carta" : " cartas")
      : "Nenhuma carta encontrada.";
  }

  el("deck-filter").addEventListener("change", render);
  el("search").addEventListener("input", render);

  async function load() {
    try {
      const response = await fetch("/api/catalog");
      if (!response.ok) throw new Error();
      const catalog = await response.json();
      for (const card of catalog.cards.sort((a, b) => a.label.localeCompare(b.label, "pt-BR"))) {
        byName.set(card.name, card);
      }
      for (const deck of catalog.decks) {
        decks.set(deck.id, deck);
        for (const name of deck.cards) configured.add(name);
        el("deck-filter").append(new Option(deck.label, deck.id));
      }
      el("decks").replaceChildren(...catalog.decks.map(deckTile));
      el("browser").hidden = false;
      render();
    } catch {
      el("status").textContent = "Catálogo indisponível. Abra a versão atualizada do painel.";
    }
  }

  load();
})();
