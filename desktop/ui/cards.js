// The `.openhydra` card UI (M2): export a signed "magnet for a model" from a shared model, and
// import someone else's card to route to their model by peer id — no live discovery. Backs the
// per-row "Card" action in the Share view and the "Add a model by card" section (#importsection).
import { $, $$, escapeHtml, peerShort } from "./dom";
import { call } from "./bridge";
import { toast } from "./chrome";
import { emit } from "./bus";
import { displayModelName } from "./models";

const CARD_TTL_SECS = 30 * 24 * 3600; // 30 days

// ── export: sign + show a card (magnet + JSON). Public for a global model; private (swarm-bound,
// M4) for a private one — the caller passes `isPrivate` and we resolve which owned swarm to bind. ──
export async function exportCardModal(model, isPrivate = false) {
  let swarm = null;
  if (isPrivate) {
    // A private card must name a swarm the user OWNS. Pick from owned swarms; guide them to Swarms if
    // none. (One owned swarm → use it; several → ask.)
    let owned = [];
    try {
      owned = (await call("list_swarms")).filter((s) => s.role === "owner");
    } catch {}
    if (!owned.length) {
      toast("Create a swarm first (Swarms tab) — a private card is bound to a swarm you own");
      return;
    }
    swarm = owned.length === 1 ? owned[0].swarm_public_key : await pickSwarm(owned);
    if (!swarm) return; // cancelled
  }
  let out;
  try {
    out = await call("export_card", { model, ttl_secs: CARD_TTL_SECS, region: null, swarm });
  } catch (e) {
    toast(`Export failed: ${e}`);
    return;
  }
  const { card, magnet } = out;
  const json = JSON.stringify(card, null, 2);
  const priv = !!card.swarm_public_key;
  const blurb = priv
    ? `Send this to a swarm member and they can add your model — connecting to you directly. It's a <b>private</b> card: only a member of your swarm can actually serve from it (others are refused). Signed, carries no secret, expires in 30 days.`
    : `Send this to someone and they can add your model to their OpenHydra — connecting to you directly, no discovery needed. It's signed, carries no secret, and expires in 30 days.`;
  const back = document.createElement("div");
  back.className = "cmodal-back";
  back.innerHTML =
    `<div class="cmodal" style="max-width:560px"><div class="cmodal-h"><b>Share “${escapeHtml(displayModelName(model))}” as a ${priv ? "private " : ""}card</b></div>` +
    `<div class="cmodal-b">` +
      `<div class="cmodal-path">${blurb}</div>` +
      `<label class="mut" style="font-size:11px;margin-top:10px;display:block">Magnet string</label>` +
      `<textarea class="input mono cardmagnet" rows="3" readonly style="width:100%;font-size:10.5px">${escapeHtml(magnet)}</textarea>` +
    `</div>` +
    `<div class="cmodal-f"><button class="btn ghost sm ccopyjson">Copy JSON</button><button class="btn sm brand ccopymag">Copy magnet</button></div></div>`;
  document.body.appendChild(back);
  const close = () => back.remove();
  $(".ccopymag", back).onclick = () => { navigator.clipboard?.writeText(magnet); toast("Card copied — send it to add your model"); close(); };
  $(".ccopyjson", back).onclick = () => { navigator.clipboard?.writeText(json); toast("Card JSON copied"); };
  back.onclick = (e) => { if (e.target === back) close(); };
}

// Choose which owned swarm a private card binds to (when the user owns more than one). Resolves to a
// swarm_public_key, or null if cancelled.
function pickSwarm(owned) {
  return new Promise((resolve) => {
    const back = document.createElement("div");
    back.className = "cmodal-back";
    back.innerHTML =
      `<div class="cmodal" style="max-width:460px"><div class="cmodal-h"><b>Which swarm is this card for?</b></div>` +
      `<div class="cmodal-b"><div class="cmodal-path">Only members of the swarm you pick will be able to serve from this card.</div>` +
      owned
        .map(
          (s) =>
            `<button class="btn outline sm pickswarm" data-pk="${escapeHtml(s.swarm_public_key)}" style="display:block;width:100%;text-align:left;margin-top:8px">` +
            `<b>${escapeHtml(s.label || "Swarm")}</b> <span class="mut mono" style="font-size:10.5px">${escapeHtml(s.fingerprint)}</span></button>`,
        )
        .join("") +
      `</div><div class="cmodal-f"><button class="btn ghost sm pickcancel">Cancel</button></div></div>`;
    document.body.appendChild(back);
    const done = (v) => { back.remove(); resolve(v); };
    $$(".pickswarm", back).forEach((b) => (b.onclick = () => done(b.dataset.pk)));
    $(".pickcancel", back).onclick = () => done(null);
    back.onclick = (e) => { if (e.target === back) done(null); };
  });
}

// ── M2.1: launch-on-open. A double-clicked `.openhydra` file or an `openhydra:` magnet link opens
// the app (or focuses the running one) and routes the card text here → the Share import box, verified.
// Rate-limited entry point (review fix #2): a malicious site can fire `openhydra:` links in a loop,
// each of which would otherwise nav-to-Share + run a full signature verify. Leading-edge cooldown —
// the first open processes IMMEDIATELY (no delay for a legit single open), and any further opens
// within OPEN_COOLDOWN_MS are ignored, so a flood can trigger the work at most once per window.
const OPEN_COOLDOWN_MS = 500;
let lastCardOpenAt = 0;
export function openCardText(text) {
  if (!text || !text.trim()) return;
  const now = Date.now();
  if (now - lastCardOpenAt < OPEN_COOLDOWN_MS) return; // within cooldown → drop (flood guard)
  lastCardOpenAt = now;
  doOpenCard(text.trim());
}

async function doOpenCard(text) {
  emit("nav", "share"); // switch to the Share view
  await new Promise((r) => setTimeout(r, 60)); // let the view render
  await ensureImportSection(); // build the import UI if it isn't yet
  const input = document.querySelector("#cardinput");
  if (!input) return;
  input.value = text;
  input.scrollIntoView({ behavior: "smooth", block: "center" });
  document.querySelector("#cardpreviewbtn")?.click(); // auto-verify → shows the preview + Add button
}

/// Wire the two OS delivery channels: a `card-opened` event (app already running) and a one-shot
/// `take_pending_card` pickup (the app was launched BY a card, before the UI was ready). No-op in the
/// browser mock (no `window.__TAURI__`).
export async function wireCardDeepLinks() {
  try {
    await window.__TAURI__?.event?.listen?.("card-opened", (e) => {
      // Also drain the pending slot the backend set alongside this event, so a later webview reload
      // can't re-take a stale card and re-preview it (review fix #3).
      call("take_pending_card").catch(() => {});
      openCardText(e.payload);
    });
  } catch {}
  try {
    const pending = await call("take_pending_card");
    if (pending) openCardText(pending);
  } catch {}
}

// ── import: paste → preview → add, + the imported-cards list ──
// Built once (the panel persists across nav), so a status refresh never wipes the user's paste.
let built = false;

export async function ensureImportSection() {
  const host = $("#importsection");
  if (!host || built) return;
  built = true;
  host.innerHTML =
    `<div class="card">` +
      `<div class="row" style="padding:14px 16px 8px"><div class="ctitle">Add a model by card</div></div>` +
      `<div style="padding:0 16px 14px">` +
        `<div class="mut" style="font-size:11.5px;margin-bottom:8px">Paste an <b>openhydra:card:…</b> magnet (or a <b>.openhydra</b> file's contents) to add someone's model. You connect to them directly by peer id — no discovery needed.</div>` +
        `<textarea id="cardinput" class="input mono" rows="3" placeholder="openhydra:card:…   or the .openhydra JSON" style="width:100%;font-size:11px"></textarea>` +
        `<div class="row" style="margin-top:8px;gap:8px"><button class="btn outline sm" id="cardpreviewbtn">Preview</button><button class="btn brand sm" id="cardaddbtn" style="display:none">Add model</button></div>` +
        `<div id="cardpreviewout" style="margin-top:8px"></div>` +
      `</div>` +
      `<div id="importedlist"></div>` +
    `</div>`;

  const input = $("#cardinput", host);
  const previewOut = $("#cardpreviewout", host);
  const addBtn = $("#cardaddbtn", host);
  let previewed = null; // the exact text that last previewed OK (imported verbatim on Add)

  $("#cardpreviewbtn", host).onclick = async () => {
    const text = (input.value || "").trim();
    if (!text) { toast("Paste a card first"); return; }
    previewOut.innerHTML = `<span class="mut" style="font-size:11.5px">Verifying…</span>`;
    addBtn.style.display = "none";
    previewed = null;
    try {
      const c = await call("preview_card", { input: text });
      previewOut.innerHTML = cardPreviewHtml(c);
      previewed = text;
      addBtn.style.display = "";
    } catch (e) {
      previewOut.innerHTML = `<div class="cardfail">✕ ${escapeHtml(String(e))}</div>`;
    }
  };
  addBtn.onclick = async () => {
    if (!previewed) return;
    try {
      const c = await call("import_card", { input: previewed });
      toast(`Added ${displayModelName(c.model_id)} via card`);
      input.value = "";
      previewOut.innerHTML = "";
      addBtn.style.display = "none";
      previewed = null;
      await renderImportedList();
      emit("refresh"); // pull fresh status so the model surfaces where it's routable
    } catch (e) {
      toast(`Import failed: ${e}`);
    }
  };

  // Convenience: dropping a .openhydra file onto the textarea fills it in.
  input.addEventListener("drop", (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) {
      e.preventDefault();
      f.text().then((t) => { input.value = t.trim(); });
    }
  });

  await renderImportedList();
}

function cardPreviewHtml(c) {
  const exp = new Date(Number(c.expires_at)).toLocaleDateString();
  const price =
    c.pricing_mode === "paid" ? "paid" : c.pricing_mode === "ad_supported" ? "ad-supported" : "free (reciprocal)";
  const region = c.region ? `${escapeHtml(c.region)} · ` : "";
  // M4: a swarm-bound card is private — serving needs a membership credential for that swarm.
  const priv = c.swarm_public_key
    ? ` <span class="badge secondary" title="Private — you must be a member of this swarm to serve from it">🔒 private</span>`
    : "";
  return (
    `<div class="card pad" style="font-size:11.5px">` +
    `<div><b>${escapeHtml(displayModelName(c.model_id))}</b> <span class="badge ok">✔ signature valid</span>${priv}</div>` +
    `<div class="mut" style="margin-top:4px">provider ${escapeHtml(peerShort(c.libp2p_peer_id))} · ${escapeHtml(price)} · ${region}expires ${escapeHtml(exp)}</div>` +
    (c.swarm_public_key ? `<div class="mut" style="margin-top:2px">swarm ${escapeHtml(peerShort(c.swarm_public_key))} — you'll serve from it only if you're a member</div>` : "") +
    (c.canonical_id ? `<div class="mut" style="margin-top:2px">${escapeHtml(c.canonical_id)}</div>` : "") +
    `</div>`
  );
}

async function renderImportedList() {
  const host = $("#importedlist");
  if (!host) return;
  let cards = [];
  try { cards = await call("list_cards"); } catch {}
  if (!cards.length) { host.innerHTML = ""; return; }
  const rows = cards
    .map((c) => {
      const exp = new Date(Number(c.expires_at)).toLocaleDateString();
      return (
        `<tr>` +
        `<td><b>${escapeHtml(displayModelName(c.model_id))}</b> <span class="badge secondary" title="Added by importing a signed card">via card</span></td>` +
        `<td class="mut">${escapeHtml(peerShort(c.libp2p_peer_id))}</td>` +
        `<td class="mut">expires ${escapeHtml(exp)}</td>` +
        // escapeHtml (not esc) — a card is untrusted remote input; esc doesn't escape the `"` that
        // would break out of the attribute (event-handler injection XSS). model_id is free-form.
        `<td style="text-align:right"><button class="btn ghost sm cardrm" data-lib="${escapeHtml(c.libp2p_peer_id)}" data-model="${escapeHtml(c.model_id)}">Remove</button></td>` +
        `</tr>`
      );
    })
    .join("");
  host.innerHTML =
    `<div class="row" style="padding:8px 16px 4px"><div class="ctitle" style="font-size:12px">Added via card</div></div>` +
    `<table style="margin:0 6px 8px"><tbody>${rows}</tbody></table>`;
  $$(".cardrm", host).forEach((b) => {
    b.onclick = async () => {
      try {
        await call("remove_imported_card", { libp2p_peer_id: b.dataset.lib, model_id: b.dataset.model });
        toast("Removed");
        await renderImportedList();
        emit("refresh");
      } catch (e) {
        toast(`Remove failed: ${e}`);
      }
    };
  });
}
