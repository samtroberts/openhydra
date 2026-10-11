// Network data tables: the Ledger (recent co-signed transactions), the Diagnostics/Peers view
// (peer list + DHT records + swarm + logs), and the provider/gateway log pane. peerLimit and logTab
// are module-local; setLogTab lets the log chips switch the pane. infraPeerIds is shared from ./state.
import { $, $$, peerShort } from "./dom";
import { html, setHTML } from "./html";
import { state, snap, infraPeerIds } from "./state";
import { repByLibp2p } from "./econ";
import { displayModelName } from "./models";
import { modelIcon, relTime, repBadge } from "./format";
import { injectIcons } from "./icons";
import { menu, toast } from "./chrome";

  // #5: real ledger rows from the agent's recent-transaction ring (served rows from the provider
  // process, used rows from the gateway; merged + newest-first by the desktop). The credit column
  // is a signed contribution unit (served +, used −; "not money"), not a wallet balance.
  export function renderLedger() {
    const t = snap?.transfers, rows = t?.recent || [];
    $("#v-ledger .row .mut").textContent = rows.length
      ? `${rows.length} recent · ${t?.receipts_ledgered ?? 0} co-signed · ${t?.tokens_served ?? 0} served / ${t?.tokens_consumed ?? "—"} used tokens`
      : `${t?.receipts_ledgered ?? 0} receipts · ${t?.tokens_served ?? 0} tokens served`;
    setHTML($("#ledgertable tbody"), rows.length
      ? html`${rows.slice(0, 100).map((r) => {
          // Credits on the same cost-weighted scale as Activity's net balance (rows from before it was
          // recorded have none).
          const served = r.kind === "served", cr = r.credits == null ? "—" : (served ? "+" : "−") + Number(r.credits).toLocaleString();
          return html`<tr><td class="mut">${relTime(r.ts_ms)}</td><td><span class="badge ${served ? "ok" : "secondary"}">${served ? "served" : "used"}</span></td><td title="${displayModelName(r.model)}">${modelIcon(r.model)}${displayModelName(r.model)}</td><td class="mono" title="${r.counterparty} (the other machine's id)">${peerShort(r.counterparty)}</td><td class="num">${r.tokens}</td><td class="num ${served ? "up" : ""}" style="${served ? "" : "color:hsl(var(--danger))"}">${cr}</td></tr>`;
        })}`
      : html`<tr><td colspan="6" class="mut">No transactions yet — serve a model or run a chat, and co-signed receipts appear here. Recent activity is kept in memory (launch the agent with a ledger DB for full history).</td></tr>`);
  }

  // infraPeerIds (bootstraps + relay hops we hide from the Peers list) now lives in ./state so the
  // status-bar peer count can share it — see the note there. Keeps the two "peers" counts identical.
  let peerLimit = 10;   // "View more" bumps this
  export function renderPeers() {
    if (!snap) { setHTML($("#peertable tbody"), html`<tr><td colspan="5" class="mut">Turn on Sharing or chat to connect, then peers appear here.</td></tr>`); const pc = $("#peercount"); if (pc) pc.textContent = "0 peers"; const pm = $("#peermore"); if (pm) pm.style.display = "none"; return; }
    const n = snap.network;
    const infra = infraPeerIds();
    const peers = (n.peers || []).filter((p) => !infra.has(p.peer_id));   // hide bootstraps/relays
    const repL = repByLibp2p();
    const shown = peers.slice(0, peerLimit);
    setHTML($("#peertable tbody"), shown.length ? html`${shown.map((p) => html`<tr data-p="${p.path}"><td class="mono">${peerShort(p.peer_id)}</td><td><span class="badge ${p.path === "direct" ? "ok" : p.path === "relay" ? "warn" : "secondary"}">${p.path}</span></td><td class="num">${p.quic_direct_v6}</td><td>${repBadge(repL[p.peer_id])}</td><td class="rowmenu mut" data-peer="${p.peer_id}"><span class="icon" data-i="more"></span></td></tr>`)}` : html`<tr><td colspan="5" class="mut">${n.peers.length ? "Only infrastructure connected — waiting for network peers." : "No peers connected yet — connecting."}</td></tr>`);
    injectIcons($("#peertable"));
    // dynamic count + View more
    const pc = $("#peercount"); if (pc) pc.textContent = peers.length <= peerLimit ? `${peers.length} peer${peers.length === 1 ? "" : "s"}` : `${shown.length} of ${peers.length} peers`;
    const pm = $("#peermore"); if (pm) { pm.style.display = peers.length > peerLimit ? "" : "none"; pm.onclick = () => { peerLimit += 10; renderPeers(); }; }
    // re-apply the active Direct/Relay/Mixed chip so the 2.5s poll doesn't reset it to All
    const pp = $("#peerchips .chip.on")?.dataset.p || "all";
    // …and the header search, so a filter survives the refresh too.
    const q = ($("#search")?.value || "").toLowerCase();
    $$("#peertable tbody tr").forEach((r) => r.style.display = (pp === "all" || r.dataset.p === pp) && (!q || r.textContent.toLowerCase().includes(q)) ? "" : "none");
    $("#actchips .chip .num") && ($("#actchips .chip .num").textContent = peers.length);
    $$("#peertable .rowmenu").forEach((cell) => cell.onclick = (e) => { e.stopPropagation(); menu(cell, [{ label: "Copy peer id", fn: async () => { try { await navigator.clipboard.writeText(cell.dataset.peer); toast("Peer id copied"); } catch (err) { toast(`Copy failed: ${err}`); } } }]); });
    // DHT
    setHTML($$("#v-peers .acttab")[1].querySelector("tbody"), (n.known_models || []).length ? html`${(snap.network.known_models).map((m) => html`<tr><td class="mono" title="${displayModelName(m)}">/oh/model/${displayModelName(m)}</td><td><span class="badge secondary">provider</span></td><td class="num">${(snap.network.known_providers || []).filter((p) => p.model_id === m).length || 1}</td></tr>`)}` : html`<tr><td colspan="3" class="mut">No records yet.</td></tr>`);
    setHTML($$("#v-peers .acttab")[1].querySelector(".card.pad"), html`kad_routing_peers: <span class="num">${n.kad_routing_peers}</span> · server mode: ${n.kad_server_mode ? "yes" : "no"}`);
    // Swarm
    const sw = $$("#v-peers .acttab")[2].querySelectorAll(".kpi .val"); sw[0].textContent = n.listen_addrs.length; sw[1].textContent = n.relay_reservations.length; sw[2].textContent = n.counters.dcutr_successes; sw[3].textContent = n.autonat_private ? "private" : "public";
    // Logs
    renderLogs();
  }
  let logTab = "provider";
  // Warnings from the network layer (libp2p, QUIC) are routine on real networks: dimmed, with a note,
  // so they don't read as faults.
  const NOISY = /\b(WARN|DEBUG)\b.*\b(libp2p|quinn|quinn_udp|yamux|multistream|kad|mdns|netlink)\b/i;
  export function renderLogs() {
    const logs = (logTab === "provider" ? state?.provider?.logs : state?.gateway?.logs) || [];
    setHTML($("#logbody"), logs.length
      ? html`<div class="mut" style="font-family:var(--sans);margin-bottom:6px">Dimmed lines are routine network-layer warnings.</div>${logs.map((l, i) => html`${i ? html`<br>` : ""}${NOISY.test(l) ? html`<span class="mut">${l}</span>` : l}`)}`
      : html`No log lines yet.`);
  }

export function setLogTab(t) { logTab = t; }
