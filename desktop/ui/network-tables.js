// Network data tables: the Ledger (recent co-signed transactions), the Diagnostics/Peers view
// (peer list + DHT records + swarm + logs), and the provider/gateway log pane. infraPeerIds,
// peerLimit, and logTab are module-local; setLogTab lets the log chips switch the pane.
import { $, $$, esc, peerShort } from "./dom";
import { state, snap } from "./state";
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
      ? `${rows.length} recent · ${t?.receipts_ledgered ?? 0} co-signed · ${t?.tokens_served ?? 0} served / ${t?.tokens_consumed ?? 0} used tokens`
      : `${t?.receipts_ledgered ?? 0} receipts · ${t?.tokens_served ?? 0} tokens served`;
    $("#ledgertable tbody").innerHTML = rows.length
      ? rows.slice(0, 100).map((r) => {
          const served = r.kind === "served", cr = (served ? "+" : "−") + (r.tokens / 100).toFixed(1);
          return `<tr><td class="mut">${relTime(r.ts_ms)}</td><td><span class="badge ${served ? "ok" : "secondary"}">${served ? "served" : "used"}</span></td><td title="${esc(r.model)}">${modelIcon(r.model)}${esc(displayModelName(r.model))}</td><td class="mono">${peerShort(r.counterparty)}</td><td class="num">${r.tokens}</td><td class="num ${served ? "up" : ""}"${served ? "" : ' style="color:hsl(var(--danger))"'}>${cr}</td></tr>`;
        }).join("")
      : `<tr><td colspan="6" class="mut">No transactions yet — serve a model or run a chat, and co-signed receipts appear here. Recent activity is kept in memory (launch the agent with a ledger DB for full history).</td></tr>`;
  }

  // libp2p ids of the infrastructure we're connected to (bootstraps + circuit relays) — these
  // aren't "peers" a user cares about, so we hide them from the Peers list.
  function infraPeerIds() {
    const s = new Set();
    (snap?.network?.relay_reservations || []).forEach((a) => { const m = a.match(/\/p2p\/([^/]+)\/p2p-circuit/); if (m) s.add(m[1]); });
    (state?.settings?.bootstraps || []).forEach((a) => { const m = a.match(/\/p2p\/([^/]+)/); if (m) s.add(m[1]); });
    return s;
  }
  let peerLimit = 10;   // "View more" bumps this
  export function renderPeers() {
    if (!snap) { $("#peertable tbody").innerHTML = `<tr><td colspan="5" class="mut">Turn on Sharing or chat to connect, then peers appear here.</td></tr>`; const pc = $("#peercount"); if (pc) pc.textContent = "0 peers"; const pm = $("#peermore"); if (pm) pm.style.display = "none"; return; }
    const n = snap.network;
    const infra = infraPeerIds();
    const peers = (n.peers || []).filter((p) => !infra.has(p.peer_id));   // hide bootstraps/relays
    const repL = repByLibp2p();
    const shown = peers.slice(0, peerLimit);
    $("#peertable tbody").innerHTML = shown.length ? shown.map((p) => `<tr data-p="${p.path}"><td class="mono">${peerShort(p.peer_id)}</td><td><span class="badge ${p.path === "direct" ? "ok" : p.path === "relay" ? "warn" : "secondary"}">${p.path}</span></td><td class="num">${p.quic_direct_v6}</td><td>${repBadge(repL[p.peer_id])}</td><td class="rowmenu mut"><span class="icon" data-i="more"></span></td></tr>`).join("") : `<tr><td colspan="5" class="mut">${n.peers.length ? "Only infrastructure connected — waiting for network peers." : "No peers connected yet — connecting."}</td></tr>`;
    injectIcons($("#peertable"));
    // dynamic count + View more
    const pc = $("#peercount"); if (pc) pc.textContent = peers.length <= peerLimit ? `${peers.length} peer${peers.length === 1 ? "" : "s"}` : `${shown.length} of ${peers.length} peers`;
    const pm = $("#peermore"); if (pm) { pm.style.display = peers.length > peerLimit ? "" : "none"; pm.onclick = () => { peerLimit += 10; renderPeers(); }; }
    // re-apply the active Direct/Relay/Mixed chip so the 2.5s poll doesn't reset it to All
    const pp = $("#peerchips .chip.on")?.dataset.p || "all";
    $$("#peertable tbody tr").forEach((r) => r.style.display = (pp === "all" || r.dataset.p === pp) ? "" : "none");
    $("#actchips .chip .num") && ($("#actchips .chip .num").textContent = peers.length);
    $$("#peertable .rowmenu").forEach((cell) => cell.onclick = (e) => { e.stopPropagation(); menu(cell, [{ label: "Copy peer id", fn: () => toast("Copied") }, { sep: 1 }, { label: "Drop connection", fn: () => toast("Dropped") }]); });
    // DHT
    $$("#v-peers .acttab")[1].querySelector("tbody").innerHTML = (n.known_models || []).length ? (snap.network.known_models).map((m) => `<tr><td class="mono" title="${esc(m)}">/oh/model/${esc(displayModelName(m))}</td><td><span class="badge secondary">provider</span></td><td class="num">${(snap.network.known_providers || []).filter((p) => p.model_id === m).length || 1}</td><td class="num">—</td></tr>`).join("") : `<tr><td colspan="4" class="mut">No records yet.</td></tr>`;
    $$("#v-peers .acttab")[1].querySelector(".card.pad").innerHTML = `kad_routing_peers: <span class="num">${n.kad_routing_peers}</span> · server mode: ${n.kad_server_mode ? "yes" : "no"}`;
    // Swarm
    const sw = $$("#v-peers .acttab")[2].querySelectorAll(".kpi .val"); sw[0].textContent = n.listen_addrs.length; sw[1].textContent = n.relay_reservations.length; sw[2].textContent = n.counters.dcutr_successes; sw[3].textContent = n.autonat_private ? "private" : "public";
    // Logs
    renderLogs();
  }
  let logTab = "provider";
  export function renderLogs() { const logs = (logTab === "provider" ? state?.provider?.logs : state?.gateway?.logs) || []; $("#logbody").innerHTML = logs.length ? logs.map(esc).join("<br>") : "—"; }

export function setLogTab(t) { logTab = t; }
