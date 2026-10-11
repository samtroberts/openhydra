// Activity view (home mode): lifetime served/used KPIs + net contribution + give-to-get ratio,
// the rolling recent-chat TPS/RTT + node uptime line, and the served-vs-used timeline chart.
import { $, $$ } from "./dom";
import { html, setHTML } from "./html";
import { snap } from "./state";
import { totalServed, totalUsed, totalCreditsEarned, totalCreditsSpent } from "./stats";
import { fmtNum, fmtUptime } from "./format";
import { renderChart } from "./chart";
import { mean, tpsSamples, rttSamples } from "./telemetry";

  export function renderActivity() {
    const t = snap?.transfers, k = $$("#v-activity .g4 .kpi .val");
    const served = totalServed(), used = totalUsed();   // #7 durable lifetime totals (tokens)
    // Credit-tiering: net balance is now COST-WEIGHTED credits (earned − spent), so serving a 30B
    // model counts for far more than a 1B one — not the flat served − used tokens.
    const netCredits = totalCreditsEarned() - totalCreditsSpent();
    const ratio = used > 0 ? (served / used) : (served > 0 ? null : 0);
    k[0].textContent = fmtNum(served);
    k[1].textContent = fmtNum(used);
    k[2].textContent = (netCredits >= 0 ? "+" : "") + Math.round(netCredits).toLocaleString();
    k[3].textContent = ratio == null ? "∞" : ratio ? ratio.toFixed(1) + "×" : "—";
    setHTML($$("#v-activity .g4 .kpi .sub")[0], html`<span class="dot ok"></span>${t?.receipts_ledgered ?? 0} receipts co-signed`);
    // Replies your own Mac served for your own chats aren't a network transaction: shown apart.
    const local = t?.tokens_local ?? 0;
    $$("#v-activity .g4 .kpi .sub")[1].textContent = local ? `from the network · ${fmtNum(local)} more answered by this Mac` : "from the network";
    $$("#v-activity .g4 .kpi .sub")[2].textContent = "net credits · cost-weighted, earned − spent";
    $$("#v-activity .g4 .kpi .sub")[3].textContent = "served ÷ used";
    // Rolling per-chat throughput/latency — the only place an aggregated TPS/RTT is honest,
    // since the agent emits these per-request. Uptime rounds out the "your node" picture.
    const note = $("#actnote");
    if (note) {
      const at = mean(tpsSamples), ar = mean(rttSamples);
      const parts = [];
      if (at != null) parts.push(`avg ${Math.round(at)} t/s`);
      if (ar != null) parts.push(`${Math.round(ar)} ms RTT`);
      if (snap?.uptime_secs != null) parts.push(`node up ${fmtUptime(snap.uptime_secs)}`);
      note.textContent = (parts.length ? `Your recent chats: ${parts.join(" · ")}. ` : "") + "Full transaction history lives in Network › Ledger.";
    }
    renderChart("#actchart", "activity");   // #10 timeline
  }
