// Activity view (home mode): lifetime served/used KPIs + net contribution + give-to-get ratio,
// the rolling recent-chat TPS/RTT + node uptime line, and the served-vs-used timeline chart.
import { $, $$ } from "./dom";
import { snap } from "./state";
import { totalServed, totalUsed } from "./stats";
import { fmtNum, fmtUptime } from "./format";
import { renderChart } from "./chart";
import { mean, tpsSamples, rttSamples } from "./telemetry";

  export function renderActivity() {
    const t = snap?.transfers, k = $$("#v-activity .g4 .kpi .val");
    const served = totalServed(), used = totalUsed();   // #7 durable lifetime totals
    const netCredits = served - used;   // #3: net balance rises with serving, falls with using
    const ratio = used > 0 ? (served / used) : (served > 0 ? null : 0);
    k[0].textContent = fmtNum(served);
    k[1].textContent = fmtNum(used);
    k[2].textContent = (netCredits >= 0 ? "+" : "") + Math.round(netCredits).toLocaleString();
    k[3].textContent = ratio == null ? "∞" : ratio ? ratio.toFixed(1) + "×" : "—";
    $$("#v-activity .g4 .kpi .sub")[0].innerHTML = `<span class="dot ok"></span>${t?.receipts_ledgered ?? 0} receipts co-signed`;
    $$("#v-activity .g4 .kpi .sub")[1].textContent = "this device";
    $$("#v-activity .g4 .kpi .sub")[2].textContent = "net contribution · served − used";
    $$("#v-activity .g4 .kpi .sub")[3].textContent = "served ÷ used";
    // Rolling per-chat throughput/latency — the only place an aggregated TPS/RTT is honest,
    // since the agent emits these per-request. Uptime rounds out the "your node" picture.
    const note = $("#v-activity .mut");
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
