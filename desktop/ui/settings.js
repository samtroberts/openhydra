// Settings view + the Engine-endpoint helper (#3: the endpoint follows the selected engine). The
// device-name / gateway-port / bootstraps fields are not clobbered while focused (the poll re-renders
// Settings). updateEngineEndpoint is also called by the engine dropdown in the controller.
import { $, $$ } from "./dom";
import { state, engines, deviceName } from "./state";

  // #3: the Settings › Engine endpoint follows the selected engine (was pinned to engines[0]).
  const ENGINE_ENDPOINTS = { "auto-detect": "", ollama: "http://127.0.0.1:11434", "vLLM": "http://127.0.0.1:8000", "LM Studio": "http://127.0.0.1:1234", "llama.cpp": "http://127.0.0.1:8080", "Exo": "http://127.0.0.1:52415", "ComfyUI": "http://127.0.0.1:8188" };
  export function updateEngineEndpoint() {
    const drop = $("#enginedrop"), field = $("#engineendpoint"); if (!drop || !field) return;
    const sel = (drop.querySelector("span")?.textContent || "auto-detect").trim();
    // auto-detect → the first detected engine's live URL; a specific pick → that engine's live URL
    // if it's currently running, else its standard endpoint.
    const live = engines.find((e) => (e.label || "").toLowerCase() === sel.toLowerCase());
    const url = sel === "auto-detect"
      ? (engines[0]?.url || "http://127.0.0.1:11434")
      : (live?.url || ENGINE_ENDPOINTS[sel] || "http://127.0.0.1:11434");
    field.textContent = url;
  }

  export function renderSettings() {
    const p = state; if (!p) return;
    // #9: don't clobber the device-name field while the user is editing it (the 2.5s poll
    // re-renders Settings; overwriting a focused field was why it "couldn't be changed").
    const id = $('.setpanel[data-p="identity"]'); const dn = id.querySelector('[contenteditable]');
    if (document.activeElement !== dn) dn.textContent = deviceName;
    id.querySelectorAll(".input")[1].childNodes[0].textContent = (p.provider.status.peer_id || p.gateway.status.peer_id || "—");
    const netp = $('.setpanel[data-p="network"]');
    const gwp = netp.querySelector('#gwport'); if (gwp && document.activeElement !== gwp) gwp.textContent = p.settings.gateway_port;
    const bsEl = netp.querySelector('#bootstraps'); if (bsEl && document.activeElement !== bsEl) bsEl.textContent = (p.settings.bootstraps || []).join("\n");
    const eng = $('.setpanel[data-p="engine"]');
    updateEngineEndpoint(); // #3: endpoint follows the selected engine (auto-detect ⇒ engines[0])
    $("#engineautostartsw").classList.toggle("on", !!p.settings.engine_autostart);
    $("#resumelaunchsw") && $("#resumelaunchsw").classList.toggle("on", p.settings.resume_on_launch !== false);
    $("#advsw").classList.toggle("on", $("#app").hasAttribute("data-adv"));
    $("#verboselogsw") && $("#verboselogsw").classList.toggle("on", !!p.settings.verbose_logs);   // #4
  }
