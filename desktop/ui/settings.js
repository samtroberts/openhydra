// Settings view + the Engine-endpoint helper (#3: the endpoint follows the selected engine). Settings
// save as they change (the controller's saveSetting); text fields are not clobbered while focused, and
// unsaved values in `pendingSettings` win over the poll's re-render. updateEngineEndpoint is also called by the engine dropdown in the controller.
import { $ } from "./dom";
import { html, setHTML } from "./html";
import { state, engines, deviceName } from "./state";
import { call } from "./bridge";

  // Settings › Engine: the saved engine kind (agent `--engine-kind` value) ↔ the dropdown label, and
  // each engine's standard endpoint (shown as the placeholder when no custom endpoint is saved).
  const ENGINE_LABEL = { auto: "auto-detect", ollama: "ollama", vllm: "vLLM", "lm-studio": "LM Studio", "llama-cpp": "llama.cpp", exo: "Exo", comfyui: "ComfyUI" };
  export const ENGINE_ENDPOINTS = { "auto-detect": "", ollama: "http://127.0.0.1:11434", "vLLM": "http://127.0.0.1:8000", "LM Studio": "http://127.0.0.1:1234", "llama.cpp": "http://127.0.0.1:8080", "Exo": "http://127.0.0.1:52415", "ComfyUI": "http://127.0.0.1:8188" };
  // The endpoint field's value to save: the engine's default address shown in it isn't a custom
  // endpoint (saving it would pin the engine to it), so it saves as "use the default".
  export function engineEndpointValue(el) {
    const v = (el.textContent || "").trim();
    const sel = ($("#enginedrop")?.querySelector("span")?.textContent || "").trim();
    return v === ENGINE_ENDPOINTS[sel] ? "" : v;
  }
  export function updateEngineEndpoint() {
    const drop = $("#enginedrop"), field = $("#engineendpoint"); if (!drop || !field) return;
    const sel = (drop.querySelector("span")?.textContent || "auto-detect").trim();
    const auto = sel === "auto-detect";
    // auto-detect serves every running engine at its own address: nothing to edit.
    field.contentEditable = auto ? "false" : "true";
    field.classList.toggle("mut", auto);
    if (document.activeElement === field) return;
    const saved = ((pendingSettings.engine_endpoint ?? state?.settings?.engine_endpoint) || "").trim();
    field.textContent = auto
      ? (engines.length ? engines.map((e) => `${e.label} ${e.url}`).join(" · ") : "detected automatically")
      : (saved || ENGINE_ENDPOINTS[sel] || "");
  }

  // Settings changes waiting for save_settings to finish. renderSettings shows these over the saved
  // values, so the 2.5 s poll can't flip a control back while its save is in flight.
  export const pendingSettings = {};

  // Settings › Identity: both keys with their peer ids (identity_info resolves them once).
  let identities = null, identitiesLoading = false;
  function renderIdentities() {
    const host = $("#identrows"); if (!host) return;
    if (!identities) {
      if (!identitiesLoading) { identitiesLoading = true; call("identity_info").then((r) => { identities = r; renderIdentities(); }).catch(() => {}).finally(() => { identitiesLoading = false; }); }
      return;
    }
    if (host.dataset.done) return;
    const row = (title, desc, v) => html`<div class="field"><label>${title}</label><div class="desc" style="margin:-2px 0 6px">${desc}</div>
      <div class="input mono" style="justify-content:space-between;gap:10px;font-size:11.5px"><span style="overflow:hidden;text-overflow:ellipsis">${v?.peer_id || "—"}</span>
      <button class="btn ghost sm idcopy" data-v="${v?.peer_id || ""}">Copy</button></div>
      <div class="desc mono" style="margin-top:4px">${v?.key_file || ""}</div></div>`;
    setHTML(host, html`${row("Sharing identity", "The peer id other machines see when they use your models.", identities.sharing)}${row("Chat identity", "Used when you chat, and for swarms you join.", identities.chat)}`);
    host.dataset.done = "1";
    host.querySelectorAll(".idcopy").forEach((b) => b.onclick = async () => { try { await navigator.clipboard.writeText(b.dataset.v); b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1200); } catch {} });
  }

  export function renderSettings() {
    const p = state; if (!p) return;
    const s = { ...p.settings, ...pendingSettings };
    // #9: don't clobber the device-name field while the user is editing it (the 2.5s poll
    // re-renders Settings; overwriting a focused field was why it "couldn't be changed").
    const id = $('.setpanel[data-p="identity"]'); const dn = id.querySelector('[contenteditable]');
    if (document.activeElement !== dn) dn.textContent = deviceName;
    renderIdentities();
    const netp = $('.setpanel[data-p="network"]');
    const gwp = netp.querySelector('#gwport'); if (gwp && document.activeElement !== gwp) gwp.textContent = s.gateway_port;
    const bsEl = netp.querySelector('#bootstraps'); if (bsEl && document.activeElement !== bsEl) bsEl.textContent = (s.bootstraps || []).join("\n");
    // Built-in bootstrap peers exist only in release builds; say which case this is.
    const builtin = p.builtin_bootstraps || 0;
    const publicNet = s.public_network !== false;
    $("#netsub") && ($("#netsub").textContent = !publicNet
      ? "Private deployment: your node connects only through the bootstrap peers you list below (and machines on your local network). OpenHydra's public servers are never contacted."
      : builtin
        ? "Your node joins the network through OpenHydra's bootstrap peers. You can add more and change the local API port."
        : "This build has no built-in bootstrap peer: add one below to join the public network. Without one, only machines on your local network are found.");
    $("#bootdesc") && ($("#bootdesc").textContent = !publicNet
      ? "Your deployment's bootstrap peers, one per line."
      : builtin ? "Extra bootstrap peer addresses, one per line (used as well as the built-in ones)." : "Bootstrap peer addresses, one per line.");
    $("#relaysw") && $("#relaysw").classList.toggle("on", !!s.peer_relay);
    // Absent in a config from before the setting existed: on (the public network).
    $("#publicnetsw") && $("#publicnetsw").classList.toggle("on", s.public_network !== false);
    const ed = $("#enginedrop span");
    if (ed && !document.querySelector(".menu")) ed.textContent = ENGINE_LABEL[s.engine_kind] || "auto-detect";
    updateEngineEndpoint();
    $("#engineautostartsw").classList.toggle("on", !!s.engine_autostart);
    $("#authchecksw") && $("#authchecksw").classList.toggle("on", !!s.authenticity_checks);
    $("#auditreplaysw") && $("#auditreplaysw").classList.toggle("on", !!s.audit_replay);
    // Absent in a config from before the setting existed: on, as the agent defaults.
    $("#tokdlsw") && $("#tokdlsw").classList.toggle("on", s.tokenizer_downloads !== false);
    // A saved "A2" (offered by an unreleased build) is used as A1: this app's gateway can't run the
    // model checks A2 needs.
    const floor = s.min_authenticity || "off";
    $("#minauthsel") && ($("#minauthsel").value = floor === "A2" ? "A1" : floor);
    $("#minauthnote") && ($("#minauthnote").style.display = floor === "A2" ? "" : "none");
    $("#resumelaunchsw") && $("#resumelaunchsw").classList.toggle("on", s.resume_on_launch !== false);
    $("#advsw").classList.toggle("on", $("#app").hasAttribute("data-adv"));
    $("#verboselogsw") && $("#verboselogsw").classList.toggle("on", !!s.verbose_logs);   // #4
  }
