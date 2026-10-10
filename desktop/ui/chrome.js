// Shared UI chrome: transient toast + popover menu (wireframe verbatim). Module-local ephemeral
// state (the toast element/timer, the currently-open menu anchor); no app state.
import { $, $$, esc, escapeHtml } from "./dom";

let toastEl, toastT;
export function toast(m) { if (!toastEl) { toastEl = document.createElement("div"); toastEl.className = "toast"; document.body.appendChild(toastEl); } toastEl.textContent = m; toastEl.classList.add("show"); clearTimeout(toastT); toastT = setTimeout(() => toastEl.classList.remove("show"), 1600); }

let menuFor = null;
export function closeMenus() { $$(".menu").forEach((m) => m.remove()); menuFor = null; }
export function menu(anchor, items) {
  if (menuFor === anchor) { closeMenus(); return; }
  closeMenus(); menuFor = anchor;
  const m = document.createElement("div"); m.className = "menu";
  items.forEach((it) => { if (it.sep) { const s = document.createElement("div"); s.className = "msep"; m.appendChild(s); return; } const mi = document.createElement("div"); mi.className = "mi"; mi.innerHTML = `<span class="ck">${it.on ? "✓" : ""}</span>${esc(it.label)}`; mi.onclick = (e) => { e.stopPropagation(); closeMenus(); it.fn && it.fn(); }; m.appendChild(mi); });
  document.body.appendChild(m);
  const r = anchor.getBoundingClientRect(); m.style.left = Math.min(r.left, innerWidth - m.offsetWidth - 10) + "px"; let t = r.bottom + 5; if (t + m.offsetHeight > innerHeight - 8) t = r.top - m.offsetHeight - 5; m.style.top = Math.max(8, t) + "px";
}

// In-app confirm dialog. window.confirm() does nothing in the Tauri webview, so every "are you sure"
// goes through this. Resolves true (OK) or false (Cancel, backdrop click, Escape).
export function confirmModal(title, body, okLabel) {
  return new Promise((resolve) => {
    const back = document.createElement("div");
    back.className = "cmodal-back";
    back.innerHTML =
      `<div class="cmodal"><div class="cmodal-h"><b>${escapeHtml(title)}</b></div>` +
      `<div class="cmodal-b"><div class="cmodal-path" style="white-space:pre-line">${escapeHtml(body)}</div></div>` +
      `<div class="cmodal-f"><button class="btn ghost sm cx">Cancel</button><button class="btn sm cok">${escapeHtml(okLabel || "OK")}</button></div></div>`;
    document.body.appendChild(back);
    const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); done(false); } };
    const done = (v) => { document.removeEventListener("keydown", onKey, true); back.remove(); resolve(v); };
    document.addEventListener("keydown", onKey, true);
    $(".cx", back).onclick = () => done(false);
    $(".cok", back).onclick = () => done(true);
    back.onclick = (e) => { if (e.target === back) done(false); };
  });
}
