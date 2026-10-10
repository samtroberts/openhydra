// Shared UI chrome: transient toast + popover menu (wireframe verbatim). Module-local ephemeral
// state (the toast element/timer, the currently-open menu anchor); no app state.
import { $, $$, esc, escapeHtml } from "./dom";

let toastEl, toastT;
export function toast(m) { if (!toastEl) { toastEl = document.createElement("div"); toastEl.className = "toast"; document.body.appendChild(toastEl); } toastEl.textContent = m; toastEl.classList.add("show"); clearTimeout(toastT); toastT = setTimeout(() => toastEl.classList.remove("show"), 1600); }

let menuFor = null;
// Keyboard for an open menu: ↑/↓ move, Enter picks, Escape closes.
document.addEventListener("keydown", (e) => {
  const m = document.querySelector(".menu"); if (!m) return;
  // While typing, only Escape reaches the menu (Enter sends the message, arrows move the caret).
  const a = document.activeElement, typing = a && (a.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName));
  if (typing && e.key !== "Escape") return;
  const items = [...m.querySelectorAll(".mi")]; if (!items.length) return;
  let i = items.findIndex((x) => x.classList.contains("kb"));
  if (e.key === "Escape") { e.preventDefault(); closeMenus(); return; }
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    i = e.key === "ArrowDown" ? (i + 1) % items.length : (i <= 0 ? items.length - 1 : i - 1);
    items.forEach((x, k) => x.classList.toggle("kb", k === i));
    items[i].scrollIntoView({ block: "nearest" });
  } else if (e.key === "Enter" && i >= 0) { e.preventDefault(); items[i].click(); }
});
export function closeMenus() { $$(".menu").forEach((m) => m.remove()); menuFor = null; }
export function menu(anchor, items) {
  if (menuFor === anchor) { closeMenus(); return; }
  closeMenus(); menuFor = anchor;
  const m = document.createElement("div"); m.className = "menu";
  items.forEach((it) => { if (it.sep) { const s = document.createElement("div"); s.className = "msep"; m.appendChild(s); return; } const mi = document.createElement("div"); mi.className = "mi"; mi.innerHTML = `<span class="ck">${it.on ? "✓" : ""}</span>${esc(it.label)}`; mi.onclick = (e) => { e.stopPropagation(); closeMenus(); it.fn && it.fn(); }; m.appendChild(mi); });
  document.body.appendChild(m);
  const r = anchor.getBoundingClientRect(); m.style.left = Math.max(8, Math.min(r.left, innerWidth - m.offsetWidth - 10)) + "px"; let t = r.bottom + 5; if (t + m.offsetHeight > innerHeight - 8) t = r.top - m.offsetHeight - 5; m.style.top = Math.max(8, t) + "px";
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
    $(".cx", back).focus();   // a stray Enter/Space cancels instead of re-triggering the button behind
  });
}
