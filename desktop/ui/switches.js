// Switches (`.switch`) as controls a keyboard and a screen reader can use (review 2026-10-14, L17):
// each gets a role, a tab stop, its state in aria-checked and a name, kept in step with its "on"
// class however the page changes it; Space or Enter toggles the focused one. No imports, so the UI
// tests can load it.

// What a screen reader calls the switch: the label it sits in, else its row's label in Settings, else
// its tooltip.
export function switchName(sw) {
  const label = (sw.closest?.("label") || sw.closest?.(".field")?.querySelector?.("label"))?.textContent?.trim();
  return label || sw.getAttribute("title") || "";
}

// Brings one switch's role, tab stop, state and name up to date. A switch with three states (the
// share-all switch, `data-tristate`) is a checkbox: aria-checked="mixed" isn't valid on a switch.
export function syncSwitch(sw) {
  const tri = sw.hasAttribute("data-tristate");
  sw.setAttribute("role", tri ? "checkbox" : "switch");
  if (!sw.hasAttribute("tabindex")) sw.setAttribute("tabindex", "0");
  const state = sw.classList.contains("on") ? "true" : tri && sw.classList.contains("indeterminate") ? "mixed" : "false";
  if (sw.getAttribute("aria-checked") !== state) sw.setAttribute("aria-checked", state);
  const name = switchName(sw);
  if (name && sw.getAttribute("aria-label") !== name) sw.setAttribute("aria-label", name);
}

// The keys that toggle a focused switch.
export function isToggleKey(e) {
  return e.key === " " || e.key === "Enter";
}

// Every switch in `doc`, now and as the page adds or changes them, plus the keyboard.
export function wireSwitches(doc) {
  doc.querySelectorAll(".switch").forEach(syncSwitch);
  const each = (n) => {
    if (n.nodeType !== 1) return;
    if (n.classList.contains("switch")) syncSwitch(n);
    n.querySelectorAll(".switch").forEach(syncSwitch);
  };
  // Only class and title changes are watched, so the attributes set here never wake it again.
  new MutationObserver((records) => {
    for (const r of records) {
      if (r.type === "attributes") { if (r.target.classList?.contains("switch")) syncSwitch(r.target); }
      else r.addedNodes.forEach(each);
    }
  }).observe(doc.body, { subtree: true, childList: true, attributes: true, attributeFilter: ["class", "title"] });
  doc.addEventListener("keydown", (e) => {
    const sw = e.target?.closest?.(".switch");
    if (!sw || !isToggleKey(e)) return;
    e.preventDefault(); // Space would scroll the page
    sw.click();
  });
}
