// Minimal event bus. Feature modules emit coarse app signals (navigate, refresh, …) instead of
// importing the controller's orchestration functions — which would create an import cycle, since
// the controller's renderView dispatches back into those same feature modules. The controller
// subscribes with on(); features fire with emit().
const listeners = {};
export function on(evt, fn) { (listeners[evt] ||= []).push(fn); return () => { const a = listeners[evt]; const i = a ? a.indexOf(fn) : -1; if (i >= 0) a.splice(i, 1); }; }
export function emit(evt, ...args) { (listeners[evt] || []).slice().forEach((fn) => fn(...args)); }
