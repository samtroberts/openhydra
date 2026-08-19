// Shared, reassignable app state — the cross-module core that was module-level `let`s in the IIFE.
// ES modules can't reassign an imported binding, so writers call the setters; readers import the
// live bindings (which reflect each reassignment on read). View-local ephemeral state (peerLimit,
// logTab, statsDB, the nav history, coachmark refs, …) stays inside its own module instead.
export let state = null;            // get_state(): provider/gateway/settings snapshot
export let snap = null;             // status_snapshot(): network/transfers/economy
export let engines = [];            // detect_engines_now(): running engines + their models
export let installedEngines = [];   // installed_engines(): engine labels present on disk
export let sessions = {};           // { id: { t, m } } chat sessions
export let sessionOrder = [];       // recents order — ids, newest first
export let curChat = null;          // active chat id
export let activeView = "home";     // current view key
export let deviceName = "";          // #9 friendly node name
export let usedTokens = 0;           // legacy in-app consumed-token counter

export const setState = (v) => { state = v; };
export const setSnap = (v) => { snap = v; };
export const setEngines = (v) => { engines = v; };
export const setInstalledEngines = (v) => { installedEngines = v; };
export const setSessions = (v) => { sessions = v; };
export const setSessionOrder = (v) => { sessionOrder = v; };
export const setCurChat = (v) => { curChat = v; };
export const setActiveView = (v) => { activeView = v; };
export const setDeviceName = (v) => { deviceName = v; };
export const setUsedTokens = (v) => { usedTokens = v; };
