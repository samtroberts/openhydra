// First-run coachmark tour (spotlight; first launch + after updates only). Requests navigation via
// the bus (emit "nav") instead of importing the controller's go(). Module-local overlay refs;
// maybeStartTour auto-runs it once on boot, coachShow(0) replays it.
import { $ } from "./dom";
import { injectIcons } from "./icons";
import { store } from "./storage";
import { emit } from "./bus";

  const COACH = [
    { v: "home", a: "#homecard", t: "Chat with the network", d: "Ask anything — requests route to models served by peers. The first connection takes a few seconds; watch the status bar fill in." },
    { v: "providers", a: "#modeswitch", t: "Two sides of the app", d: "Home is where you use AI. Network is where you browse models, share your machine, and manage engines." },
    { v: "engines", a: '.nav[data-v="engines"]', t: "Engines & models", d: "OpenHydra wraps any engine already on your machine — whatever it can run, you can share." },
    { v: "share", a: "#sharecta", t: "Share when you're ready", d: "Click ‘Share your models’ to open the Share tab and pick what to serve. You're always connected; sharing is your choice." },
  ];
  const TOUR_KEY = "oh_tour_v2";
  let coachEl = null, coachRing = null, coachOv = [];
  function coachEnd() { coachEl?.remove(); coachEl = null; coachRing?.classList.remove("coachring"); coachRing = null; coachOv.forEach((d) => d.remove()); coachOv = []; store.set(TOUR_KEY, true); }
  function coachSpot(a, r) {
    const p = 5, W = innerWidth, H = innerHeight;
    if (!coachOv.length) { const s = document.createElement("div"); s.className = "covspot"; document.body.appendChild(s); coachOv.push(s); for (let i = 0; i < 4; i++) { const d = document.createElement("div"); d.className = "covstrip"; document.body.appendChild(d); coachOv.push(d); } }
    const x1 = Math.max(0, r.left - p), y1 = Math.max(0, r.top - p), x2 = Math.min(W, r.right + p), y2 = Math.min(H, r.bottom + p);
    const set = (d, l, t, w, h) => { d.style.left = l + "px"; d.style.top = t + "px"; d.style.width = Math.max(0, w) + "px"; d.style.height = Math.max(0, h) + "px"; };
    const rad = a ? (parseFloat(getComputedStyle(a).borderRadius) || 8) : 8;
    set(coachOv[0], x1, y1, x2 - x1, y2 - y1); coachOv[0].style.borderRadius = (rad + p) + "px";
    set(coachOv[1], 0, 0, W, y1); set(coachOv[2], 0, y2, W, H - y2); set(coachOv[3], 0, y1, x1, y2 - y1); set(coachOv[4], x2, y1, W - x2, y2 - y1);
  }
  export function coachShow(i) {
    const s = COACH[i]; emit("nav", s.v);
    coachRing?.classList.remove("coachring"); const a = $(s.a); coachRing = a; a?.classList.add("coachring");
    if (!coachEl) { coachEl = document.createElement("div"); coachEl.className = "coach"; document.body.appendChild(coachEl); }
    coachEl.innerHTML = `<button class="iconbtn cx" id="coachx" title="Close"><span class="icon" data-i="x"></span></button><div style="font-weight:600;margin-bottom:4px;padding-right:24px">${s.t}</div><div class="mut" style="font-size:12px;line-height:1.55">${s.d}</div><div class="row" style="margin-top:11px;gap:8px">${i > 0 ? '<button class="btn outline sm" id="coachback">Back</button>' : ""}<span class="mut num" style="font-size:11px">${i + 1} / ${COACH.length}</span><div class="grow"></div><button class="btn primary sm" id="coachnext">${i < COACH.length - 1 ? "Next" : "Done"}</button></div>`;
    injectIcons(coachEl);
    const r = a ? a.getBoundingClientRect() : { left: innerWidth / 2 - 136, top: innerHeight / 2, right: innerWidth / 2 + 136, bottom: innerHeight / 2 };
    coachSpot(a, r);
    let left = Math.min(Math.max(10, r.left), innerWidth - 292), top = r.bottom + 14; if (top + 160 > innerHeight) top = Math.max(10, r.top - 166);
    coachEl.style.left = left + "px"; coachEl.style.top = top + "px";
    $("#coachx").onclick = coachEnd; const bk = $("#coachback"); if (bk) bk.onclick = () => coachShow(i - 1); $("#coachnext").onclick = () => (i < COACH.length - 1 ? coachShow(i + 1) : coachEnd());
  }

export function maybeStartTour() { if (!store.get(TOUR_KEY, false)) setTimeout(() => coachShow(0), 900); }
