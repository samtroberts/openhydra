// Swarms UI (M3): private sharing by membership credential. A swarm owner creates a group (its group
// key stays on this machine), approves members by their PUBLIC key, and hands back a signed
// credential. A member generates an enrollment request, sends it to the owner out-of-band, and pastes
// the returned credential to join. No shared secret ever leaves a machine — the whole exchange is
// public keys + signatures. Enrollment is offline copy/paste in v1 (no wire protocol).
//
// Every value interpolated into HTML goes through escapeHtml — swarm/member labels are user-entered
// and untrusted (the M2 card XSS lesson). Key fingerprints are shown prominently so users can confirm
// a key out-of-band (voice/in-person) before approving — the anti-mis-enrollment control.
import { $, $$, escapeHtml, peerShort } from "./dom";
import { call } from "./bridge";
import { toast } from "./chrome";

// Default credential lifetime the owner grants (matches the agent default; surfaced so it's honest).
const CRED_TTL_SECS = 90 * 24 * 3600; // 90 days

// ── main view ──
export async function renderSwarms() {
  const host = $("#v-swarms");
  if (!host) return;
  let swarms = [];
  try {
    swarms = await call("list_swarms");
  } catch (e) {
    host.innerHTML = `<div class="cardfail">Couldn't load swarms: ${escapeHtml(String(e))}</div>`;
    return;
  }

  const intro =
    `<div class="row" style="margin-bottom:10px">` +
    `<div class="ctitle">Swarms</div>` +
    `<span class="mut" style="margin-left:auto;font-size:11.5px">Private sharing · no shared secret leaves your machine</span>` +
    `</div>` +
    `<div class="card pad" style="margin-bottom:14px">` +
    `<div class="mut" style="font-size:12px;line-height:1.5">A <b>swarm</b> is a private group you share models with. <b>Create</b> one to become its owner and approve members by their key; <b>join</b> someone else's by sending them an enrollment request and pasting the credential they return. Everything exchanged is a public key or a signature — the group's private key never leaves the owner's machine.</div>` +
    `<div class="row" style="gap:8px;margin-top:12px"><button class="btn brand sm" id="swcreate"><span class="icon" data-i="plus"></span>Create swarm</button><button class="btn outline sm" id="swjoin">Join a swarm</button></div>` +
    `</div>`;

  const list = swarms.length
    ? swarms.map(swarmCardHtml).join("")
    : `<div class="card pad mut" style="font-size:12.5px">No swarms yet. Create one to share privately, or join someone else's.</div>`;

  host.innerHTML = intro + list;

  $("#swcreate", host).onclick = createSwarmModal;
  $("#swjoin", host).onclick = joinSwarmModal;
  wireSwarmCards(host, swarms);
}

// ── one swarm card (owner shows members + approve/revoke; member shows the credential) ──
function swarmCardHtml(s) {
  const owner = s.role === "owner";
  const roleBadge = owner
    ? `<span class="badge ok" title="You own this swarm">owner</span>`
    : `<span class="badge secondary" title="You're a member of this swarm">member</span>`;
  const sub = owner
    ? `${s.member_count} member${s.member_count === 1 ? "" : "s"}${s.revoked_count ? ` · ${s.revoked_count} revoked` : ""}`
    : s.credential_expires_at
      ? `credential expires ${new Date(Number(s.credential_expires_at)).toLocaleDateString()}`
      : "no credential";

  const memberRows =
    owner && s.members.length
      ? `<table style="margin:6px 6px 8px"><tbody>` +
        s.members
          .map(
            (m) =>
              `<tr>` +
              `<td><b>${escapeHtml(m.label || "member")}</b><div class="mut mono" style="font-size:10.5px">${escapeHtml(m.fingerprint)}</div></td>` +
              `<td class="mut" style="font-size:11px">expires ${new Date(Number(m.expires_at)).toLocaleDateString()}</td>` +
              `<td style="text-align:right"><button class="btn ghost sm swrevoke" data-swarm="${escapeHtml(s.swarm_public_key)}" data-member="${escapeHtml(m.member_public_key)}" data-label="${escapeHtml(m.label || "member")}">Revoke</button></td>` +
              `</tr>`,
          )
          .join("") +
        `</tbody></table>`
      : owner
        ? `<div class="mut" style="font-size:11.5px;padding:2px 8px 8px">No members yet. Approve someone's enrollment request to add them.</div>`
        : "";

  const actions = owner
    ? `<button class="btn brand sm swaddmember" data-swarm="${escapeHtml(s.swarm_public_key)}" data-label="${escapeHtml(s.label)}">Add member</button>`
    : s.member_can_control
      ? `<button class="btn brand sm swremote" data-swarm="${escapeHtml(s.swarm_public_key)}" data-label="${escapeHtml(s.label)}" title="Flip one of this rig's models between Private and Global from here">Control rig</button>`
      : "";

  return (
    `<div class="card" style="margin-bottom:14px">` +
    `<div class="row" style="padding:14px 16px 8px;gap:8px">` +
    `<div><div class="ctitle" style="display:flex;align-items:center;gap:8px">${escapeHtml(s.label || "Swarm")} ${roleBadge}</div>` +
    `<div class="mut mono" style="font-size:10.5px;margin-top:3px" title="Group key fingerprint — confirm this out-of-band">${escapeHtml(s.fingerprint)}</div></div>` +
    `<div class="grow"></div>` +
    `<span class="mut" style="font-size:11.5px">${escapeHtml(sub)}</span>` +
    actions +
    `<button class="btn ghost sm swforget" data-swarm="${escapeHtml(s.swarm_public_key)}" data-label="${escapeHtml(s.label || "this swarm")}" data-owner="${owner ? "1" : "0"}" title="${owner ? "Delete this swarm and its group key" : "Leave this swarm"}">${owner ? "Delete" : "Leave"}</button>` +
    `</div>` +
    memberRows +
    `</div>`
  );
}

function wireSwarmCards(host, swarms) {
  $$(".swaddmember", host).forEach((b) => {
    b.onclick = () => approveMemberModal(b.dataset.swarm, b.dataset.label);
  });
  $$(".swremote", host).forEach((b) => {
    b.onclick = () => remoteScopeModal(b.dataset.swarm, b.dataset.label);
  });
  $$(".swrevoke", host).forEach((b) => {
    b.onclick = async () => {
      if (!confirm(`Revoke "${b.dataset.label}"? They lose access immediately and can't be re-approved without un-revoking.`)) return;
      try {
        await call("swarm_revoke_member", { swarm_public_key: b.dataset.swarm, member_public_key: b.dataset.member });
        toast("Member revoked");
        renderSwarms();
      } catch (e) {
        toast(`Revoke failed: ${e}`);
      }
    };
  });
  $$(".swforget", host).forEach((b) => {
    b.onclick = async () => {
      const owner = b.dataset.owner === "1";
      const msg = owner
        ? `Delete "${b.dataset.label}"? This destroys the group key — you can't issue any more credentials for it (existing members keep their credential until it expires).`
        : `Leave "${b.dataset.label}"? You'll drop your credential for it.`;
      if (!confirm(msg)) return;
      try {
        await call("forget_swarm", { swarm_public_key: b.dataset.swarm });
        toast(owner ? "Swarm deleted" : "Left swarm");
        renderSwarms();
      } catch (e) {
        toast(`Failed: ${e}`);
      }
    };
  });
}

// ── modal scaffold (mirrors cards.js exportCardModal) ──
function modal(title, bodyHtml, footHtml, maxWidth = 560) {
  const back = document.createElement("div");
  back.className = "cmodal-back";
  back.innerHTML =
    `<div class="cmodal" style="max-width:${maxWidth}px"><div class="cmodal-h"><b>${escapeHtml(title)}</b></div>` +
    `<div class="cmodal-b">${bodyHtml}</div>` +
    `<div class="cmodal-f">${footHtml}</div></div>`;
  document.body.appendChild(back);
  const close = () => back.remove();
  back.onclick = (e) => {
    if (e.target === back) close();
  };
  return { back, close };
}

function copyBtnRow(text, okMsg) {
  return () => {
    navigator.clipboard?.writeText(text);
    toast(okMsg);
  };
}

// A small segmented (single-choice) control. `options` = [{v,label,title}]; `selected` is the default
// value. Wire the change with `wireSegmented`. Used for the three-way Consume/Control/Both grant + the
// member's request role (M5 dedicated control identity).
function segmented(name, options, selected) {
  return (
    `<div class="row" data-seg="${escapeHtml(name)}" role="group" style="gap:6px;margin-top:6px">` +
    options
      .map(
        (o) =>
          `<button type="button" class="btn ${o.v === selected ? "brand" : "outline"} sm segbtn" ` +
          `data-seg-val="${escapeHtml(o.v)}" title="${escapeHtml(o.title || "")}">${escapeHtml(o.label)}</button>`,
      )
      .join("") +
    `</div>`
  );
}

function wireSegmented(root, name, onChange) {
  const wrap = root.querySelector(`[data-seg="${name}"]`);
  if (!wrap) return;
  wrap.querySelectorAll(".segbtn").forEach((b) => {
    b.onclick = () => {
      wrap.querySelectorAll(".segbtn").forEach((x) => x.classList.replace("brand", "outline"));
      b.classList.replace("outline", "brand");
      onChange(b.dataset.segVal);
    };
  });
}

// A read-only credential/request box with a copy button. `idx` makes the copy button class unique so
// several boxes can coexist in one modal (a "Both" grant returns two credentials).
function credBox(title, magnet, note, idx) {
  return (
    `<div style="margin-top:12px">` +
    `<div style="font-size:11.5px"><b>${escapeHtml(title)}</b></div>` +
    `<div class="mut" style="font-size:11px;margin:2px 0 4px">${escapeHtml(note)}</div>` +
    `<textarea class="input mono" rows="3" readonly style="width:100%;font-size:10.5px">${escapeHtml(magnet)}</textarea>` +
    `<div class="row" style="margin-top:4px"><button class="btn sm brand swcopy${idx}">Copy</button></div>` +
    `</div>`
  );
}

// ── create swarm (owner) ──
function createSwarmModal() {
  const { back, close } = modal(
    "Create a swarm",
    `<div class="cmodal-path">You'll be the owner. The group's private key is generated here and never leaves this machine.</div>` +
      `<label class="mut" style="font-size:11px;margin-top:10px;display:block">Swarm name</label>` +
      `<input class="input swname" placeholder="e.g. Home rig" style="width:100%" maxlength="128"/>`,
    `<button class="btn ghost sm swcancel">Cancel</button><button class="btn sm brand swdo">Create</button>`,
  );
  const name = $(".swname", back);
  name.focus();
  $(".swcancel", back).onclick = close;
  $(".swdo", back).onclick = async () => {
    const label = (name.value || "").trim();
    if (!label) {
      toast("Give the swarm a name");
      return;
    }
    try {
      await call("create_swarm", { label });
      toast(`Created “${label}”`);
      close();
      renderSwarms();
    } catch (e) {
      toast(`Create failed: ${e}`);
    }
  };
}

// ── approve member (owner): choose grant → paste request(s) → preview fingerprint(s) → approve →
// hand back credential(s). Grant is three-way (M5 dedicated control identity): Consume (serve), Control
// (a CAP_CONTROL-only credential on the member's control key), or Both (two credentials — control and
// consume bind different keys, so there is no single-key "both"). ──
const GRANT_OPTIONS = [
  { v: "consume", label: "Use models", title: "A member who can use your private models (no rig control)" },
  { v: "control", label: "Control rig", title: "A control-only device: can flip your rigs' model scope (incl. publish). No serve access." },
  { v: "both", label: "Both", title: "Two credentials — consume on their consumer key, control on their control key" },
];

function approveMemberModal(swarmPublicKey, swarmLabel) {
  const { back, close } = modal(
    `Add a member to “${swarmLabel}”`,
    `<div class="cmodal-path">Choose what to grant, then paste the enrollment request(s) the person sent you. Confirm their fingerprint with them (voice/in person) before approving — that's what stops you enrolling the wrong key.</div>` +
      `<label class="mut" style="font-size:11px;margin-top:10px;display:block">What to grant</label>` +
      segmented("grant", GRANT_OPTIONS, "consume") +
      `<label class="mut swreqlabel" style="font-size:11px;margin-top:10px;display:block">Enrollment request</label>` +
      `<textarea class="input mono swreq" rows="3" placeholder="openhydra:enroll:…   or the request JSON" style="width:100%;font-size:11px"></textarea>` +
      `<div class="swctlreqwrap" style="display:none">` +
      `<label class="mut" style="font-size:11px;margin-top:10px;display:block">Control request (their second magnet, from their control key)</label>` +
      `<textarea class="input mono swctlreq" rows="3" placeholder="openhydra:enroll:…   (control key)" style="width:100%;font-size:11px"></textarea>` +
      `</div>` +
      `<div class="row" style="margin-top:8px"><button class="btn outline sm swpreview">Preview</button></div>` +
      `<div class="swpreviewout" style="margin-top:8px"></div>`,
    `<button class="btn ghost sm swcancel">Close</button>`,
  );
  let grant = "consume";
  const out = $(".swpreviewout", back);
  const foot = $(".cmodal-f", back);
  $(".swcancel", back).onclick = close;
  wireSegmented(back, "grant", (v) => {
    grant = v;
    $(".swctlreqwrap", back).style.display = v === "both" ? "" : "none";
    $(".swreqlabel", back).textContent =
      v === "control" ? "Control request (from their control key)" : "Enrollment request";
    // A grant change invalidates any prior preview.
    out.innerHTML = "";
    foot.innerHTML = `<button class="btn ghost sm swcancel">Close</button>`;
    $(".swcancel", back).onclick = close;
  });

  $(".swpreview", back).onclick = async () => {
    const req = ($(".swreq", back).value || "").trim();
    const ctlReq = ($(".swctlreq", back)?.value || "").trim();
    if (!req) {
      toast("Paste a request first");
      return;
    }
    if (grant === "both" && !ctlReq) {
      toast("Paste the control request too (their second magnet)");
      return;
    }
    out.innerHTML = `<span class="mut" style="font-size:11.5px">Verifying…</span>`;
    try {
      // Preview the primary request; for "both" preview the control request as well and show both.
      const primary = await call("preview_enroll_request", { request: req });
      const cards = [await requestCardHtml(primary, grant === "control" ? "control" : "consume")];
      if (grant === "both") {
        const ctl = await call("preview_enroll_request", { request: ctlReq });
        cards.push(await requestCardHtml(ctl, "control"));
      }
      out.innerHTML =
        cards.join("") +
        `<label class="mut" style="font-size:11px;margin-top:10px;display:block">Label for this member</label>` +
        `<input class="input swmlabel" value="${escapeHtml(primary.label || "")}" placeholder="e.g. Sam's MacBook" style="width:100%" maxlength="128"/>`;
      foot.innerHTML = `<button class="btn ghost sm swcancel2">Close</button><button class="btn sm brand swapprove">Approve &amp; issue</button>`;
      $(".swcancel2", back).onclick = close;
      $(".swapprove", back).onclick = () => doApprove(back, close, swarmPublicKey, grant, req, ctlReq);
    } catch (e) {
      out.innerHTML = `<div class="cardfail">✕ ${escapeHtml(String(e))}</div>`;
      foot.innerHTML = `<button class="btn ghost sm swcancel3">Close</button>`;
      $(".swcancel3", back).onclick = close;
    }
  };
}

// A verified-request preview card. `kind` labels what this key will be granted so the owner sees, per
// request, whether it's the consume key or the control key.
async function requestCardHtml(r, kind) {
  const fp = await fingerprintOf(r.member_public_key);
  const tag =
    kind === "control"
      ? `<span class="badge secondary" title="This key will get a control-only credential">🔒 control key</span>`
      : `<span class="badge secondary" title="This key will get a serve/consume credential">consume key</span>`;
  return (
    `<div class="card pad" style="font-size:11.5px;margin-bottom:6px">` +
    `<div><b>${escapeHtml(r.label || "member")}</b> <span class="badge ok">✔ request signed</span> ${tag}</div>` +
    `<div class="mut mono" style="margin-top:4px;font-size:11px" title="Confirm this with them out-of-band">${escapeHtml(fp)}</div>` +
    `<div class="mut" style="margin-top:2px">id ${escapeHtml(peerShort(r.member_openhydra_peer_id))}</div>` +
    `</div>`
  );
}

async function doApprove(back, close, swarmPublicKey, grant, request, controlRequest) {
  const memberLabel = ($(".swmlabel", back)?.value || "").trim();
  try {
    const approved = await call("swarm_approve_member", {
      swarm_public_key: swarmPublicKey,
      request,
      control_request: grant === "both" ? controlRequest : null,
      member_label: memberLabel,
      ttl_secs: CRED_TTL_SECS,
      grant,
    });
    const boxes = [];
    if (approved.consume)
      boxes.push(
        credBox(
          "Consume credential",
          approved.consume.magnet,
          "Lets the device use your private models. Send it to the member.",
          0,
        ),
      );
    if (approved.control)
      boxes.push(
        credBox(
          "🔒 Control credential",
          approved.control.magnet,
          "Lets the device remote-set your rigs' model scope (incl. publish). Control only — no serve access. Send it to the member.",
          1,
        ),
      );
    $(".cmodal-b", back).innerHTML =
      `<div class="cmodal-path">Approved. Send the credential${boxes.length > 1 ? "s" : ""} back to the member — they paste ${boxes.length > 1 ? "them" : "it"} into “Join a swarm” to finish. Signed, no secret inside.</div>` +
      boxes.join("");
    $(".cmodal-f", back).innerHTML = `<button class="btn ghost sm swdone">Done</button>`;
    if (approved.consume)
      $(".swcopy0", back).onclick = copyBtnRow(approved.consume.magnet, "Consume credential copied");
    if (approved.control)
      $(".swcopy1", back).onclick = copyBtnRow(approved.control.magnet, "Control credential copied");
    $(".swdone", back).onclick = () => {
      close();
      renderSwarms();
    };
  } catch (e) {
    toast(`Approve failed: ${e}`);
  }
}

// ── control a rig (member, M5): send a signed REMOTE_SCOPE_SET to flip a model's scope ──
function remoteScopeModal(swarmPublicKey, swarmLabel) {
  const lab = `<label class="mut" style="font-size:11px;margin-top:10px;display:block">`;
  const { back, close } = modal(
    `Control a rig in “${swarmLabel}”`,
    `<div class="cmodal-path">Flip one of this rig owner's shared models between Private and Global from here. You'll need the rig's peer id (shown in its app's Peers view) and the model's name. The rig verifies your control credential before applying — and can refuse to publish if its owner turned remote publishing off.</div>` +
      `${lab}Rig peer id</label>` +
      `<input class="input mono swrig" placeholder="12D3KooW…" style="width:100%;font-size:10.5px"/>` +
      `${lab}Model</label>` +
      `<input class="input swrmodel" placeholder="e.g. llama3.1:8b" style="width:100%"/>` +
      `${lab}New scope</label>` +
      `<select class="input swrscope" style="width:100%">` +
      `<option value="private">Private — swarm members only</option>` +
      `<option value="global">Global — public / marketplace</option>` +
      `<option value="device">Device — loopback only</option>` +
      `</select>` +
      `<div class="swrout" style="margin-top:10px"></div>`,
    `<button class="btn ghost sm swcancel">Cancel</button><button class="btn sm brand swrsend">Send to rig</button>`,
  );
  $(".swcancel", back).onclick = close;
  $(".swrig", back).focus();
  $(".swrsend", back).onclick = async () => {
    const provider = ($(".swrig", back).value || "").trim();
    const model = ($(".swrmodel", back).value || "").trim();
    const scope = $(".swrscope", back).value;
    if (!provider || !model) {
      toast("Enter the rig peer id and the model");
      return;
    }
    // M1 consent surface: making a model Global is public exposure — confirm before sending.
    if (
      scope === "global" &&
      !confirm(
        `Publish “${model}” GLOBALLY on the rig?\n\n` +
          `The model becomes discoverable and routable on the public network / marketplace ` +
          `(the rig serves and earns/spends against it). The rig will refuse if its owner has ` +
          `disabled remote publishing.`,
      )
    ) {
      return;
    }
    const out = $(".swrout", back);
    out.innerHTML = `<span class="mut" style="font-size:11.5px">Dialing the rig and sending…</span>`;
    try {
      const ack = await call("swarm_remote_scope", {
        swarm_public_key: swarmPublicKey,
        provider,
        model,
        scope,
      });
      out.innerHTML =
        `<div class="card pad" style="font-size:11.5px"><span class="badge ok">✔ applied</span> ${escapeHtml(ack)}</div>`;
      toast("Rig updated");
    } catch (e) {
      out.innerHTML = `<div class="cardfail">✕ ${escapeHtml(String(e))}</div>`;
    }
  };
}

// ── join a swarm (member): pick a role, generate the request(s) to send, then paste the credential(s)
// you get back. Role (M5 dedicated control identity): Use models (consume key), Control rig (a
// dedicated control key — no serve access), or Both (a request from each key). "Both" is inherently two
// credentials because control and consume bind different keys. ──
const ROLE_OPTIONS = [
  { v: "consume", label: "Use models", title: "Request access to use this swarm's private models" },
  { v: "control", label: "Control rig", title: "Request control-only access — flip the owner's rig model scope. No serve access." },
  { v: "both", label: "Both", title: "Request both — one credential per key (consume + control)" },
];

function joinSwarmModal() {
  const { back, close } = modal(
    "Join a swarm",
    `<div class="cmodal-path">Two steps. Pick what you want, generate your request(s) and send them to the swarm's owner (any channel — they're public). When they send the credential(s) back, paste them below to finish.</div>` +
      `<label class="mut" style="font-size:11px;margin-top:10px;display:block">What you want</label>` +
      segmented("role", ROLE_OPTIONS, "consume") +
      `<label class="mut" style="font-size:11px;margin-top:10px;display:block">A name for this device (the owner sees it)</label>` +
      `<input class="input swdev" placeholder="e.g. Sam's MacBook" style="width:100%" maxlength="128"/>` +
      `<label class="mut" style="font-size:11px;margin-top:8px;display:block">Swarm key (optional — pin which swarm you're joining)</label>` +
      `<input class="input mono swhint" placeholder="group public key (optional)" style="width:100%;font-size:10.5px"/>` +
      `<div class="row" style="margin-top:8px"><button class="btn outline sm swgen">Generate my request</button></div>` +
      `<div class="swgenout" style="margin-top:8px"></div>` +
      `<hr style="border:0;border-top:1px solid var(--line);margin:14px 0"/>` +
      `<label class="mut" style="font-size:11px;display:block">Paste the credential the owner sent back</label>` +
      `<textarea class="input mono swcred" rows="3" placeholder="openhydra:cred:…   or the credential JSON" style="width:100%;font-size:11px;margin-top:6px"></textarea>` +
      `<div class="swcred2wrap" style="display:none">` +
      `<label class="mut" style="font-size:11px;margin-top:8px;display:block">Second credential (the 🔒 control one, for “Both”)</label>` +
      `<textarea class="input mono swcred2" rows="3" placeholder="openhydra:cred:…   (control)" style="width:100%;font-size:11px;margin-top:6px"></textarea>` +
      `</div>` +
      `<div class="row" style="margin-top:8px"><button class="btn outline sm swcredpreview">Preview</button><button class="btn brand sm swaccept" style="display:none">Add swarm</button></div>` +
      `<div class="swcredout" style="margin-top:8px"></div>`,
    `<button class="btn ghost sm swcancel">Close</button>`,
    600,
  );
  let role = "consume";
  const dev = $(".swdev", back);
  const hint = $(".swhint", back);
  const genOut = $(".swgenout", back);
  const acceptBtn = $(".swaccept", back);
  dev.focus();
  $(".swcancel", back).onclick = close;
  wireSegmented(back, "role", (v) => {
    role = v;
    $(".swcred2wrap", back).style.display = v === "both" ? "" : "none";
  });

  $(".swgen", back).onclick = async () => {
    const label = (dev.value || "").trim();
    if (!label) {
      toast("Name this device first");
      return;
    }
    genOut.innerHTML = `<span class="mut" style="font-size:11.5px">Signing…</span>`;
    try {
      const r = await call("swarm_enroll_request", {
        swarm: (hint.value || "").trim() || null,
        label,
        role,
      });
      const boxes = [];
      if (r.consume)
        boxes.push(credBox("Enrollment request", r.consume.magnet, "Send this to the owner.", 0));
      if (r.control)
        boxes.push(
          credBox("🔒 Control request", r.control.magnet, "Send this too — it's from your control key.", 1),
        );
      genOut.innerHTML = boxes.join("");
      if (r.consume) $(".swcopy0", back).onclick = copyBtnRow(r.consume.magnet, "Request copied");
      if (r.control) $(".swcopy1", back).onclick = copyBtnRow(r.control.magnet, "Control request copied");
    } catch (e) {
      genOut.innerHTML = `<div class="cardfail">✕ ${escapeHtml(String(e))}</div>`;
    }
  };

  const cred = $(".swcred", back);
  const cred2 = $(".swcred2", back);
  const credOut = $(".swcredout", back);
  let previewed = null; // { credential, control_credential|null }

  $(".swcredpreview", back).onclick = async () => {
    const text = (cred.value || "").trim();
    const text2 = (cred2?.value || "").trim();
    if (!text) {
      toast("Paste a credential first");
      return;
    }
    if (role === "both" && !text2) {
      toast("Paste the second (control) credential too");
      return;
    }
    credOut.innerHTML = `<span class="mut" style="font-size:11.5px">Verifying…</span>`;
    acceptBtn.style.display = "none";
    previewed = null;
    try {
      const c1 = await call("preview_swarm_credential", { credential: text });
      const cards = [credentialCardHtml(c1, await fingerprintOf(c1.swarm_public_key))];
      if (role === "both") {
        const c2 = await call("preview_swarm_credential", { credential: text2 });
        cards.push(credentialCardHtml(c2, await fingerprintOf(c2.swarm_public_key)));
      }
      credOut.innerHTML = cards.join("");
      previewed = { credential: text, control_credential: role === "both" ? text2 : null };
      acceptBtn.style.display = "";
    } catch (e) {
      credOut.innerHTML = `<div class="cardfail">✕ ${escapeHtml(String(e))}</div>`;
    }
  };

  acceptBtn.onclick = async () => {
    if (!previewed) return;
    try {
      const v = await call("swarm_accept_credential", {
        credential: previewed.credential,
        control_credential: previewed.control_credential,
        label: null,
      });
      toast(`Joined “${v.label || "swarm"}”`);
      close();
      renderSwarms();
    } catch (e) {
      toast(`Couldn't join: ${e}`);
    }
  };
}

// A verified-credential preview card, tagged by the capability it grants so the member sees whether a
// pasted credential is a consume or a 🔒 control credential.
function credentialCardHtml(c, fp) {
  const caps = Number(c.capabilities || 0);
  const isControl = (caps & 2) !== 0 && (caps & 1) === 0; // CAP_CONTROL without CAP_SERVE
  const tag = isControl
    ? ` <span class="badge secondary">🔒 control</span>`
    : (caps & 2) !== 0
      ? ` <span class="badge secondary">consume + control</span>`
      : "";
  return (
    `<div class="card pad" style="font-size:11.5px;margin-bottom:6px">` +
    `<div><b>${escapeHtml(c.swarm_label || "swarm")}</b> <span class="badge ok">✔ signature valid</span>${tag}</div>` +
    `<div class="mut mono" style="margin-top:4px;font-size:11px">${escapeHtml(fp)}</div>` +
    `<div class="mut" style="margin-top:2px">expires ${new Date(Number(c.expires_at)).toLocaleDateString()}</div>` +
    `</div>`
  );
}

// A fingerprint helper: the backend already computes fingerprints for stored keys, but a freshly
// previewed request/credential carries the raw public key — derive its display fingerprint the same
// way the backend does (SHA-256[..8], grouped upper-hex) so both agree.
async function fingerprintOf(publicKeyHex) {
  try {
    if (!/^[0-9a-fA-F]+$/.test(publicKeyHex)) return "invalid-key";
    const bytes = new Uint8Array(publicKeyHex.match(/.{1,2}/g).map((h) => parseInt(h, 16)));
    const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
    const hex = [...digest.slice(0, 8)].map((b) => b.toString(16).padStart(2, "0").toUpperCase()).join("");
    return hex.match(/.{1,4}/g).join(" ");
  } catch {
    return "";
  }
}
