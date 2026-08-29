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

// ── approve member (owner): paste request → preview fingerprint → approve → hand back credential ──
function approveMemberModal(swarmPublicKey, swarmLabel) {
  const { back, close } = modal(
    `Add a member to “${swarmLabel}”`,
    `<div class="cmodal-path">Paste the enrollment request the person sent you. Confirm their fingerprint with them (voice/in person) before approving — that's what stops you enrolling the wrong key.</div>` +
      `<textarea class="input mono swreq" rows="3" placeholder="openhydra:enroll:…   or the request JSON" style="width:100%;font-size:11px;margin-top:10px"></textarea>` +
      `<div class="row" style="margin-top:8px"><button class="btn outline sm swpreview">Preview</button></div>` +
      `<div class="swpreviewout" style="margin-top:8px"></div>`,
    `<button class="btn ghost sm swcancel">Close</button>`,
  );
  const req = $(".swreq", back);
  const out = $(".swpreviewout", back);
  const foot = $(".cmodal-f", back);
  $(".swcancel", back).onclick = close;

  $(".swpreview", back).onclick = async () => {
    const text = (req.value || "").trim();
    if (!text) {
      toast("Paste a request first");
      return;
    }
    out.innerHTML = `<span class="mut" style="font-size:11.5px">Verifying…</span>`;
    try {
      const r = await call("preview_enroll_request", { request: text });
      const fp = await fingerprintOf(r.member_public_key);
      out.innerHTML =
        `<div class="card pad" style="font-size:11.5px">` +
        `<div><b>${escapeHtml(r.label || "member")}</b> <span class="badge ok">✔ request signed</span></div>` +
        `<div class="mut mono" style="margin-top:4px;font-size:11px" title="Confirm this with them out-of-band">${escapeHtml(fp)}</div>` +
        `<div class="mut" style="margin-top:2px">id ${escapeHtml(peerShort(r.member_openhydra_peer_id))}</div>` +
        `</div>` +
        `<label class="mut" style="font-size:11px;margin-top:10px;display:block">Label for this member</label>` +
        `<input class="input swmlabel" value="${escapeHtml(r.label || "")}" placeholder="e.g. Sam's MacBook" style="width:100%" maxlength="128"/>`;
      // Swap the footer to an Approve action now that a valid request is loaded.
      foot.innerHTML = `<button class="btn ghost sm swcancel2">Close</button><button class="btn sm brand swapprove">Approve &amp; issue credential</button>`;
      $(".swcancel2", back).onclick = close;
      $(".swapprove", back).onclick = () => doApprove(back, close, swarmPublicKey, text);
    } catch (e) {
      out.innerHTML = `<div class="cardfail">✕ ${escapeHtml(String(e))}</div>`;
      foot.innerHTML = `<button class="btn ghost sm swcancel3">Close</button>`;
      $(".swcancel3", back).onclick = close;
    }
  };
}

async function doApprove(back, close, swarmPublicKey, requestText) {
  const memberLabel = ($(".swmlabel", back)?.value || "").trim();
  try {
    const approved = await call("swarm_approve_member", {
      swarm_public_key: swarmPublicKey,
      request: requestText,
      member_label: memberLabel,
      ttl_secs: CRED_TTL_SECS,
    });
    // Show the credential to send back — replace the body with a copy box.
    const magnet = approved.magnet;
    $(".cmodal-b", back).innerHTML =
      `<div class="cmodal-path">Approved. Send this credential back to the member — they paste it into “Join a swarm” to finish. It's signed and carries no secret.</div>` +
      `<textarea class="input mono" rows="3" readonly style="width:100%;font-size:10.5px;margin-top:10px">${escapeHtml(magnet)}</textarea>`;
    $(".cmodal-f", back).innerHTML =
      `<button class="btn ghost sm swdone">Done</button><button class="btn sm brand swcopycred">Copy credential</button>`;
    $(".swcopycred", back).onclick = copyBtnRow(magnet, "Credential copied — send it to the member");
    $(".swdone", back).onclick = () => {
      close();
      renderSwarms();
    };
  } catch (e) {
    toast(`Approve failed: ${e}`);
  }
}

// ── join a swarm (member): generate a request to send, then paste the credential you get back ──
function joinSwarmModal() {
  const { back, close } = modal(
    "Join a swarm",
    `<div class="cmodal-path">Two steps. First generate your enrollment request and send it to the swarm's owner (any channel — it's public). When they send a credential back, paste it below to finish.</div>` +
      `<label class="mut" style="font-size:11px;margin-top:10px;display:block">A name for this device (the owner sees it)</label>` +
      `<input class="input swdev" placeholder="e.g. Sam's MacBook" style="width:100%" maxlength="128"/>` +
      `<label class="mut" style="font-size:11px;margin-top:8px;display:block">Swarm key (optional — pin which swarm you're joining)</label>` +
      `<input class="input mono swhint" placeholder="group public key (optional)" style="width:100%;font-size:10.5px"/>` +
      `<div class="row" style="margin-top:8px"><button class="btn outline sm swgen">Generate my request</button></div>` +
      `<div class="swgenout" style="margin-top:8px"></div>` +
      `<hr style="border:0;border-top:1px solid var(--line);margin:14px 0"/>` +
      `<label class="mut" style="font-size:11px;display:block">Paste the credential the owner sent back</label>` +
      `<textarea class="input mono swcred" rows="3" placeholder="openhydra:cred:…   or the credential JSON" style="width:100%;font-size:11px;margin-top:6px"></textarea>` +
      `<div class="row" style="margin-top:8px"><button class="btn outline sm swcredpreview">Preview</button><button class="btn brand sm swaccept" style="display:none">Add swarm</button></div>` +
      `<div class="swcredout" style="margin-top:8px"></div>`,
    `<button class="btn ghost sm swcancel">Close</button>`,
    600,
  );
  const dev = $(".swdev", back);
  const hint = $(".swhint", back);
  const genOut = $(".swgenout", back);
  dev.focus();
  $(".swcancel", back).onclick = close;

  $(".swgen", back).onclick = async () => {
    const label = (dev.value || "").trim();
    if (!label) {
      toast("Name this device first");
      return;
    }
    genOut.innerHTML = `<span class="mut" style="font-size:11.5px">Signing…</span>`;
    try {
      const r = await call("swarm_enroll_request", { swarm: (hint.value || "").trim() || null, label });
      const magnet = r.magnet;
      genOut.innerHTML =
        `<textarea class="input mono" rows="3" readonly style="width:100%;font-size:10.5px">${escapeHtml(magnet)}</textarea>` +
        `<div class="row" style="margin-top:6px"><button class="btn sm brand swcopyreq">Copy request</button><span class="mut" style="font-size:11px;align-self:center;margin-left:8px">Send this to the owner.</span></div>`;
      $(".swcopyreq", back).onclick = copyBtnRow(magnet, "Request copied — send it to the owner");
    } catch (e) {
      genOut.innerHTML = `<div class="cardfail">✕ ${escapeHtml(String(e))}</div>`;
    }
  };

  const cred = $(".swcred", back);
  const credOut = $(".swcredout", back);
  const acceptBtn = $(".swaccept", back);
  let previewed = null;

  $(".swcredpreview", back).onclick = async () => {
    const text = (cred.value || "").trim();
    if (!text) {
      toast("Paste a credential first");
      return;
    }
    credOut.innerHTML = `<span class="mut" style="font-size:11.5px">Verifying…</span>`;
    acceptBtn.style.display = "none";
    previewed = null;
    try {
      const c = await call("preview_swarm_credential", { credential: text });
      const fp = await fingerprintOf(c.swarm_public_key);
      credOut.innerHTML =
        `<div class="card pad" style="font-size:11.5px">` +
        `<div><b>${escapeHtml(c.swarm_label || "swarm")}</b> <span class="badge ok">✔ signature valid</span></div>` +
        `<div class="mut mono" style="margin-top:4px;font-size:11px">${escapeHtml(fp)}</div>` +
        `<div class="mut" style="margin-top:2px">expires ${new Date(Number(c.expires_at)).toLocaleDateString()}</div>` +
        `</div>`;
      previewed = text;
      acceptBtn.style.display = "";
    } catch (e) {
      credOut.innerHTML = `<div class="cardfail">✕ ${escapeHtml(String(e))}</div>`;
    }
  };

  acceptBtn.onclick = async () => {
    if (!previewed) return;
    try {
      const v = await call("swarm_accept_credential", { credential: previewed, label: null });
      toast(`Joined “${v.label || "swarm"}”`);
      close();
      renderSwarms();
    } catch (e) {
      toast(`Couldn't join: ${e}`);
    }
  };
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
