// Settings › Network: which mode the network settings say, and whether the running roles still use
// another (review 2026-10-14, M1/M2). No imports, so the UI tests can load it.

// Whether the gateway downloads tokenizers under these settings: the public network's switch, or a
// private deployment's (off unless turned on). Absent fields: their defaults.
export function tokenizerDownloadsOn(s) {
  return s.public_network !== false ? s.tokenizer_downloads !== false : s.private_tokenizer_downloads === true;
}

// Which setting the "Check billed token counts" switch edits for these settings.
export function tokenizerDownloadsKey(s) {
  return s.public_network !== false ? "tokenizer_downloads" : "private_tokenizer_downloads";
}

// What still runs with the old network settings, as a sentence ("" when nothing): a running role
// started with another public-network mode, or a gateway with another tokenizer-download setting.
export function networkRestartPending(p, s) {
  const publicNet = s.public_network !== false;
  const mode = (pub) => (pub ? "on the public network" : "as a private deployment");
  const out = [];
  const gw = p.gateway?.status, pv = p.provider?.status;
  const gwStale = gw?.running && gw.started_public_network != null && gw.started_public_network !== publicNet;
  const pvStale = pv?.running && pv.started_public_network != null && pv.started_public_network !== publicNet;
  if (gwStale && pvStale) out.push(`The gateway and sharing still run ${mode(!publicNet)}.`);
  else if (gwStale) out.push(`The gateway still runs ${mode(!publicNet)}.`);
  else if (pvStale) out.push(`Sharing still runs ${mode(!publicNet)}.`);
  if (!gwStale && gw?.running && gw.started_tokenizer_downloads != null && gw.started_tokenizer_downloads !== tokenizerDownloadsOn(s)) {
    out.push(`The gateway still runs with tokenizer downloads ${gw.started_tokenizer_downloads ? "on" : "off"}.`);
  }
  return out.join(" ");
}
