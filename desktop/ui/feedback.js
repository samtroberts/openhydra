// Where "Feedback" leads (SL12): the GitHub issue form, with the app's version filled in, the
// known-issues page, and the community Discord. No imports, so the UI tests can load it.
export const REPO = "https://github.com/samtroberts/openhydra";

/** The bug-report form, its version field filled in when the version is known. */
export function reportUrl(version) {
  const v = String(version || "").replace(/^v/, "").trim();
  return `${REPO}/issues/new?template=bug_report.yml` + (/^[0-9A-Za-z.+-]{1,40}$/.test(v) ? `&version=${encodeURIComponent(v)}` : "");
}

export const KNOWN_ISSUES_URL = `${REPO}/blob/main/KNOWN_ISSUES.md`;

/** The community Discord: an invite that doesn't expire. */
export const DISCORD_URL = "https://discord.gg/vNn3fWxAqP";
