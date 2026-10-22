# Known issues

What's already known in the current release, and what to do about it. If you hit something that isn't
here, please [report it](https://github.com/samtroberts/openhydra/issues/new/choose): the desktop app's
**Feedback** link (bottom right) fills in your version for you.

## Using the network

- **Payments aren't live.** Sharing your machine earns credits that let you use the network more; it
  doesn't earn money yet, and nothing you do in the app costs money.
- **The provider that answers can read your prompt.** Prompts travel encrypted, but the machine running
  the model sees them in plain text, as with any hosted AI service. Don't send anything you wouldn't
  give a stranger's computer. Private groups keep a model to machines you choose.
- **The command-line gateway sometimes double-checks an answer.** With `openhydra serve`, a request
  may also go to up to four other providers of the same model, to check that the answers match
  (`--audit-replay on`, the default). Turn it off with `--audit-replay off`. The desktop app has it off
  unless you switch on *Check providers with my own requests* in Settings.
- **The model list can take about a minute to fill after you start.** Providers announce their models
  every two minutes, so a freshly started app or gateway lists them as the announcements arrive
  (30–60 seconds in our tests). Asking for a model by name works sooner.

## Choosing a model

- **"Best available" only picks models it can trust:** ones on your own machine, from a card you
  imported, from a provider that has answered you well before, or from OpenHydra's own providers. If
  none of those is online, it asks you to pick a model by name.
- **"Best available" doesn't give coding tools a very small model.** When a request offers tools
  (Claude Code, OpenCode and other coding agents send them), it skips models of 4B parameters or fewer;
  if only those are online, you get an error that names them. Pick a larger model, or wait for one. A
  model you name yourself, or one your own machine shares, is always used as asked.

## Platforms

- **Windows: the installer isn't code-signed yet.** SmartScreen shows "Windows protected your PC".
  Choose *More info*, then *Run anyway*.
- **Linux: `openhydra:` links and `.openhydra` files may not open the app.** The AppImage doesn't
  register them at all; with the `.deb` or `.rpm` it hasn't been tested yet. Paste the card into
  *Add a model by card* instead.
- **Behind a strict router, connections may go through a relay** and be slower. This is expected.

## Reporting security issues

Please don't file those publicly: email sam@openhydra.co (see [SECURITY.md](SECURITY.md)).
