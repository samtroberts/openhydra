import { defineConfig } from "vite";

// Vite config for the OpenHydra desktop frontend (Tauri v2, vanilla — no framework).
// The app talks to the Rust backend through the global `window.__TAURI__` (withGlobalTauri:true),
// so there are no @tauri-apps/api imports to bundle. `logo-mark.png` lives in ui/public and is
// referenced by a stable absolute path (/logo-mark.png) from both index.html and app.js.
export default defineConfig({
  root: "ui",
  // Tauri serves the bundle over a custom protocol → asset URLs must be relative, not root-absolute.
  base: "./",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    target: "es2021",
    // Keep the wireframe's inline base64 logos as-is; don't let Vite re-inline other small assets.
    assetsInlineLimit: 0,
  },
  // Don't clobber the Tauri CLI's output when run as beforeDevCommand.
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  // Tauri injects TAURI_ENV_* at build time; expose them (plus the usual VITE_*) to the client.
  envPrefix: ["VITE_", "TAURI_ENV_"],
});
