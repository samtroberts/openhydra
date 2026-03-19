import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Fade out splash screen
const splash = document.getElementById("splash");
if (splash) {
  splash.style.opacity = "0";
  setTimeout(() => (splash.style.display = "none"), 500);
}
