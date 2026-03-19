import { useState, useEffect, useRef } from "react";
import { Command } from "@tauri-apps/plugin-shell";
import { CheckCircle, Loader, AlertCircle } from "lucide-react";

type StepStatus = "pending" | "active" | "done" | "error";

interface Step {
  label: string;
  status: StepStatus;
}

const INITIAL_STEPS: Step[] = [
  { label: "Checking Python...", status: "pending" },
  { label: "Creating virtual environment...", status: "pending" },
  { label: "Installing dependencies...", status: "pending" },
  { label: "Ready!", status: "pending" },
];

function StepIcon({ status }: { status: StepStatus }) {
  switch (status) {
    case "active":
      return <Loader size={14} className="animate-spin text-[#00d4b8]" />;
    case "done":
      return <CheckCircle size={14} className="text-emerald-400" />;
    case "error":
      return <AlertCircle size={14} className="text-red-400" />;
    default:
      return <div className="w-3.5 h-3.5 rounded-full border border-gray-700" />;
  }
}

export default function BootstrapScreen({ onComplete }: { onComplete: () => void }) {
  const [steps, setSteps] = useState<Step[]>(INITIAL_STEPS);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return;
    started.current = true;

    const run = async () => {
      try {
        const command = Command.sidecar("binaries/openhydra-node", ["--bootstrap"]);

        command.stdout.on("data", (line: string) => {
          const trimmed = line.trim();
          setLogLines((prev) => [...prev.slice(-50), trimmed]);

          if (trimmed.includes("checking_python") || trimmed.includes("Checking Python")) {
            setSteps((s) => s.map((st, i) => ({ ...st, status: i === 0 ? "active" : st.status })));
          } else if (trimmed.includes("creating_venv") || trimmed.includes("Creating")) {
            setSteps((s) =>
              s.map((st, i) => ({
                ...st,
                status: i === 0 ? "done" : i === 1 ? "active" : st.status,
              }))
            );
          } else if (trimmed.includes("installing") || trimmed.includes("Installing")) {
            setSteps((s) =>
              s.map((st, i) => ({
                ...st,
                status: i <= 1 ? "done" : i === 2 ? "active" : st.status,
              }))
            );
          } else if (trimmed.includes("bootstrap_complete") || trimmed.includes("Ready")) {
            setSteps((s) => s.map((st) => ({ ...st, status: "done" })));
          }
        });

        command.stderr.on("data", (line: string) => {
          setLogLines((prev) => [...prev.slice(-50), line.trim()]);
        });

        command.on("close", (data) => {
          if (data.code === 0) {
            setSteps((s) => s.map((st) => ({ ...st, status: "done" })));
            setTimeout(onComplete, 1000);
          } else {
            setError(`Bootstrap failed (exit code ${data.code})`);
            setSteps((s) =>
              s.map((st) => (st.status === "active" ? { ...st, status: "error" } : st))
            );
          }
        });

        command.on("error", (err) => {
          setError(String(err));
        });

        await command.spawn();
        setSteps((s) => s.map((st, i) => ({ ...st, status: i === 0 ? "active" : st.status })));
      } catch (err) {
        setError(`Failed to start bootstrap: ${err}`);
      }
    };

    run();
  }, [onComplete]);

  return (
    <div className="h-screen flex flex-col items-center justify-center bg-[#0a0a0a] text-gray-100 p-8">
      <div className="text-4xl mb-4">🦙</div>
      <h1 className="text-xl font-semibold mb-1">Setting up OpenHydra</h1>
      <p className="text-sm text-gray-500 mb-8">First-time setup — this only happens once.</p>

      <div className="w-full max-w-sm space-y-3 mb-8">
        {steps.map((step, i) => (
          <div key={i} className="flex items-center gap-3">
            <StepIcon status={step.status} />
            <span
              className={`text-sm ${
                step.status === "active"
                  ? "text-[#00d4b8]"
                  : step.status === "done"
                  ? "text-gray-400"
                  : step.status === "error"
                  ? "text-red-400"
                  : "text-gray-600"
              }`}
            >
              {step.label}
            </span>
          </div>
        ))}
      </div>

      {error && (
        <div className="w-full max-w-sm bg-red-950/30 border border-red-800 rounded-lg p-3 mb-4">
          <p className="text-xs text-red-400">{error}</p>
        </div>
      )}

      <div className="w-full max-w-sm bg-gray-950 border border-gray-800 rounded-lg p-3 max-h-32 overflow-auto font-mono text-[10px] text-gray-600 custom-scrollbar">
        {logLines.map((line, i) => (
          <div key={i}>{line}</div>
        ))}
      </div>
    </div>
  );
}
