import { Activity, MessageSquare, LayoutDashboard } from "lucide-react";
import { useNode } from "../../contexts/NodeContext";

interface Props {
  activeView: string;
  onViewChange: (view: "dashboard" | "swarm" | "chat") => void;
}

export default function Header({ activeView, onViewChange }: Props) {
  const { state } = useNode();

  const statusColor =
    state.status === "running" ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.4)]" :
    state.status === "starting" ? "bg-amber-400 animate-pulse" :
    state.status === "error" ? "bg-red-400" :
    "bg-gray-600";

  const statusText =
    state.status === "running" ? "Running" :
    state.status === "starting" ? "Starting..." :
    state.status === "error" ? "Error" :
    state.status === "bootstrapping" ? "Setting up..." :
    "Stopped";

  const modeLabel = state.mode === "local" ? "Local" : "Swarm";

  const tabs = [
    { id: "dashboard" as const, label: "Dashboard", icon: LayoutDashboard },
    { id: "swarm" as const, label: "Swarm", icon: Activity },
    { id: "chat" as const, label: "Chat", icon: MessageSquare },
  ];

  return (
    <header
      className="h-11 border-b border-white/[0.06] flex items-center px-4 gap-4 shrink-0 bg-zinc-950/90 backdrop-blur-sm"
      data-tauri-drag-region
    >
      <div className="flex items-center gap-2 mr-3" data-tauri-drag-region>
        <span className="text-base">🦙</span>
        <span className="font-semibold text-[13px] tracking-tight text-gray-200">
          OpenHydra
        </span>
        <span className="text-[9px] text-gray-600 font-mono">v1.1</span>
      </div>

      <nav className="flex gap-0.5">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onViewChange(tab.id)}
            className={`
              flex items-center gap-1.5 px-3 py-1 rounded-md text-[12px] font-medium
              transition-all duration-200
              ${activeView === tab.id
                ? "bg-white/[0.07] text-gray-100"
                : "text-gray-500 hover:text-gray-300 hover:bg-white/[0.03]"
              }
            `}
          >
            <tab.icon size={13} strokeWidth={activeView === tab.id ? 2 : 1.5} />
            {tab.label}
          </button>
        ))}
      </nav>

      <div className="ml-auto flex items-center gap-3">
        <span className="text-[10px] text-gray-600 font-mono">{modeLabel}</span>
        <div className="flex items-center gap-1.5">
          <div className={`w-1.5 h-1.5 rounded-full ${statusColor}`} />
          <span className="text-[11px] text-gray-400">{statusText}</span>
        </div>
      </div>
    </header>
  );
}
