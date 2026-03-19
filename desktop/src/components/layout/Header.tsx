import { Activity, MessageSquare, LayoutDashboard } from "lucide-react";
import { useNode } from "../../contexts/NodeContext";

interface Props {
  activeView: string;
  onViewChange: (view: "dashboard" | "swarm" | "chat") => void;
}

export default function Header({ activeView, onViewChange }: Props) {
  const { state } = useNode();

  const statusColor =
    state.status === "running" ? "bg-emerald-400" :
    state.status === "starting" ? "bg-amber-400 animate-pulse" :
    state.status === "error" ? "bg-red-400" :
    "bg-gray-500";

  const statusText =
    state.status === "running" ? "Running" :
    state.status === "starting" ? "Starting..." :
    state.status === "error" ? "Error" :
    state.status === "bootstrapping" ? "Setting up..." :
    "Stopped";

  const tabs = [
    { id: "dashboard" as const, label: "Dashboard", icon: LayoutDashboard },
    { id: "swarm" as const, label: "Swarm", icon: Activity },
    { id: "chat" as const, label: "Chat", icon: MessageSquare },
  ];

  return (
    <header className="h-12 border-b border-gray-800 flex items-center px-4 gap-4 shrink-0 bg-[#0f0f0f]"
            data-tauri-drag-region>
      <div className="flex items-center gap-2 mr-4">
        <span className="text-lg">🦙</span>
        <span className="font-semibold text-sm tracking-wide">OpenHydra</span>
        <span className="text-[10px] text-gray-500 font-mono">v0.1.0</span>
      </div>

      <nav className="flex gap-1">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onViewChange(tab.id)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeView === tab.id
                ? "bg-gray-800 text-white"
                : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/50"
            }`}
          >
            <tab.icon size={14} />
            {tab.label}
          </button>
        ))}
      </nav>

      <div className="ml-auto flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${statusColor}`} />
        <span className="text-xs text-gray-400">{statusText}</span>
      </div>
    </header>
  );
}
