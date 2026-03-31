import { Activity, Globe, Coins } from "lucide-react";
import { useNode } from "../../contexts/NodeContext";

export default function StatusCards() {
  const { state } = useNode();

  const statusColor =
    state.status === "running" ? "text-emerald-400" :
    state.status === "starting" ? "text-amber-400" :
    state.status === "error" ? "text-red-400" :
    "text-gray-500";

  const statusLabel =
    state.status === "running" ? "Running" :
    state.status === "starting" ? "Starting..." :
    state.status === "error" ? "Error" :
    "Stopped";

  const statusDot =
    state.status === "running" ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]" :
    state.status === "starting" ? "bg-amber-400 animate-pulse" :
    "bg-gray-600";

  return (
    <div className="flex gap-3">
      <div className="bg-zinc-900/50 border border-white/[0.06] rounded-xl px-4 py-3 min-w-[120px]">
        <div className="flex items-center gap-1.5 text-gray-500 text-[10px] font-medium uppercase tracking-wider mb-1.5">
          <div className={`w-1.5 h-1.5 rounded-full ${statusDot}`} />
          Status
        </div>
        <div className={`text-sm font-semibold ${statusColor}`}>
          {statusLabel}
        </div>
      </div>

      <div className="bg-zinc-900/50 border border-white/[0.06] rounded-xl px-4 py-3 min-w-[100px]">
        <div className="flex items-center gap-1.5 text-gray-500 text-[10px] font-medium uppercase tracking-wider mb-1.5">
          <Globe size={10} />
          Port
        </div>
        <div className="text-sm font-semibold font-mono text-gray-200">
          :{state.config.apiPort}
        </div>
      </div>

      <div className="bg-zinc-900/50 border border-white/[0.06] rounded-xl px-4 py-3 min-w-[120px]">
        <div className="flex items-center gap-1.5 text-gray-500 text-[10px] font-medium uppercase tracking-wider mb-1.5">
          <Coins size={10} />
          HYDRA
        </div>
        <div className="text-sm font-semibold font-mono text-[#00d4b8]">
          {state.balance.hydra.toFixed(2)}
        </div>
      </div>
    </div>
  );
}
