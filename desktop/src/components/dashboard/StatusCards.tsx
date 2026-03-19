import { Activity, Globe, Coins } from "lucide-react";
import { useNode } from "../../contexts/NodeContext";

export default function StatusCards() {
  const { state } = useNode();

  const statusColor = state.status === "running" ? "text-emerald-400" : "text-gray-500";
  const statusLabel = state.status === "running" ? "Running" : state.status === "starting" ? "Starting..." : "Stopped";

  return (
    <div className="grid grid-cols-3 gap-3">
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="flex items-center gap-2 text-gray-400 text-xs mb-2">
          <Activity size={14} />
          Node Status
        </div>
        <div className={`text-lg font-semibold ${statusColor}`}>
          {statusLabel}
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="flex items-center gap-2 text-gray-400 text-xs mb-2">
          <Globe size={14} />
          API Port
        </div>
        <div className="text-lg font-semibold font-mono">
          :{state.config.apiPort}
        </div>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
        <div className="flex items-center gap-2 text-gray-400 text-xs mb-2">
          <Coins size={14} />
          HYDRA Earned
        </div>
        <div className="text-lg font-semibold font-mono text-[#00d4b8]">
          {state.balance.hydra.toFixed(2)}
        </div>
      </div>
    </div>
  );
}
