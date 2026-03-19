import { Zap, Users, ArrowRight } from "lucide-react";
import { useNode } from "../../contexts/NodeContext";
import { MODEL_CATALOG, recommendModel, rewardMultiplier, type ModelInfo } from "../../lib/models";

function ModelCard({ model, recommended, active, onJoin }: {
  model: ModelInfo;
  recommended: boolean;
  active: boolean;
  onJoin: () => void;
}) {
  const mult = rewardMultiplier(model);
  const tierColors: Record<string, string> = {
    Nano: "border-gray-600 bg-gray-900",
    Basic: "border-blue-800 bg-blue-950/30",
    Standard: "border-purple-800 bg-purple-950/30",
    Advanced: "border-amber-800 bg-amber-950/30",
    Frontier: "border-emerald-700 bg-emerald-950/30",
  };
  const borderClass = tierColors[model.tier] || tierColors.Nano;

  return (
    <div className={`border rounded-lg p-4 ${borderClass} ${recommended ? "ring-1 ring-[#00d4b8]" : ""}`}>
      {recommended && (
        <div className="text-[10px] uppercase tracking-wider text-[#00d4b8] font-semibold mb-2 flex items-center gap-1">
          <Zap size={10} /> Recommended for your hardware
        </div>
      )}
      <div className="flex items-center justify-between mb-1">
        <span className="font-semibold text-gray-100">{model.name}</span>
        <span className="text-xs font-mono text-gray-400">{model.tier}</span>
      </div>
      <p className="text-xs text-gray-500 mb-3">{model.description}</p>
      <div className="flex items-center gap-3 text-xs text-gray-400 mb-3">
        <span>{model.minVramGb} GB RAM</span>
        <span className="text-gray-700">|</span>
        <span className="flex items-center gap-1">
          <Users size={10} /> {model.requiredPeers} peer{model.requiredPeers > 1 ? "s" : ""}
        </span>
        <span className="text-gray-700">|</span>
        <span className="text-[#00d4b8] font-semibold">{mult}x rewards</span>
      </div>
      <button
        onClick={onJoin}
        disabled={active}
        className={`w-full py-1.5 rounded text-xs font-medium transition-colors ${
          active
            ? "bg-[#00d4b8]/20 text-[#00d4b8] border border-[#00d4b8]/30 cursor-default"
            : "bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white"
        }`}
      >
        {active ? "Currently Running" : "Join Swarm"}
        {!active && <ArrowRight size={10} className="inline ml-1" />}
      </button>
    </div>
  );
}

export default function SwarmView() {
  const { state, startNode, stopNode, updateConfig } = useNode();
  const rec = recommendModel(state.systemRam);

  const handleJoin = async (model: ModelInfo) => {
    if (state.status === "running" || state.status === "starting") {
      await stopNode();
    }
    updateConfig({ modelId: model.hfId });
    await startNode({ modelId: model.hfId });
  };

  return (
    <div className="h-full overflow-auto p-4">
      <div className="mb-6">
        <h2 className="text-lg font-semibold text-gray-100 mb-1">Network Demand</h2>
        <p className="text-sm text-gray-500">
          Your Mac has <span className="text-gray-200 font-semibold">{state.systemRam} GB</span> — you can
          power the <span className="text-[#00d4b8] font-semibold">{rec.tier} Swarm</span>.
          Drop into a model swarm below to start earning.
        </p>
      </div>

      <div className="grid gap-3">
        {MODEL_CATALOG.map((m) => (
          <ModelCard
            key={m.id}
            model={m}
            recommended={m.id === rec.id}
            active={state.config.modelId === m.hfId && state.status === "running"}
            onJoin={() => handleJoin(m)}
          />
        ))}
      </div>
    </div>
  );
}
