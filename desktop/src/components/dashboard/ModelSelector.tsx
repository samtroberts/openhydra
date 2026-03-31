import { ChevronDown } from "lucide-react";
import { useNode } from "../../contexts/NodeContext";
import { MODEL_CATALOG, rewardMultiplier } from "../../lib/models";

export default function ModelSelector() {
  const { state, updateConfig } = useNode();
  const disabled = state.status === "running" || state.status === "starting";

  return (
    <div>
      <label className="block text-[10px] text-gray-500 font-medium uppercase tracking-wider mb-1.5">
        Model
      </label>
      <div className="relative">
        <select
          value={state.config.modelId}
          onChange={(e) => updateConfig({ modelId: e.target.value })}
          disabled={disabled}
          className="
            w-full appearance-none
            bg-zinc-900/50 border border-white/[0.06] rounded-xl
            pl-3 pr-8 py-2.5 text-sm text-gray-200
            disabled:opacity-40 disabled:cursor-not-allowed
            focus:outline-none focus:border-cyan-500/40 focus:ring-1 focus:ring-cyan-500/20
            transition-all duration-200
          "
        >
          {MODEL_CATALOG.map((m) => (
            <option key={m.id} value={m.hfId}>
              {m.name} — {m.tier} ({m.minVramGb} GB)
            </option>
          ))}
        </select>
        <ChevronDown
          size={14}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 pointer-events-none"
        />
      </div>
    </div>
  );
}
