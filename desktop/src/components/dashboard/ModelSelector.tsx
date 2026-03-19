import { useNode } from "../../contexts/NodeContext";
import { MODEL_CATALOG, rewardMultiplier } from "../../lib/models";

export default function ModelSelector() {
  const { state, updateConfig } = useNode();
  const disabled = state.status === "running" || state.status === "starting";

  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1.5">Model</label>
      <select
        value={state.config.modelId}
        onChange={(e) => updateConfig({ modelId: e.target.value })}
        disabled={disabled}
        className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 disabled:opacity-50 focus:outline-none focus:border-[#00d4b8]"
      >
        {MODEL_CATALOG.map((m) => (
          <option key={m.id} value={m.hfId}>
            {m.name} — {m.tier} ({m.minVramGb} GB, {rewardMultiplier(m)}x rewards)
          </option>
        ))}
      </select>
    </div>
  );
}
