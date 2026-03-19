import { useNode } from "../../contexts/NodeContext";

export default function RamSlider() {
  const { state, updateConfig } = useNode();
  const disabled = state.status === "running" || state.status === "starting";

  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1.5">
        RAM Allocation: <span className="text-gray-100 font-mono">{state.config.ramAllocation} GB</span>
      </label>
      <input
        type="range"
        min={2}
        max={Math.max(state.systemRam, 32)}
        value={state.config.ramAllocation}
        onChange={(e) => updateConfig({ ramAllocation: Number(e.target.value) })}
        disabled={disabled}
        className="w-full accent-[#00d4b8] disabled:opacity-50"
      />
    </div>
  );
}
