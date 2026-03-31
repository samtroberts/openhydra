import { useNode } from "../../contexts/NodeContext";

export default function RamSlider() {
  const { state, updateConfig } = useNode();
  const disabled = state.status === "running" || state.status === "starting";

  return (
    <div>
      <label className="block text-[10px] text-gray-500 font-medium uppercase tracking-wider mb-1.5">
        RAM Allocation
        <span className="ml-1.5 text-gray-300 font-mono text-xs normal-case">
          {state.config.ramAllocation} GB
        </span>
        <span className="ml-1 text-gray-600 font-normal normal-case">
          / {state.systemRam} GB
        </span>
      </label>
      <input
        type="range"
        min={2}
        max={Math.max(state.systemRam, 32)}
        value={state.config.ramAllocation}
        onChange={(e) => updateConfig({ ramAllocation: Number(e.target.value) })}
        disabled={disabled}
        className="w-full accent-cyan-500 disabled:opacity-40 disabled:cursor-not-allowed h-1.5"
      />
    </div>
  );
}
