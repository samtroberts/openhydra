import { Wifi, WifiOff } from "lucide-react";
import { useNode } from "../../contexts/NodeContext";
import type { NodeMode } from "../../contexts/NodeContext";

export default function ModeToggle() {
  const { state, switchMode } = useNode();
  const isLocal = state.mode === "local";
  const disabled = state.modeSwitching || state.status === "starting";

  const handleToggle = (target: NodeMode) => {
    if (disabled || state.mode === target) return;
    switchMode(target);
  };

  return (
    <div className="flex items-center gap-3">
      <div className="relative flex h-9 rounded-lg bg-zinc-900/80 border border-white/[0.06] p-0.5">
        {/* Sliding indicator */}
        <div
          className={`
            absolute top-0.5 h-8 w-[calc(50%-2px)] rounded-md
            transition-all duration-300 ease-in-out
            ${isLocal
              ? "left-0.5 bg-zinc-800 shadow-sm"
              : "left-[calc(50%+2px)] bg-cyan-500/15 shadow-[0_0_12px_rgba(0,212,184,0.1)]"
            }
          `}
        />

        {/* Local button */}
        <button
          onClick={() => handleToggle("local")}
          disabled={disabled}
          className={`
            relative z-10 flex items-center gap-1.5 px-4 py-1.5 rounded-md text-xs font-medium
            transition-colors duration-200 min-w-[6rem] justify-center
            ${disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"}
            ${isLocal ? "text-gray-200" : "text-gray-500 hover:text-gray-400"}
          `}
        >
          <WifiOff size={12} />
          Local
        </button>

        {/* Swarm button */}
        <button
          onClick={() => handleToggle("swarm")}
          disabled={disabled}
          className={`
            relative z-10 flex items-center gap-1.5 px-4 py-1.5 rounded-md text-xs font-medium
            transition-colors duration-200 min-w-[6rem] justify-center
            ${disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer"}
            ${!isLocal ? "text-cyan-300" : "text-gray-500 hover:text-gray-400"}
          `}
        >
          <Wifi size={12} />
          Swarm
        </button>
      </div>

      {/* Status label */}
      {state.modeSwitching ? (
        <span className="text-[10px] text-amber-400/80 font-medium animate-pulse">
          Switching engines...
        </span>
      ) : (
        <span className="text-[10px] text-gray-500">
          {isLocal ? "Private, offline inference" : "Contributing to the global network"}
        </span>
      )}
    </div>
  );
}
