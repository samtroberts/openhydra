import { Play, Square, Loader } from "lucide-react";
import { useNode } from "../../contexts/NodeContext";

export default function NodeToggle() {
  const { state, startNode, stopNode } = useNode();
  const { status } = state;

  const isDisabled = status === "starting" || status === "bootstrapping";

  if (status === "running") {
    return (
      <button
        onClick={stopNode}
        className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-red-600 hover:bg-red-700 text-white font-medium text-sm transition-colors"
      >
        <Square size={16} />
        Stop Node
      </button>
    );
  }

  if (status === "starting" || status === "bootstrapping") {
    return (
      <button disabled className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-amber-600/80 text-white font-medium text-sm cursor-wait">
        <Loader size={16} className="animate-spin" />
        {status === "bootstrapping" ? "Setting up..." : "Starting..."}
      </button>
    );
  }

  return (
    <button
      onClick={() => startNode()}
      disabled={isDisabled}
      className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white font-medium text-sm transition-colors"
    >
      <Play size={16} />
      Start Node
    </button>
  );
}
