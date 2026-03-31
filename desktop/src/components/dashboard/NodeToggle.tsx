import { Play, Square, Loader } from "lucide-react";
import { useNode } from "../../contexts/NodeContext";

export default function NodeToggle() {
  const { state, startNode, stopNode } = useNode();
  const { status } = state;

  if (status === "running") {
    return (
      <button
        onClick={stopNode}
        className="
          flex items-center gap-2 px-5 py-2.5 rounded-xl
          bg-red-500/10 border border-red-500/20
          text-red-400 hover:text-red-300 hover:bg-red-500/15 hover:border-red-500/30
          font-medium text-sm transition-all duration-200
        "
      >
        <Square size={14} />
        Stop
      </button>
    );
  }

  if (status === "starting" || status === "bootstrapping") {
    return (
      <button
        disabled
        className="
          flex items-center gap-2 px-5 py-2.5 rounded-xl
          bg-amber-500/10 border border-amber-500/20
          text-amber-400 font-medium text-sm cursor-wait
        "
      >
        <Loader size={14} className="animate-spin" />
        {status === "bootstrapping" ? "Setting up..." : "Starting..."}
      </button>
    );
  }

  return (
    <button
      onClick={() => startNode()}
      className="
        flex items-center gap-2 px-5 py-2.5 rounded-xl
        bg-emerald-500/10 border border-emerald-500/20
        text-emerald-400 hover:text-emerald-300 hover:bg-emerald-500/15 hover:border-emerald-500/30
        font-medium text-sm transition-all duration-200
      "
    >
      <Play size={14} />
      Start Node
    </button>
  );
}
