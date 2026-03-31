import { useRef, useEffect } from "react";
import { Trash2 } from "lucide-react";
import { useNode } from "../../contexts/NodeContext";

export default function LogTerminal() {
  const { state, clearLogs } = useNode();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [state.logs.length]);

  return (
    <div className="flex-1 min-h-0 flex flex-col bg-zinc-950/80 border border-white/[0.04] rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/[0.04]">
        <span className="text-[10px] text-gray-500 font-medium uppercase tracking-wider">Output</span>
        <button
          onClick={clearLogs}
          className="text-gray-600 hover:text-gray-400 transition-colors p-0.5 rounded"
          title="Clear logs"
        >
          <Trash2 size={11} />
        </button>
      </div>
      <div className="flex-1 overflow-auto p-3 font-mono text-[11px] leading-5 custom-scrollbar">
        {state.logs.length === 0 && (
          <span className="text-gray-600 select-none">
            No logs yet. Start your node to see output.
          </span>
        )}
        {state.logs.map((log, i) => (
          <div
            key={i}
            className={log.stream === "stderr" ? "text-red-400/80" : "text-gray-500"}
          >
            {log.line}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
