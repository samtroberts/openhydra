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
    <div className="flex-1 min-h-0 flex flex-col bg-gray-950 border border-gray-800 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-gray-800">
        <span className="text-xs text-gray-500 font-mono">Logs</span>
        <button
          onClick={clearLogs}
          className="text-gray-600 hover:text-gray-400 transition-colors"
          title="Clear logs"
        >
          <Trash2 size={12} />
        </button>
      </div>
      <div className="flex-1 overflow-auto p-2 font-mono text-xs leading-5 custom-scrollbar">
        {state.logs.length === 0 && (
          <span className="text-gray-600">No logs yet. Start your node to see output.</span>
        )}
        {state.logs.map((log, i) => (
          <div
            key={i}
            className={log.stream === "stderr" ? "text-red-400" : "text-gray-400"}
          >
            {log.line}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
