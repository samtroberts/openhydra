import { useState, useRef, useCallback } from "react";
import { Send, Square } from "lucide-react";
import { useChat } from "../../contexts/ChatContext";

export default function ChatInput() {
  const { state, sendMessage, cancelStream } = useChat();
  const [value, setValue] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = useCallback(() => {
    const trimmed = value.trim();
    if (!trimmed || state.isStreaming) return;
    sendMessage(trimmed);
    setValue("");
    inputRef.current?.focus();
  }, [value, state.isStreaming, sendMessage]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-gray-800 p-3">
      <div className="flex items-end gap-2 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 focus-within:border-[#00d4b8]">
        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Send a message..."
          rows={1}
          className="flex-1 bg-transparent text-sm text-gray-100 placeholder-gray-600 resize-none outline-none max-h-32"
        />
        {state.isStreaming ? (
          <button
            onClick={cancelStream}
            className="p-1 text-red-400 hover:text-red-300 transition-colors"
            title="Stop generating"
          >
            <Square size={16} />
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!value.trim()}
            className="p-1 text-[#00d4b8] hover:text-[#00e8ca] disabled:text-gray-700 transition-colors"
            title="Send"
          >
            <Send size={16} />
          </button>
        )}
      </div>
    </div>
  );
}
