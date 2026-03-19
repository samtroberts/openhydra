import { useRef, useEffect } from "react";
import { MessageSquare, Trash2 } from "lucide-react";
import { useChat } from "../../contexts/ChatContext";
import { useNode } from "../../contexts/NodeContext";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";

export default function ChatView() {
  const { state: chatState, clearChat } = useChat();
  const { state: nodeState } = useNode();
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [chatState.messages.length, chatState.messages[chatState.messages.length - 1]?.content]);

  if (nodeState.status !== "running") {
    return (
      <div className="h-full flex flex-col items-center justify-center text-gray-600 gap-3 p-8">
        <MessageSquare size={32} />
        <p className="text-sm">Start your node to begin chatting.</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800">
        <span className="text-xs text-gray-500 font-mono">{nodeState.config.modelId}</span>
        {chatState.messages.length > 0 && (
          <button
            onClick={clearChat}
            className="text-gray-600 hover:text-gray-400 transition-colors"
            title="Clear chat"
          >
            <Trash2 size={12} />
          </button>
        )}
      </div>

      <div ref={scrollRef} className="flex-1 overflow-auto p-4 space-y-3 custom-scrollbar">
        {chatState.messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-gray-600 gap-2">
            <span className="text-2xl">🦙</span>
            <p className="text-sm">Ask anything. Your node is running locally.</p>
          </div>
        )}
        {chatState.messages.map((msg, i) => (
          <ChatMessage key={i} role={msg.role} content={msg.content} isStreaming={msg.isStreaming} />
        ))}
      </div>

      <ChatInput />
    </div>
  );
}
