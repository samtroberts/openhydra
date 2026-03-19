import ReactMarkdown from "react-markdown";

interface Props {
  role: "user" | "assistant" | "system";
  content: string;
  isStreaming?: boolean;
}

export default function ChatMessage({ role, content, isStreaming }: Props) {
  const isUser = role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
          isUser
            ? "bg-[#00d4b8]/15 text-gray-100"
            : "bg-gray-800 text-gray-300"
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap">{content}</p>
        ) : (
          <div className="prose prose-invert prose-sm max-w-none [&_p]:my-1 [&_pre]:bg-gray-900 [&_code]:text-[#00d4b8]">
            <ReactMarkdown>{content || (isStreaming ? "..." : "")}</ReactMarkdown>
            {isStreaming && content && (
              <span className="inline-block w-1.5 h-4 bg-[#00d4b8] animate-pulse ml-0.5 align-text-bottom" />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
