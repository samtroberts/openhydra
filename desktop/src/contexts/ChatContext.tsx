import { createContext, useContext, useReducer, useRef, useCallback, type ReactNode } from "react";
import { streamChat } from "../lib/api";

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
  ts: number;
  isStreaming?: boolean;
}

interface ChatState {
  messages: Message[];
  isStreaming: boolean;
  model: string;
}

type Action =
  | { type: "ADD_USER_MSG"; content: string }
  | { type: "START_STREAM" }
  | { type: "STREAM_CHUNK"; content: string }
  | { type: "END_STREAM" }
  | { type: "STREAM_ERROR"; error: string }
  | { type: "SET_MODEL"; model: string }
  | { type: "CLEAR" };

function reducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case "ADD_USER_MSG":
      return {
        ...state,
        messages: [...state.messages, { role: "user", content: action.content, ts: Date.now() }],
      };
    case "START_STREAM":
      return {
        ...state,
        isStreaming: true,
        messages: [...state.messages, { role: "assistant", content: "", ts: Date.now(), isStreaming: true }],
      };
    case "STREAM_CHUNK": {
      const msgs = [...state.messages];
      const last = msgs[msgs.length - 1];
      if (last?.role === "assistant") {
        msgs[msgs.length - 1] = { ...last, content: last.content + action.content };
      }
      return { ...state, messages: msgs };
    }
    case "END_STREAM": {
      const msgs = [...state.messages];
      const last = msgs[msgs.length - 1];
      if (last?.role === "assistant") {
        msgs[msgs.length - 1] = { ...last, isStreaming: false };
      }
      return { ...state, isStreaming: false, messages: msgs };
    }
    case "STREAM_ERROR":
      return {
        ...state,
        isStreaming: false,
        messages: [
          ...state.messages,
          { role: "assistant", content: `Error: ${action.error}`, ts: Date.now() },
        ],
      };
    case "SET_MODEL":
      return { ...state, model: action.model };
    case "CLEAR":
      return { ...state, messages: [] };
    default:
      return state;
  }
}

interface ChatContextValue {
  state: ChatState;
  sendMessage: (content: string) => Promise<void>;
  cancelStream: () => void;
  clearChat: () => void;
  setModel: (model: string) => void;
}

const ChatContext = createContext<ChatContextValue | null>(null);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, {
    messages: [],
    isStreaming: false,
    model: "openhydra-qwen3.5-0.8b",
  });
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (content: string) => {
    if (state.isStreaming) return;

    dispatch({ type: "ADD_USER_MSG", content });
    dispatch({ type: "START_STREAM" });

    const controller = new AbortController();
    abortRef.current = controller;

    const messages = [...state.messages, { role: "user", content }].map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      for await (const chunk of streamChat(state.model, messages, controller.signal)) {
        dispatch({ type: "STREAM_CHUNK", content: chunk });
      }
      dispatch({ type: "END_STREAM" });
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") {
        dispatch({ type: "END_STREAM" });
      } else {
        dispatch({ type: "STREAM_ERROR", error: String(err) });
      }
    } finally {
      abortRef.current = null;
    }
  }, [state.isStreaming, state.messages, state.model]);

  const cancelStream = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const clearChat = useCallback(() => dispatch({ type: "CLEAR" }), []);
  const setModel = useCallback((model: string) => dispatch({ type: "SET_MODEL", model }), []);

  return (
    <ChatContext.Provider value={{ state, sendMessage, cancelStream, clearChat, setModel }}>
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChat must be inside ChatProvider");
  return ctx;
}
