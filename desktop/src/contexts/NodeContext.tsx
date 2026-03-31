import { createContext, useContext, useReducer, useRef, useCallback, useEffect, type ReactNode } from "react";
import { Command, type Child } from "@tauri-apps/plugin-shell";
import { invoke } from "@tauri-apps/api/core";
import { fetchHealth, fetchBalance } from "../lib/api";
import { MODEL_CATALOG } from "../lib/models";

export type NodeStatus = "stopped" | "starting" | "running" | "error" | "bootstrapping";

interface LogEntry {
  stream: "stdout" | "stderr";
  line: string;
  ts: number;
}

interface NodeConfig {
  modelId: string;
  peerId: string;
  apiPort: number;
  ramAllocation: number;
}

interface NodeState {
  status: NodeStatus;
  logs: LogEntry[];
  config: NodeConfig;
  balance: { hydra: number; credits: number };
  healthy: boolean;
  systemRam: number;
}

type Action =
  | { type: "SET_STATUS"; status: NodeStatus }
  | { type: "APPEND_LOG"; entry: LogEntry }
  | { type: "CLEAR_LOGS" }
  | { type: "SET_CONFIG"; config: Partial<NodeConfig> }
  | { type: "SET_BALANCE"; balance: { hydra: number; credits: number } }
  | { type: "SET_HEALTHY"; healthy: boolean }
  | { type: "SET_RAM"; ram: number };

const MAX_LOGS = 500;

function reducer(state: NodeState, action: Action): NodeState {
  switch (action.type) {
    case "SET_STATUS":
      return { ...state, status: action.status };
    case "APPEND_LOG":
      const logs = [...state.logs, action.entry];
      return { ...state, logs: logs.length > MAX_LOGS ? logs.slice(-MAX_LOGS) : logs };
    case "CLEAR_LOGS":
      return { ...state, logs: [] };
    case "SET_CONFIG":
      return { ...state, config: { ...state.config, ...action.config } };
    case "SET_BALANCE":
      return { ...state, balance: action.balance };
    case "SET_HEALTHY":
      return { ...state, healthy: action.healthy };
    case "SET_RAM":
      return { ...state, systemRam: action.ram };
    default:
      return state;
  }
}

const defaultConfig: NodeConfig = {
  modelId: MODEL_CATALOG[0].hfId,
  peerId: `hydra-${Math.random().toString(36).slice(2, 8)}`,
  apiPort: 8080,
  ramAllocation: 4, // Updated dynamically after system RAM detection
};

const initialState: NodeState = {
  status: "stopped",
  logs: [],
  config: defaultConfig,
  balance: { hydra: 0, credits: 0 },
  healthy: false,
  systemRam: 0, // Fetched from Rust backend on startup via get_system_ram()
};

interface NodeContextValue {
  state: NodeState;
  startNode: (config?: Partial<NodeConfig>) => Promise<void>;
  stopNode: () => Promise<void>;
  clearLogs: () => void;
  updateConfig: (config: Partial<NodeConfig>) => void;
}

const NodeContext = createContext<NodeContextValue | null>(null);

export function NodeProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const childRef = useRef<Child | null>(null);
  const healthInterval = useRef<ReturnType<typeof setInterval>>();
  const balanceInterval = useRef<ReturnType<typeof setInterval>>();

  const startNode = useCallback(async (configOverrides?: Partial<NodeConfig>) => {
    if (state.status === "running" || state.status === "starting") return;

    const config = { ...state.config, ...configOverrides };
    dispatch({ type: "SET_CONFIG", config: configOverrides || {} });
    dispatch({ type: "SET_STATUS", status: "starting" });
    dispatch({ type: "CLEAR_LOGS" });

    try {
      const command = Command.sidecar("binaries/openhydra-node", [
        "--peer-id", config.peerId,
        "--model-id", config.modelId,
        "--api-port", String(config.apiPort),
      ]);

      command.stdout.on("data", (line: string) => {
        dispatch({ type: "APPEND_LOG", entry: { stream: "stdout", line: line.trim(), ts: Date.now() } });
        if (line.includes("Coordinator API listening") || line.includes("api_server")) {
          dispatch({ type: "SET_STATUS", status: "running" });
        }
      });

      command.stderr.on("data", (line: string) => {
        dispatch({ type: "APPEND_LOG", entry: { stream: "stderr", line: line.trim(), ts: Date.now() } });
      });

      command.on("close", (data) => {
        childRef.current = null;
        if (data.code === 100) {
          dispatch({ type: "SET_STATUS", status: "bootstrapping" });
        } else {
          dispatch({ type: "SET_STATUS", status: "stopped" });
        }
      });

      command.on("error", () => {
        dispatch({ type: "SET_STATUS", status: "error" });
      });

      const child = await command.spawn();
      childRef.current = child;
    } catch (err) {
      dispatch({ type: "SET_STATUS", status: "error" });
      dispatch({
        type: "APPEND_LOG",
        entry: { stream: "stderr", line: `Failed to spawn sidecar: ${err}`, ts: Date.now() },
      });
    }
  }, [state.status, state.config]);

  const stopNode = useCallback(async () => {
    if (childRef.current) {
      await childRef.current.kill();
      childRef.current = null;
    }
    dispatch({ type: "SET_STATUS", status: "stopped" });
    dispatch({ type: "SET_HEALTHY", healthy: false });
  }, []);

  const clearLogs = useCallback(() => dispatch({ type: "CLEAR_LOGS" }), []);
  const updateConfig = useCallback((c: Partial<NodeConfig>) => dispatch({ type: "SET_CONFIG", config: c }), []);

  // Health polling
  useEffect(() => {
    if (state.status === "running") {
      healthInterval.current = setInterval(async () => {
        const h = await fetchHealth();
        dispatch({ type: "SET_HEALTHY", healthy: h });
      }, 3000);
    } else {
      dispatch({ type: "SET_HEALTHY", healthy: false });
    }
    return () => clearInterval(healthInterval.current);
  }, [state.status]);

  // Balance polling
  useEffect(() => {
    if (state.status === "running") {
      balanceInterval.current = setInterval(async () => {
        const b = await fetchBalance();
        dispatch({ type: "SET_BALANCE", balance: b });
      }, 10000);
    }
    return () => clearInterval(balanceInterval.current);
  }, [state.status]);

  // Detect physical RAM from Rust backend on startup
  useEffect(() => {
    invoke<number>("get_system_ram")
      .then((ram) => {
        if (ram > 0) {
          dispatch({ type: "SET_RAM", ram });
          // Auto-set RAM allocation to half of physical RAM (sensible default)
          const allocation = Math.max(2, Math.floor(ram / 2));
          dispatch({ type: "SET_CONFIG", config: { ramAllocation: allocation } });
        }
      })
      .catch(() => {
        // Fallback for browser dev mode (no Tauri runtime): detect via navigator
        const navMem = (navigator as unknown as { deviceMemory?: number }).deviceMemory;
        if (navMem && navMem > 0) {
          dispatch({ type: "SET_RAM", ram: navMem });
        } else {
          // Last resort fallback — use 8 GB as conservative default
          dispatch({ type: "SET_RAM", ram: 8 });
        }
      });
  }, []);

  return (
    <NodeContext.Provider value={{ state, startNode, stopNode, clearLogs, updateConfig }}>
      {children}
    </NodeContext.Provider>
  );
}

export function useNode() {
  const ctx = useContext(NodeContext);
  if (!ctx) throw new Error("useNode must be inside NodeProvider");
  return ctx;
}
