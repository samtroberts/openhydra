import { useState } from "react";
import { Toaster } from "react-hot-toast";
import { NodeProvider, useNode } from "./contexts/NodeContext";
import { ChatProvider } from "./contexts/ChatContext";
import Header from "./components/layout/Header";
import Dashboard from "./components/dashboard/Dashboard";
import SwarmView from "./components/swarm/SwarmView";
import ChatView from "./components/chat/ChatView";
import BootstrapScreen from "./components/bootstrap/BootstrapScreen";

type View = "dashboard" | "swarm" | "chat";

function AppInner() {
  const [view, setView] = useState<View>("dashboard");
  const { state, startNode } = useNode();

  if (state.status === "bootstrapping") {
    return <BootstrapScreen onComplete={() => startNode()} />;
  }

  return (
    <>
      <div className="h-screen flex flex-col bg-[#0a0a0a] text-gray-100">
        <Header activeView={view} onViewChange={setView} />
        <main className="flex-1 overflow-hidden">
          {view === "dashboard" && <Dashboard />}
          {view === "swarm" && <SwarmView />}
          {view === "chat" && <ChatView />}
        </main>
      </div>
      <Toaster
        position="bottom-right"
        toastOptions={{
          style: { background: "#1a1a1a", color: "#e5e5e5", border: "1px solid #333" },
        }}
      />
    </>
  );
}

export default function App() {
  return (
    <NodeProvider>
      <ChatProvider>
        <AppInner />
      </ChatProvider>
    </NodeProvider>
  );
}
