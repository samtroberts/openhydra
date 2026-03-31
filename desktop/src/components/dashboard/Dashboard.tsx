import StatusCards from "./StatusCards";
import NodeToggle from "./NodeToggle";
import ModelSelector from "./ModelSelector";
import RamSlider from "./RamSlider";
import ModeToggle from "./ModeToggle";
import LogTerminal from "./LogTerminal";

export default function Dashboard() {
  return (
    <div className="h-full flex flex-col p-5 gap-5 overflow-auto">
      {/* Top row: status + mode */}
      <div className="flex items-start justify-between gap-4">
        <StatusCards />
        <ModeToggle />
      </div>

      {/* Controls row */}
      <div className="flex items-end gap-3">
        <div className="flex-1 min-w-0">
          <ModelSelector />
        </div>
        <div className="flex-1 min-w-0">
          <RamSlider />
        </div>
        <NodeToggle />
      </div>

      {/* Logs */}
      <LogTerminal />
    </div>
  );
}
