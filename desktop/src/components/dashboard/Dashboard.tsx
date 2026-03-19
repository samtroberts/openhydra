import StatusCards from "./StatusCards";
import NodeToggle from "./NodeToggle";
import ModelSelector from "./ModelSelector";
import RamSlider from "./RamSlider";
import LogTerminal from "./LogTerminal";

export default function Dashboard() {
  return (
    <div className="h-full flex flex-col p-4 gap-4 overflow-auto">
      <StatusCards />

      <div className="grid grid-cols-[1fr_1fr_auto] gap-4 items-end">
        <ModelSelector />
        <RamSlider />
        <NodeToggle />
      </div>

      <LogTerminal />
    </div>
  );
}
