export interface ModelInfo {
  id: string;
  name: string;
  hfId: string;
  requiredPeers: number;
  minVramGb: number;
  tier: string;
  quantization: string;
  description: string;
}

export const MODEL_CATALOG: ModelInfo[] = [
  {
    id: "openhydra-qwen3.5-0.8b",
    name: "Qwen 3.5 0.8B",
    hfId: "Qwen/Qwen3.5-0.8B",
    requiredPeers: 1,
    minVramGb: 2,
    tier: "Nano",
    quantization: "fp32",
    description: "Base swarm. Runs on anything. The universal starting point.",
  },
  {
    id: "openhydra-qwen3.5-2b",
    name: "Qwen 3.5 2B",
    hfId: "Qwen/Qwen3.5-2B",
    requiredPeers: 1,
    minVramGb: 5,
    tier: "Basic",
    quantization: "fp32",
    description: "Compact multilingual reasoning. Single peer.",
  },
  {
    id: "openhydra-qwen3.5-4b",
    name: "Qwen 3.5 4B",
    hfId: "Qwen/Qwen3.5-4B",
    requiredPeers: 1,
    minVramGb: 9,
    tier: "Standard",
    quantization: "int4",
    description: "Strong reasoning on a single peer. NF4 quantized.",
  },
  {
    id: "openhydra-qwen3.5-9b",
    name: "Qwen 3.5 9B",
    hfId: "Qwen/Qwen3.5-9B",
    requiredPeers: 2,
    minVramGb: 18,
    tier: "Advanced",
    quantization: "int8",
    description: "High-quality reasoning. 2-peer pipeline.",
  },
  {
    id: "openhydra-qwen3.5-27b",
    name: "Qwen 3.5 27B",
    hfId: "Qwen/Qwen3.5-27B",
    requiredPeers: 4,
    minVramGb: 16,
    tier: "Frontier",
    quantization: "int4",
    description: "Frontier-class. 4-peer pipeline. Maximum rewards.",
  },
];

export function recommendModel(ramGb: number): ModelInfo {
  if (ramGb >= 18) return MODEL_CATALOG[3]; // 9B
  if (ramGb >= 9) return MODEL_CATALOG[2];  // 4B
  if (ramGb >= 5) return MODEL_CATALOG[1];  // 2B
  return MODEL_CATALOG[0];                   // 0.8B
}

export function rewardMultiplier(model: ModelInfo): number {
  switch (model.tier) {
    case "Frontier": return 4;
    case "Advanced": return 3;
    case "Standard": return 2;
    case "Basic": return 1.5;
    default: return 1;
  }
}
