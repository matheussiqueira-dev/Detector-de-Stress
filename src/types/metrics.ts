import type { BehaviorEventType } from "./events";
import type { VisionSource } from "./vision";

export type AnalysisState =
  | "normal"
  | "distracted"
  | "low-confidence"
  | "high-movement"
  | "face-lost"
  | "paused";

export interface MetricSample {
  id: string;
  timestamp: string;
  label: string;
  attentionScore: number;
  detectionConfidence: number;
  faceDetected: boolean;
  movement: number;
  fps: number;
  latency: number;
  stability: number;
  mouthOpen: number;
  eyesClosed: number;
  yaw: number;
  state: AnalysisState;
  source: VisionSource;
}

export interface SessionSummary {
  id: string;
  title: string;
  startedAt: string;
  durationMinutes: number;
  averageAttention: number;
  detectionRate: number;
  averageFps: number;
  alerts: number;
  distractionEvents: number;
  lowConfidenceEvents: number;
  trackingStability: number;
  source: "real" | "mock";
}

export interface DashboardKpi {
  label: string;
  value: string;
  delta: string;
  tone: "cyan" | "emerald" | "amber" | "red" | "violet";
  sparkline: number[];
}

export interface EventDistribution {
  type: BehaviorEventType;
  label: string;
  count: number;
}

export interface StateDistribution {
  state: AnalysisState;
  label: string;
  value: number;
}
