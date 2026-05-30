import type { MetricSample, StateDistribution } from "@/types/metrics";
import { round } from "@/lib/utils";

const base = new Date("2026-05-30T12:00:00-03:00").getTime();

export const mockMetrics: MetricSample[] = Array.from({ length: 42 }, (_, index) => {
  const movementWave = Math.abs(Math.sin(index / 4));
  const confidence = 0.68 + Math.sin(index / 5) * 0.14 + (index % 11 === 0 ? -0.12 : 0);
  const attention = 72 + Math.cos(index / 6) * 12 - (index % 13 === 0 ? 18 : 0) - movementWave * 6;
  const state =
    confidence < 0.62
      ? "low-confidence"
      : attention < 58
        ? "distracted"
        : movementWave > 0.82
          ? "high-movement"
          : "normal";

  return {
    id: `metric_${index + 1}`,
    timestamp: new Date(base + index * 15_000).toISOString(),
    label: `${String(Math.floor((index * 15) / 60)).padStart(2, "0")}:${String((index * 15) % 60).padStart(2, "0")}`,
    attentionScore: round(Math.max(34, Math.min(94, attention))),
    detectionConfidence: round(Math.max(0.45, Math.min(0.96, confidence)), 2),
    faceDetected: index % 17 !== 0,
    movement: round(Math.min(0.95, movementWave * 0.76 + (index % 9 === 0 ? 0.18 : 0)), 2),
    fps: round(26 + Math.sin(index / 3) * 5 - (index % 16 === 0 ? 7 : 0), 1),
    latency: round(28 + Math.abs(Math.cos(index / 4)) * 22),
    stability: round(Math.max(0.42, 0.92 - movementWave * 0.28), 2),
    mouthOpen: round(index % 10 === 0 ? 0.68 : Math.max(0.06, Math.sin(index / 7) * 0.2 + 0.18), 2),
    eyesClosed: round(index % 14 === 0 ? 0.58 : Math.max(0.04, Math.cos(index / 6) * 0.12 + 0.13), 2),
    yaw: round(Math.sin(index / 5) * 24 + (index % 13 === 0 ? 14 : 0)),
    state,
    source: "mediapipe",
  };
});

export const stateDistribution: StateDistribution[] = [
  { state: "normal", label: "Atencao normal", value: 64 },
  { state: "distracted", label: "Possivel distracao", value: 14 },
  { state: "low-confidence", label: "Baixa confianca", value: 10 },
  { state: "high-movement", label: "Movimento elevado", value: 9 },
  { state: "face-lost", label: "Face nao detectada", value: 3 },
  { state: "paused", label: "Pausado", value: 0 },
];
