import type { AnalysisState } from "@/types/metrics";
import type { AttentionScoreInput } from "@/types/vision";
import { clamp, round } from "./utils";

/**
 * Calculates an experimental attention score from local browser vision signals.
 * The score is for portfolio demonstration only and must not be interpreted as
 * a medical, biometric, psychological, or identity signal.
 */
export function calculateAttentionScore(input: AttentionScoreInput): number {
  if (!input.faceDetected) {
    return 0;
  }

  const confidence = clamp(input.detectionConfidence, 0, 1) * 34;
  const stability = clamp(input.stability, 0, 1) * 24;
  const movementPenalty = clamp(input.movement, 0, 1) * 14;
  const mouthPenalty = clamp(input.mouthOpen, 0, 1) * 8;
  const eyesPenalty = clamp(input.eyesClosed, 0, 1) * 10;
  const yawPenalty = clamp(Math.abs(input.yaw) / 45, 0, 1) * 8;
  const lostPenalty = clamp(input.lostFrames ?? 0, 0, 30) * 0.6;

  return round(
    clamp(52 + confidence + stability - movementPenalty - mouthPenalty - eyesPenalty - yawPenalty - lostPenalty),
  );
}

export function getAnalysisState(input: AttentionScoreInput, paused = false): AnalysisState {
  if (paused) {
    return "paused";
  }

  if (!input.faceDetected) {
    return "face-lost";
  }

  if (input.detectionConfidence < 0.62) {
    return "low-confidence";
  }

  if (input.movement > 0.7) {
    return "high-movement";
  }

  if (calculateAttentionScore(input) < 58 || Math.abs(input.yaw) > 28) {
    return "distracted";
  }

  return "normal";
}

export const ANALYSIS_STATE_LABELS: Record<AnalysisState, string> = {
  normal: "Atencao normal",
  distracted: "Possivel distracao",
  "low-confidence": "Baixa confianca",
  "high-movement": "Movimento facial elevado",
  "face-lost": "Face nao detectada",
  paused: "Analise pausada",
};
