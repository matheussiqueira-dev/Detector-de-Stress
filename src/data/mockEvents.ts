import type { BehaviorEvent } from "@/types/events";
import type { EventDistribution } from "@/types/metrics";

const base = new Date("2026-05-30T12:00:00-03:00").getTime();

export const mockEvents: BehaviorEvent[] = [
  {
    id: "evt_demo_001",
    timestamp: new Date(base + 2_000).toISOString(),
    type: "CAMERA_STARTED",
    severity: "info",
    message: "Captura local iniciada com permissao do navegador.",
    metadata: { source: "mock", resolution: "1280x720" },
  },
  {
    id: "evt_demo_002",
    timestamp: new Date(base + 7_500).toISOString(),
    type: "MODEL_READY",
    severity: "info",
    message: "Modelo de face tracking carregado no navegador.",
    metadata: { model: "MediaPipe FaceLandmarker", backend: "WASM" },
  },
  {
    id: "evt_demo_003",
    timestamp: new Date(base + 43_000).toISOString(),
    type: "LOW_CONFIDENCE",
    severity: "warning",
    message: "Confianca caiu por baixa iluminacao temporaria.",
    metadata: { confidence: 0.58, attentionScore: 62 },
  },
  {
    id: "evt_demo_004",
    timestamp: new Date(base + 89_000).toISOString(),
    type: "HIGH_MOVEMENT",
    severity: "info",
    message: "Movimento facial elevado detectado por deslocamento rapido.",
    metadata: { movement: 0.81, stability: 0.61 },
  },
  {
    id: "evt_demo_005",
    timestamp: new Date(base + 145_000).toISOString(),
    type: "ATTENTION_DROP",
    severity: "warning",
    message: "Score experimental de atencao ficou abaixo do limite.",
    metadata: { attentionScore: 54, yaw: 31 },
  },
  {
    id: "evt_demo_006",
    timestamp: new Date(base + 221_000).toISOString(),
    type: "FACE_LOST",
    severity: "warning",
    message: "Face saiu parcialmente do quadro por alguns segundos.",
    metadata: { durationMs: 1900, source: "mediapipe" },
  },
  {
    id: "evt_demo_007",
    timestamp: new Date(base + 248_000).toISOString(),
    type: "FACE_DETECTED",
    severity: "info",
    message: "Rastreamento facial restabelecido.",
    metadata: { confidence: 0.83 },
  },
  {
    id: "evt_demo_008",
    timestamp: new Date(base + 315_000).toISOString(),
    type: "LOW_CONFIDENCE",
    severity: "warning",
    message: "Confianca oscilou abaixo do ideal.",
    metadata: { confidence: 0.6, fps: 21 },
  },
];

export const eventDistribution: EventDistribution[] = [
  { type: "FACE_DETECTED", label: "Face detectada", count: 12 },
  { type: "FACE_LOST", label: "Face perdida", count: 3 },
  { type: "LOW_CONFIDENCE", label: "Baixa confianca", count: 5 },
  { type: "ATTENTION_DROP", label: "Queda de atencao", count: 4 },
  { type: "HIGH_MOVEMENT", label: "Movimento elevado", count: 6 },
  { type: "CAMERA_STARTED", label: "Camera iniciada", count: 1 },
  { type: "CAMERA_STOPPED", label: "Camera parada", count: 1 },
  { type: "MODEL_READY", label: "Modelo pronto", count: 1 },
  { type: "MODEL_ERROR", label: "Erro de modelo", count: 0 },
];
