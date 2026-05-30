export type EventSeverity = "info" | "warning" | "critical";

export type BehaviorEventType =
  | "FACE_DETECTED"
  | "FACE_LOST"
  | "LOW_CONFIDENCE"
  | "ATTENTION_DROP"
  | "HIGH_MOVEMENT"
  | "CAMERA_STARTED"
  | "CAMERA_STOPPED"
  | "MODEL_READY"
  | "MODEL_ERROR";

export type EventMetadata = Record<string, string | number | boolean | null>;

export interface BehaviorEvent {
  id: string;
  timestamp: string;
  type: BehaviorEventType;
  severity: EventSeverity;
  message: string;
  metadata: EventMetadata;
}

export const EVENT_LABELS: Record<BehaviorEventType, string> = {
  FACE_DETECTED: "Face detectada",
  FACE_LOST: "Face nao detectada",
  LOW_CONFIDENCE: "Baixa confianca",
  ATTENTION_DROP: "Possivel distracao",
  HIGH_MOVEMENT: "Movimento elevado",
  CAMERA_STARTED: "Camera iniciada",
  CAMERA_STOPPED: "Camera parada",
  MODEL_READY: "Modelo pronto",
  MODEL_ERROR: "Falha no modelo",
};
