import type { BehaviorEvent, BehaviorEventType, EventMetadata, EventSeverity } from "@/types/events";
import type { MetricSample } from "@/types/metrics";
import { createId } from "./utils";

const DEFAULT_MESSAGES: Record<BehaviorEventType, string> = {
  FACE_DETECTED: "Rastreamento facial restabelecido.",
  FACE_LOST: "Nenhuma face consistente no quadro.",
  LOW_CONFIDENCE: "Confianca abaixo do limite recomendado.",
  ATTENTION_DROP: "Queda no score experimental de atencao.",
  HIGH_MOVEMENT: "Movimento facial acima do padrao da sessao.",
  CAMERA_STARTED: "Captura local de camera iniciada.",
  CAMERA_STOPPED: "Captura local de camera interrompida.",
  MODEL_READY: "Modelo de visao computacional pronto no navegador.",
  MODEL_ERROR: "Modelo real indisponivel; fallback local ativado.",
};

export function createBehaviorEvent(
  type: BehaviorEventType,
  severity: EventSeverity = "info",
  metadata: EventMetadata = {},
  message = DEFAULT_MESSAGES[type],
): BehaviorEvent {
  return {
    id: createId("evt"),
    timestamp: new Date().toISOString(),
    type,
    severity,
    message,
    metadata,
  };
}

export function deriveEventFromMetric(sample: MetricSample): BehaviorEvent | null {
  if (!sample.faceDetected) {
    return createBehaviorEvent("FACE_LOST", "warning", {
      attentionScore: sample.attentionScore,
      source: sample.source,
    });
  }

  if (sample.detectionConfidence < 0.62) {
    return createBehaviorEvent("LOW_CONFIDENCE", "warning", {
      confidence: sample.detectionConfidence,
      source: sample.source,
    });
  }

  if (sample.attentionScore < 58) {
    return createBehaviorEvent("ATTENTION_DROP", "warning", {
      attentionScore: sample.attentionScore,
      yaw: sample.yaw,
    });
  }

  if (sample.movement > 0.7) {
    return createBehaviorEvent("HIGH_MOVEMENT", "info", {
      movement: sample.movement,
      stability: sample.stability,
    });
  }

  return null;
}

export function getEventSeverityClass(severity: EventSeverity) {
  const classes: Record<EventSeverity, string> = {
    info: "border-cyan-400/30 bg-cyan-400/10 text-cyan-100",
    warning: "border-amber-400/30 bg-amber-400/10 text-amber-100",
    critical: "border-red-400/30 bg-red-400/10 text-red-100",
  };

  return classes[severity];
}
