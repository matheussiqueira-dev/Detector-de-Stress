import type { BehaviorEvent } from "@/types/events";
import type { MetricSample, SessionSummary } from "@/types/metrics";
import { average, round } from "./utils";

export interface Insight {
  id: string;
  title: string;
  description: string;
  severity: "positive" | "neutral" | "warning";
}

export function generateInsights(
  metrics: MetricSample[],
  events: BehaviorEvent[],
  session: SessionSummary,
): Insight[] {
  const attentionAverage = average(metrics.map((sample) => sample.attentionScore));
  const confidenceAverage = average(metrics.map((sample) => sample.detectionConfidence * 100));
  const movementAverage = average(metrics.map((sample) => sample.movement * 100));
  const lowConfidenceEvents = events.filter((event) => event.type === "LOW_CONFIDENCE").length;
  const attentionDrops = events.filter((event) => event.type === "ATTENTION_DROP").length;
  const fpsAverage = average(metrics.map((sample) => sample.fps));

  const insights: Insight[] = [];

  if (attentionAverage >= 76 && session.trackingStability >= 82) {
    insights.push({
      id: "stable-attention",
      title: "Atencao estavel",
      description: "A atencao permaneceu estavel na maior parte da sessao simulada.",
      severity: "positive",
    });
  }

  if (lowConfidenceEvents > 4 || confidenceAverage < 72) {
    insights.push({
      id: "low-confidence",
      title: "Confianca abaixo do ideal",
      description: "Reposicione a camera ou melhore a iluminacao para reduzir perdas de rastreamento.",
      severity: "warning",
    });
  }

  if (movementAverage > 46 || events.some((event) => event.type === "HIGH_MOVEMENT")) {
    insights.push({
      id: "movement-oscillation",
      title: "Oscilacao de rastreamento",
      description: "Foram detectadas variacoes frequentes na posicao facial ao longo da sessao.",
      severity: "neutral",
    });
  }

  if (attentionDrops >= 3) {
    insights.push({
      id: "attention-drops",
      title: "Aumento de possivel distracao",
      description: "Houve aumento de eventos de possivel distracao nos ultimos minutos analisados.",
      severity: "warning",
    });
  }

  if (fpsAverage > 0 && fpsAverage < 20) {
    insights.push({
      id: "low-fps",
      title: "FPS reduzido",
      description: "A taxa media ficou baixa. Reduza efeitos visuais ou feche abas pesadas.",
      severity: "warning",
    });
  }

  insights.push({
    id: "session-summary",
    title: "Resumo operacional",
    description: `Sessao ${session.source === "mock" ? "demonstrativa" : "real"} com score medio ${round(attentionAverage)} e confianca media ${round(confidenceAverage)}%.`,
    severity: "neutral",
  });

  return insights.slice(0, 5);
}
