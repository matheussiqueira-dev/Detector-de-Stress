import type { DashboardKpi, SessionSummary } from "@/types/metrics";

export const mockSession: SessionSummary = {
  id: "session_portfolio_demo",
  title: "Sessao demonstrativa de portfolio",
  startedAt: "2026-05-30T12:00:00-03:00",
  durationMinutes: 10.5,
  averageAttention: 76,
  detectionRate: 91,
  averageFps: 27.8,
  alerts: 8,
  distractionEvents: 4,
  lowConfidenceEvents: 5,
  trackingStability: 84,
  source: "mock",
};

export const mockSessions: SessionSummary[] = [
  mockSession,
  {
    id: "session_lab_002",
    title: "Calibracao de iluminacao",
    startedAt: "2026-05-29T18:35:00-03:00",
    durationMinutes: 7.2,
    averageAttention: 81,
    detectionRate: 94,
    averageFps: 30.4,
    alerts: 4,
    distractionEvents: 1,
    lowConfidenceEvents: 2,
    trackingStability: 89,
    source: "mock",
  },
  {
    id: "session_lab_003",
    title: "Teste de baixa luz",
    startedAt: "2026-05-28T21:12:00-03:00",
    durationMinutes: 6.8,
    averageAttention: 66,
    detectionRate: 78,
    averageFps: 23.7,
    alerts: 11,
    distractionEvents: 5,
    lowConfidenceEvents: 7,
    trackingStability: 72,
    source: "mock",
  },
];

export const dashboardKpis: DashboardKpi[] = [
  {
    label: "Attention Score medio",
    value: "76",
    delta: "+8% vs. baseline",
    tone: "cyan",
    sparkline: [62, 68, 71, 75, 73, 79, 82, 76],
  },
  {
    label: "Tempo com face detectada",
    value: "91%",
    delta: "9m 34s rastreados",
    tone: "emerald",
    sparkline: [72, 74, 81, 86, 89, 88, 92, 91],
  },
  {
    label: "FPS medio",
    value: "27.8",
    delta: "latencia media 38ms",
    tone: "violet",
    sparkline: [22, 24, 28, 31, 30, 27, 29, 28],
  },
  {
    label: "Alertas recentes",
    value: "8",
    delta: "4 distracoes, 5 baixa confianca",
    tone: "amber",
    sparkline: [1, 2, 2, 3, 5, 4, 6, 8],
  },
  {
    label: "Estabilidade do tracking",
    value: "84%",
    delta: "movimento controlado",
    tone: "emerald",
    sparkline: [76, 79, 83, 81, 86, 88, 85, 84],
  },
  {
    label: "Sessoes simuladas",
    value: "3",
    delta: "mock data identificado",
    tone: "red",
    sparkline: [1, 1, 2, 2, 2, 3, 3, 3],
  },
];
