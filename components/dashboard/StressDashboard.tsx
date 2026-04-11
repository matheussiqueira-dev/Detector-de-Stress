/**
 * StressDashboard — Painel principal de monitoramento de stress em tempo real.
 *
 * Conecta-se ao backend Python via WebSocket (com fallback para polling REST),
 * exibe telemetria ao vivo, histórico gráfico dos últimos 5 minutos,
 * sistema de alertas por threshold e exportação da sessão em CSV.
 *
 * @author Matheus Siqueira <https://www.matheussiqueira.dev/>
 */
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AlertBanner, type Alert, type AlertSeverity } from "@/components/ui/AlertBanner";
import { EncomPanel } from "@/components/ui/EncomPanel";
import { StressChart, type HistoryPoint } from "@/components/ui/StressChart";
import { TronButton } from "@/components/ui/TronButton";
import { TronCard } from "@/components/ui/TronCard";

import styles from "./StressDashboard.module.css";

// ── Tipos ──────────────────────────────────────────────────────────────────

type ScorePayload = {
  score: number;
  trend: number;
  ts: number;
};

type ConnectionStatus = "connecting" | "online" | "offline" | "demo";

// ── Constantes ─────────────────────────────────────────────────────────────

const DEMO_SNAPSHOT: ScorePayload = { score: 0.52, trend: 0.012, ts: Date.now() / 1000 };
const POLL_INTERVAL_MS = 2000;
const HISTORY_WINDOW_SEC = 300; // 5 minutos
const WS_RECONNECT_DELAY_MS = 3000;

/** Limiares para emissão de alertas (espelham Config.alert_threshold_*). */
const ALERT_HIGH = 0.75;
const ALERT_MEDIUM = 0.50;

/** Cooldown entre alertas do mesmo tipo (ms). */
const ALERT_COOLDOWN_MS = 30_000;

// ── Utilitários ─────────────────────────────────────────────────────────────

function resolveApiUrl(): string {
  const explicitUrl = process.env.NEXT_PUBLIC_STRESSCAM_API_URL;
  if (explicitUrl) return explicitUrl;
  if (typeof window !== "undefined" && window.location.hostname === "localhost") {
    return "http://localhost:8000";
  }
  return "";
}

function resolveWsUrl(apiUrl: string): string {
  const explicitWs = process.env.NEXT_PUBLIC_STRESSCAM_WS_URL;
  if (explicitWs) return explicitWs;
  if (apiUrl.startsWith("http://")) return apiUrl.replace("http://", "ws://").replace(/:(\d+)$/, ":8765");
  if (apiUrl.startsWith("https://")) return apiUrl.replace("https://", "wss://").replace(/:(\d+)$/, ":8765");
  return "";
}

function clamp(v: number, min = 0, max = 1): number {
  return Math.max(min, Math.min(max, v));
}

function formatTrend(trend: number): string {
  return `${trend >= 0 ? "+" : ""}${trend.toFixed(3)}`;
}

function scoreLabel(score: number): string {
  if (score >= ALERT_HIGH) return "Critical Load";
  if (score >= ALERT_MEDIUM) return "Elevated State";
  if (score >= 0.3) return "Managed Load";
  return "Stable Baseline";
}

function generateAlertId(): string {
  return `alert-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}

// ── Hook: dados de stress ──────────────────────────────────────────────────

function useStressData(apiUrl: string, wsUrl: string) {
  const [snapshot, setSnapshot] = useState<ScorePayload | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const applySnapshot = useCallback((data: ScorePayload) => {
    setSnapshot(data);
    setStatus("online");
  }, []);

  // ── WebSocket ────────────────────────────────────────────────────────
  const connectWs = useCallback(() => {
    if (!wsUrl) return false;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => setStatus("online");

      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data) as ScorePayload;
          applySnapshot(data);
        } catch {
          /* malformed frame — ignore */
        }
      };

      ws.onerror = () => setStatus("offline");

      ws.onclose = () => {
        setStatus("offline");
        reconnectTimerRef.current = setTimeout(connectWs, WS_RECONNECT_DELAY_MS);
      };

      return true;
    } catch {
      return false;
    }
  }, [wsUrl, applySnapshot]);

  // ── REST polling fallback ────────────────────────────────────────────
  const startPolling = useCallback(() => {
    if (!apiUrl) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const res = await fetch(`${apiUrl}/score`, { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as ScorePayload;
        if (!cancelled) applySnapshot(data);
      } catch {
        if (!cancelled) setStatus("offline");
      }
    };

    poll();
    pollingTimerRef.current = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
    };
  }, [apiUrl, applySnapshot]);

  useEffect(() => {
    if (!apiUrl) {
      setSnapshot(DEMO_SNAPSHOT);
      setStatus("demo");
      return;
    }

    const wsConnected = connectWs();
    if (!wsConnected) {
      startPolling();
    }

    return () => {
      wsRef.current?.close();
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (pollingTimerRef.current) clearInterval(pollingTimerRef.current);
    };
  }, [apiUrl, connectWs, startPolling]);

  return { snapshot, status };
}

// ── Hook: histórico ────────────────────────────────────────────────────────

function useStressHistory(snapshot: ScorePayload | null): HistoryPoint[] {
  const [history, setHistory] = useState<HistoryPoint[]>([]);

  useEffect(() => {
    if (!snapshot) return;
    const now = snapshot.ts;
    setHistory((prev) => {
      const pruned = prev.filter((p) => now - p.ts <= HISTORY_WINDOW_SEC);
      return [...pruned, { ts: snapshot.ts, score: snapshot.score, trend: snapshot.trend }];
    });
  }, [snapshot]);

  return history;
}

// ── Hook: alertas por threshold ────────────────────────────────────────────

function useStressAlerts(score: number): { alerts: Alert[]; dismiss: (id: string) => void } {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const lastAlertTs = useRef<Record<AlertSeverity, number>>({ critical: 0, warning: 0, info: 0 });

  useEffect(() => {
    const now = Date.now();

    const shouldEmit = (severity: AlertSeverity): boolean => {
      return now - lastAlertTs.current[severity] > ALERT_COOLDOWN_MS;
    };

    const emit = (severity: AlertSeverity, message: string) => {
      lastAlertTs.current[severity] = now;
      setAlerts((prev) => [
        ...prev,
        { id: generateAlertId(), severity, message, autoDismissMs: 8000 },
      ]);
    };

    if (score >= ALERT_HIGH && shouldEmit("critical")) {
      emit("critical", `Stress crítico detectado: ${Math.round(score * 100)}% — faça uma pausa.`);
    } else if (score >= ALERT_MEDIUM && shouldEmit("warning")) {
      emit("warning", `Nível de stress elevado: ${Math.round(score * 100)}% — respire fundo.`);
    }
  }, [score]);

  const dismiss = useCallback((id: string) => {
    setAlerts((prev) => prev.filter((a) => a.id !== id));
  }, []);

  return { alerts, dismiss };
}

// ── Exportação CSV ─────────────────────────────────────────────────────────

function exportHistoryAsCsv(history: HistoryPoint[]): void {
  if (history.length === 0) return;

  const header = "ts,datetime,score,trend\n";
  const rows = history.map((p) => {
    const dt = new Date(p.ts * 1000).toISOString();
    return `${p.ts.toFixed(3)},${dt},${p.score.toFixed(4)},${p.trend.toFixed(4)}`;
  });
  const csv = header + rows.join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = `stress_session_${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

// ── Componente principal ───────────────────────────────────────────────────

export function StressDashboard() {
  const [apiUrl] = useState(() => resolveApiUrl());
  const wsUrl = useMemo(() => resolveWsUrl(apiUrl), [apiUrl]);

  const { snapshot, status } = useStressData(apiUrl, wsUrl);
  const history = useStressHistory(snapshot);

  const score = clamp(snapshot?.score ?? DEMO_SNAPSHOT.score);
  const trend = snapshot?.trend ?? 0;
  const percent = Math.round(score * 100);

  const { alerts, dismiss } = useStressAlerts(score);

  const updatedAt = snapshot
    ? new Date(snapshot.ts * 1000).toLocaleTimeString("pt-BR")
    : "--";

  const ringStyle = useMemo(
    () => ({
      background: `conic-gradient(var(--encom-neon) ${percent}%, rgba(0, 229, 255, 0.1) ${percent}% 100%)`,
    }),
    [percent],
  );

  const signalBars = Array.from({ length: 18 }, (_, i) => {
    const active = i <= Math.round(score * 17);
    return (
      <span
        key={i}
        aria-hidden="true"
        className={active ? styles.signalBarActive : styles.signalBar}
      />
    );
  });

  return (
    <div className={styles.page}>
      {/* Sistema de Alertas */}
      {alerts.length > 0 && (
        <section aria-label="Alertas">
          <AlertBanner alerts={alerts} onDismiss={dismiss} />
        </section>
      )}

      {/* Hero */}
      <section className={styles.hero}>
        <div className={styles.heroCopy}>
          <span className={`${styles.kicker} encom-label`}>ENCOM Neural Monitor</span>
          <h1 className={`${styles.title} encom-heading`}>Stress Detector Interface</h1>
          <p className={styles.lead}>
            Pipeline de visão computacional em Python com frontend Next.js em tempo real.
            Estima stress fisiológico via EAR, tensão facial e área pupilar.
          </p>
          <div className={styles.actions}>
            <TronButton
              disabled={!apiUrl}
              onClick={() => window.open(`${apiUrl}/score`, "_blank", "noopener,noreferrer")}
            >
              Open API Feed
            </TronButton>
            <TronButton
              onClick={() =>
                window.open("https://www.matheussiqueira.dev/", "_blank", "noopener,noreferrer")
              }
            >
              Contact Architect
            </TronButton>
            <TronButton
              disabled={history.length === 0}
              onClick={() => exportHistoryAsCsv(history)}
              title="Exportar histórico da sessão em CSV"
            >
              Export CSV
            </TronButton>
          </div>
        </div>

        <EncomPanel className={styles.heroPanel} eyebrow="Realtime Uplink" title="Signal Core">
          <div className={styles.coreGrid}>
            <div className={styles.ring} style={ringStyle}>
              <div className={styles.ringInner}>
                <span className={`${styles.ringLabel} encom-label`}>Load</span>
                <strong className={styles.ringValue}>{percent}%</strong>
              </div>
            </div>
            <div className={styles.coreStats}>
              <div>
                <span className={`${styles.statLabel} encom-label`}>State</span>
                <strong className={styles.statValue}>{scoreLabel(score)}</strong>
              </div>
              <div>
                <span className={`${styles.statLabel} encom-label`}>Trend</span>
                <strong className={styles.statValue}>{formatTrend(trend)}</strong>
              </div>
              <div>
                <span className={`${styles.statLabel} encom-label`}>Last Sync</span>
                <strong className={styles.statValue}>{updatedAt}</strong>
              </div>
            </div>
          </div>
        </EncomPanel>
      </section>

      {/* Cards de telemetria */}
      <section className={styles.cards}>
        <TronCard
          eyebrow="Telemetry"
          title="Core Score"
          value={`${percent}%`}
          detail="Leitura contínua do endpoint REST / WebSocket exposto pelo backend Python."
          thumbnail={<div className={styles.glyphCircle} />}
        />
        <TronCard
          eyebrow="Status"
          title="Node Health"
          value={status.toUpperCase()}
          detail="WebSocket com reconexão automática. Sem endpoint configurado: modo demonstração."
          thumbnail={
            <div
              className={
                status === "online"
                  ? styles.nodeOnline
                  : status === "demo"
                    ? styles.nodeDemo
                    : styles.nodeOffline
              }
            />
          }
        />
        <TronCard
          eyebrow="Trend"
          title="Signal Drift"
          value={formatTrend(trend)}
          detail="Delta de score entre leituras consecutivas, indicando direção do stress."
          thumbnail={<div className={styles.waveform}>{signalBars}</div>}
        />
      </section>

      {/* Gráfico de histórico */}
      <section>
        <EncomPanel eyebrow="Session Telemetry" title={`Stress History (last ${HISTORY_WINDOW_SEC / 60} min)`}>
          <StressChart data={history} windowSec={HISTORY_WINDOW_SEC} />
        </EncomPanel>
      </section>

      {/* Grid inferior */}
      <section className={styles.lowerGrid}>
        <EncomPanel eyebrow="Control Grid" title="Deployment Notes">
          <ul className={styles.list}>
            <li>WebSocket em tempo real com fallback automático para polling REST (2 s).</li>
            <li>Sistema de alertas com cooldown de 30 s por severidade (warning / critical).</li>
            <li>Histórico de 5 min renderizado em SVG puro, sem dependências pesadas.</li>
            <li>Exportação da sessão em CSV via botão "Export CSV" ou endpoint <code>/history</code>.</li>
          </ul>
        </EncomPanel>

        <EncomPanel eyebrow="Diagnostics" title="System Guidance">
          <div className={styles.guidance}>
            <p>
              Inicie o backend Python e exponha <code>GET /score</code> e <code>ws://…:8765</code>.
              Configure <code>NEXT_PUBLIC_STRESSCAM_API_URL</code> e{" "}
              <code>NEXT_PUBLIC_STRESSCAM_WS_URL</code> para ambiente de produção.
            </p>
            <div className={styles.signalRail}>
              <span className={`${styles.railLabel} encom-label`}>Scan</span>
              <div className={styles.railTrack}>{signalBars}</div>
            </div>
          </div>
        </EncomPanel>
      </section>
    </div>
  );
}
