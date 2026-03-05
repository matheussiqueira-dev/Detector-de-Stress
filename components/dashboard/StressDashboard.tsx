"use client";

import { useEffect, useMemo, useState } from "react";

import { EncomPanel } from "@/components/ui/EncomPanel";
import { TronButton } from "@/components/ui/TronButton";
import { TronCard } from "@/components/ui/TronCard";

import styles from "./StressDashboard.module.css";

type ScorePayload = {
  score: number;
  trend: number;
  ts: number;
};

const DEFAULT_DEMO_SNAPSHOT: ScorePayload = {
  score: 0.52,
  trend: 0.012,
  ts: Date.now() / 1000,
};

function resolveApiUrl() {
  const explicitUrl = process.env.NEXT_PUBLIC_STRESSCAM_API_URL;
  if (explicitUrl) {
    return explicitUrl;
  }

  if (typeof window !== "undefined" && window.location.hostname === "localhost") {
    return "http://localhost:8000";
  }

  return "";
}

function clampScore(score: number) {
  return Math.max(0, Math.min(1, score));
}

function formatTrend(trend: number) {
  return `${trend >= 0 ? "+" : ""}${trend.toFixed(3)}`;
}

function scoreLabel(score: number) {
  if (score >= 0.75) {
    return "Critical Load";
  }
  if (score >= 0.5) {
    return "Elevated State";
  }
  if (score >= 0.3) {
    return "Managed Load";
  }
  return "Stable Baseline";
}

export function StressDashboard() {
  const [apiUrl, setApiUrl] = useState<string>("");
  const [snapshot, setSnapshot] = useState<ScorePayload | null>(null);
  const [status, setStatus] = useState<"connecting" | "online" | "offline" | "demo">("connecting");
  const [updatedAt, setUpdatedAt] = useState<string>("--");

  useEffect(() => {
    const resolvedUrl = resolveApiUrl();
    setApiUrl(resolvedUrl);

    if (!resolvedUrl) {
      setSnapshot(DEFAULT_DEMO_SNAPSHOT);
      setStatus("demo");
      setUpdatedAt("Demo mode");
      return;
    }

    let cancelled = false;

    const pullScore = async () => {
      try {
        const response = await fetch(`${resolvedUrl}/score`, { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = (await response.json()) as ScorePayload;
        if (cancelled) {
          return;
        }

        setSnapshot(data);
        setStatus("online");
        setUpdatedAt(new Date(data.ts * 1000).toLocaleTimeString("pt-BR"));
      } catch {
        if (!cancelled) {
          setStatus("offline");
        }
      }
    };

    pullScore();
    const intervalId = window.setInterval(pullScore, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  const score = clampScore(snapshot?.score ?? DEFAULT_DEMO_SNAPSHOT.score);
  const trend = snapshot?.trend ?? 0;
  const percent = Math.round(score * 100);

  const ringStyle = useMemo(
    () => ({
      background: `conic-gradient(var(--encom-neon) ${percent}%, rgba(0, 229, 255, 0.1) ${percent}% 100%)`,
    }),
    [percent],
  );

  const signalBars = Array.from({ length: 18 }, (_, index) => {
    const active = index <= Math.round(score * 17);
    return (
      <span
        key={index}
        aria-hidden="true"
        className={active ? styles.signalBarActive : styles.signalBar}
      />
    );
  });

  return (
    <div className={styles.page}>
      <section className={styles.hero}>
        <div className={styles.heroCopy}>
          <span className={`${styles.kicker} encom-label`}>ENCOM Neural Monitor</span>
          <h1 className={`${styles.title} encom-heading`}>Stress Detector Interface</h1>
          <p className={styles.lead}>
            Uma camada visual nova em Next.js para o pipeline existente, com telemetria em tempo
            real, painéis holográficos e estrutura pronta para deploy no Vercel.
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

      <section className={styles.cards}>
        <TronCard
          eyebrow="Telemetry"
          title="Core Score"
          value={`${percent}%`}
          detail="Leitura contínua do endpoint REST exposto pelo backend Python, normalizada para um HUD enxuto."
          thumbnail={<div className={styles.glyphCircle} />}
        />
        <TronCard
          eyebrow="Status"
          title="Node Health"
          value={status.toUpperCase()}
          detail="Reconexão automática a cada 2 segundos. Sem endpoint configurado em produção, o painel entra em modo de demonstração."
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
          detail="Ajustado para leitura rápida em desktop, tablet e mobile, sem dependências pesadas."
          thumbnail={<div className={styles.waveform}>{signalBars}</div>}
        />
      </section>

      <section className={styles.lowerGrid}>
        <EncomPanel eyebrow="Control Grid" title="Deployment Notes">
          <ul className={styles.list}>
            <li>App Router com componentes de servidor e clientes separados.</li>
            <li>Fontes Orbitron, Rajdhani e Exo 2 carregadas por `next/font`.</li>
            <li>Tokens globais em CSS variables para manter escalabilidade do tema ENCOM.</li>
            <li>Consumo do backend existente via `NEXT_PUBLIC_STRESSCAM_API_URL`.</li>
          </ul>
        </EncomPanel>

        <EncomPanel eyebrow="Diagnostics" title="System Guidance">
          <div className={styles.guidance}>
            <p>
              Inicie o backend Python normalmente e exponha `GET /score`. Em produção, aponte a
              variável pública para a URL do serviço que estiver emitindo a telemetria.
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
