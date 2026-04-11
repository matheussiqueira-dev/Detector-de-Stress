/**
 * StressChart — Gráfico SVG de histórico de stress em tempo real.
 *
 * Renderiza uma linha animada com gradiente de cor baseado na intensidade
 * (verde → amarelo → vermelho), exibindo os últimos N segundos de dados.
 *
 * @author Matheus Siqueira <https://www.matheussiqueira.dev/>
 */
"use client";

import { useMemo } from "react";

import styles from "./StressChart.module.css";

export type HistoryPoint = {
  ts: number;    // timestamp UNIX (segundos)
  score: number; // score normalizado [0, 1]
  trend: number;
};

type StressChartProps = {
  /** Pontos do histórico a renderizar. */
  data: HistoryPoint[];
  /** Número de segundos exibidos na janela (eixo X). */
  windowSec?: number;
  /** Altura SVG em pixels. */
  height?: number;
};

/** Mapeia score → cor ENCOM (verde / ciano / vermelho). */
function scoreToColor(score: number): string {
  if (score < 0.35) return "#00c853";   // verde
  if (score < 0.70) return "#00e5ff";   // ciano
  return "#ff1744";                      // vermelho
}

/** Gera um id único para gradientes SVG. */
let _gradientCounter = 0;

/**
 * Componente de gráfico SVG para histórico de stress.
 */
export function StressChart({
  data,
  windowSec = 300,
  height = 120,
}: StressChartProps) {
  const gradientId = useMemo(() => `sg-${++_gradientCounter}`, []);
  const areaId = useMemo(() => `sa-${_gradientCounter}`, []);

  const chartData = useMemo(() => {
    if (data.length < 2) return null;

    const now = data[data.length - 1].ts;
    const visible = data.filter((p) => now - p.ts <= windowSec);
    if (visible.length < 2) return null;

    const tMin = visible[0].ts;
    const tMax = visible[visible.length - 1].ts;
    const tSpan = Math.max(tMax - tMin, 1);

    const W = 100; // coordenadas percentuais
    const H = 100;
    const PAD = 4;

    const toX = (t: number) => ((t - tMin) / tSpan) * (W - PAD * 2) + PAD;
    const toY = (s: number) => H - PAD - s * (H - PAD * 2);

    const points = visible.map((p) => ({ x: toX(p.ts), y: toY(p.score), score: p.score }));

    // Polyline para a linha de score
    const linePoints = points.map((p) => `${p.x},${p.y}`).join(" ");

    // Área preenchida (fecha no fundo)
    const firstX = points[0].x;
    const lastX = points[points.length - 1].x;
    const areaPoints =
      linePoints + ` ${lastX},${H - PAD} ${firstX},${H - PAD}`;

    // Score atual (último ponto) para a cor dominante
    const lastScore = visible[visible.length - 1].score;

    return { linePoints, areaPoints, lastScore, points, tMin, tMax, tSpan };
  }, [data, windowSec]);

  return (
    <div className={styles.wrapper} aria-label="Gráfico de histórico de stress">
      {/* Eixo Y labels */}
      <div className={styles.yLabels} aria-hidden="true">
        <span>100%</span>
        <span>75%</span>
        <span>50%</span>
        <span>25%</span>
        <span>0%</span>
      </div>

      <svg
        className={styles.svg}
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <defs>
          {/* Gradiente de cor dinâmico */}
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#ff1744" stopOpacity="0.9" />
            <stop offset="50%" stopColor="#00e5ff" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#00c853" stopOpacity="0.5" />
          </linearGradient>

          {/* Gradiente para área preenchida */}
          <linearGradient id={areaId} x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="0%"
              stopColor={chartData ? scoreToColor(chartData.lastScore) : "#00e5ff"}
              stopOpacity="0.18"
            />
            <stop offset="100%" stopColor="transparent" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Grade horizontal */}
        {[25, 50, 75].map((pct) => (
          <line
            key={pct}
            x1="4"
            y1={100 - 4 - pct}
            x2="96"
            y2={100 - 4 - pct}
            stroke="rgba(0,229,255,0.1)"
            strokeWidth="0.3"
          />
        ))}

        {/* Limiar médio (50%) */}
        <line
          x1="4" y1="50" x2="96" y2="50"
          stroke="rgba(255,196,0,0.3)"
          strokeWidth="0.4"
          strokeDasharray="2 2"
        />

        {/* Limiar crítico (75%) */}
        <line
          x1="4" y1="29" x2="96" y2="29"
          stroke="rgba(255,50,50,0.3)"
          strokeWidth="0.4"
          strokeDasharray="2 2"
        />

        {chartData ? (
          <>
            {/* Área preenchida */}
            <polygon
              points={chartData.areaPoints}
              fill={`url(#${areaId})`}
            />

            {/* Linha principal */}
            <polyline
              points={chartData.linePoints}
              fill="none"
              stroke={`url(#${gradientId})`}
              strokeWidth="1.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Ponto atual */}
            {chartData.points.length > 0 && (
              <circle
                cx={chartData.points[chartData.points.length - 1].x}
                cy={chartData.points[chartData.points.length - 1].y}
                r="1.8"
                fill={scoreToColor(chartData.lastScore)}
                filter="url(#glow)"
              />
            )}
          </>
        ) : (
          <text
            x="50"
            y="54"
            textAnchor="middle"
            fill="rgba(0,229,255,0.4)"
            fontSize="6"
            fontFamily="var(--font-ui, sans-serif)"
          >
            Aguardando dados...
          </text>
        )}
      </svg>

      {/* Eixo X labels */}
      <div className={styles.xLabels} aria-hidden="true">
        <span>−{windowSec}s</span>
        <span>agora</span>
      </div>
    </div>
  );
}
