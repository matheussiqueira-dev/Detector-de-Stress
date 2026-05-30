"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  Gauge,
  Pause,
  Play,
  RefreshCw,
  ScanFace,
  ShieldCheck,
  Square,
  Waves,
} from "lucide-react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { EVENT_LABELS } from "@/types/events";
import type { BoundingBox, FaceLandmarkPoint, VisionModelStatus } from "@/types/vision";
import { useCamera } from "@/hooks/useCamera";
import { useFaceTracking } from "@/hooks/useFaceTracking";
import { useSessionMetrics } from "@/hooks/useSessionMetrics";
import { ANALYSIS_STATE_LABELS } from "@/lib/scoring";
import { trackAppEvent } from "@/lib/analytics";
import { formatPercent, round } from "@/lib/utils";
import { getEventSeverityClass } from "@/lib/events";
import { MetricCard } from "@/components/ui/MetricCard";
import { StatusBadge } from "@/components/ui/StatusBadge";

const modelStatusLabel: Record<VisionModelStatus, string> = {
  idle: "Aguardando",
  loading: "Carregando modelo",
  ready: "Modelo real ativo",
  fallback: "Fallback local",
  error: "Erro no modelo",
};

export function RealtimeDemo() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const previousCameraStatus = useRef<string>("idle");
  const previousModelStatus = useRef<VisionModelStatus>("idle");
  const [paused, setPaused] = useState(false);
  const camera = useCamera();
  const { result: tracking } = useFaceTracking({
    enabled: camera.isActive,
    paused,
    videoRef,
  });
  const session = useSessionMetrics({
    tracking,
    running: camera.isActive,
    paused,
  });

  useEffect(() => {
    const video = videoRef.current;

    if (!video) {
      return;
    }

    video.srcObject = camera.stream;

    if (camera.stream) {
      void video.play();
    }
  }, [camera.stream]);

  useEffect(() => {
    if (camera.status === previousCameraStatus.current) {
      return;
    }

    if (camera.status === "active") {
      session.registerEvent("CAMERA_STARTED", "info", { localProcessing: true });
      trackAppEvent("camera_permission_granted");
    }

    if (camera.status === "blocked") {
      session.registerEvent("MODEL_ERROR", "warning", { reason: "camera_permission_denied" });
      trackAppEvent("camera_permission_denied");
    }

    if (previousCameraStatus.current === "active" && camera.status !== "active") {
      session.registerEvent("CAMERA_STOPPED", "info");
    }

    previousCameraStatus.current = camera.status;
  }, [camera.status, session]);

  useEffect(() => {
    if (tracking.modelStatus === previousModelStatus.current) {
      return;
    }

    if (tracking.modelStatus === "ready") {
      session.registerEvent("MODEL_READY", "info", { source: tracking.source });
    }

    if (tracking.modelStatus === "fallback" || tracking.modelStatus === "error") {
      session.registerEvent("MODEL_ERROR", "warning", { source: tracking.source });
    }

    previousModelStatus.current = tracking.modelStatus;
  }, [session, tracking.modelStatus, tracking.source]);

  useEffect(() => {
    drawOverlay(canvasRef.current, tracking.boundingBox, tracking.landmarks, tracking.confidence, tracking.source);
  }, [tracking.boundingBox, tracking.confidence, tracking.landmarks, tracking.source]);

  const chartData = useMemo(
    () =>
      session.samples.slice(-32).map((sample) => ({
        label: sample.label,
        attention: sample.attentionScore,
        confidence: Math.round(sample.detectionConfidence * 100),
      })),
    [session.samples],
  );

  const currentState = session.currentMetric?.state ?? (paused ? "paused" : camera.isActive ? "face-lost" : "paused");

  async function handleStart() {
    trackAppEvent("demo_started");
    await camera.startCamera();
    setPaused(false);
  }

  function handleStop() {
    camera.stopCamera();
    setPaused(false);
  }

  function handleReset() {
    session.resetSession();
    setPaused(false);
    trackAppEvent("session_reset");
  }

  return (
    <section className="mx-auto grid max-w-7xl gap-5 px-4 py-6 sm:px-6 lg:grid-cols-[minmax(0,1fr)_390px] lg:px-8">
      <div className="space-y-5">
        <div className="panel overflow-hidden">
          <div className="flex flex-col gap-3 border-b border-white/10 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs font-medium uppercase tracking-[0.18em] text-cyan-200/80">Live Vision Console</p>
              <h1 className="mt-1 text-2xl font-semibold tracking-normal text-white">Demo em tempo real</h1>
            </div>
            <StatusBadge state={currentState} pulse={camera.isActive && !paused} />
          </div>

          <div className="relative aspect-video bg-black">
            <video
              ref={videoRef}
              muted
              playsInline
              className="h-full w-full object-contain"
              aria-label="Feed local da webcam"
            />
            <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden />

            {!camera.isActive ? (
              <div className="absolute inset-0 grid place-items-center bg-black/78 p-6 text-center">
                <div className="max-w-md">
                  <div className="mx-auto grid h-14 w-14 place-items-center rounded-xl border border-cyan-300/30 bg-cyan-300/10 text-cyan-100">
                    <ScanFace className="h-7 w-7" aria-hidden />
                  </div>
                  <h2 className="mt-5 text-xl font-semibold text-white">Captura local de webcam</h2>
                  <p className="mt-2 text-sm leading-6 text-zinc-400">
                    A imagem e processada no navegador. Frames nao sao enviados para servidor e nenhuma biometria e armazenada.
                  </p>
                </div>
              </div>
            ) : null}

            {camera.isActive ? (
              <>
                <div className="absolute left-3 top-3 grid grid-cols-2 gap-2 text-xs sm:left-4 sm:top-4 sm:grid-cols-1">
                  <OverlayMetric label="Attention" value={`${session.currentMetric?.attentionScore ?? 0}`} />
                  <OverlayMetric label="Confidence" value={formatPercent(round(tracking.confidence * 100))} />
                  <OverlayMetric label="FPS" value={`${round(tracking.fps, 1)}`} />
                  <OverlayMetric label="Latency" value={`${round(tracking.latency)}ms`} />
                </div>

                <div className="absolute bottom-3 left-3 right-3 flex flex-wrap gap-2 text-xs sm:bottom-4 sm:left-4 sm:right-4">
                  <span className="rounded-full border border-white/10 bg-black/60 px-3 py-1 text-zinc-200 backdrop-blur">
                    {modelStatusLabel[tracking.modelStatus]}
                  </span>
                  <span className="rounded-full border border-white/10 bg-black/60 px-3 py-1 text-zinc-200 backdrop-blur">
                    Fonte: {tracking.source}
                  </span>
                  <span className="rounded-full border border-white/10 bg-black/60 px-3 py-1 text-zinc-200 backdrop-blur">
                    {tracking.faceDetected ? "Face detectada" : "Face nao detectada"}
                  </span>
                </div>
              </>
            ) : null}
          </div>

          <div className="grid gap-3 border-t border-white/10 p-4 sm:grid-cols-2 lg:grid-cols-4">
            <button className="control-button bg-cyan-300 text-black hover:bg-cyan-200" onClick={handleStart} type="button">
              <Camera className="h-4 w-4" aria-hidden />
              Iniciar camera
            </button>
            <button
              className="control-button border-white/10 bg-white/[0.04] text-white hover:bg-white/[0.08]"
              onClick={() => setPaused((value) => !value)}
              type="button"
              disabled={!camera.isActive}
            >
              {paused ? <Play className="h-4 w-4" aria-hidden /> : <Pause className="h-4 w-4" aria-hidden />}
              {paused ? "Retomar analise" : "Pausar analise"}
            </button>
            <button
              className="control-button border-white/10 bg-white/[0.04] text-white hover:bg-white/[0.08]"
              onClick={handleReset}
              type="button"
            >
              <RefreshCw className="h-4 w-4" aria-hidden />
              Reiniciar sessao
            </button>
            <button
              className="control-button border-red-300/20 bg-red-300/10 text-red-100 hover:bg-red-300/15"
              onClick={handleStop}
              type="button"
              disabled={!camera.isActive}
            >
              <Square className="h-4 w-4" aria-hidden />
              Parar camera
            </button>
          </div>
        </div>

        {camera.error ? (
          <div className="rounded-lg border border-red-300/25 bg-red-300/10 p-4 text-sm text-red-100">
            <div className="flex items-start gap-3">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
              <p>{camera.error}</p>
            </div>
          </div>
        ) : null}

        <div className="grid gap-4 md:grid-cols-3">
          <MetricCard label="Attention Score" value={session.currentMetric?.attentionScore ?? 0} detail="Metrica experimental" icon={<Gauge className="h-4 w-4" />} />
          <MetricCard label="Deteccao" value={`${session.averages.detectionRate}%`} detail="Tempo com face no quadro" tone="emerald" icon={<ScanFace className="h-4 w-4" />} />
          <MetricCard label="Movimento" value={formatPercent(round(tracking.movement * 100))} detail="Variacao facial estimada" tone="violet" icon={<Waves className="h-4 w-4" />} />
        </div>

        <div className="panel p-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold text-white">Evolucao temporal</h2>
              <p className="text-sm text-zinc-500">Attention Score e confianca de deteccao durante a sessao.</p>
            </div>
          </div>
          <div className="mt-4 h-56">
            {chartData.length > 1 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
                  <XAxis dataKey="label" stroke="#a1a1aa" tickLine={false} axisLine={false} minTickGap={24} />
                  <YAxis stroke="#a1a1aa" tickLine={false} axisLine={false} domain={[0, 100]} width={32} />
                  <Tooltip
                    contentStyle={{
                      border: "1px solid rgba(255,255,255,0.12)",
                      borderRadius: 8,
                      background: "rgba(9,9,11,0.94)",
                      color: "#f4f4f5",
                    }}
                    labelStyle={{ color: "#f4f4f5" }}
                  />
                  <Line type="monotone" dataKey="attention" stroke="#22d3ee" strokeWidth={2.2} dot={false} />
                  <Line type="monotone" dataKey="confidence" stroke="#34d399" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="grid h-full place-items-center rounded-lg border border-dashed border-white/10 text-sm text-zinc-500">
                Aguardando amostras da sessao.
              </div>
            )}
          </div>
        </div>
      </div>

      <aside className="space-y-5">
        <div className="panel p-4">
          <h2 className="text-lg font-semibold text-white">Status da analise</h2>
          <div className="mt-4 grid gap-3 text-sm">
            <StatusRow label="Estado" value={ANALYSIS_STATE_LABELS[currentState]} />
            <StatusRow label="Modelo" value={modelStatusLabel[tracking.modelStatus]} />
            <StatusRow label="Confianca" value={formatPercent(round(tracking.confidence * 100))} />
            <StatusRow label="FPS medio" value={`${session.averages.fps || round(tracking.fps, 1)}`} />
            <StatusRow label="Latencia media" value={`${session.averages.latency || round(tracking.latency)}ms`} />
            <StatusRow label="Estabilidade" value={`${session.averages.stability || round(tracking.stability * 100)}%`} />
          </div>
        </div>

        <div className="panel p-4">
          <div className="flex items-start gap-3">
            <ShieldCheck className="mt-1 h-5 w-5 shrink-0 text-emerald-200" aria-hidden />
            <p className="text-sm leading-6 text-zinc-300">
              Esta aplicacao e uma demonstracao tecnica de visao computacional e nao realiza diagnostico medico,
              biometrico ou psicologico.
            </p>
          </div>
        </div>

        <div className="panel p-4">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-lg font-semibold text-white">Event log</h2>
            <span className="rounded-full border border-white/10 px-2 py-1 text-xs text-zinc-400">{session.events.length}</span>
          </div>
          <div className="mt-4 max-h-[460px] space-y-3 overflow-y-auto pr-1">
            {session.events.length ? (
              session.events.map((event) => (
                <article key={event.id} className={`rounded-lg border p-3 ${getEventSeverityClass(event.severity)}`}>
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-xs font-semibold">{EVENT_LABELS[event.type]}</p>
                    <time className="font-mono text-[11px] opacity-75">
                      {new Intl.DateTimeFormat("pt-BR", {
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                      }).format(new Date(event.timestamp))}
                    </time>
                  </div>
                  <p className="mt-2 text-xs leading-5 opacity-85">{event.message}</p>
                </article>
              ))
            ) : (
              <div className="rounded-lg border border-dashed border-white/10 p-5 text-center text-sm text-zinc-500">
                Eventos aparecerao aqui durante a captura.
              </div>
            )}
          </div>
        </div>
      </aside>
    </section>
  );
}

function OverlayMetric({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="w-28 rounded-lg border border-cyan-300/20 bg-black/58 px-2 py-1.5 text-cyan-50 shadow-[0_0_24px_rgba(34,211,238,0.12)] backdrop-blur sm:w-32 sm:px-3 sm:py-2">
      <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-cyan-200/80">{label}</p>
      <p className="mt-1 font-mono text-base font-semibold sm:text-lg">{value}</p>
    </div>
  );
}

function StatusRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2">
      <span className="text-zinc-500">{label}</span>
      <span className="text-right font-medium text-zinc-100">{value}</span>
    </div>
  );
}

function drawOverlay(
  canvas: HTMLCanvasElement | null,
  box: BoundingBox | null,
  landmarks: FaceLandmarkPoint[],
  confidence: number,
  source: string,
) {
  if (!canvas) {
    return;
  }

  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));

  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);

  if (!box) {
    drawScanner(ctx, rect.width, rect.height);
    return;
  }

  const x = box.x * rect.width;
  const y = box.y * rect.height;
  const width = box.width * rect.width;
  const height = box.height * rect.height;

  ctx.save();
  ctx.strokeStyle = source === "visual-fallback" ? "rgba(251,191,36,0.95)" : "rgba(34,211,238,0.95)";
  ctx.lineWidth = 2;
  ctx.shadowColor = source === "visual-fallback" ? "rgba(251,191,36,0.35)" : "rgba(34,211,238,0.35)";
  ctx.shadowBlur = 14;
  ctx.strokeRect(x, y, width, height);

  const corner = Math.min(width, height) * 0.18;
  ctx.lineWidth = 4;
  drawCorner(ctx, x, y, corner, "tl");
  drawCorner(ctx, x + width, y, corner, "tr");
  drawCorner(ctx, x, y + height, corner, "bl");
  drawCorner(ctx, x + width, y + height, corner, "br");

  ctx.shadowBlur = 0;
  ctx.fillStyle = "rgba(34,211,238,0.85)";
  landmarks.forEach((point) => {
    ctx.beginPath();
    ctx.arc(point.x * rect.width, point.y * rect.height, 1.7, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = "rgba(0,0,0,0.72)";
  ctx.fillRect(x, Math.max(0, y - 24), 170, 24);
  ctx.fillStyle = "#ecfeff";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.fillText(`face ${Math.round(confidence * 100)}% | ${source}`, x + 8, Math.max(16, y - 7));
  ctx.restore();
}

function drawCorner(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, corner: "tl" | "tr" | "bl" | "br") {
  ctx.beginPath();
  if (corner === "tl") {
    ctx.moveTo(x + size, y);
    ctx.lineTo(x, y);
    ctx.lineTo(x, y + size);
  }
  if (corner === "tr") {
    ctx.moveTo(x - size, y);
    ctx.lineTo(x, y);
    ctx.lineTo(x, y + size);
  }
  if (corner === "bl") {
    ctx.moveTo(x, y - size);
    ctx.lineTo(x, y);
    ctx.lineTo(x + size, y);
  }
  if (corner === "br") {
    ctx.moveTo(x - size, y);
    ctx.lineTo(x, y);
    ctx.lineTo(x, y - size);
  }
  ctx.stroke();
}

function drawScanner(ctx: CanvasRenderingContext2D, width: number, height: number) {
  ctx.save();
  ctx.strokeStyle = "rgba(239,68,68,0.45)";
  ctx.lineWidth = 1;
  for (let y = 0; y < height; y += 18) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(239,68,68,0.85)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.fillText("FACE_NOT_DETECTED", 18, 28);
  ctx.restore();
}
