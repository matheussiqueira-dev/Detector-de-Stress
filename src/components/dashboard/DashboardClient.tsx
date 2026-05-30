"use client";

import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  BrainCircuit,
  Filter,
  Gauge,
  MonitorCheck,
  ScanFace,
  TableProperties,
  Zap,
} from "lucide-react";
import type { BehaviorEvent, BehaviorEventType } from "@/types/events";
import { EVENT_LABELS } from "@/types/events";
import type { DashboardKpi, EventDistribution, MetricSample, SessionSummary, StateDistribution } from "@/types/metrics";
import { AttentionTimelineChart, ConfidenceLineChart, EventBarChart, StateDistributionChart } from "@/components/charts/AnalyticsCharts";
import { Sparkline } from "@/components/charts/Sparkline";
import { MetricCard } from "@/components/ui/MetricCard";
import { generateInsights } from "@/lib/insights";
import { trackAppEvent } from "@/lib/analytics";
import { getEventSeverityClass } from "@/lib/events";
import { formatPercent } from "@/lib/utils";

interface DashboardClientProps {
  metrics: MetricSample[];
  events: BehaviorEvent[];
  session: SessionSummary;
  kpis: DashboardKpi[];
  eventDistribution: EventDistribution[];
  stateDistribution: StateDistribution[];
}

const sparklineColors: Record<DashboardKpi["tone"], string> = {
  cyan: "#22d3ee",
  emerald: "#34d399",
  amber: "#fbbf24",
  red: "#f87171",
  violet: "#a78bfa",
};

export function DashboardClient({
  metrics,
  events,
  session,
  kpis,
  eventDistribution,
  stateDistribution,
}: DashboardClientProps) {
  const [filter, setFilter] = useState<BehaviorEventType | "ALL">("ALL");
  const insights = useMemo(() => generateInsights(metrics, events, session), [events, metrics, session]);
  const filteredEvents = useMemo(
    () => (filter === "ALL" ? events : events.filter((event) => event.type === filter)),
    [events, filter],
  );

  useEffect(() => {
    trackAppEvent("dashboard_opened");
  }, []);

  return (
    <section className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.18em] text-emerald-200/80">Intelligent Analytics</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-normal text-white sm:text-4xl">Dashboard inteligente</h1>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-zinc-400">
            KPIs, eventos, graficos e insights gerados por regras sobre uma sessao demonstrativa identificada como mock data.
          </p>
        </div>
        <div className="rounded-lg border border-amber-300/20 bg-amber-300/10 px-4 py-3 text-sm text-amber-100">
          Dados exibidos: {session.source === "mock" ? "mock data profissional" : "sessao real"}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {kpis.map((kpi, index) => (
          <MetricCard
            key={kpi.label}
            label={kpi.label}
            value={kpi.value}
            detail={kpi.delta}
            tone={kpi.tone}
            icon={getKpiIcon(index)}
          >
            <Sparkline data={kpi.sparkline} color={sparklineColors[kpi.tone]} />
          </MetricCard>
        ))}
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(0,1.25fr)_minmax(320px,0.75fr)]">
        <section className="panel p-4">
          <div className="mb-4 flex items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold text-white">Attention Score</h2>
              <p className="text-sm text-zinc-500">Evolucao principal da metrica experimental.</p>
            </div>
            <span className="rounded-full border border-cyan-300/25 bg-cyan-300/10 px-3 py-1 text-xs text-cyan-100">
              Media {session.averageAttention}
            </span>
          </div>
          <AttentionTimelineChart data={metrics} />
        </section>

        <section className="panel p-4">
          <h2 className="text-lg font-semibold text-white">Insights acionaveis</h2>
          <div className="mt-4 space-y-3">
            {insights.map((insight) => (
              <article
                key={insight.id}
                className={`rounded-lg border p-3 ${
                  insight.severity === "positive"
                    ? "border-emerald-300/20 bg-emerald-300/10"
                    : insight.severity === "warning"
                      ? "border-amber-300/20 bg-amber-300/10"
                      : "border-white/10 bg-white/[0.035]"
                }`}
              >
                <div className="flex items-center gap-2 text-sm font-semibold text-white">
                  <BrainCircuit className="h-4 w-4 text-cyan-200" aria-hidden />
                  {insight.title}
                </div>
                <p className="mt-2 text-sm leading-6 text-zinc-400">{insight.description}</p>
              </article>
            ))}
          </div>
        </section>
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-2">
        <section className="panel p-4">
          <h2 className="text-lg font-semibold text-white">Confianca e FPS</h2>
          <p className="text-sm text-zinc-500">Comparacao entre confianca de deteccao e performance estimada.</p>
          <div className="mt-4">
            <ConfidenceLineChart data={metrics} />
          </div>
        </section>

        <section className="panel p-4">
          <h2 className="text-lg font-semibold text-white">Eventos por tipo</h2>
          <p className="text-sm text-zinc-500">Volume de sinais detectados durante a sessao.</p>
          <div className="mt-4">
            <EventBarChart data={eventDistribution} />
          </div>
        </section>
      </div>

      <div className="mt-5 grid gap-5 xl:grid-cols-[380px_minmax(0,1fr)]">
        <section className="panel p-4">
          <h2 className="text-lg font-semibold text-white">Distribuicao de estados</h2>
          <StateDistributionChart data={stateDistribution} />
          <div className="mt-3 grid gap-2">
            {stateDistribution.filter((item) => item.value > 0).map((item) => (
              <div key={item.state} className="flex items-center justify-between rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 text-sm">
                <span className="text-zinc-400">{item.label}</span>
                <span className="font-medium text-white">{formatPercent(item.value)}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="panel p-4">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-white">Tabela de eventos</h2>
              <p className="text-sm text-zinc-500">Historico estruturado com severidade, mensagem e metadata.</p>
            </div>
            <div className="flex items-center gap-2 overflow-x-auto">
              <Filter className="h-4 w-4 shrink-0 text-zinc-500" aria-hidden />
              <button className={filterButtonClass(filter === "ALL")} onClick={() => setFilter("ALL")} type="button">
                Todos
              </button>
              {eventDistribution
                .filter((item) => item.count > 0)
                .map((item) => (
                  <button
                    className={filterButtonClass(filter === item.type)}
                    key={item.type}
                    onClick={() => setFilter(item.type)}
                    type="button"
                  >
                    {EVENT_LABELS[item.type]}
                  </button>
                ))}
            </div>
          </div>

          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[720px] border-separate border-spacing-y-2 text-left text-sm">
              <thead className="text-xs uppercase tracking-[0.14em] text-zinc-500">
                <tr>
                  <th className="px-3 py-2">Horario</th>
                  <th className="px-3 py-2">Tipo</th>
                  <th className="px-3 py-2">Severidade</th>
                  <th className="px-3 py-2">Mensagem</th>
                  <th className="px-3 py-2">Metadata</th>
                </tr>
              </thead>
              <tbody>
                {filteredEvents.map((event) => (
                  <tr key={event.id} className="rounded-lg bg-white/[0.035]">
                    <td className="rounded-l-lg border-y border-l border-white/10 px-3 py-3 font-mono text-xs text-zinc-400">
                      {new Intl.DateTimeFormat("pt-BR", {
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                      }).format(new Date(event.timestamp))}
                    </td>
                    <td className="border-y border-white/10 px-3 py-3 text-zinc-100">{EVENT_LABELS[event.type]}</td>
                    <td className="border-y border-white/10 px-3 py-3">
                      <span className={`rounded-full border px-2 py-1 text-xs ${getEventSeverityClass(event.severity)}`}>
                        {event.severity}
                      </span>
                    </td>
                    <td className="border-y border-white/10 px-3 py-3 text-zinc-300">{event.message}</td>
                    <td className="rounded-r-lg border-y border-r border-white/10 px-3 py-3 font-mono text-xs text-zinc-500">
                      {JSON.stringify(event.metadata)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </section>
  );
}

function getKpiIcon(index: number) {
  const icons = [
    <Gauge className="h-4 w-4" aria-hidden key="gauge" />,
    <ScanFace className="h-4 w-4" aria-hidden key="scan" />,
    <Zap className="h-4 w-4" aria-hidden key="zap" />,
    <AlertTriangle className="h-4 w-4" aria-hidden key="alert" />,
    <MonitorCheck className="h-4 w-4" aria-hidden key="monitor" />,
    <TableProperties className="h-4 w-4" aria-hidden key="table" />,
  ];

  return icons[index % icons.length];
}

function filterButtonClass(active: boolean) {
  return `h-8 shrink-0 rounded-lg border px-3 text-xs transition ${
    active
      ? "border-cyan-300/30 bg-cyan-300/10 text-cyan-100"
      : "border-white/10 bg-white/[0.03] text-zinc-400 hover:text-white"
  }`;
}
