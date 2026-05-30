"use client";

import { useEffect, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EventDistribution, MetricSample, StateDistribution } from "@/types/metrics";

const grid = "rgba(255,255,255,0.08)";
const text = "#a1a1aa";
const stateColors = ["#34d399", "#fbbf24", "#fb923c", "#a78bfa", "#f87171", "#71717a"];

export function AttentionTimelineChart({ data }: { data: MetricSample[] }) {
  const { ref, size } = useChartSize();

  return (
    <div ref={ref} className="h-72 w-full">
      {size.width > 0 && size.height > 0 ? (
        <AreaChart data={data} width={size.width} height={size.height}>
          <defs>
            <linearGradient id="attentionFill" x1="0" x2="0" y1="0" y2="1">
              <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={grid} vertical={false} />
          <XAxis dataKey="label" stroke={text} tickLine={false} axisLine={false} minTickGap={28} />
          <YAxis stroke={text} tickLine={false} axisLine={false} domain={[0, 100]} width={32} />
          <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#f4f4f5" }} />
          <Area
            type="monotone"
            dataKey="attentionScore"
            name="Attention Score"
            stroke="#22d3ee"
            strokeWidth={2.4}
            fill="url(#attentionFill)"
          />
        </AreaChart>
      ) : (
        <ChartPlaceholder />
      )}
    </div>
  );
}

export function ConfidenceLineChart({ data }: { data: MetricSample[] }) {
  const { ref, size } = useChartSize();
  const normalized = data.map((sample) => ({
    ...sample,
    confidencePercent: Math.round(sample.detectionConfidence * 100),
  }));

  return (
    <div ref={ref} className="h-64 w-full">
      {size.width > 0 && size.height > 0 ? (
        <LineChart data={normalized} width={size.width} height={size.height}>
          <CartesianGrid stroke={grid} vertical={false} />
          <XAxis dataKey="label" stroke={text} tickLine={false} axisLine={false} minTickGap={28} />
          <YAxis stroke={text} tickLine={false} axisLine={false} domain={[0, 100]} width={32} />
          <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#f4f4f5" }} />
          <Line type="monotone" dataKey="confidencePercent" name="Confianca" stroke="#34d399" strokeWidth={2.2} dot={false} />
          <Line type="monotone" dataKey="fps" name="FPS" stroke="#a78bfa" strokeWidth={1.8} dot={false} />
        </LineChart>
      ) : (
        <ChartPlaceholder />
      )}
    </div>
  );
}

export function EventBarChart({ data }: { data: EventDistribution[] }) {
  const { ref, size } = useChartSize();

  return (
    <div ref={ref} className="h-64 w-full">
      {size.width > 0 && size.height > 0 ? (
        <BarChart data={data} width={size.width} height={size.height}>
          <CartesianGrid stroke={grid} vertical={false} />
          <XAxis dataKey="label" stroke={text} tickLine={false} axisLine={false} interval={0} tick={{ fontSize: 11 }} />
          <YAxis stroke={text} tickLine={false} axisLine={false} width={28} />
          <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#f4f4f5" }} />
          <Bar dataKey="count" name="Eventos" radius={[6, 6, 2, 2]} fill="#22d3ee" />
        </BarChart>
      ) : (
        <ChartPlaceholder />
      )}
    </div>
  );
}

export function StateDistributionChart({ data }: { data: StateDistribution[] }) {
  const { ref, size } = useChartSize();

  return (
    <div ref={ref} className="h-64 w-full">
      {size.width > 0 && size.height > 0 ? (
        <PieChart width={size.width} height={size.height}>
          <Pie data={data} dataKey="value" nameKey="label" innerRadius={54} outerRadius={86} paddingAngle={2}>
            {data.map((entry, index) => (
              <Cell key={entry.state} fill={stateColors[index % stateColors.length]} />
            ))}
          </Pie>
          <Tooltip contentStyle={tooltipStyle} labelStyle={{ color: "#f4f4f5" }} />
        </PieChart>
      ) : (
        <ChartPlaceholder />
      )}
    </div>
  );
}

const tooltipStyle = {
  border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: 8,
  background: "rgba(9,9,11,0.94)",
  color: "#f4f4f5",
};

function ChartPlaceholder() {
  return <div className="h-full w-full rounded-lg border border-white/10 bg-white/[0.03]" />;
}

function useChartSize() {
  const ref = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const node = ref.current;
    if (!node) {
      return;
    }

    const update = () => {
      const rect = node.getBoundingClientRect();
      setSize({
        width: Math.floor(rect.width),
        height: Math.floor(rect.height),
      });
    };
    const frame = window.requestAnimationFrame(update);
    const observer = new ResizeObserver(update);
    observer.observe(node);

    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, []);

  return { ref, size };
}
