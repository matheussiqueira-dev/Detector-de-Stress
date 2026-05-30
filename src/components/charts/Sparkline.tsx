"use client";

import { useEffect, useRef, useState } from "react";
import { Line, LineChart } from "recharts";

interface SparklineProps {
  data: number[];
  color?: string;
}

export function Sparkline({ data, color = "#67e8f9" }: SparklineProps) {
  const { ref, size } = useChartSize();
  const chartData = data.map((value, index) => ({ index, value }));

  return (
    <div ref={ref} className="h-12 w-full">
      {size.width > 0 && size.height > 0 ? (
        <LineChart data={chartData} width={size.width} height={size.height}>
          <Line type="monotone" dataKey="value" stroke={color} strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      ) : (
        <div className="h-full w-full rounded bg-white/[0.03]" />
      )}
    </div>
  );
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
