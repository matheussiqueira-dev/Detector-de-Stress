import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  label: string;
  value: string | number;
  detail?: string;
  icon?: ReactNode;
  tone?: "cyan" | "emerald" | "amber" | "red" | "violet";
  children?: ReactNode;
}

const toneClasses = {
  cyan: "text-cyan-100 bg-cyan-300/10 border-cyan-300/25",
  emerald: "text-emerald-100 bg-emerald-300/10 border-emerald-300/25",
  amber: "text-amber-100 bg-amber-300/10 border-amber-300/25",
  red: "text-red-100 bg-red-300/10 border-red-300/25",
  violet: "text-violet-100 bg-violet-300/10 border-violet-300/25",
};

export function MetricCard({ label, value, detail, icon, tone = "cyan", children }: MetricCardProps) {
  return (
    <section className="panel min-w-0 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-xs font-medium uppercase tracking-[0.18em] text-zinc-500">{label}</p>
          <p className="mt-2 text-3xl font-semibold tracking-normal text-white">{value}</p>
          {detail ? <p className="mt-1 text-sm text-zinc-400">{detail}</p> : null}
        </div>
        {icon ? (
          <div className={cn("grid h-9 w-9 shrink-0 place-items-center rounded-lg border", toneClasses[tone])}>
            {icon}
          </div>
        ) : null}
      </div>
      {children ? <div className="mt-4">{children}</div> : null}
    </section>
  );
}
