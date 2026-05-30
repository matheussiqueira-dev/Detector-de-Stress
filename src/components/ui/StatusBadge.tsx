import type { AnalysisState } from "@/types/metrics";
import { ANALYSIS_STATE_LABELS } from "@/lib/scoring";
import { cn } from "@/lib/utils";

interface StatusBadgeProps {
  state: AnalysisState;
  pulse?: boolean;
}

const stateClasses: Record<AnalysisState, string> = {
  normal: "border-emerald-300/30 bg-emerald-300/10 text-emerald-100",
  distracted: "border-amber-300/30 bg-amber-300/10 text-amber-100",
  "low-confidence": "border-orange-300/30 bg-orange-300/10 text-orange-100",
  "high-movement": "border-violet-300/30 bg-violet-300/10 text-violet-100",
  "face-lost": "border-red-300/30 bg-red-300/10 text-red-100",
  paused: "border-zinc-300/20 bg-zinc-300/10 text-zinc-100",
};

export function StatusBadge({ state, pulse = false }: StatusBadgeProps) {
  return (
    <span className={cn("inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium", stateClasses[state])}>
      <span className={cn("h-2 w-2 rounded-full bg-current", pulse && "animate-pulse")} aria-hidden />
      {ANALYSIS_STATE_LABELS[state]}
    </span>
  );
}
