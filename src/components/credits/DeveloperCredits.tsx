import { ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";

interface DeveloperCreditsProps {
  compact?: boolean;
  className?: string;
}

export function DeveloperCredits({ compact = false, className }: DeveloperCreditsProps) {
  return (
    <a
      href="https://www.matheussiqueira.dev"
      target="_blank"
      rel="noreferrer"
      className={cn(
        "group inline-flex max-w-full items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3 py-2 text-xs font-medium text-zinc-300 transition hover:border-cyan-300/40 hover:bg-cyan-300/10 hover:text-white",
        compact && "w-full flex-col gap-1 justify-center text-center",
        className,
      )}
      aria-label="Desenvolvido por Matheus Siqueira - www.matheussiqueira.dev"
    >
      <span className="min-w-0">Desenvolvido por Matheus Siqueira</span>
      <span className={cn("min-w-0 break-all text-cyan-200/80", compact ? "inline" : "hidden sm:inline")}>
        www.matheussiqueira.dev
      </span>
      <ExternalLink className="h-3.5 w-3.5 text-cyan-200/80 transition group-hover:translate-x-0.5" aria-hidden />
    </a>
  );
}
