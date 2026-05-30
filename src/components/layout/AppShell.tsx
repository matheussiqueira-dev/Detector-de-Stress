import Link from "next/link";
import type { ReactNode } from "react";
import { Activity, BarChart3, Camera, GitBranch, ShieldCheck } from "lucide-react";
import { DeveloperCredits } from "@/components/credits/DeveloperCredits";

const navItems = [
  { href: "/", label: "Home", icon: Activity },
  { href: "/demo", label: "Demo", icon: Camera },
  { href: "/dashboard", label: "Dashboard", icon: BarChart3 },
  { href: "/about", label: "Privacidade", icon: ShieldCheck },
];

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="fixed inset-0 -z-10 bg-[radial-gradient(circle_at_top_left,rgba(34,211,238,0.13),transparent_32%),radial-gradient(circle_at_bottom_right,rgba(16,185,129,0.12),transparent_30%)]" />
      <div className="fixed inset-0 -z-10 tech-grid opacity-45" />
      <header className="sticky top-0 z-50 border-b border-white/10 bg-black/72 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-3 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between gap-3">
            <Link href="/" className="flex min-w-0 items-center gap-3" aria-label="AI Face Behavior Dashboard">
              <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-cyan-300/30 bg-cyan-300/10 text-cyan-100">
                <Activity className="h-4 w-4" aria-hidden />
              </span>
              <span className="min-w-0">
                <span className="block truncate text-sm font-semibold tracking-normal text-white">
                  AI Face Behavior Dashboard
                </span>
                <span className="block truncate text-xs text-zinc-500">Computer vision portfolio app</span>
              </span>
            </Link>
            <div className="hidden items-center gap-3 md:flex">
              <DeveloperCredits />
              <a
                href="https://github.com/matheussiqueira-dev/Detector-de-Stress"
                target="_blank"
                rel="noreferrer"
                className="grid h-9 w-9 place-items-center rounded-lg border border-white/10 bg-white/[0.04] text-zinc-300 transition hover:border-white/25 hover:text-white"
                aria-label="Abrir repositorio no GitHub"
                title="GitHub"
              >
                <GitBranch className="h-4 w-4" aria-hidden />
              </a>
            </div>
          </div>
          <nav className="flex gap-2 overflow-x-auto pb-1 md:pb-0" aria-label="Navegacao principal">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="inline-flex h-9 shrink-0 items-center gap-2 rounded-lg border border-white/10 bg-white/[0.035] px-3 text-sm text-zinc-300 transition hover:border-cyan-300/30 hover:bg-cyan-300/10 hover:text-white"
              >
                <item.icon className="h-4 w-4" aria-hidden />
                {item.label}
              </Link>
            ))}
          </nav>
          <DeveloperCredits compact className="justify-center md:hidden" />
        </div>
      </header>
      <main>{children}</main>
      <footer className="border-t border-white/10 bg-black/55">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-6 text-sm text-zinc-500 sm:px-6 md:flex-row md:items-center md:justify-between lg:px-8">
          <p>Demonstracao tecnica local-first. Nenhum frame da camera e enviado para servidor.</p>
          <DeveloperCredits />
        </div>
      </footer>
    </div>
  );
}
