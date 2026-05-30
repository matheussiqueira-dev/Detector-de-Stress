import Link from "next/link";
import { ArrowRight, BarChart3, BrainCircuit, Camera, LockKeyhole, Radar, ScanFace, ShieldCheck } from "lucide-react";
import { DeveloperCredits } from "@/components/credits/DeveloperCredits";

const features = [
  {
    title: "Webcam em tempo real",
    description: "Captura local com getUserMedia, estados de permissao e erro, sem backend obrigatorio.",
    icon: Camera,
  },
  {
    title: "Face tracking no navegador",
    description: "MediaPipe Tasks Vision com fallback nativo/visual para manter a demo testavel.",
    icon: ScanFace,
  },
  {
    title: "Indicadores comportamentais",
    description: "Attention Score experimental, movimento, confianca, olhos, boca, FPS e latencia.",
    icon: Radar,
  },
  {
    title: "Dashboard inteligente",
    description: "KPIs, series temporais, distribuicao de estados, eventos e insights por regras.",
    icon: BarChart3,
  },
];

export default function Home() {
  return (
    <div>
      <section className="relative overflow-hidden border-b border-white/10">
        <div className="mx-auto grid min-h-[calc(100vh-138px)] max-w-7xl min-w-0 gap-10 overflow-hidden px-4 py-12 sm:px-6 lg:grid-cols-[0.92fr_1.08fr] lg:px-8 lg:py-16">
          <div className="flex min-w-0 max-w-[calc(100vw-2rem)] flex-col justify-center sm:max-w-full">
            <p className="text-xs font-medium uppercase tracking-[0.2em] text-cyan-200/80">Portfolio computer vision app</p>
            <h1 className="mt-5 max-w-[21rem] text-4xl font-semibold leading-tight tracking-normal text-white sm:max-w-3xl sm:text-6xl lg:text-7xl">
              AI Face Behavior Dashboard
            </h1>
            <p className="mt-6 max-w-[21rem] text-base leading-8 text-zinc-300 sm:max-w-2xl sm:text-lg">
              Aplicacao web profissional para demonstrar captura de webcam, deteccao facial, acompanhamento do rosto,
              metricas em tempo real, alertas visuais e um dashboard analitico com insights acionaveis.
            </p>
            <div className="mt-8 flex flex-col gap-3 sm:flex-row">
              <Link className="control-button bg-cyan-300 text-black hover:bg-cyan-200" href="/demo">
                Abrir Demonstracao
                <ArrowRight className="h-4 w-4" aria-hidden />
              </Link>
              <Link className="control-button border-white/10 bg-white/[0.04] text-white hover:bg-white/[0.08]" href="/dashboard">
                Ver Dashboard
                <BarChart3 className="h-4 w-4" aria-hidden />
              </Link>
            </div>
            <div className="mt-8 flex flex-wrap gap-3 text-sm text-zinc-400">
              <span className="rounded-full border border-emerald-300/20 bg-emerald-300/10 px-3 py-1 text-emerald-100">
                Local-first
              </span>
              <span className="rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-cyan-100">
                Next.js + TypeScript
              </span>
              <span className="rounded-full border border-violet-300/20 bg-violet-300/10 px-3 py-1 text-violet-100">
                Vercel-ready
              </span>
            </div>
          </div>

          <div className="flex min-w-0 items-center">
            <div className="panel min-w-0 max-w-full overflow-hidden">
              <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-red-400" />
                  <span className="h-2.5 w-2.5 rounded-full bg-amber-300" />
                  <span className="h-2.5 w-2.5 rounded-full bg-emerald-300" />
                </div>
                <p className="font-mono text-xs text-zinc-500">face-behavior/live</p>
              </div>
              <div className="grid gap-0 lg:grid-cols-[1.2fr_0.8fr]">
                <div className="relative aspect-video bg-black">
                  <div className="absolute inset-0 bg-[linear-gradient(120deg,rgba(34,211,238,0.10),transparent_42%),linear-gradient(0deg,rgba(255,255,255,0.04),transparent)]" />
                  <div className="absolute left-[33%] top-[18%] h-[46%] w-[31%] rounded-[8px] border-2 border-cyan-200 shadow-[0_0_34px_rgba(34,211,238,0.32)]" />
                  <div className="absolute left-[36%] top-[26%] grid h-[28%] w-[25%] grid-cols-5 gap-2 opacity-80">
                    {Array.from({ length: 20 }, (_, index) => (
                      <span key={index} className="h-1.5 w-1.5 rounded-full bg-cyan-100/80" />
                    ))}
                  </div>
                  <div className="absolute left-4 top-4 space-y-2">
                    {["ATTENTION 82", "CONF 94%", "FPS 29.8", "LAT 34MS"].map((item) => (
                      <div key={item} className="rounded-md border border-cyan-300/20 bg-black/62 px-3 py-2 font-mono text-xs text-cyan-50">
                        {item}
                      </div>
                    ))}
                  </div>
                  <div className="absolute bottom-4 left-4 right-4 h-16 rounded-lg border border-white/10 bg-black/58 p-3">
                    <div className="flex h-full items-end gap-1">
                      {[35, 49, 42, 68, 72, 66, 78, 84, 76, 88, 82, 91].map((value, index) => (
                        <span
                          key={index}
                          className="flex-1 rounded-t bg-cyan-300/70"
                          style={{ height: `${value}%` }}
                        />
                      ))}
                    </div>
                  </div>
                </div>
                <div className="grid gap-3 border-t border-white/10 p-4 lg:border-l lg:border-t-0">
                  {[
                    ["Face detectada", "online", "emerald"],
                    ["Atencao normal", "82/100", "cyan"],
                    ["Movimento", "baixo", "violet"],
                    ["Alertas", "2 recentes", "amber"],
                  ].map(([label, value, tone]) => (
                    <div key={label} className="rounded-lg border border-white/10 bg-white/[0.035] p-3">
                      <p className="text-xs text-zinc-500">{label}</p>
                      <p className={`mt-1 text-lg font-semibold ${tone === "emerald" ? "text-emerald-200" : tone === "amber" ? "text-amber-200" : tone === "violet" ? "text-violet-200" : "text-cyan-200"}`}>
                        {value}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {features.map((feature) => (
            <article className="panel p-5" key={feature.title}>
              <div className="grid h-10 w-10 place-items-center rounded-lg border border-cyan-300/25 bg-cyan-300/10 text-cyan-100">
                <feature.icon className="h-5 w-5" aria-hidden />
              </div>
              <h2 className="mt-4 text-lg font-semibold text-white">{feature.title}</h2>
              <p className="mt-2 text-sm leading-6 text-zinc-400">{feature.description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="border-y border-white/10 bg-white/[0.025]">
        <div className="mx-auto grid max-w-7xl gap-5 px-4 py-12 sm:px-6 lg:grid-cols-3 lg:px-8">
          <div className="panel p-5">
            <BrainCircuit className="h-6 w-6 text-cyan-200" aria-hidden />
            <h2 className="mt-4 text-xl font-semibold text-white">Inteligencia por regras</h2>
            <p className="mt-3 text-sm leading-6 text-zinc-400">
              Os insights correlacionam score, confianca, FPS, perda facial e movimento para gerar recomendacoes simples.
            </p>
          </div>
          <div className="panel p-5">
            <LockKeyhole className="h-6 w-6 text-emerald-200" aria-hidden />
            <h2 className="mt-4 text-xl font-semibold text-white">Privacidade por design</h2>
            <p className="mt-3 text-sm leading-6 text-zinc-400">
              O processamento acontece no navegador sempre que possivel. A aplicacao nao identifica pessoas nem armazena biometria.
            </p>
          </div>
          <div className="panel p-5">
            <ShieldCheck className="h-6 w-6 text-amber-200" aria-hidden />
            <h2 className="mt-4 text-xl font-semibold text-white">Aviso tecnico</h2>
            <p className="mt-3 text-sm leading-6 text-zinc-400">
              Esta aplicacao e uma demonstracao tecnica de visao computacional e nao realiza diagnostico medico,
              biometrico ou psicologico.
            </p>
          </div>
        </div>
      </section>

      <section className="mx-auto flex max-w-7xl flex-col gap-5 px-4 py-10 sm:px-6 lg:px-8">
        <div className="panel flex flex-col gap-4 p-5 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-xl font-semibold text-white">Pronto para demonstracao publica</h2>
            <p className="mt-2 text-sm text-zinc-400">Rotas, PWA, SEO, analytics e documentacao preparados para deploy na Vercel.</p>
          </div>
          <DeveloperCredits />
        </div>
      </section>
    </div>
  );
}
