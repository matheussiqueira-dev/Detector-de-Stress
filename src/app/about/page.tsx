import type { Metadata } from "next";
import { LockKeyhole, ServerOff, ShieldAlert, ShieldCheck } from "lucide-react";

export const metadata: Metadata = {
  title: "Privacidade e limites",
  description: "Como a demo processa webcam localmente, quais dados nao sao armazenados e quais sao as limitacoes tecnicas.",
  alternates: {
    canonical: "/about",
  },
};

const items = [
  {
    title: "Processamento local",
    description: "A camera e acessada via Web APIs e os frames sao processados no navegador sempre que o modelo esta disponivel.",
    icon: LockKeyhole,
  },
  {
    title: "Sem identificacao",
    description: "O app nao reconhece identidade, nao compara faces e nao armazena biometria facial.",
    icon: ShieldCheck,
  },
  {
    title: "Sem diagnostico",
    description: "Os scores sao experimentais e nao representam avaliacao medica, psicologica, clinica ou biometrica.",
    icon: ShieldAlert,
  },
  {
    title: "Sem backend obrigatorio",
    description: "O dashboard usa mock data profissional quando nao ha persistencia real de sessao.",
    icon: ServerOff,
  },
];

export default function AboutPage() {
  return (
    <section className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
      <div className="max-w-3xl">
        <p className="text-xs font-medium uppercase tracking-[0.18em] text-cyan-200/80">Privacidade e seguranca</p>
        <h1 className="mt-3 text-4xl font-semibold tracking-normal text-white">Limites claros para uma demo tecnica</h1>
        <p className="mt-4 text-base leading-8 text-zinc-400">
          Esta aplicacao foi criada para portfolio e demonstracao publica de engenharia frontend, visao computacional no
          navegador e dashboards analiticos. Ela nao deve ser usada para decisao clinica, psicologica, biometrica ou de
          identidade.
        </p>
      </div>

      <div className="mt-8 grid gap-4 md:grid-cols-2">
        {items.map((item) => (
          <article className="panel p-5" key={item.title}>
            <div className="grid h-10 w-10 place-items-center rounded-lg border border-cyan-300/25 bg-cyan-300/10 text-cyan-100">
              <item.icon className="h-5 w-5" aria-hidden />
            </div>
            <h2 className="mt-4 text-xl font-semibold text-white">{item.title}</h2>
            <p className="mt-3 text-sm leading-6 text-zinc-400">{item.description}</p>
          </article>
        ))}
      </div>

      <div className="panel mt-8 p-5">
        <h2 className="text-xl font-semibold text-white">Aviso obrigatorio</h2>
        <p className="mt-3 text-sm leading-6 text-zinc-300">
          Esta aplicacao e uma demonstracao tecnica de visao computacional e nao realiza diagnostico medico, biometrico
          ou psicologico.
        </p>
      </div>
    </section>
  );
}
