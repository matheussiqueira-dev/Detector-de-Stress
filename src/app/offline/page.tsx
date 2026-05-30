import Link from "next/link";
import { WifiOff } from "lucide-react";

export default function OfflinePage() {
  return (
    <section className="mx-auto grid min-h-[60vh] max-w-3xl place-items-center px-4 py-10 text-center">
      <div className="panel p-8">
        <WifiOff className="mx-auto h-10 w-10 text-amber-200" aria-hidden />
        <h1 className="mt-5 text-3xl font-semibold text-white">Voce esta offline</h1>
        <p className="mt-3 text-sm leading-6 text-zinc-400">
          A interface basica pode ser carregada do cache, mas a demo em tempo real precisa de permissao de camera e dos
          assets do navegador.
        </p>
        <Link className="control-button mt-6 border-white/10 bg-white/[0.04] text-white hover:bg-white/[0.08]" href="/">
          Voltar para Home
        </Link>
      </div>
    </section>
  );
}
