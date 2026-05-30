"use client";

import { AlertTriangle, RefreshCw } from "lucide-react";

export default function Error({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <section className="mx-auto grid min-h-[60vh] max-w-3xl place-items-center px-4 py-10 text-center">
      <div className="panel p-8">
        <AlertTriangle className="mx-auto h-10 w-10 text-red-200" aria-hidden />
        <h1 className="mt-5 text-3xl font-semibold text-white">Algo falhou</h1>
        <p className="mt-3 text-sm leading-6 text-zinc-400">{error.message || "Erro inesperado na interface."}</p>
        <button className="control-button mt-6 bg-cyan-300 text-black hover:bg-cyan-200" onClick={reset} type="button">
          <RefreshCw className="h-4 w-4" aria-hidden />
          Tentar novamente
        </button>
      </div>
    </section>
  );
}
