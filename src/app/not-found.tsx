import Link from "next/link";
import { SearchX } from "lucide-react";

export default function NotFound() {
  return (
    <section className="mx-auto grid min-h-[60vh] max-w-3xl place-items-center px-4 py-10 text-center">
      <div className="panel p-8">
        <SearchX className="mx-auto h-10 w-10 text-zinc-300" aria-hidden />
        <h1 className="mt-5 text-3xl font-semibold text-white">Pagina nao encontrada</h1>
        <p className="mt-3 text-sm text-zinc-400">A rota solicitada nao existe nesta demonstracao.</p>
        <Link className="control-button mt-6 bg-cyan-300 text-black hover:bg-cyan-200" href="/">
          Voltar para Home
        </Link>
      </div>
    </section>
  );
}
