export default function Loading() {
  return (
    <div className="mx-auto grid min-h-[50vh] max-w-7xl place-items-center px-4 py-10">
      <div className="panel p-6 text-center">
        <div className="mx-auto h-10 w-10 animate-spin rounded-full border-2 border-cyan-200 border-t-transparent" />
        <p className="mt-4 text-sm text-zinc-400">Carregando interface...</p>
      </div>
    </div>
  );
}
