# AI Face Behavior Dashboard

Aplicacao web profissional de visao computacional no navegador, criada para demonstrar webcam em tempo real, deteccao facial, acompanhamento do rosto, metricas comportamentais experimentais, alertas visuais, historico temporal e dashboard inteligente.

Desenvolvido por **Matheus Siqueira**  
Portfolio: [www.matheussiqueira.dev](https://www.matheussiqueira.dev)

## Visao Geral

O projeto reproduz em formato web app a experiencia de um sistema tecnico de monitoramento facial: feed da webcam, bounding box dinamica, pontos faciais, indicadores no overlay, status de analise, grafico temporal, eventos e dashboard analitico.

A aplicacao foi estruturada para rodar sem backend obrigatorio. O processamento da camera acontece no navegador sempre que possivel e o dashboard usa mock data profissional quando nao ha persistencia real de sessao.

## Rotas

- `/` - landing page premium com hero, CTAs, recursos, explicacao tecnica, privacidade e creditos.
- `/demo` - demonstracao em tempo real com webcam, face tracking, overlay, metricas, alertas e event log.
- `/dashboard` - KPIs, graficos, distribuicao de estados, tabela filtravel, alertas e insights.
- `/about` - privacidade, limites tecnicos e aviso de nao diagnostico.
- `/offline` - fallback basico para PWA.

## Funcionalidades

- Captura de webcam com `getUserMedia`.
- Tratamento de permissao negada, navegador sem suporte e falha de modelo.
- Detecao facial client-side com MediaPipe Tasks Vision.
- Fallback para `FaceDetector` nativo quando disponivel.
- Fallback visual local identificado quando o modelo real nao puder carregar.
- Canvas overlay com bounding box, pontos faciais e metricas no video.
- Attention Score experimental de 0 a 100.
- Indicadores de confianca, FPS, latencia, movimento, boca aberta, olhos parcialmente fechados e yaw aproximado.
- Registro de eventos: `FACE_DETECTED`, `FACE_LOST`, `LOW_CONFIDENCE`, `ATTENTION_DROP`, `HIGH_MOVEMENT`, `CAMERA_STARTED`, `CAMERA_STOPPED`, `MODEL_READY`, `MODEL_ERROR`.
- Dashboard com KPIs, graficos Recharts, filtros, tabela de eventos e insights por regras.
- PWA basico com manifest, icones, service worker e pagina offline.
- SEO com Metadata API, canonical, Open Graph dinamico, Twitter Card, robots e sitemap.
- Vercel Analytics e estrutura para eventos customizados.
- Creditos visiveis e clicaveis para Matheus Siqueira em todas as telas principais.

## Stack

- Next.js App Router
- React
- TypeScript
- Tailwind CSS
- MediaPipe Tasks Vision
- Recharts
- Lucide React
- Vercel Analytics

## Arquitetura

```txt
src/
  app/
    page.tsx
    demo/page.tsx
    dashboard/page.tsx
    about/page.tsx
    offline/page.tsx
    layout.tsx
    robots.ts
    sitemap.ts
    opengraph-image.tsx
  components/
    charts/
    credits/
    dashboard/
    layout/
    ui/
    webcam/
  data/
    mockEvents.ts
    mockMetrics.ts
    mockSessions.ts
  hooks/
    useCamera.ts
    useFaceTracking.ts
    useSessionMetrics.ts
  lib/
    analytics.ts
    events.ts
    insights.ts
    scoring.ts
    utils.ts
  types/
    events.ts
    metrics.ts
    vision.ts
```

## Como Rodar Localmente

```bash
npm install
npm run dev
```

Abra `http://localhost:3000`.

Para validar producao:

```bash
npm run lint
npm run build
npm run start
```

## Como Funciona a Deteccao Facial

A rota `/demo` tenta carregar o MediaPipe FaceLandmarker no navegador usando WASM e modelo remoto oficial. Quando o modelo esta pronto, cada frame do video e analisado localmente e transformado em:

- bounding box normalizada;
- pontos faciais amostrados;
- confianca estimada;
- movimento e estabilidade;
- sinais aproximados de boca aberta, olhos parcialmente fechados e yaw.

Se o modelo nao carregar, o app tenta usar a API nativa `FaceDetector`. Se nenhuma das duas estiver disponivel, ativa um fallback visual local identificado na interface, mantendo a demonstracao testavel sem fingir diagnostico real.

## Attention Score

O score e calculado em `src/lib/scoring.ts` por `calculateAttentionScore(input): number`.

Ele considera:

- presenca de face;
- confianca da deteccao;
- estabilidade do rosto;
- movimento brusco;
- boca aberta;
- olhos parcialmente fechados;
- yaw aproximado;
- perda de rastreamento.

O score e uma metrica experimental de demonstracao. Nao e diagnostico medico, psicologico, clinico, biometrico ou de identidade.

## Dashboard e Insights

O dashboard usa dados estruturados em:

- `src/data/mockMetrics.ts`
- `src/data/mockEvents.ts`
- `src/data/mockSessions.ts`

Os insights sao gerados por regras em `src/lib/insights.ts`, avaliando media de atencao, confianca, movimento, FPS, estabilidade e eventos.

Exemplos:

- atencao estavel;
- confianca abaixo do ideal;
- oscilacao de rastreamento;
- aumento de possivel distracao;
- FPS reduzido.

## Privacidade e Limites

Esta aplicacao e uma demonstracao tecnica de visao computacional e nao realiza diagnostico medico, biometrico ou psicologico.

Regras aplicadas:

- nao envia frames da webcam para servidor;
- nao armazena biometria facial;
- nao identifica pessoas;
- nao faz reconhecimento de identidade;
- nao promete avaliacao clinica;
- trata camera bloqueada, navegador sem suporte e falha de modelo.

## Limitacoes Conhecidas

- O MediaPipe FaceLandmarker baixa WASM e modelo no cliente; se a rede bloquear esses assets, a demo usa fallback nativo ou visual identificado.
- O dashboard usa mock data profissional enquanto nao houver persistencia real de sessoes.
- `npm audit` pode reportar alerta moderado transitive de `postcss` dentro do `next@16.2.6`. Em 30/05/2026, `next@16.2.6` e a versao latest no npm e a correcao sugerida por `npm audit fix --force` faz downgrade quebrado para Next 9, por isso nao foi aplicada.

## PWA

Arquivos principais:

- `public/manifest.json`
- `public/icon.svg`
- `public/apple-icon.svg`
- `public/sw.js`
- `src/app/offline/page.tsx`

O service worker e registrado apenas em producao para evitar cache agressivo durante desenvolvimento.

## SEO e Compartilhamento

Implementado com Metadata API do Next.js:

- title e description;
- canonical;
- Open Graph;
- Twitter Card;
- imagem social dinamica em `src/app/opengraph-image.tsx`;
- robots;
- sitemap;
- icones do projeto.

## Deploy na Vercel

1. Suba o repositorio para o GitHub.
2. Importe o projeto na Vercel.
3. Use o preset automatico de Next.js.
4. Garanta que o build command seja `npm run build`.
5. Publique.

Observacao: webcam em producao exige HTTPS. A Vercel fornece HTTPS automaticamente.

## Variaveis de Ambiente

Veja `.env.example`.

Nao ha variaveis obrigatorias para rodar a demo. Nao inclua tokens, chaves ou segredos no repositorio.

## Roadmap Futuro

- Persistir sessoes reais em banco local-first ou backend opcional.
- Exportar relatorios CSV/JSON.
- Calibracao guiada de camera e iluminacao.
- Testes automatizados de scoring e geracao de insights.
- Modo comparativo entre sessoes.
- Worker dedicado para inferencia e melhor isolamento de performance.

## Creditos

Desenvolvido por **Matheus Siqueira**  
[www.matheussiqueira.dev](https://www.matheussiqueira.dev)
