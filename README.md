# StressCam — Real-Time Physiological Stress Detector

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=flat-square&logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6?style=flat-square&logo=typescript&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-58%20cases-brightgreen?style=flat-square)

Pipeline modular de visão computacional que estima stress fisiológico em tempo real usando apenas uma webcam comum. Combina um backend Python de alta performance com um dashboard Next.js de tema ENCOM/Tron, exponível via REST e WebSocket.

> ⚠️ **Aviso:** Este sistema é uma ferramenta de pesquisa e demonstração técnica, **não um dispositivo médico**. Os scores são estimativas baseadas em sinais visuais não-invasivos e não substituem avaliação clínica.

---

## Índice

- [Funcionalidades](#funcionalidades)
- [Arquitetura](#arquitetura)
- [Instalação](#instalação)
- [Uso](#uso)
- [API Reference](#api-reference)
- [Configuração](#configuração)
- [Testes](#testes)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Roadmap](#roadmap)
- [Autor](#autor)

---

## Funcionalidades

### Backend Python
- **Detecção facial em tempo real** via MediaPipe (468 landmarks + íris)
- **3 sinais fisiológicos não-invasivos:**
  - Eye Aspect Ratio (EAR) — abertura ocular e taxa de piscadas
  - Tensão facial geométrica — sobrancelha, lábios, mandíbula
  - Área pupilar — estimativa de dilatação da íris
- **Calibração individual** com 15 s de baseline neutro por usuário
- **2 modelos de regressão:** SGD (default) ou Random Forest
- **Normalização de iluminação** via CLAHE adaptativo
- **Persistência de modelo** — salva e carrega modelo treinado com joblib
- **Gravação de sessão** — exporta histórico em JSON e CSV com resumo estatístico
- **API REST + WebSocket** — score, trend, health e histórico expostos
- **Encerramento limpo** via SIGINT/SIGTERM e tecla `q`
- **Logging estruturado** com rotação de arquivo (5 MB / 3 backups)

### Frontend Next.js
- **Conexão WebSocket** com fallback automático para polling REST (2 s)
- **Gráfico SVG em tempo real** — histórico dos últimos 5 minutos com gradiente dinâmico
- **Sistema de alertas** — banners warning/critical com cooldown de 30 s
- **Exportação CSV** da sessão atual com um clique
- **Tema ENCOM/Tron** — Orbitron + Rajdhani + Exo 2, animações CSS puras
- **Responsivo** — desktop, tablet e mobile
- **Deploy Vercel** — zero configuração

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    Navegador (Next.js / Vercel)              │
│                                                             │
│  StressDashboard ──► useStressData ──► WebSocket ─────────┐ │
│  StressChart (SVG)                    └─► REST /score     │ │
│  AlertBanner                          (fallback 2 s)      │ │
│  Export CSV                                               │ │
└───────────────────────────────────────────────────────────┼─┘
                                                            │ WS :8765
                                                            │ HTTP :8000
┌───────────────────────────────────────────────────────────▼─┐
│                    Backend Python                            │
│                                                             │
│  ScoreServer ──► GET /score  │ /health │ /history           │
│  (REST + WS)                                               │
│       │                                                     │
│       ▼                                                     │
│  Main Loop (app.py)                                         │
│  ├── VideoStream ─── CLAHE normalization ── frame mirror    │
│  ├── FaceProcessor ─ MediaPipe 468 landmarks + iris         │
│  ├── FeatureExtractor ─ EAR · facial_tension · pupil_area   │
│  ├── TemporalBuffer ─ 10 s rolling window · stats · entropy │
│  ├── BaselineNormalizer ─ 15 s calibration                  │
│  ├── StressRegressor ─ SGD | RF | NumPy fallback            │
│  ├── SessionRecorder ─ JSON + CSV export on exit            │
│  └── DiagLogger ─ FPS · inference time · latency spikes     │
└─────────────────────────────────────────────────────────────┘
```

---

## Instalação

### Pré-requisitos

| Requisito | Versão Mínima |
|-----------|--------------|
| Python    | 3.10         |
| Node.js   | 18           |
| Webcam    | Qualquer câmera compatível com OpenCV |

### Backend Python

```bash
# 1. Clone o repositório
git clone https://github.com/matheussiqueira-dev/Detector-de-Stress.git
cd Detector-de-Stress

# 2. Crie e ative o ambiente virtual (Python 3.10 recomendado)
py -3.10 -m venv .venv          # Windows
python3.10 -m venv .venv        # Linux/macOS
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. (Opcional) Configure variáveis de ambiente
cp .env.example .env
# Edite .env conforme necessário
```

### Frontend Next.js

```bash
# Na raiz do projeto
npm install
npm run dev   # http://localhost:3000
```

---

## Uso

### Iniciar o pipeline principal

```bash
# HUD OpenCV com servidor REST/WebSocket
python -m stresscam.app

# Com modelo Random Forest
STRESSCAM_MODEL_TYPE=rf python -m stresscam.app

# Com câmera específica (Windows/DirectShow)
STRESSCAM_DEVICE="BRIO 305" python -m stresscam.app
```

### Dashboard Streamlit (local)

```bash
streamlit run stresscam/streamlit_app.py
```

### Via Makefile

```bash
make setup          # Cria .venv e instala dependências
make run            # Inicia pipeline principal
make run-streamlit  # Inicia dashboard Streamlit
make run-rf         # Inicia com Random Forest
make test           # Executa todos os testes
make test-cov       # Testes com relatório de cobertura
make frontend-dev   # Inicia Next.js em modo dev
```

### Atalhos na janela OpenCV

| Tecla | Ação |
|-------|------|
| `q`   | Encerrar pipeline (salva sessão e modelo) |
| `s`   | Salvar frame atual como `frame_NNN.png` |
| `d`   | Ativar / desativar modo demonstração (alta sensibilidade) |

### Protocolo de calibração

1. Ao iniciar, mantenha **expressão neutra** por 15 segundos (período de baseline)
2. O modelo é treinado automaticamente com suas medidas de repouso
3. Ajuste `blink_ear_thresh` se piscadas não forem detectadas corretamente
4. Se o score oscilar muito: aumente `ema_alpha` ou `win_size_sec`
5. Se reagir lentamente: reduza `ema_alpha` ou ative o modo demonstração (`d`)

---

## API Reference

### REST — `http://localhost:8000`

#### `GET /score`

Retorna o score atual de stress.

```json
{
  "score": 0.53,
  "trend": -0.012,
  "ts": 1738280000.1
}
```

#### `GET /health`

Metadados de saúde do servidor.

```json
{
  "status": "ok",
  "uptime_s": 142.3,
  "frame_count": 4269
}
```

#### `GET /history?limit=N`

Retorna as últimas N leituras (padrão: 300 ≈ 1 min @ 5 Hz, máx: 3000).

```json
{
  "history": [
    { "ts": 1738280000.1, "score": 0.42, "trend": 0.01 },
    { "ts": 1738280000.3, "score": 0.44, "trend": 0.02 }
  ]
}
```

### WebSocket — `ws://localhost:8765`

Envia o mesmo payload de `/score` a cada 200 ms (5 Hz padrão).

---

## Configuração

Copie `.env.example` para `.env` e ajuste conforme necessário.
Todas as variáveis são **opcionais** — os defaults estão em `stresscam/config.py`.

| Variável | Default | Descrição |
|----------|---------|-----------|
| `STRESSCAM_DEVICE_INDEX` | `0` | Índice OpenCV da câmera |
| `STRESSCAM_DEVICE` | — | Nome DirectShow (Windows) |
| `STRESSCAM_FPS` | `30` | Taxa de captura |
| `STRESSCAM_HTTP_PORT` | `8000` | Porta REST |
| `STRESSCAM_WS_PORT` | `8765` | Porta WebSocket |
| `STRESSCAM_MODEL_TYPE` | `sgd` | `sgd` ou `rf` |
| `STRESSCAM_BASELINE_SEC` | `15` | Duração da calibração (s) |
| `STRESSCAM_HIGH_SENSITIVITY` | `true` | Modo demo/alta sensibilidade |
| `STRESSCAM_ENABLE_RECORDING` | `true` | Gravar sessão em JSON/CSV |
| `STRESSCAM_AUTO_SAVE_MODEL` | `true` | Salvar modelo após treino |
| `STRESSCAM_LOG_LEVEL` | `INFO` | `debug`/`info`/`warning`/`error` |
| `NEXT_PUBLIC_STRESSCAM_API_URL` | — | URL do backend (produção) |
| `NEXT_PUBLIC_STRESSCAM_WS_URL` | — | URL WebSocket (produção) |

---

## Testes

```bash
# Executar todos os testes
pytest

# Com cobertura de código
pytest --cov=stresscam --cov-report=term-missing

# Módulo específico
pytest tests/test_server.py -v
```

**Cobertura atual:** 58 casos de teste distribuídos em 7 módulos.

| Módulo | Casos | Cobertura |
|--------|-------|-----------|
| `config` | 12 | validação, env, device_name |
| `features` | 13 | EAR, blink_rate, tension, pupil, pack |
| `temporal` | 11 | buffer, features, entropy, baseline |
| `model` | 10 | SGD, RF, persistence, fallback |
| `server` | 8 | thread-safety, REST endpoints |
| `recorder` | 8 | JSON/CSV, summary stats |
| `logger` | 6 | setup, get_logger, idempotência |

---

## Estrutura do Projeto

```
Detector-de-Stress/
├── stresscam/                  # Pacote Python principal
│   ├── __init__.py
│   ├── app.py                  # Orquestrador do pipeline + loop principal
│   ├── config.py               # Dataclass de configuração com validação
│   ├── diag.py                 # Diagnósticos de performance (FPS, latência)
│   ├── features.py             # Extração de features fisiológicas
│   ├── logger.py               # Logging centralizado com rotação
│   ├── model.py                # Regressores de stress + persistência
│   ├── recorder.py             # Gravação de sessão em JSON/CSV
│   ├── server.py               # Servidor REST + WebSocket
│   ├── streamlit_app.py        # Dashboard Streamlit interativo
│   ├── temporal.py             # Buffers temporais + baseline normalizer
│   ├── video.py                # VideoStream + FaceProcessor
│   └── viz.py                  # Visualização HUD + gráfico rolling
│
├── app/                        # Next.js App Router
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
│
├── components/
│   ├── dashboard/
│   │   ├── StressDashboard.tsx     # Dashboard principal (WS + alertas + CSV)
│   │   └── StressDashboard.module.css
│   ├── system/
│   │   ├── BackgroundGrid.tsx
│   │   ├── Footer.tsx
│   │   └── WhatsAppButton.tsx
│   └── ui/
│       ├── AlertBanner.tsx         # Sistema de alertas por threshold
│       ├── EncomPanel.tsx
│       ├── StressChart.tsx         # Gráfico SVG de histórico (5 min)
│       ├── TronButton.tsx
│       └── TronCard.tsx
│
├── styles/
│   └── encom-theme.css
│
├── tests/                      # Suite pytest (58 casos)
│   ├── test_config.py
│   ├── test_features.py
│   ├── test_logger.py
│   ├── test_model.py
│   ├── test_recorder.py
│   ├── test_server.py
│   └── test_temporal.py
│
├── .env.example                # Template de variáveis de ambiente
├── .gitignore
├── Makefile                    # Comandos de desenvolvimento
├── next.config.ts
├── package.json
├── pytest.ini
├── requirements.txt
├── tsconfig.json
└── vercel.json
```

---

## Roadmap

- [ ] Persistência do baseline por usuário (login simples)
- [ ] Detecção de emoções com FER+ / modelos leves
- [ ] Relatório automático pós-sessão com gráficos matplotlib
- [ ] Feedback sonoro quando score > 0.7 (TTS)
- [ ] Exportação contínua para APIs externas (MQTT / HTTP POST)
- [ ] Testes Jest para componentes React

---

## Autor

Desenvolvido por **Matheus Siqueira**

- Portfolio: [www.matheussiqueira.dev](https://www.matheussiqueira.dev/)
- GitHub: [@matheussiqueira-dev](https://github.com/matheussiqueira-dev)

---

*[www.matheussiqueira.dev](https://www.matheussiqueira.dev/)*
