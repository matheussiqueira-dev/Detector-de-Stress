# StressCam (esqueleto)

Pipeline modular para estimar estresse fisiológico em tempo real usando apenas webcam.

## Rodar
```bash
# recomendado Python 3.10 (MediaPipe solutions)
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt  # instala mediapipe + opencv compatíveis
python -m stresscam.app  # tecla q para sair (HUD OpenCV)
```

### Dashboard (Streamlit)
```bash
streamlit run stresscam/streamlit_app.py
# se streamlit.exe não estiver no PATH: .venv\Scripts\streamlit run stresscam/streamlit_app.py
```

### Escolher a webcam (BRIO 305)
- Opção 1 (nome da câmera, Windows/DirectShow):  
  `set STRESSCAM_DEVICE=video=BRIO 305` e depois `python -m stresscam.app`
- Opção 2 (índice OpenCV): edite `device_index` em `stresscam/config.py` ou exporte `STRESSCAM_DEVICE_INDEX=1`.
- O código falha ao iniciar se não conseguir abrir a câmera, para evitar usar a câmera errada.

### API externa (REST/WebSocket)
- REST: `GET http://localhost:8000/score` → `{"score":0.53,"trend":-0.01,"ts":1738280000.1}`
- WebSocket: `ws://localhost:8765/` envia o mesmo JSON periodicamente (5 Hz). Útil para dashboards externos.

## Módulos
- `stresscam/video.py`: captura, normalização de luz, detecção facial (MediaPipe) e suavização de bbox.
- `stresscam/features.py`: sinais visuais (EAR, tensão facial geométrica, área pupilar).
- `stresscam/temporal.py`: buffers deslizantes, estatísticas e baseline individual.
- `stresscam/model.py`: modelo leve (SGD ou RandomForest) com clipping para [0,1].
- `stresscam/viz.py`: HUD em tempo real.
- `stresscam/app.py`: orquestra o loop principal.
- `stresscam/streamlit_app.py`: dashboard com gráfico e controles.
- `stresscam/server.py`: expõe o score por REST/WS para integração externa.
- `stresscam/diag.py`: logging de FPS/tempo de inferência.

Notas:
- Usa baseline inicial (`Config.baseline_sec`) para normalizar por indivíduo.
- Saída é contínua e suavizada por EMA (`ema_alpha`).
- Não é diagnóstico médico; sensível a iluminação e qualidade da câmera.

### Calibração rápida
1) Fique em repouso por ~15 s ao iniciar (baseline).
2) Ajuste `blink_ear_thresh` (config ou slider no Streamlit) até detecções de piscada ficarem consistentes.
3) Se o score oscilar muito, aumente `ema_alpha` ou `win_size_sec`; se reagir lento, reduza.
4) Ative/ajuste logs: `log_diag=True` e `log_interval_sec` em `stresscam/config.py`.
