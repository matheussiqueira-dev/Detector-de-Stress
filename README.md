# StressCam (esqueleto)

Pipeline modular para estimar estresse fisiológico em tempo real usando apenas webcam.

## Rodar
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m stresscam.app  # tecla q para sair
```

### Dashboard (Streamlit)
```bash
streamlit run stresscam/streamlit_app.py
# se streamlit.exe não estiver no PATH: .venv\Scripts\streamlit run stresscam/streamlit_app.py
```

## Módulos
- `stresscam/video.py`: captura, normalização de luz, detecção facial (MediaPipe) e suavização de bbox.
- `stresscam/features.py`: sinais visuais (EAR, tensão facial geométrica, área pupilar).
- `stresscam/temporal.py`: buffers deslizantes, estatísticas e baseline individual.
- `stresscam/model.py`: modelo leve (SGD ou RandomForest) com clipping para [0,1].
- `stresscam/viz.py`: HUD em tempo real.
- `stresscam/app.py`: orquestra o loop principal.

Notas:
- Usa baseline inicial (`Config.baseline_sec`) para normalizar por indivíduo.
- Saída é contínua e suavizada por EMA (`ema_alpha`).
- Não é diagnóstico médico; sensível a iluminação e qualidade da câmera.
