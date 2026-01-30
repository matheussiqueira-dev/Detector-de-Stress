"""
Dashboard Streamlit para visualizar o score de estresse em tempo real.
Rode com: streamlit run stresscam/streamlit_app.py
"""
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from .config import Config
from .video import VideoStream, FaceProcessor, normalize_lighting
from .features import eye_aspect_ratio, facial_tension, pupil_area, EAR_LEFT, EAR_RIGHT
from .temporal import TemporalBuffer, BaselineNormalizer
from .model import StressRegressor
from .server import ScoreServer


def pack_features(feats):
    return np.hstack([
        feats["blink_rate"],
        feats["ear_mean"], feats["ear_std"],
        feats["tension_mean"], feats["tension_std"],
        feats["pupil_mean"], feats["pupil_std"],
        feats["entropy_tension"]
    ])


def init_state():
    if "cfg" not in st.session_state:
        st.session_state.cfg = Config()
    if "stream" not in st.session_state:
        st.session_state.stream = VideoStream(st.session_state.cfg)
    if "face" not in st.session_state:
        st.session_state.face = FaceProcessor(st.session_state.cfg)
    if "buf" not in st.session_state:
        st.session_state.buf = TemporalBuffer(st.session_state.cfg)
    if "base" not in st.session_state:
        st.session_state.base = BaselineNormalizer()
    if "model" not in st.session_state:
        st.session_state.model = StressRegressor(st.session_state.cfg)
    if "server" not in st.session_state:
        st.session_state.server = ScoreServer(
            st.session_state.cfg.http_port,
            st.session_state.cfg.ws_port,
            st.session_state.cfg.broadcast_hz,
        ) if st.session_state.cfg.enable_server else None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "t0" not in st.session_state:
        st.session_state.t0 = time.time()
    if "history" not in st.session_state:
        st.session_state.history = []


def start():
    st.session_state.running = True
    st.session_state.t0 = time.time()
    if st.session_state.server:
        st.session_state.server.start()


def stop():
    st.session_state.running = False
    try:
        st.session_state.stream.release()
    except Exception:
        pass
    if st.session_state.server:
        st.session_state.server.stop()


def main():
    st.set_page_config(page_title="StressCam Dashboard", layout="wide")
    st.title("StressCam – Estresse Visual em Tempo Real")
    st.markdown("Pressione **Iniciar** para capturar webcam. Baseline nos primeiros segundos.")

    init_state()
    cfg = st.session_state.cfg

    col1, col2 = st.columns([2, 1])
    with col2:
        if st.button("Iniciar captura", type="primary", disabled=st.session_state.running):
            start()
        if st.button("Parar", disabled=not st.session_state.running):
            stop()
        cfg.blink_ear_thresh = st.slider("Threshold EAR (blink)", 0.15, 0.35, cfg.blink_ear_thresh, 0.01)
        cfg.ema_alpha = st.slider("Suavização EMA", 0.05, 0.5, cfg.ema_alpha, 0.01)
        cfg.model_type = st.selectbox("Modelo", ["sgd", "rf"], index=0 if cfg.model_type == "sgd" else 1)

    frame_ph = col1.empty()
    chart_ph = col1.empty()

    prev_score = 0.5

    while st.session_state.running:
        frame = st.session_state.stream.read()
        frame = normalize_lighting(frame, cfg)

        results = st.session_state.face.detect(frame)
        lms = st.session_state.face.landmarks(results)

        if lms:
            bbox = st.session_state.face.bbox_from_landmarks(lms, frame.shape)
            ear_L = eye_aspect_ratio(lms, EAR_LEFT)
            ear_R = eye_aspect_ratio(lms, EAR_RIGHT)
            ear = (ear_L + ear_R) / 2.0
            tension_vec = facial_tension(lms)
            pupil = pupil_area(lms)

            st.session_state.buf.append(ear, tension_vec, pupil)
            feats = st.session_state.buf.features()

            if feats:
                fvec = pack_features(feats)
                # baseline nos primeiros segundos
                if (time.time() - st.session_state.t0) < cfg.baseline_sec:
                    st.session_state.base.collect(fvec)
                elif (not st.session_state.model.trained) and st.session_state.base.ready():
                    X0, y0 = st.session_state.base.dump()
                    st.session_state.model = StressRegressor(cfg)
                    st.session_state.model.fit_baseline(X0, y0)

                score_raw = st.session_state.model.predict(fvec)
                if st.session_state.buf.score_ema is None:
                    st.session_state.buf.score_ema = score_raw
                else:
                    st.session_state.buf.score_ema = st.session_state.buf.score_ema + cfg.ema_alpha * (score_raw - st.session_state.buf.score_ema)
                trend = st.session_state.buf.score_ema - prev_score
                prev_score = st.session_state.buf.score_ema

                if st.session_state.server:
                    st.session_state.server.update(st.session_state.buf.score_ema, trend)

                # anotar histórico para gráfico
                st.session_state.history.append(
                    {"t": time.time(), "score": st.session_state.buf.score_ema}
                )

                # HUD simples
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(frame, f"Stress {st.session_state.buf.score_ema:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Tend {trend:+.3f}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 200, 0) if trend <= 0 else (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Face nao detectada", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # desenhar frame no dashboard
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_ph.image(frame_rgb, channels="RGB")

        # gráfico de score
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            df["t"] = df["t"] - df["t"].iloc[0]
            chart_ph.line_chart(df, x="t", y="score", height=240)

        # pequeno sleep evita sobrecarga
        time.sleep(0.01)

    # se não estiver rodando, mostrar placeholder
    if not st.session_state.running:
        frame_ph.info("Clique em **Iniciar captura** para começar.")


if __name__ == "__main__":
    main()
