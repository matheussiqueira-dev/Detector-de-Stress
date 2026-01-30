import numpy as np

# Índices dos olhos na malha do MediaPipe (468+ tem íris)
EAR_LEFT = [33, 160, 158, 133, 153, 144]
EAR_RIGHT = [362, 385, 387, 263, 373, 380]
IRIS_RIGHT = [468, 469, 470, 471, 472]


def eye_aspect_ratio(landmarks, idx):
    pts = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in idx])
    vert = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3]) * 2
    return (vert / horiz) if horiz > 0 else 0.0


def blink_rate(buffer_ear, fps, thresh):
    if len(buffer_ear) == 0:
        return 0.0
    closed = buffer_ear < thresh
    # transições open -> closed contam piscadas; diff evita loop python
    blinks = np.count_nonzero(np.diff(closed.astype(np.int8)) == 1)
    minutes = len(buffer_ear) / (fps * 60.0)
    return blinks / minutes if minutes > 0 else 0.0


def facial_tension(landmarks):
    def dist(a, b):
        return np.linalg.norm(np.array([a.x - b.x, a.y - b.y]))

    # gaps escolhidos por sensibilidade a contração/relaxamento
    brow_gap = dist(landmarks.landmark[70], landmarks.landmark[105])
    lip_stretch = dist(landmarks.landmark[61], landmarks.landmark[291])
    jaw_drop = dist(landmarks.landmark[17], landmarks.landmark[152])
    return np.array([brow_gap, lip_stretch, jaw_drop], dtype=float)


def pupil_area(landmarks):
    pts = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y] for i in IRIS_RIGHT])
    w = pts[:, 0].max() - pts[:, 0].min()
    h = pts[:, 1].max() - pts[:, 1].min()
    return float(w * h)


def pack_features(feats):
    """
    Empacota o dicionário de features em vetor 1D pronto para o modelo.
    Mantido aqui para reutilização em app e dashboard.
    """
    return np.hstack([
        feats["blink_rate"],
        feats["ear_mean"], feats["ear_std"],
        feats["tension_mean"], feats["tension_std"],
        feats["pupil_mean"], feats["pupil_std"],
        feats["entropy_tension"],
    ])
