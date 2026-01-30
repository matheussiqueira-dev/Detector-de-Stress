import cv2


def draw_hud(frame, score, trend, bbox=None):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (260, 90), (20, 20, 20), -1)
    cv2.putText(overlay, f"Stress: {score:.2f}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(overlay, f"Tendencia: {trend:+.3f}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 200, 0) if trend <= 0 else (0, 0, 255), 2)
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 2)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    return frame
