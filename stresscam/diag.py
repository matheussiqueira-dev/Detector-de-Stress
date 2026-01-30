import time


class DiagLogger:
    """
    Registra FPS e tempo médio de inferência para ajudar a detectar gargalos.
    """
    def __init__(self, interval_sec=5):
        self.interval = interval_sec
        self.last = time.time()
        self.frame_count = 0
        self.infer_sum = 0.0

    def tick(self, infer_ms: float):
        self.frame_count += 1
        self.infer_sum += infer_ms
        now = time.time()
        if (now - self.last) >= self.interval:
            fps = self.frame_count / (now - self.last)
            avg_ms = (self.infer_sum / max(1, self.frame_count)) * 1000
            print(f"[diag] fps={fps:.1f} avg_infer_ms={avg_ms:.2f}")
            self.last = now
            self.frame_count = 0
            self.infer_sum = 0.0
