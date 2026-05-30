"use client";

import { useCallback, useEffect, useState } from "react";

export type CameraStatus = "idle" | "requesting" | "active" | "blocked" | "unsupported" | "error";

export function useCamera() {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [status, setStatus] = useState<CameraStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const stopCamera = useCallback(() => {
    setStream((current) => {
      current?.getTracks().forEach((track) => track.stop());
      return null;
    });
    setStatus((current) => (current === "active" ? "idle" : current));
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("unsupported");
      setError("Este navegador nao oferece suporte a getUserMedia.");
      return null;
    }

    try {
      setStatus("requesting");
      const nextStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 60 },
        },
      });

      setStream((current) => {
        current?.getTracks().forEach((track) => track.stop());
        return nextStream;
      });
      setStatus("active");
      return nextStream;
    } catch (requestError) {
      const isDenied =
        requestError instanceof DOMException &&
        (requestError.name === "NotAllowedError" || requestError.name === "PermissionDeniedError");

      setStatus(isDenied ? "blocked" : "error");
      setError(
        isDenied
          ? "Permissao da camera bloqueada. Libere o acesso no navegador para testar a demo."
          : "Nao foi possivel iniciar a camera neste dispositivo.",
      );
      return null;
    }
  }, []);

  useEffect(() => {
    return () => {
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, [stream]);

  return {
    stream,
    status,
    error,
    isActive: status === "active" && Boolean(stream),
    startCamera,
    stopCamera,
  };
}
