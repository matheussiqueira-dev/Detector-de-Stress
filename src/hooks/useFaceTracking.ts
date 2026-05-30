"use client";

import { useEffect, useRef, useState } from "react";
import type { MutableRefObject, RefObject } from "react";
import type { FaceLandmarker, FaceLandmarkerResult, NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { BoundingBox, FaceLandmarkPoint, FaceTrackingResult, VisionModelStatus } from "@/types/vision";
import { clamp, round } from "@/lib/utils";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";

type NativeFaceDetectorResult = {
  boundingBox: DOMRectReadOnly | { x: number; y: number; width: number; height: number };
};

type NativeFaceDetector = {
  detect(source: HTMLVideoElement): Promise<NativeFaceDetectorResult[]>;
};

type NativeFaceDetectorConstructor = new (options?: {
  fastMode?: boolean;
  maxDetectedFaces?: number;
}) => NativeFaceDetector;

declare global {
  interface Window {
    FaceDetector?: NativeFaceDetectorConstructor;
  }
}

const idleResult: FaceTrackingResult = {
  faceDetected: false,
  confidence: 0,
  boundingBox: null,
  landmarks: [],
  movement: 0,
  stability: 1,
  mouthOpen: 0,
  eyesClosed: 0,
  yaw: 0,
  fps: 0,
  latency: 0,
  source: "idle",
  modelStatus: "idle",
  message: "Aguardando camera.",
};

interface UseFaceTrackingOptions {
  enabled: boolean;
  paused: boolean;
  videoRef: RefObject<HTMLVideoElement | null>;
}

export function useFaceTracking({ enabled, paused, videoRef }: UseFaceTrackingOptions) {
  const [result, setResult] = useState<FaceTrackingResult>(idleResult);
  const [modelStatus, setModelStatus] = useState<VisionModelStatus>("idle");
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const nativeDetectorRef = useRef<NativeFaceDetector | null>(null);
  const lastBoxRef = useRef<BoundingBox | null>(null);
  const lastFrameTimeRef = useRef<number>(0);
  const fallbackPhaseRef = useRef<number>(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadModel() {
      if (!enabled || paused || landmarkerRef.current || modelStatus === "loading") {
        return;
      }

      setModelStatus("loading");

      try {
        const vision = await import("@mediapipe/tasks-vision");
        const fileset = await vision.FilesetResolver.forVisionTasks(WASM_URL);
        const landmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath: MODEL_URL,
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numFaces: 1,
          minFaceDetectionConfidence: 0.45,
          minFacePresenceConfidence: 0.45,
          minTrackingConfidence: 0.45,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        });

        if (!cancelled) {
          landmarkerRef.current = landmarker;
          setModelStatus("ready");
        }
      } catch {
        if (!cancelled) {
          if (window.FaceDetector) {
            nativeDetectorRef.current = new window.FaceDetector({
              fastMode: true,
              maxDetectedFaces: 1,
            });
          }

          setModelStatus(window.FaceDetector ? "fallback" : "fallback");
        }
      }
    }

    void loadModel();

    return () => {
      cancelled = true;
    };
  }, [enabled, paused, modelStatus]);

  useEffect(() => {
    if (!enabled) {
      const frame = window.requestAnimationFrame(() => setResult(idleResult));
      return () => window.cancelAnimationFrame(frame);
    }

    if (paused) {
      return;
    }

    let cancelled = false;

    async function detectFrame(now: number) {
      const video = videoRef.current;

      if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.videoWidth === 0) {
        rafRef.current = window.requestAnimationFrame(detectFrame);
        return;
      }

      const start = performance.now();
      const fps = calculateFps(now, lastFrameTimeRef.current);
      lastFrameTimeRef.current = now;

      try {
        if (landmarkerRef.current && modelStatus === "ready") {
          const output = landmarkerRef.current.detectForVideo(video, now);
          setResult(fromMediaPipeResult(output, video, fps, performance.now() - start, lastBoxRef));
        } else if (nativeDetectorRef.current) {
          const detections = await nativeDetectorRef.current.detect(video);
          setResult(fromNativeFaceDetectorResult(detections, video, fps, performance.now() - start, lastBoxRef));
        } else {
          setResult(fromVisualFallback(video, fps, performance.now() - start, fallbackPhaseRef, lastBoxRef));
        }
      } catch {
        setModelStatus("error");
        setResult({
          ...idleResult,
          modelStatus: "error",
          message: "Falha temporaria na inferencia do modelo.",
        });
      }

      if (!cancelled) {
        rafRef.current = window.requestAnimationFrame(detectFrame);
      }
    }

    rafRef.current = window.requestAnimationFrame(detectFrame);

    return () => {
      cancelled = true;
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
      }
    };
  }, [enabled, paused, modelStatus, videoRef]);

  return {
    result: {
      ...result,
      modelStatus,
    },
    modelStatus,
  };
}

function fromMediaPipeResult(
  output: FaceLandmarkerResult,
  video: HTMLVideoElement,
  fps: number,
  latency: number,
  lastBoxRef: MutableRefObject<BoundingBox | null>,
): FaceTrackingResult {
  const landmarks = output.faceLandmarks[0] ?? [];
  const boundingBox = landmarks.length > 0 ? boxFromLandmarks(landmarks) : null;
  const movement = calculateMovement(boundingBox, lastBoxRef.current);
  const blendshape = output.faceBlendshapes[0];
  const mouthOpen = getBlendshapeScore(blendshape, ["jawOpen", "mouthFunnel", "mouthPucker"]);
  const eyesClosed = Math.max(
    getBlendshapeScore(blendshape, ["eyeBlinkLeft", "eyeSquintLeft"]),
    getBlendshapeScore(blendshape, ["eyeBlinkRight", "eyeSquintRight"]),
  );
  const yaw = estimateYaw(landmarks);
  const confidence = landmarks.length > 0 ? clamp(0.86 - movement * 0.12 + landmarks.length / 4000, 0.58, 0.98) : 0;

  lastBoxRef.current = boundingBox;

  return {
    faceDetected: Boolean(boundingBox),
    confidence: round(confidence, 2),
    boundingBox,
    landmarks: landmarks.map(toPoint).filter((_, index) => index % 6 === 0),
    movement: round(movement, 2),
    stability: round(clamp(1 - movement, 0, 1), 2),
    mouthOpen: round(mouthOpen, 2),
    eyesClosed: round(eyesClosed, 2),
    yaw: round(yaw),
    fps,
    latency: round(latency),
    source: "mediapipe",
    modelStatus: "ready",
    message: "MediaPipe FaceLandmarker ativo.",
  };
}

function fromNativeFaceDetectorResult(
  detections: NativeFaceDetectorResult[],
  video: HTMLVideoElement,
  fps: number,
  latency: number,
  lastBoxRef: MutableRefObject<BoundingBox | null>,
): FaceTrackingResult {
  const detection = detections[0];
  const boundingBox = detection
    ? {
        x: clamp(detection.boundingBox.x / video.videoWidth, 0, 1),
        y: clamp(detection.boundingBox.y / video.videoHeight, 0, 1),
        width: clamp(detection.boundingBox.width / video.videoWidth, 0, 1),
        height: clamp(detection.boundingBox.height / video.videoHeight, 0, 1),
      }
    : null;
  const movement = calculateMovement(boundingBox, lastBoxRef.current);

  lastBoxRef.current = boundingBox;

  return {
    faceDetected: Boolean(boundingBox),
    confidence: boundingBox ? round(clamp(0.78 - movement * 0.1, 0.5, 0.86), 2) : 0,
    boundingBox,
    landmarks: boundingBox ? syntheticLandmarksFromBox(boundingBox) : [],
    movement: round(movement, 2),
    stability: round(clamp(1 - movement, 0, 1), 2),
    mouthOpen: 0.16,
    eyesClosed: 0.1,
    yaw: boundingBox ? round((boundingBox.x + boundingBox.width / 2 - 0.5) * 42) : 0,
    fps,
    latency: round(latency),
    source: "native-face-detector",
    modelStatus: "fallback",
    message: "Fallback via FaceDetector nativo do navegador.",
  };
}

function fromVisualFallback(
  video: HTMLVideoElement,
  fps: number,
  latency: number,
  phaseRef: MutableRefObject<number>,
  lastBoxRef: MutableRefObject<BoundingBox | null>,
): FaceTrackingResult {
  phaseRef.current += 0.045;
  const x = 0.34 + Math.sin(phaseRef.current) * 0.035;
  const y = 0.17 + Math.cos(phaseRef.current * 0.75) * 0.025;
  const boundingBox = {
    x,
    y,
    width: video.videoWidth > video.videoHeight ? 0.28 : 0.34,
    height: 0.45,
  };
  const movement = calculateMovement(boundingBox, lastBoxRef.current);

  lastBoxRef.current = boundingBox;

  return {
    faceDetected: true,
    confidence: 0.54,
    boundingBox,
    landmarks: syntheticLandmarksFromBox(boundingBox),
    movement: round(movement, 2),
    stability: round(clamp(1 - movement, 0.35, 1), 2),
    mouthOpen: round(0.22 + Math.max(0, Math.sin(phaseRef.current * 2)) * 0.24, 2),
    eyesClosed: round(0.12 + Math.max(0, Math.cos(phaseRef.current * 1.6)) * 0.18, 2),
    yaw: round(Math.sin(phaseRef.current) * 18),
    fps,
    latency: round(latency),
    source: "visual-fallback",
    modelStatus: "fallback",
    message: "Fallback visual local: modelo real indisponivel.",
  };
}

function boxFromLandmarks(landmarks: NormalizedLandmark[]): BoundingBox {
  const xs = landmarks.map((point) => point.x);
  const ys = landmarks.map((point) => point.y);
  const minX = clamp(Math.min(...xs), 0, 1);
  const maxX = clamp(Math.max(...xs), 0, 1);
  const minY = clamp(Math.min(...ys), 0, 1);
  const maxY = clamp(Math.max(...ys), 0, 1);

  return {
    x: minX,
    y: minY,
    width: clamp(maxX - minX, 0, 1),
    height: clamp(maxY - minY, 0, 1),
  };
}

function calculateMovement(next: BoundingBox | null, previous: BoundingBox | null) {
  if (!next || !previous) {
    return 0.08;
  }

  const nextCenterX = next.x + next.width / 2;
  const nextCenterY = next.y + next.height / 2;
  const previousCenterX = previous.x + previous.width / 2;
  const previousCenterY = previous.y + previous.height / 2;
  const distance = Math.hypot(nextCenterX - previousCenterX, nextCenterY - previousCenterY);
  const scaleChange = Math.abs(next.width * next.height - previous.width * previous.height);

  return clamp(distance * 8 + scaleChange * 4, 0, 1);
}

function calculateFps(now: number, lastFrameTime: number) {
  if (!lastFrameTime) {
    return 0;
  }

  return round(clamp(1000 / Math.max(1, now - lastFrameTime), 0, 60), 1);
}

function getBlendshapeScore(
  blendshape: FaceLandmarkerResult["faceBlendshapes"][number] | undefined,
  names: string[],
) {
  if (!blendshape) {
    return 0;
  }

  return names.reduce((highest, name) => {
    const score = blendshape.categories.find((category) => category.categoryName === name)?.score ?? 0;
    return Math.max(highest, score);
  }, 0);
}

function estimateYaw(landmarks: NormalizedLandmark[]) {
  if (landmarks.length < 455) {
    return 0;
  }

  const nose = landmarks[1];
  const leftCheek = landmarks[234];
  const rightCheek = landmarks[454];

  if (!nose || !leftCheek || !rightCheek) {
    return 0;
  }

  const faceCenter = (leftCheek.x + rightCheek.x) / 2;
  return clamp((nose.x - faceCenter) * 120, -45, 45);
}

function toPoint(landmark: NormalizedLandmark): FaceLandmarkPoint {
  return {
    x: landmark.x,
    y: landmark.y,
    z: landmark.z,
  };
}

function syntheticLandmarksFromBox(box: BoundingBox): FaceLandmarkPoint[] {
  const points: FaceLandmarkPoint[] = [];
  const centerX = box.x + box.width / 2;
  const centerY = box.y + box.height / 2;

  for (let index = 0; index < 24; index += 1) {
    const angle = (Math.PI * 2 * index) / 24;
    points.push({
      x: centerX + Math.cos(angle) * box.width * 0.34,
      y: centerY + Math.sin(angle) * box.height * 0.38,
    });
  }

  return points;
}
