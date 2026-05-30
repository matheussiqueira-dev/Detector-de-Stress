export type VisionSource = "mediapipe" | "native-face-detector" | "visual-fallback" | "idle";

export type VisionModelStatus = "idle" | "loading" | "ready" | "fallback" | "error";

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface FaceLandmarkPoint {
  x: number;
  y: number;
  z?: number;
}

export interface FaceTrackingResult {
  faceDetected: boolean;
  confidence: number;
  boundingBox: BoundingBox | null;
  landmarks: FaceLandmarkPoint[];
  movement: number;
  stability: number;
  mouthOpen: number;
  eyesClosed: number;
  yaw: number;
  fps: number;
  latency: number;
  source: VisionSource;
  modelStatus: VisionModelStatus;
  message: string;
}

export interface AttentionScoreInput {
  faceDetected: boolean;
  detectionConfidence: number;
  stability: number;
  movement: number;
  mouthOpen: number;
  eyesClosed: number;
  yaw: number;
  lostFrames?: number;
}
