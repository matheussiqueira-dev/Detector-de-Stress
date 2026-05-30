"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { BehaviorEvent, BehaviorEventType } from "@/types/events";
import type { MetricSample } from "@/types/metrics";
import type { FaceTrackingResult } from "@/types/vision";
import { calculateAttentionScore, getAnalysisState } from "@/lib/scoring";
import { createBehaviorEvent, deriveEventFromMetric } from "@/lib/events";
import { average, createId, formatTimeLabel, round } from "@/lib/utils";

interface UseSessionMetricsOptions {
  tracking: FaceTrackingResult;
  running: boolean;
  paused: boolean;
}

export function useSessionMetrics({ tracking, running, paused }: UseSessionMetricsOptions) {
  const [samples, setSamples] = useState<MetricSample[]>([]);
  const [events, setEvents] = useState<BehaviorEvent[]>([]);
  const lastSampleAtRef = useRef(0);
  const lostFramesRef = useRef(0);
  const lastEventAtRef = useRef<Record<BehaviorEventType, number>>({} as Record<BehaviorEventType, number>);

  const pushEvent = useCallback((event: BehaviorEvent) => {
    setEvents((current) => [event, ...current].slice(0, 80));
  }, []);

  const registerEvent = useCallback(
    (type: BehaviorEventType, severity: BehaviorEvent["severity"] = "info", metadata: BehaviorEvent["metadata"] = {}) => {
      pushEvent(createBehaviorEvent(type, severity, metadata));
    },
    [pushEvent],
  );

  const resetSession = useCallback(() => {
    setSamples([]);
    setEvents([createBehaviorEvent("CAMERA_STARTED", "info", { reset: true })]);
    lastSampleAtRef.current = 0;
    lostFramesRef.current = 0;
    lastEventAtRef.current = {} as Record<BehaviorEventType, number>;
  }, []);

  useEffect(() => {
    if (!running || paused) {
      return;
    }

    const now = performance.now();

    if (now - lastSampleAtRef.current < 450) {
      return;
    }

    lastSampleAtRef.current = now;
    lostFramesRef.current = tracking.faceDetected ? 0 : lostFramesRef.current + 1;

    const scoreInput = {
      faceDetected: tracking.faceDetected,
      detectionConfidence: tracking.confidence,
      stability: tracking.stability,
      movement: tracking.movement,
      mouthOpen: tracking.mouthOpen,
      eyesClosed: tracking.eyesClosed,
      yaw: tracking.yaw,
      lostFrames: lostFramesRef.current,
    };
    const timestamp = new Date().toISOString();
    const sample: MetricSample = {
      id: createId("metric"),
      timestamp,
      label: formatTimeLabel(timestamp),
      attentionScore: calculateAttentionScore(scoreInput),
      detectionConfidence: tracking.confidence,
      faceDetected: tracking.faceDetected,
      movement: tracking.movement,
      fps: tracking.fps,
      latency: tracking.latency,
      stability: tracking.stability,
      mouthOpen: tracking.mouthOpen,
      eyesClosed: tracking.eyesClosed,
      yaw: tracking.yaw,
      state: getAnalysisState(scoreInput, paused),
      source: tracking.source,
    };

    setSamples((current) => [...current, sample].slice(-90));

    const derivedEvent = deriveEventFromMetric(sample);
    if (derivedEvent && canEmitEvent(derivedEvent.type, lastEventAtRef.current)) {
      pushEvent(derivedEvent);
    }
  }, [paused, pushEvent, running, tracking]);

  const currentMetric = samples.at(-1) ?? null;
  const averages = useMemo(
    () => ({
      attention: round(average(samples.map((sample) => sample.attentionScore))),
      confidence: round(average(samples.map((sample) => sample.detectionConfidence * 100))),
      fps: round(average(samples.map((sample) => sample.fps)), 1),
      latency: round(average(samples.map((sample) => sample.latency))),
      detectionRate: round(
        samples.length === 0 ? 0 : (samples.filter((sample) => sample.faceDetected).length / samples.length) * 100,
      ),
      stability: round(average(samples.map((sample) => sample.stability * 100))),
    }),
    [samples],
  );

  return {
    samples,
    events,
    currentMetric,
    averages,
    registerEvent,
    resetSession,
  };
}

function canEmitEvent(type: BehaviorEventType, cache: Record<BehaviorEventType, number>) {
  const now = performance.now();
  const last = cache[type] ?? 0;

  if (last > 0 && now - last < 3_000) {
    return false;
  }

  cache[type] = now;
  return true;
}
