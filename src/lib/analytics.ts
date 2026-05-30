"use client";

import { track } from "@vercel/analytics";

type AnalyticsEvent =
  | "demo_started"
  | "camera_permission_granted"
  | "camera_permission_denied"
  | "dashboard_opened"
  | "attention_alert_triggered"
  | "face_lost"
  | "session_reset";

export function trackAppEvent(event: AnalyticsEvent, properties?: Record<string, string | number | boolean>) {
  try {
    track(event, properties);
  } catch {
    // Analytics must never break the local computer-vision demo.
  }
}
