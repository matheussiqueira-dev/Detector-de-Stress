import type { Metadata } from "next";
import { RealtimeDemo } from "@/components/webcam/RealtimeDemo";

export const metadata: Metadata = {
  title: "Demo em tempo real",
  description: "Webcam, deteccao facial, overlay dinamico, metricas e alertas em tempo real no navegador.",
  alternates: {
    canonical: "/demo",
  },
};

export default function DemoPage() {
  return <RealtimeDemo />;
}
