import type { Metadata } from "next";
import { DashboardClient } from "@/components/dashboard/DashboardClient";
import { mockMetrics, stateDistribution } from "@/data/mockMetrics";
import { eventDistribution, mockEvents } from "@/data/mockEvents";
import { dashboardKpis, mockSession } from "@/data/mockSessions";

export const metadata: Metadata = {
  title: "Dashboard inteligente",
  description: "KPIs, graficos, eventos, alertas e insights acionaveis para a sessao de face behavior analytics.",
  alternates: {
    canonical: "/dashboard",
  },
};

export default function DashboardPage() {
  return (
    <DashboardClient
      metrics={mockMetrics}
      events={mockEvents}
      session={mockSession}
      kpis={dashboardKpis}
      eventDistribution={eventDistribution}
      stateDistribution={stateDistribution}
    />
  );
}
