import type { ReactNode } from "react";
import type { Metadata } from "next";
import { Exo_2, Orbitron, Rajdhani } from "next/font/google";

import "./globals.css";
import { BackgroundGrid } from "@/components/system/BackgroundGrid";
import { Footer } from "@/components/system/Footer";
import { WhatsAppButton } from "@/components/system/WhatsAppButton";

const orbitron = Orbitron({
  subsets: ["latin"],
  variable: "--font-heading",
  weight: ["700", "800"],
  display: "swap",
});

const rajdhani = Rajdhani({
  subsets: ["latin"],
  variable: "--font-ui",
  weight: ["500", "600", "700"],
  display: "swap",
});

const exo2 = Exo_2({
  subsets: ["latin"],
  variable: "--font-body",
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Stress Detector | ENCOM Interface",
  description: "Tron: Legacy inspired ENCOM dashboard for real-time stress monitoring.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="pt-BR">
      <body className={`${orbitron.variable} ${rajdhani.variable} ${exo2.variable}`}>
        <div className="encom-app-shell">
          <BackgroundGrid />
          <main className="encom-main">{children}</main>
          <Footer />
          <WhatsAppButton />
        </div>
      </body>
    </html>
  );
}
