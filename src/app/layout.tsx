import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import { AppShell } from "@/components/layout/AppShell";
import { PwaRegister } from "@/components/layout/PwaRegister";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://detector-de-stress.vercel.app"),
  title: {
    default: "AI Face Behavior Dashboard | Matheus Siqueira",
    template: "%s | AI Face Behavior Dashboard",
  },
  description:
    "Web app profissional de visao computacional no navegador com webcam, face tracking, metricas de atencao, alertas e dashboard inteligente.",
  applicationName: "AI Face Behavior Dashboard",
  authors: [{ name: "Matheus Siqueira", url: "https://www.matheussiqueira.dev" }],
  creator: "Matheus Siqueira",
  publisher: "Matheus Siqueira",
  alternates: {
    canonical: "/",
  },
  icons: {
    icon: "/icon.svg",
    apple: "/apple-icon.svg",
  },
  manifest: "/manifest.json",
  openGraph: {
    type: "website",
    locale: "pt_BR",
    url: "/",
    siteName: "AI Face Behavior Dashboard",
    title: "AI Face Behavior Dashboard",
    description:
      "Demo local-first de webcam, deteccao facial, indicadores comportamentais e dashboard analitico para portfolio.",
    images: [
      {
        url: "/opengraph-image",
        width: 1200,
        height: 630,
        alt: "AI Face Behavior Dashboard por Matheus Siqueira",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "AI Face Behavior Dashboard",
    description: "Visao computacional no navegador com dashboard inteligente e foco em privacidade.",
    images: ["/opengraph-image"],
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="pt-BR" className={`${geistSans.variable} ${geistMono.variable} h-full`}>
      <body className="min-h-full antialiased">
        <PwaRegister />
        <AppShell>{children}</AppShell>
        <Analytics />
      </body>
    </html>
  );
}
