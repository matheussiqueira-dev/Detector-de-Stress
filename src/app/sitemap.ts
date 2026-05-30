import type { MetadataRoute } from "next";

const baseUrl = "https://detector-de-stress.vercel.app";

export default function sitemap(): MetadataRoute.Sitemap {
  return ["", "/demo", "/dashboard", "/about"].map((path) => ({
    url: `${baseUrl}${path}`,
    lastModified: new Date("2026-05-30T00:00:00-03:00"),
    changeFrequency: "weekly",
    priority: path === "" ? 1 : 0.8,
  }));
}
