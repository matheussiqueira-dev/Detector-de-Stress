import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "AI Face Behavior Dashboard por Matheus Siqueira";
export const size = {
  width: 1200,
  height: 630,
};
export const contentType = "image/png";

export default function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          background: "#050506",
          color: "#f4f4f5",
          fontFamily: "Arial, sans-serif",
          padding: 72,
          position: "relative",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 0,
            background:
              "radial-gradient(circle at 82% 18%, rgba(34, 211, 238, 0.22), transparent 32%), radial-gradient(circle at 8% 92%, rgba(52, 211, 153, 0.16), transparent 30%)",
          }}
        />
        <div style={{ display: "flex", flexDirection: "column", justifyContent: "center", width: 680, zIndex: 1 }}>
          <div style={{ color: "#67e8f9", fontSize: 28, letterSpacing: 6, textTransform: "uppercase" }}>
            Portfolio computer vision app
          </div>
          <div style={{ marginTop: 28, fontSize: 78, fontWeight: 800, lineHeight: 0.95 }}>
            AI Face Behavior Dashboard
          </div>
          <div style={{ marginTop: 32, color: "#d4d4d8", fontSize: 28, lineHeight: 1.35 }}>
            Webcam, face tracking, metricas em tempo real e dashboard inteligente.
          </div>
          <div style={{ marginTop: 58, color: "#67e8f9", fontSize: 28 }}>Desenvolvido por Matheus Siqueira</div>
          <div style={{ marginTop: 10, color: "#d4d4d8", fontSize: 24 }}>www.matheussiqueira.dev</div>
        </div>
        <div
          style={{
            zIndex: 1,
            marginLeft: "auto",
            width: 330,
            height: 390,
            alignSelf: "center",
            borderRadius: 32,
            border: "8px solid #22d3ee",
            background: "#08090a",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 0 60px rgba(34, 211, 238, 0.28)",
          }}
        >
          <div
            style={{
              width: 190,
              height: 230,
              borderRadius: 60,
              border: "8px solid #34d399",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#e0f2fe",
              fontSize: 44,
            }}
          >
            FACE
          </div>
        </div>
      </div>
    ),
    size,
  );
}
