/**
 * AlertBanner — Exibe alertas visuais baseados no nível de stress detectado.
 *
 * Renderiza um banner temático ENCOM com severidade (info, warning, critical)
 * e suporte a auto-dismiss configurável.
 *
 * @author Matheus Siqueira <https://www.matheussiqueira.dev/>
 */
"use client";

import { useEffect, useState } from "react";

import styles from "./AlertBanner.module.css";

export type AlertSeverity = "info" | "warning" | "critical";

export type Alert = {
  id: string;
  severity: AlertSeverity;
  message: string;
  /** Se definido, o banner se dispensa automaticamente após esse período (ms). */
  autoDismissMs?: number;
};

type AlertBannerProps = {
  alerts: Alert[];
  onDismiss: (id: string) => void;
};

function SeverityIcon({ severity }: { severity: AlertSeverity }) {
  if (severity === "critical") return <span aria-hidden="true">⚠</span>;
  if (severity === "warning") return <span aria-hidden="true">◈</span>;
  return <span aria-hidden="true">◉</span>;
}

function SingleAlert({
  alert,
  onDismiss,
}: {
  alert: Alert;
  onDismiss: (id: string) => void;
}) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    if (!alert.autoDismissMs) return;
    const timer = setTimeout(() => {
      setVisible(false);
      onDismiss(alert.id);
    }, alert.autoDismissMs);
    return () => clearTimeout(timer);
  }, [alert.id, alert.autoDismissMs, onDismiss]);

  if (!visible) return null;

  return (
    <div
      className={`${styles.banner} ${styles[alert.severity]}`}
      role="alert"
      aria-live="assertive"
    >
      <span className={styles.icon}>
        <SeverityIcon severity={alert.severity} />
      </span>
      <span className={styles.message}>{alert.message}</span>
      <button
        className={styles.dismiss}
        onClick={() => {
          setVisible(false);
          onDismiss(alert.id);
        }}
        aria-label="Dispensar alerta"
      >
        ✕
      </button>
    </div>
  );
}

/**
 * Renderiza a pilha de alertas ativos do sistema.
 */
export function AlertBanner({ alerts, onDismiss }: AlertBannerProps) {
  if (alerts.length === 0) return null;

  return (
    <div className={styles.stack} aria-label="Alertas do sistema">
      {alerts.map((alert) => (
        <SingleAlert key={alert.id} alert={alert} onDismiss={onDismiss} />
      ))}
    </div>
  );
}
