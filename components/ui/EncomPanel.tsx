import type { HTMLAttributes, ReactNode } from "react";

import styles from "./EncomPanel.module.css";

type EncomPanelProps = HTMLAttributes<HTMLDivElement> & {
  eyebrow?: string;
  title?: string;
  children: ReactNode;
};

export function EncomPanel({
  eyebrow,
  title,
  children,
  className,
  ...props
}: EncomPanelProps) {
  return (
    <section className={[styles.panel, className].filter(Boolean).join(" ")} {...props}>
      {(eyebrow || title) && (
        <header className={styles.header}>
          {eyebrow ? <span className={`${styles.eyebrow} encom-label`}>{eyebrow}</span> : null}
          {title ? <h2 className={`${styles.title} encom-heading`}>{title}</h2> : null}
        </header>
      )}
      <div className={styles.content}>{children}</div>
    </section>
  );
}
