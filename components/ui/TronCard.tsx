import type { ReactNode } from "react";

import { EncomPanel } from "./EncomPanel";
import styles from "./TronCard.module.css";

type TronCardProps = {
  title: string;
  eyebrow: string;
  value: string;
  detail: string;
  thumbnail?: ReactNode;
};

export function TronCard({ title, eyebrow, value, detail, thumbnail }: TronCardProps) {
  return (
    <EncomPanel className={styles.card} eyebrow={eyebrow} title={title}>
      <div className={styles.body}>
        <div className={styles.thumbnail}>{thumbnail}</div>
        <div className={styles.info}>
          <strong className={styles.value}>{value}</strong>
          <p className={styles.detail}>{detail}</p>
        </div>
      </div>
    </EncomPanel>
  );
}
