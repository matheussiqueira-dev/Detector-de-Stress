import styles from "./WhatsAppButton.module.css";

export function WhatsAppButton() {
  return (
    <a
      aria-label="Abrir conversa no WhatsApp"
      className={styles.button}
      href="https://wa.me/5581999203683"
      rel="noreferrer"
      target="_blank"
    >
      <span className={styles.label}>WA</span>
    </a>
  );
}
