import styles from "./Footer.module.css";

export function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        <p className={styles.text}>
          Desenvolvido por{" "}
          <a
            className={styles.name}
            href="https://www.matheussiqueira.dev/"
            target="_blank"
            rel="noreferrer"
          >
            Matheus Siqueira
          </a>
        </p>
        <a
          className={styles.site}
          href="https://www.matheussiqueira.dev/"
          target="_blank"
          rel="noreferrer"
          aria-label="Abrir site de Matheus Siqueira"
        >
          www.matheussiqueira.dev
        </a>
      </div>
    </footer>
  );
}
