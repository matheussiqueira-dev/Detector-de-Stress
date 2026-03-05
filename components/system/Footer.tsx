import styles from "./Footer.module.css";

export function Footer() {
  return (
    <footer className={styles.footer}>
      <p className={styles.text}>
        Desenvolvido por{" "}
        <a href="https://www.matheussiqueira.dev/" target="_blank" rel="noreferrer">
          Matheus Siqueira
        </a>
      </p>
    </footer>
  );
}
