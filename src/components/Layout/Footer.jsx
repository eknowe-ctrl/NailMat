import styles from './Footer.module.css'

export default function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={`container ${styles.inner}`}>
        <span className={styles.brand}>NailMat</span>
        <span className={styles.copy}>© {new Date().getFullYear()} — инструменты для мастеров маникюра</span>
      </div>
    </footer>
  )
}
