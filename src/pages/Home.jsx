import { Link } from 'react-router-dom'
import styles from './Home.module.css'

const TOOLS = [
  {
    to: '/visualizer',
    icon: '💅',
    tag: 'Интерактив',
    title: 'Визуализатор',
    desc: 'Примерьте форму ногтей, цвет лака и дизайн — и отправьте готовый референс мастеру.',
  },
  {
    to: '/calculator',
    icon: '🧮',
    tag: 'Финансы',
    title: 'Калькулятор',
    desc: 'Посчитайте себестоимость процедуры, прибыль и рекомендуемую цену за услугу.',
  },
  {
    to: '/catalog',
    icon: '🗂️',
    tag: 'Каталог',
    title: 'Где купить',
    desc: 'Официальные дистрибьюторы брендов Luxio, Uno, Grattol и других — по городам России.',
  },
]

export default function Home() {
  return (
    <div className={styles.page}>
      {/* Hero */}
      <section className={styles.hero}>
        <div className="container">
          <p className="tag">Для мастеров маникюра</p>
          <h1 className={styles.heroTitle}>
            Инструменты,<br />
            которых не хватало
          </h1>
          <p className={styles.heroSub}>
            Визуализатор дизайна, калькулятор прибыли и каталог
            дистрибьюторов — в одном месте.
          </p>
          <div className={styles.heroCta}>
            <Link to="/visualizer" className="btn-primary">Попробовать</Link>
            <Link to="/calculator" className="btn-outline">Считать прибыль</Link>
          </div>
        </div>
      </section>

      {/* Ad banner top */}
      <div className="container">
        <div className={`ad-banner ${styles.adTop}`}>РЕКЛАМА</div>
      </div>

      {/* Tools grid */}
      <section className={styles.tools}>
        <div className="container">
          <h2 className={styles.sectionTitle}>Инструменты</h2>
          <div className={styles.grid}>
            {TOOLS.map(({ to, icon, tag, title, desc }) => (
              <Link key={to} to={to} className={styles.toolCard}>
                <span className={styles.toolIcon}>{icon}</span>
                <span className="tag">{tag}</span>
                <h3 className={styles.toolTitle}>{title}</h3>
                <p className={styles.toolDesc}>{desc}</p>
                <span className={styles.toolArrow}>→</span>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}
