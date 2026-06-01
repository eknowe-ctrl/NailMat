import { NavLink } from 'react-router-dom'
import styles from './Header.module.css'

const NAV = [
  { to: '/visualizer', label: 'Визуализатор' },
  { to: '/calculator', label: 'Калькулятор' },
  { to: '/catalog',    label: 'Каталог' },
]

export default function Header() {
  return (
    <header className={styles.header}>
      <div className={`container ${styles.inner}`}>
        <NavLink to="/" className={styles.logo}>
          <svg width="28" height="28" viewBox="0 0 32 32" fill="none">
            <rect width="32" height="32" rx="8" fill="#1c1c28"/>
            <ellipse cx="16" cy="21" rx="5" ry="7" fill="#B06EFF"/>
            <ellipse cx="16" cy="15" rx="5" ry="3.5" fill="#7C3AED"/>
          </svg>
          NailMat
        </NavLink>
        <nav className={styles.nav}>
          {NAV.map(({ to, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                isActive ? `${styles.link} ${styles.active}` : styles.link
              }
            >
              {label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}
