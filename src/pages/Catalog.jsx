import { useState, useMemo } from 'react'
import styles from './Catalog.module.css'

const BRANDS = [
  {
    id: 'luxio',
    name: 'Luxio',
    logo: '💜',
    country: 'Канада',
    desc: 'Профессиональные гель-лаки с широкой палитрой. Популярны за стойкость и пигментацию.',
    distributors: [
      { city: 'Москва',          name: 'NailPro',      address: 'ул. Мясницкая, 15',        phone: '+7 495 123-45-67', url: '' },
      { city: 'Санкт-Петербург', name: 'BeautyHouse',  address: 'Невский пр., 88',           phone: '+7 812 234-56-78', url: '' },
      { city: 'Екатеринбург',    name: 'NailShop Ural', address: 'ул. Ленина, 24a',          phone: '+7 343 345-67-89', url: '' },
    ],
  },
  {
    id: 'uno',
    name: 'Uno',
    logo: '⚡',
    country: 'Россия',
    desc: 'Отечественный бренд с хорошим соотношением цена/качество. Базы и топы пользуются спросом.',
    distributors: [
      { city: 'Москва',          name: 'Cosmoprofi',   address: 'ул. Профсоюзная, 3',        phone: '+7 495 987-65-43', url: '' },
      { city: 'Краснодар',       name: 'NailWorld',    address: 'ул. Красная, 136',           phone: '+7 861 456-78-90', url: '' },
      { city: 'Новосибирск',     name: 'BeautyCity',   address: 'Красный пр., 52',            phone: '+7 383 567-89-01', url: '' },
    ],
  },
  {
    id: 'grattol',
    name: 'Grattol',
    logo: '✨',
    country: 'Россия',
    desc: 'Богатая линейка топов, баз, гель-лаков и дизайн-продуктов. Широко представлен в РФ.',
    distributors: [
      { city: 'Москва',          name: 'Nail Club',    address: 'Шоссе Энтузиастов, 31',     phone: '+7 495 111-22-33', url: '' },
      { city: 'Санкт-Петербург', name: 'ProNail SPb',  address: 'ул. Садовая, 20',            phone: '+7 812 444-55-66', url: '' },
      { city: 'Ростов-на-Дону',  name: 'NailDon',     address: 'пр. Ворошиловский, 10',      phone: '+7 863 678-90-12', url: '' },
    ],
  },
  {
    id: 'cnd',
    name: 'CND',
    logo: '🌟',
    country: 'США',
    desc: 'Shellac — один из первых гель-лаков на рынке. Профессиональный стандарт во многих салонах.',
    distributors: [
      { city: 'Москва',          name: 'Lena White',   address: 'ул. Тверская, 10',          phone: '+7 495 222-33-44', url: '' },
      { city: 'Санкт-Петербург', name: 'SalonPro',     address: 'Лиговский пр., 50',          phone: '+7 812 555-66-77', url: '' },
    ],
  },
  {
    id: 'neonail',
    name: 'NeoNail',
    logo: '🔮',
    country: 'Польша',
    desc: 'Доступный европейский бренд. Отличается трендовыми палитрами и хорошей доступностью.',
    distributors: [
      { city: 'Москва',          name: 'NeoNail Russia', address: 'ул. Полянка, 5',           phone: '+7 495 333-44-55', url: '' },
      { city: 'Казань',          name: 'NailCity KZN',  address: 'ул. Баумана, 44',            phone: '+7 843 789-01-23', url: '' },
      { city: 'Уфа',             name: 'BeautyUfa',    address: 'пр. Октября, 12',             phone: '+7 347 890-12-34', url: '' },
    ],
  },
  {
    id: 'irisk',
    name: 'Irisk',
    logo: '🌸',
    country: 'Россия',
    desc: 'Широкая аудитория среди начинающих мастеров. Доступные цены, большой ассортимент.',
    distributors: [
      { city: 'Москва',          name: 'Irisk Store',   address: 'ул. Новослободская, 3',    phone: '+7 495 444-55-66', url: '' },
      { city: 'Самара',          name: 'NailSamara',    address: 'ул. Молодогвардейская, 61', phone: '+7 846 901-23-45', url: '' },
    ],
  },
]

const CITIES = ['Все города', ...Array.from(new Set(BRANDS.flatMap(b => b.distributors.map(d => d.city)))).sort()]

export default function Catalog() {
  const [search,      setSearch]      = useState('')
  const [activeCity,  setActiveCity]  = useState('Все города')
  const [activeBrand, setActiveBrand] = useState(null)

  const filtered = useMemo(() => {
    return BRANDS.filter(b => {
      const matchSearch = b.name.toLowerCase().includes(search.toLowerCase()) ||
        b.desc.toLowerCase().includes(search.toLowerCase())
      const matchCity = activeCity === 'Все города' ||
        b.distributors.some(d => d.city === activeCity)
      return matchSearch && matchCity
    })
  }, [search, activeCity])

  const selected = activeBrand ? BRANDS.find(b => b.id === activeBrand) : null

  return (
    <div className="page">
      <div className="container">
        <div className={styles.pageHeader}>
          <span className="tag">Каталог</span>
          <h1 className={styles.title}>Где купить бренд</h1>
          <p className={styles.sub}>Официальные дистрибьюторы популярных брендов по городам России.</p>
        </div>

        {/* Filters */}
        <div className={styles.filters}>
          <input
            className={styles.search}
            type="search"
            placeholder="Поиск бренда…"
            value={search}
            onChange={e => { setSearch(e.target.value); setActiveBrand(null) }}
          />
          <div className={styles.cityRow}>
            {CITIES.map(city => (
              <button
                key={city}
                onClick={() => { setActiveCity(city); setActiveBrand(null) }}
                className={`${styles.cityPill} ${activeCity === city ? styles.cityActive : ''}`}
              >
                {city}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.layout}>
          {/* Brand cards */}
          <div className={styles.brandList}>
            {filtered.length === 0 && (
              <p className={styles.empty}>Ничего не найдено. Попробуйте другой запрос.</p>
            )}
            {filtered.map(brand => (
              <button
                key={brand.id}
                onClick={() => setActiveBrand(brand.id === activeBrand ? null : brand.id)}
                className={`card ${styles.brandCard} ${activeBrand === brand.id ? styles.brandActive : ''}`}
              >
                <div className={styles.brandHeader}>
                  <span className={styles.brandLogo}>{brand.logo}</span>
                  <div className={styles.brandInfo}>
                    <span className={styles.brandName}>{brand.name}</span>
                    <span className={styles.brandCountry}>{brand.country}</span>
                  </div>
                  <span className={styles.brandCount}>{brand.distributors.length} точки</span>
                </div>
                <p className={styles.brandDesc}>{brand.desc}</p>
              </button>
            ))}
          </div>

          {/* Distributor detail */}
          <div className={styles.detail}>
            {selected ? (
              <div className={`card ${styles.detailCard}`}>
                <div className={styles.detailHeader}>
                  <span style={{ fontSize: '2rem' }}>{selected.logo}</span>
                  <div>
                    <h2 className={styles.detailName}>{selected.name}</h2>
                    <span className={styles.brandCountry}>{selected.country}</span>
                  </div>
                </div>
                <p className={styles.brandDesc} style={{ marginBottom: 8 }}>{selected.desc}</p>
                <h3 className={styles.distTitle}>Дистрибьюторы</h3>
                <div className={styles.distList}>
                  {selected.distributors
                    .filter(d => activeCity === 'Все города' || d.city === activeCity)
                    .map((d, i) => (
                      <div key={i} className={styles.distItem}>
                        <div className={styles.distCity}>{d.city}</div>
                        <div className={styles.distName}>{d.name}</div>
                        <div className={styles.distAddress}>{d.address}</div>
                        <a href={`tel:${d.phone}`} className={styles.distPhone}>{d.phone}</a>
                      </div>
                    ))}
                </div>
              </div>
            ) : (
              <div className={`card ${styles.detailEmpty}`}>
                <span style={{ fontSize: '2.5rem' }}>🗂️</span>
                <p>Выберите бренд, чтобы увидеть дистрибьюторов</p>
              </div>
            )}
            <div className={`ad-banner ${styles.ad}`}>РЕКЛАМА</div>
          </div>
        </div>
      </div>
    </div>
  )
}
