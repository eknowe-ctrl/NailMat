import { useState, useMemo } from 'react'
import styles from './Calculator.module.css'

const FIELDS = [
  { id: 'base',     label: 'База',                 unit: '₽', hint: 'Стоимость базового покрытия' },
  { id: 'top',      label: 'Топ',                  unit: '₽', hint: 'Стоимость топового покрытия' },
  { id: 'gel',      label: 'Гель-лак / лак',       unit: '₽', hint: 'Один флакон за процедуру' },
  { id: 'file',     label: 'Пилочка / расходники', unit: '₽', hint: 'Пилочки, апельсиновые палочки и т.д.' },
  { id: 'gloves',   label: 'Перчатки',             unit: '₽', hint: 'Пара перчаток' },
  { id: 'rent',     label: 'Аренда (за час)',       unit: '₽', hint: 'Стоимость аренды места в час' },
  { id: 'duration', label: 'Длительность процедуры', unit: 'ч', hint: 'В часах, например 1.5' },
  { id: 'salary',   label: 'Ваш час работы',       unit: '₽', hint: 'Сколько стоит ваш час времени' },
  { id: 'markup',   label: 'Желаемая наценка',      unit: '%', hint: 'Наценка поверх себестоимости' },
]

const DEFAULTS = {
  base: 120, top: 90, gel: 200, file: 40,
  gloves: 15, rent: 200, duration: 2,
  salary: 500, markup: 40,
}

function fmt(n) {
  return n.toLocaleString('ru-RU', { maximumFractionDigits: 0 })
}

export default function Calculator() {
  const [vals, setVals] = useState(DEFAULTS)

  const set = (id, raw) => {
    const v = parseFloat(raw) || 0
    setVals(prev => ({ ...prev, [id]: v }))
  }

  const result = useMemo(() => {
    const materials = vals.base + vals.top + vals.gel + vals.file + vals.gloves
    const rentCost  = vals.rent * vals.duration
    const laborCost = vals.salary * vals.duration
    const cost      = materials + rentCost + laborCost
    const price     = cost * (1 + vals.markup / 100)
    const profit    = price - cost
    const margin    = price > 0 ? (profit / price) * 100 : 0
    return { materials, rentCost, laborCost, cost, price, profit, margin }
  }, [vals])

  const profitClass = result.profit > 0 ? styles.positive : styles.negative

  return (
    <div className="page">
      <div className="container">
        <div className={styles.pageHeader}>
          <span className="tag">Калькулятор</span>
          <h1 className={styles.title}>Себестоимость процедуры</h1>
          <p className={styles.sub}>Введите затраты — получите чистую прибыль и рекомендуемую цену.</p>
        </div>

        <div className={styles.layout}>
          {/* Inputs */}
          <div className={styles.form}>
            {FIELDS.map(({ id, label, unit, hint }) => (
              <div key={id} className={`card ${styles.fieldCard}`}>
                <label className={styles.fieldLabel}>
                  {label}
                  <span className={styles.fieldUnit}>{unit}</span>
                </label>
                <p className={styles.fieldHint}>{hint}</p>
                <input
                  type="number"
                  min="0"
                  step={id === 'duration' ? '0.5' : '1'}
                  value={vals[id]}
                  onChange={e => set(id, e.target.value)}
                />
              </div>
            ))}
          </div>

          {/* Result panel */}
          <div className={styles.resultPanel}>
            <div className={`card ${styles.resultCard}`}>
              <h2 className={styles.resultTitle}>Результат</h2>

              <div className={styles.breakdown}>
                <div className={styles.breakdownRow}>
                  <span>Материалы</span>
                  <span>{fmt(result.materials)} ₽</span>
                </div>
                <div className={styles.breakdownRow}>
                  <span>Аренда</span>
                  <span>{fmt(result.rentCost)} ₽</span>
                </div>
                <div className={styles.breakdownRow}>
                  <span>Оплата труда</span>
                  <span>{fmt(result.laborCost)} ₽</span>
                </div>
                <div className={`${styles.breakdownRow} ${styles.breakdownTotal}`}>
                  <span>Себестоимость</span>
                  <span>{fmt(result.cost)} ₽</span>
                </div>
              </div>

              <div className={styles.highlight}>
                <p className={styles.highlightLabel}>Рекомендуемая цена</p>
                <p className={styles.highlightValue}>{fmt(result.price)} ₽</p>
              </div>

              <div className={styles.metrics}>
                <div className={styles.metric}>
                  <p className={styles.metricLabel}>Прибыль</p>
                  <p className={`${styles.metricValue} ${profitClass}`}>{fmt(result.profit)} ₽</p>
                </div>
                <div className={styles.metric}>
                  <p className={styles.metricLabel}>Маржа</p>
                  <p className={`${styles.metricValue} ${profitClass}`}>{result.margin.toFixed(1)}%</p>
                </div>
              </div>

              {/* Visual bar */}
              <div className={styles.barWrap}>
                <div
                  className={styles.barFill}
                  style={{ width: `${Math.min(result.margin, 100)}%` }}
                />
                <span className={styles.barLabel}>Маржа {result.margin.toFixed(1)}%</span>
              </div>

              <p className={styles.tip}>
                {result.margin < 20
                  ? '⚠️ Маржа ниже 20% — стоит пересмотреть цену или сократить затраты.'
                  : result.margin < 40
                  ? '✓ Неплохой результат. Можно работать.'
                  : '🔥 Отличная маржа! Цена выставлена грамотно.'}
              </p>

              <button
                className="btn-outline"
                style={{ width: '100%', marginTop: 8 }}
                onClick={() => {
                  const text = `Себестоимость: ${fmt(result.cost)} ₽ | Цена: ${fmt(result.price)} ₽ | Прибыль: ${fmt(result.profit)} ₽`
                  navigator.clipboard.writeText(text)
                }}
              >
                Скопировать расчёт
              </button>
            </div>

            <div className={`ad-banner ${styles.ad}`}>РЕКЛАМА</div>
          </div>
        </div>
      </div>
    </div>
  )
}
