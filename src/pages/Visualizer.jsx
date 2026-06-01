import { useState, useCallback } from 'react'
import styles from './Visualizer.module.css'

// Nail shape path generators (relative to each finger group origin)
const SHAPES = {
  round: (w, h) => `M${w*0.1},${h} Q0,${h*0.5} ${w*0.5},0 Q${w},${h*0.5} ${w*0.9},${h} Z`,
  square: (w, h) => `M0,${h} L0,${h*0.1} L${w},${h*0.1} L${w},${h} Z`,
  almond: (w, h) => `M${w*0.05},${h} Q${w*0.1},${h*0.6} ${w*0.5},0 Q${w*0.9},${h*0.6} ${w*0.95},${h} Z`,
  stiletto: (w, h) => `M${w*0.1},${h} Q${w*0.2},${h*0.7} ${w*0.5},0 Q${w*0.8},${h*0.7} ${w*0.9},${h} Z`,
  coffin: (w, h) => `M${w*0.05},${h} L${w*0.2},${h*0.15} L${w*0.8},${h*0.15} L${w*0.95},${h} Z`,
}

// Each finger: cx = center-x of nail, cy = top-y of nail, w, h = nail size
const FINGERS = [
  { id: 'thumb',  cx: 62,  cy: 175, w: 34, h: 52, angle: -28 },
  { id: 'index',  cx: 128, cy: 88,  w: 30, h: 58, angle: -5  },
  { id: 'middle', cx: 172, cy: 72,  w: 30, h: 62, angle: 1   },
  { id: 'ring',   cx: 216, cy: 84,  w: 28, h: 58, angle: 6   },
  { id: 'pinky',  cx: 254, cy: 108, w: 24, h: 48, angle: 12  },
]

const PALETTE = [
  '#FADADD','#F4A7B9','#E8A0BF','#C780C8','#9B59B6',
  '#7C3AED','#B06EFF','#5C6BC0','#42A5F5','#26C6DA',
  '#66BB6A','#D4E157','#FFD54F','#FFA726','#FF7043',
  '#8D6E63','#BDBDBD','#F5F5F5','#1A1A2E','#2d2d2d',
  '#FF3D7F','#FF6E7F','#C0392B','#E74C3C','#ECF0F1',
]

const FINISHES = ['Глянцевый', 'Матовый']

const DESIGNS = [
  { id: 'plain',  label: 'Однотонный' },
  { id: 'french', label: 'Френч'      },
  { id: 'foil',   label: 'Втирка'     },
  { id: 'luna',   label: 'Лунки'      },
]

function NailSVG({ shape, color, finish, design }) {
  return (
    <svg width="300" height="280" viewBox="0 0 300 280" className={styles.handSvg}>
      {/* Palm silhouette */}
      <path
        d="M55,260 Q30,240 40,190 L48,160 Q50,140 60,130 L62,220
           Q70,210 80,200 L88,120 Q90,100 100,95 L108,185
           Q118,170 125,160 L130,95 Q132,75 145,72 L148,155
           Q158,140 165,130 L168,72 Q170,60 183,62 L186,155
           Q196,145 204,138 L204,85 Q206,65 218,68 L220,155
           Q228,148 234,143 L236,105 Q238,88 250,92 L256,155
           Q270,158 275,175 Q285,205 270,240 Q250,268 200,272
           L100,272 Q65,270 55,260 Z"
        fill="#f4c5a0"
        stroke="#e0a882"
        strokeWidth="1.5"
      />

      {/* Finger highlights */}
      {FINGERS.map(f => (
        <ellipse
          key={f.id + '-hi'}
          cx={f.cx}
          cy={f.cy + f.h + 18}
          rx={f.w * 0.42}
          ry={f.h * 0.18}
          fill="rgba(255,255,255,0.08)"
        />
      ))}

      {/* Nails */}
      {FINGERS.map(finger => {
        const { id, cx, cy, w, h, angle } = finger
        const x = cx - w / 2
        const pathD = SHAPES[shape]?.(w, h) ?? SHAPES.round(w, h)
        const isGlossy = finish === 'Глянцевый'

        // Design overlay paths
        let designEl = null
        if (design === 'french') {
          designEl = (
            <path
              d={`M${w*0.05},${h*0.2} Q${w*0.5},-${h*0.08} ${w*0.95},${h*0.2}`}
              fill="none"
              stroke="rgba(255,255,255,0.85)"
              strokeWidth="4"
              strokeLinecap="round"
              clipPath={`url(#clip-${id})`}
            />
          )
        } else if (design === 'foil') {
          designEl = (
            <rect x={0} y={0} width={w} height={h}
              fill="url(#foilGrad)"
              opacity="0.45"
              clipPath={`url(#clip-${id})`}
            />
          )
        } else if (design === 'luna') {
          designEl = (
            <ellipse cx={w/2} cy={h*0.82} rx={w*0.32} ry={h*0.16}
              fill="rgba(255,255,255,0.22)"
              clipPath={`url(#clip-${id})`}
            />
          )
        }

        return (
          <g key={id} transform={`rotate(${angle}, ${cx}, ${cy + h})`}>
            <defs>
              <clipPath id={`clip-${id}`}>
                <path d={pathD} transform={`translate(${x},${cy})`}/>
              </clipPath>
            </defs>
            {/* Base nail */}
            <path
              d={pathD}
              transform={`translate(${x},${cy})`}
              fill={color}
              stroke="rgba(0,0,0,0.25)"
              strokeWidth="1"
              style={{ filter: isGlossy ? 'none' : 'contrast(0.9) saturate(0.85)' }}
            />
            {/* Design */}
            {designEl && (
              <g transform={`translate(${x},${cy})`}>{designEl}</g>
            )}
            {/* Gloss shine */}
            {isGlossy && (
              <ellipse
                cx={cx - w * 0.12}
                cy={cy + h * 0.22}
                rx={w * 0.18}
                ry={h * 0.12}
                fill="rgba(255,255,255,0.35)"
                style={{ pointerEvents: 'none' }}
              />
            )}
            {/* Matte texture overlay */}
            {!isGlossy && (
              <path
                d={pathD}
                transform={`translate(${x},${cy})`}
                fill="url(#matteGrad)"
                opacity="0.18"
                style={{ pointerEvents: 'none' }}
              />
            )}
          </g>
        )
      })}

      <defs>
        <linearGradient id="foilGrad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%"   stopColor="#FFD700"/>
          <stop offset="40%"  stopColor="#C0C0C0"/>
          <stop offset="70%"  stopColor="#B06EFF"/>
          <stop offset="100%" stopColor="#FFD700"/>
        </linearGradient>
        <linearGradient id="matteGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor="#fff" stopOpacity="0.12"/>
          <stop offset="100%" stopColor="#000" stopOpacity="0.08"/>
        </linearGradient>
      </defs>
    </svg>
  )
}

export default function Visualizer() {
  const [color,   setColor]   = useState('#B06EFF')
  const [shape,   setShape]   = useState('round')
  const [finish,  setFinish]  = useState('Глянцевый')
  const [design,  setDesign]  = useState('plain')

  const handleCopy = useCallback(() => {
    const text = `Форма: ${shape} | Цвет: ${color} | Покрытие: ${finish} | Дизайн: ${design}`
    navigator.clipboard.writeText(text)
  }, [shape, color, finish, design])

  return (
    <div className="page">
      <div className="container">
        <div className={styles.pageHeader}>
          <span className="tag">Визуализатор</span>
          <h1 className={styles.title}>Конструктор дизайна</h1>
          <p className={styles.sub}>Настройте форму, цвет и дизайн — получите референс для мастера.</p>
        </div>

        <div className={styles.layout}>
          {/* Preview */}
          <div className={`card ${styles.preview}`}>
            <NailSVG shape={shape} color={color} finish={finish} design={design} />
            <div className={styles.previewMeta}>
              <span>{DESIGNS.find(d => d.id === design)?.label}</span>
              <span style={{ color: 'var(--text-muted)' }}>·</span>
              <span>{finish}</span>
              <span style={{ color: 'var(--text-muted)' }}>·</span>
              <span style={{ textTransform: 'capitalize' }}>{shape}</span>
            </div>
            <button className="btn-primary" onClick={handleCopy} style={{ marginTop: 8, width: '100%' }}>
              Скопировать референс
            </button>
          </div>

          {/* Controls */}
          <div className={styles.controls}>
            {/* Shape */}
            <div className={`card ${styles.section}`}>
              <h3 className={styles.sectionLabel}>Форма ногтей</h3>
              <div className={styles.shapeGrid}>
                {Object.keys(SHAPES).map(s => (
                  <button
                    key={s}
                    onClick={() => setShape(s)}
                    className={`${styles.shapeBtn} ${shape === s ? styles.shapeBtnActive : ''}`}
                  >
                    <svg width="32" height="40" viewBox="0 0 32 40">
                      <path d={SHAPES[s](32, 40)} fill={shape === s ? 'var(--accent)' : 'var(--text-muted)'}/>
                    </svg>
                    <span>{s.charAt(0).toUpperCase() + s.slice(1)}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Finish */}
            <div className={`card ${styles.section}`}>
              <h3 className={styles.sectionLabel}>Покрытие</h3>
              <div className={styles.row}>
                {FINISHES.map(f => (
                  <button
                    key={f}
                    onClick={() => setFinish(f)}
                    className={`${styles.pill} ${finish === f ? styles.pillActive : ''}`}
                  >
                    {f}
                  </button>
                ))}
              </div>
            </div>

            {/* Design */}
            <div className={`card ${styles.section}`}>
              <h3 className={styles.sectionLabel}>Дизайн</h3>
              <div className={styles.row}>
                {DESIGNS.map(d => (
                  <button
                    key={d.id}
                    onClick={() => setDesign(d.id)}
                    className={`${styles.pill} ${design === d.id ? styles.pillActive : ''}`}
                  >
                    {d.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Palette */}
            <div className={`card ${styles.section}`}>
              <h3 className={styles.sectionLabel}>Цвет</h3>
              <div className={styles.palette}>
                {PALETTE.map(c => (
                  <button
                    key={c}
                    onClick={() => setColor(c)}
                    className={`${styles.swatch} ${color === c ? styles.swatchActive : ''}`}
                    style={{ background: c }}
                    title={c}
                  />
                ))}
                <label className={styles.swatchCustom} title="Свой цвет">
                  <input type="color" value={color} onChange={e => setColor(e.target.value)} />
                  +
                </label>
              </div>
            </div>
          </div>
        </div>

        {/* Ad */}
        <div className={`ad-banner ${styles.ad}`}>РЕКЛАМА</div>
      </div>
    </div>
  )
}
