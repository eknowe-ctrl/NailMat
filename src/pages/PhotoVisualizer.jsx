import { useState, useRef, useEffect, useCallback } from 'react'
import styles from './PhotoVisualizer.module.css'

const MP_CDN    = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18'
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

// t = fingertip, b = DIP joint (last knuckle — where the nail starts)
const NAIL_CFG = [
  { t: 4,  b: 3  }, // thumb
  { t: 8,  b: 7  }, // index
  { t: 12, b: 11 }, // middle
  { t: 16, b: 15 }, // ring
  { t: 20, b: 19 }, // pinky
]

// Width fractions relative to MCP knuckle span (lm5→lm17).
// MCP joints are stable regardless of finger spread → reliable width reference.
// Empirical ratios: adult nail width / hand knuckle span.
const NAIL_W_FRACS = [0.20, 0.16, 0.18, 0.16, 0.12]

function computeNailWidths(lms, W, H) {
  const px = i => [lms[i].x * W, lms[i].y * H]
  const span = Math.hypot(px(5)[0]-px(17)[0], px(5)[1]-px(17)[1])
  return NAIL_W_FRACS.map(f => span * f)
}

const PALETTE = [
  '#FADADD','#F4A7B9','#E8A0BF','#C780C8','#9B59B6',
  '#7C3AED','#B06EFF','#5C6BC0','#42A5F5','#26C6DA',
  '#66BB6A','#FFD54F','#FFA726','#FF7043','#8D6E63',
  '#BDBDBD','#F5F5F5','#1A1A2E','#FF3D7F','#E74C3C',
]

const SHAPES   = ['round','square','almond','stiletto','coffin']
const FINISHES = ['Глянцевый','Матовый']
const DESIGNS  = [
  { id:'plain',  l:'Однотонный' },
  { id:'french', l:'Френч' },
  { id:'foil',   l:'Втирка' },
  { id:'luna',   l:'Лунки' },
]

// ─── Nail shape paths ─────────────────────────────────────────────────────────
// Local coords: -Y = free edge (fingertip side), +Y = cuticle (palm side)

function buildPath(shape, w, h) {
  const p = new Path2D(), hw = w / 2, hh = h / 2
  switch (shape) {
    case 'square':
      p.roundRect(-hw, -hh, w, h, [hw * 0.12, hw * 0.12, hw * 0.12, hw * 0.12])
      break
    case 'almond':
      p.moveTo(0, -hh)
      p.bezierCurveTo( hw * .58, -hh * .5,  hw * .9,  -hh * .02,  hw * .88, hh)
      p.quadraticCurveTo(0, hh * 1.06, -hw * .88, hh)
      p.bezierCurveTo(-hw * .9,  -hh * .02, -hw * .58, -hh * .5,  0, -hh)
      p.closePath()
      break
    case 'stiletto':
      p.moveTo(0, -hh)
      p.bezierCurveTo( hw * .3,  -hh * .22,  hw * .8,  hh * .28,  hw * .86, hh)
      p.quadraticCurveTo(0, hh * 1.05, -hw * .86, hh)
      p.bezierCurveTo(-hw * .8,  hh * .28, -hw * .3,  -hh * .22, 0, -hh)
      p.closePath()
      break
    case 'coffin':
      p.moveTo(-hw * .4, -hh)
      p.lineTo( hw * .4, -hh)
      p.bezierCurveTo( hw * .72, -hh * .35,  hw * .88, hh * .1,  hw * .86, hh)
      p.quadraticCurveTo(0, hh * 1.04, -hw * .86, hh)
      p.bezierCurveTo(-hw * .88, hh * .1, -hw * .72, -hh * .35, -hw * .4, -hh)
      p.closePath()
      break
    default: // round
      p.roundRect(-hw, -hh, w, h, [hw * .72, hw * .72, hw * .18, hw * .18])
  }
  return p
}

// ─── Nail renderer ────────────────────────────────────────────────────────────

function drawNail(ctx, tip, dip, nailW, s, W, H) {
  const tx = tip.x * W, ty = tip.y * H
  const bx = dip.x * W, by = dip.y * H
  const dx = tx - bx, dy = ty - by
  const segLen = Math.hypot(dx, dy)
  if (segLen < 4) return

  // Nail height: DIP→TIP covers exactly the distal phalanx where the nail lives.
  // Factor 0.90 keeps the nail within that segment; free edge aligns with fingertip.
  const nH = segLen * 0.92
  const nW = nailW           // width comes from MCP-span reference
  const hh = nH / 2, hw = nW / 2
  const angle = Math.atan2(dy, dx)

  // cx = free edge at ~fingertip, cuticle at ~DIP
  // With nH=0.92*segLen and center at 0.54*tip+0.46*dip:
  //   free_edge = center + dir*hH ≈ tip + 0.08*segLen past tip  (tiny overhang)
  //   cuticle   = center - dir*hH ≈ dip + 0.08*segLen toward tip (near DIP)
  const cx = tx * 0.54 + bx * 0.46
  const cy = ty * 0.54 + by * 0.46

  const path = buildPath(s.shape, nW, nH)

  ctx.save()
  ctx.translate(cx, cy)
  ctx.rotate(angle + Math.PI / 2)

  // ── 1. Base color ──
  ctx.globalAlpha = s.opacity
  ctx.fillStyle   = s.color
  ctx.fill(path)

  // ── 2. Design overlay ──
  ctx.save()
  ctx.clip(path)

  if (s.design === 'french') {
    // White arch near free edge (-Y side)
    ctx.globalAlpha = s.opacity * .93
    ctx.fillStyle   = 'rgba(255,251,246,.94)'
    ctx.beginPath()
    ctx.ellipse(0, -hh * .7, hw * .92, hh * .23, 0, 0, Math.PI * 2)
    ctx.fill()
  } else if (s.design === 'foil') {
    const g = ctx.createLinearGradient(-hw, -hh, hw, hh)
    g.addColorStop(0,   'rgba(255,215,0,.62)')
    g.addColorStop(.33, 'rgba(210,210,210,.55)')
    g.addColorStop(.66, 'rgba(176,110,255,.62)')
    g.addColorStop(1,   'rgba(255,215,0,.52)')
    ctx.globalAlpha = s.opacity * .82
    ctx.fillStyle   = g
    ctx.fill(path)
  } else if (s.design === 'luna') {
    // Half-moon near cuticle (+Y side)
    ctx.globalAlpha = s.opacity * .5
    ctx.fillStyle   = 'rgba(255,255,255,.45)'
    ctx.beginPath()
    ctx.ellipse(0, hh * .62, hw * .55, hh * .2, 0, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.restore()

  // ── 3. 3-D edge shading ──
  ctx.save()
  ctx.clip(path)

  // Side shadows — makes nail look cylindrically curved
  const sideSh = ctx.createLinearGradient(-hw, 0, hw, 0)
  sideSh.addColorStop(0,    'rgba(0,0,0,.22)')
  sideSh.addColorStop(.18,  'rgba(0,0,0,0)')
  sideSh.addColorStop(.82,  'rgba(0,0,0,0)')
  sideSh.addColorStop(1,    'rgba(0,0,0,.22)')
  ctx.globalAlpha = s.opacity
  ctx.fillStyle   = sideSh
  ctx.fill(path)

  // Cuticle fade — nail fades into skin at base (+Y)
  const cutFade = ctx.createLinearGradient(0, hh * .25, 0, hh)
  cutFade.addColorStop(0, 'rgba(0,0,0,0)')
  cutFade.addColorStop(1, 'rgba(0,0,0,.28)')
  ctx.fillStyle = cutFade
  ctx.fill(path)

  ctx.restore()

  // ── 4. Finish (gloss / matte) ──
  ctx.save()
  ctx.clip(path)

  if (s.finish === 'Глянцевый') {
    // Bright specular highlight — upper-left of nail
    const gloss = ctx.createRadialGradient(-hw * .14, -hh * .32, 0, -hw * .14, -hh * .32, hw * .88)
    gloss.addColorStop(0,   'rgba(255,255,255,.42)')
    gloss.addColorStop(.38, 'rgba(255,255,255,.12)')
    gloss.addColorStop(1,   'rgba(255,255,255,0)')
    ctx.globalAlpha = s.opacity
    ctx.fillStyle   = gloss
    ctx.fill(path)
  } else {
    // Matte: slight vignette to reduce reflectivity
    const matte = ctx.createRadialGradient(0, 0, hw * .1, 0, 0, hw * .95)
    matte.addColorStop(0, 'rgba(0,0,0,0)')
    matte.addColorStop(1, 'rgba(0,0,0,.18)')
    ctx.globalAlpha = s.opacity
    ctx.fillStyle   = matte
    ctx.fill(path)
  }

  ctx.restore()

  // ── 5. Thin outline ──
  ctx.globalAlpha    = s.opacity * .28
  ctx.strokeStyle    = 'rgba(0,0,0,.6)'
  ctx.lineWidth      = .7
  ctx.stroke(path)

  ctx.restore()
}

function renderFrame(canvas, source, hands, s, mirror = false) {
  const W = canvas.width, H = canvas.height
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, W, H)
  if (mirror) {
    ctx.save(); ctx.scale(-1, 1); ctx.drawImage(source, -W, 0, W, H); ctx.restore()
  } else {
    ctx.drawImage(source, 0, 0, W, H)
  }
  for (const hand of hands) {
    const lms = mirror ? hand.map(lm => ({ x: 1 - lm.x, y: lm.y, z: lm.z })) : hand
    const widths = computeNailWidths(lms, W, H)
    for (let i = 0; i < NAIL_CFG.length; i++) {
      const cfg = NAIL_CFG[i]
      drawNail(ctx, lms[cfg.t], lms[cfg.b], widths[i], s, W, H)
    }
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function PhotoVisualizer() {
  const [phase,    setPhase]    = useState('idle')
  const [detected, setDetected] = useState(0)
  const [error,    setError]    = useState(null)

  const [color,   setColor]   = useState('#B06EFF')
  const [shape,   setShape]   = useState('round')
  const [finish,  setFinish]  = useState('Глянцевый')
  const [design,  setDesign]  = useState('plain')
  const [opacity, setOpacity] = useState(0.92)

  const modelRef    = useRef(null)
  const canvasRef   = useRef(null)
  const videoRef    = useRef(null)
  const imageRef    = useRef(null)
  const handsRef    = useRef([])
  const rafRef      = useRef(null)
  const streamRef   = useRef(null)
  const settingsRef = useRef({})

  const settings = { color, shape, finish, design, opacity }
  useEffect(() => { settingsRef.current = settings })

  // Redraw photo when settings change (camera loop reads settingsRef directly)
  useEffect(() => {
    if (phase !== 'ready' || !imageRef.current || !handsRef.current.length) return
    renderFrame(canvasRef.current, imageRef.current, handsRef.current, settingsRef.current)
  }, [color, shape, finish, design, opacity, phase])

  const loadModel = useCallback(async (mode = 'IMAGE') => {
    if (modelRef.current) {
      if (modelRef.current._mode !== mode) {
        await modelRef.current.setOptions({ runningMode: mode })
        modelRef.current._mode = mode
      }
      return modelRef.current
    }
    const { HandLandmarker, FilesetResolver } =
      await import(/* @vite-ignore */ `${MP_CDN}/+esm`)
    const fs = await FilesetResolver.forVisionTasks(`${MP_CDN}/wasm`)
    const m  = await HandLandmarker.createFromOptions(fs, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      runningMode: mode,
      numHands: 2,
    })
    m._mode = mode
    modelRef.current = m
    return m
  }, [])

  const stopCamera = useCallback(() => {
    if (rafRef.current)   { cancelAnimationFrame(rafRef.current); rafRef.current = null }
    if (streamRef.current){ streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null }
    if (videoRef.current)   videoRef.current.srcObject = null
    setPhase(p => p === 'camera' ? 'idle' : p)
  }, [])

  const handleFile = useCallback(async (file) => {
    if (!file?.type.startsWith('image/')) return
    stopCamera()
    setError(null); setPhase('loading')
    try {
      const model = await loadModel('IMAGE')
      const url   = URL.createObjectURL(file)
      const img   = new Image()
      img.src     = url
      await new Promise((res, rej) => { img.onload = res; img.onerror = rej })

      const canvas      = canvasRef.current
      canvas.width      = img.naturalWidth
      canvas.height     = img.naturalHeight
      imageRef.current  = img

      const results        = model.detect(img)
      handsRef.current     = results.landmarks ?? []
      setDetected(handsRef.current.length)

      renderFrame(canvas, img, handsRef.current, settingsRef.current)
      setPhase('ready')
    } catch (e) {
      console.error(e)
      setError('Не удалось обработать фото. Попробуйте другое изображение.')
      setPhase('idle')
    }
  }, [loadModel, stopCamera])

  const startCamera = useCallback(async () => {
    setError(null); setPhase('loading')
    try {
      const model  = await loadModel('VIDEO')
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }
      })
      streamRef.current      = stream
      videoRef.current.srcObject = stream
      await videoRef.current.play()

      const canvas  = canvasRef.current
      canvas.width  = videoRef.current.videoWidth  || 1280
      canvas.height = videoRef.current.videoHeight || 720
      setPhase('camera')

      let last = -1
      const loop = (now) => {
        const video = videoRef.current
        if (video && video.readyState >= 2 && now !== last) {
          last = now
          canvas.width  = video.videoWidth
          canvas.height = video.videoHeight
          const res = model.detectForVideo(video, now)
          handsRef.current = res.landmarks ?? []
          renderFrame(canvas, video, handsRef.current, settingsRef.current, true)
        }
        rafRef.current = requestAnimationFrame(loop)
      }
      rafRef.current = requestAnimationFrame(loop)
    } catch (e) {
      console.error(e)
      setError('Не удалось получить доступ к камере.')
      setPhase('idle')
    }
  }, [loadModel])

  useEffect(() => () => {
    if (rafRef.current)    cancelAnimationFrame(rafRef.current)
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
  }, [])

  const download = () => {
    const a    = document.createElement('a')
    a.href     = canvasRef.current.toDataURL('image/jpeg', 0.93)
    a.download = 'nailmat.jpg'
    a.click()
  }

  const share = async () => {
    canvasRef.current.toBlob(async (blob) => {
      const file = new File([blob], 'nailmat.jpg', { type: 'image/jpeg' })
      if (navigator.share && navigator.canShare?.({ files: [file] })) {
        await navigator.share({ files: [file], title: 'Мой дизайн ногтей — NailMat' })
      } else {
        try {
          await navigator.clipboard.write([new ClipboardItem({ 'image/jpeg': blob })])
          alert('Скопировано в буфер обмена!')
        } catch { download() }
      }
    }, 'image/jpeg', 0.93)
  }

  const hasResult  = phase === 'ready' || phase === 'camera'
  const showCanvas = phase !== 'idle'

  return (
    <div className="page">
      <div className="container">

        <div className={styles.header}>
          <span className="tag">AR-режим</span>
          <h1 className={styles.title}>Визуализатор на фото</h1>
          <p className={styles.sub}>
            Загрузите фото или включите камеру — ногти определяются автоматически.
            <span className={styles.privacy}>🔒 Фото не уходит на сервер</span>
          </p>
        </div>

        <div className={styles.layout}>

          {/* ── Canvas / Upload ── */}
          <div className={styles.canvasArea}>
            {!showCanvas && (
              <label
                className={styles.dropzone}
                onDragOver={e => e.preventDefault()}
                onDrop={e => { e.preventDefault(); handleFile(e.dataTransfer.files[0]) }}
              >
                <input type="file" accept="image/*" style={{ display:'none' }}
                  onChange={e => handleFile(e.target.files[0])} />
                <div className={styles.dropIcon}>📷</div>
                <p className={styles.dropText}>Перетащите фото или нажмите для выбора</p>
                <p className={styles.dropHint}>Рука должна быть хорошо видна и освещена</p>
              </label>
            )}

            <canvas ref={canvasRef} className={styles.canvas}
              style={{ display: showCanvas ? 'block' : 'none' }} />
            <video ref={videoRef} style={{ display:'none' }} playsInline muted />

            {phase === 'loading' && (
              <div className={styles.overlay}>
                <div className={styles.spinner} />
                <p className={styles.overlayText}>Загружаем модель и ищем ногти…</p>
                <p className={styles.overlayHint}>Первый запуск ~10 сек</p>
              </div>
            )}

            {phase === 'ready' && (
              <div className={styles.badge}>
                {detected > 0
                  ? `✓ ${detected === 1 ? '1 рука' : '2 руки'} · ${detected * 5} ногтей`
                  : '⚠️ Руки не найдены — попробуйте другое фото'}
              </div>
            )}
            {phase === 'camera' && (
              <div className={`${styles.badge} ${styles.badgeLive}`}>🔴 LIVE</div>
            )}
            {error && <div className={styles.errorBar}>{error}</div>}
          </div>

          {/* ── Controls ── */}
          <div className={styles.sidebar}>

            <div className={`card ${styles.section}`}>
              <h3 className={styles.sLabel}>Источник</h3>
              <div className={styles.row}>
                <label className={`btn-primary ${styles.fileBtn}`}>
                  <input type="file" accept="image/*" style={{ display:'none' }}
                    onChange={e => handleFile(e.target.files[0])} />
                  📁 Фото
                </label>
                {phase === 'camera'
                  ? <button className="btn-outline" onClick={stopCamera}>⏹ Стоп</button>
                  : <button className="btn-outline" onClick={startCamera}>📹 Камера</button>
                }
              </div>
            </div>

            <div className={`card ${styles.section}`}>
              <h3 className={styles.sLabel}>Форма</h3>
              <div className={styles.pills}>
                {SHAPES.map(s => (
                  <button key={s} onClick={() => setShape(s)}
                    className={`${styles.pill} ${shape===s ? styles.pillOn:''}`}>
                    {s.charAt(0).toUpperCase()+s.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div className={`card ${styles.section}`}>
              <h3 className={styles.sLabel}>Покрытие</h3>
              <div className={styles.pills}>
                {FINISHES.map(f => (
                  <button key={f} onClick={() => setFinish(f)}
                    className={`${styles.pill} ${finish===f ? styles.pillOn:''}`}>
                    {f}
                  </button>
                ))}
              </div>
            </div>

            <div className={`card ${styles.section}`}>
              <h3 className={styles.sLabel}>Дизайн</h3>
              <div className={styles.pills}>
                {DESIGNS.map(d => (
                  <button key={d.id} onClick={() => setDesign(d.id)}
                    className={`${styles.pill} ${design===d.id ? styles.pillOn:''}`}>
                    {d.l}
                  </button>
                ))}
              </div>
            </div>

            <div className={`card ${styles.section}`}>
              <h3 className={styles.sLabel}>Цвет</h3>
              <div className={styles.palette}>
                {PALETTE.map(c => (
                  <button key={c} onClick={() => setColor(c)}
                    style={{ background:c }}
                    className={`${styles.swatch} ${color===c ? styles.swatchOn:''}`} />
                ))}
                <label className={styles.swatchCustom} title="Свой цвет">
                  <input type="color" value={color} onChange={e => setColor(e.target.value)} />
                  +
                </label>
              </div>
            </div>

            <div className={`card ${styles.section}`}>
              <h3 className={styles.sLabel}>Прозрачность — {Math.round(opacity*100)}%</h3>
              <input type="range" min="0.3" max="1" step="0.02"
                value={opacity} onChange={e => setOpacity(+e.target.value)}
                className={styles.slider} />
            </div>

            {hasResult && (
              <div className={styles.actions}>
                <button className="btn-primary"  onClick={download}>↓ Скачать</button>
                <button className="btn-outline"  onClick={share}>↗ Поделиться</button>
              </div>
            )}

          </div>
        </div>
      </div>
    </div>
  )
}
