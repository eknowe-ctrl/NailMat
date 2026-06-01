import { useState, useRef, useEffect, useCallback } from 'react'
import styles from './PhotoVisualizer.module.css'

const MP_CDN   = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18'
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

// tip index, base index (2 joints below tip), width-scale
const NAIL_CFG = [
  { t: 4,  b: 2,  ws: 0.58 },
  { t: 8,  b: 6,  ws: 0.72 },
  { t: 12, b: 10, ws: 0.74 },
  { t: 16, b: 14, ws: 0.70 },
  { t: 20, b: 18, ws: 0.58 },
]

const PALETTE = [
  '#FADADD','#F4A7B9','#E8A0BF','#C780C8','#9B59B6',
  '#7C3AED','#B06EFF','#5C6BC0','#42A5F5','#26C6DA',
  '#66BB6A','#FFD54F','#FFA726','#FF7043','#8D6E63',
  '#BDBDBD','#F5F5F5','#1A1A2E','#FF3D7F','#E74C3C',
]

const SHAPES  = ['round','square','almond','stiletto','coffin']
const FINISHES = ['Глянцевый','Матовый']
const DESIGNS = [
  { id:'plain',  l:'Однотонный' },
  { id:'french', l:'Френч' },
  { id:'foil',   l:'Втирка' },
  { id:'luna',   l:'Лунки' },
]

// ─── Canvas nail drawing ──────────────────────────────────────────────────────

function buildPath(shape, w, h) {
  const p = new Path2D(), hw = w/2, hh = h/2
  if (shape === 'square') {
    p.rect(-hw, -hh, w, h)
  } else if (shape === 'almond') {
    p.moveTo(0, -hh)
    p.bezierCurveTo( hw*.65, -hh*.4, hw,      hh*.1, hw*.88,  hh)
    p.quadraticCurveTo(0, hh*1.07, -hw*.88, hh)
    p.bezierCurveTo(-hw,     hh*.1, -hw*.65, -hh*.4, 0, -hh)
  } else if (shape === 'stiletto') {
    p.moveTo(0, -hh)
    p.bezierCurveTo( hw*.36, -hh*.12, hw*.82, hh*.35, hw*.86, hh)
    p.quadraticCurveTo(0, hh*1.05, -hw*.86, hh)
    p.bezierCurveTo(-hw*.82,  hh*.35, -hw*.36, -hh*.12, 0, -hh)
  } else if (shape === 'coffin') {
    p.moveTo(-hw*.42, -hh); p.lineTo(hw*.42, -hh)
    p.bezierCurveTo( hw*.76, -hh*.35, hw*.88, hh*.12, hw*.86,  hh)
    p.quadraticCurveTo(0, hh*1.05, -hw*.86, hh)
    p.bezierCurveTo(-hw*.88, hh*.12, -hw*.76, -hh*.35, -hw*.42, -hh)
  } else {
    try {
      p.roundRect(-hw, -hh, w, h, [hw*.85, hw*.85, hw*.14, hw*.14])
    } catch {
      p.moveTo(0, -hh); p.arcTo(hw, -hh, hw, hh, hw*.85)
      p.arcTo(hw, hh, -hw, hh, hw*.14); p.arcTo(-hw, hh, -hw, -hh, hw*.14)
      p.arcTo(-hw, -hh, hw, -hh, hw*.85); p.closePath()
    }
  }
  return p
}

function drawNail(ctx, tip, base, ws, s, W, H) {
  const tx = tip.x*W, ty = tip.y*H
  const bx = base.x*W, by = base.y*H
  const dx = tx-bx, dy = ty-by
  const len = Math.hypot(dx, dy)
  if (len < 8) return

  const nH = len * 0.52, nW = nH * ws
  const angle = Math.atan2(dy, dx)
  const cx = tx - (dx/len)*nH*0.36
  const cy = ty - (dy/len)*nH*0.36
  const path = buildPath(s.shape, nW, nH)

  ctx.save()
  ctx.translate(cx, cy)
  ctx.rotate(angle + Math.PI/2)

  // Base color
  ctx.globalAlpha = s.opacity
  ctx.fillStyle = s.color
  ctx.fill(path)

  // Design
  if (s.design === 'french') {
    ctx.save(); ctx.clip(path)
    ctx.globalAlpha = s.opacity * .93
    ctx.fillStyle = 'rgba(255,250,240,.93)'
    ctx.beginPath(); ctx.ellipse(0, -nH*.41, nW*.6, nH*.19, 0, 0, Math.PI*2); ctx.fill()
    ctx.restore()
  } else if (s.design === 'foil') {
    ctx.save(); ctx.clip(path)
    const g = ctx.createLinearGradient(-nW*.5,-nH*.5, nW*.5,nH*.5)
    g.addColorStop(0,   'rgba(255,215,0,.6)')
    g.addColorStop(.33, 'rgba(192,192,192,.55)')
    g.addColorStop(.66, 'rgba(176,110,255,.6)')
    g.addColorStop(1,   'rgba(255,215,0,.5)')
    ctx.globalAlpha = s.opacity*.8; ctx.fillStyle = g; ctx.fill(path)
    ctx.restore()
  } else if (s.design === 'luna') {
    ctx.save(); ctx.clip(path)
    ctx.globalAlpha = s.opacity*.5; ctx.fillStyle = 'rgba(255,255,255,.45)'
    ctx.beginPath(); ctx.ellipse(0, nH*.37, nW*.35, nH*.18, 0, 0, Math.PI*2); ctx.fill()
    ctx.restore()
  }

  // Gloss / matte
  ctx.save(); ctx.clip(path)
  if (s.finish === 'Глянцевый') {
    const g = ctx.createRadialGradient(-nW*.12,-nH*.28,0, -nW*.12,-nH*.28,nW*.75)
    g.addColorStop(0, 'rgba(255,255,255,.4)'); g.addColorStop(.45,'rgba(255,255,255,.1)'); g.addColorStop(1,'rgba(255,255,255,0)')
    ctx.globalAlpha = s.opacity; ctx.fillStyle = g; ctx.fill(path)
  } else {
    const g = ctx.createRadialGradient(0,0,nW*.1, 0,0,nW*.9)
    g.addColorStop(0,'rgba(0,0,0,0)'); g.addColorStop(1,'rgba(0,0,0,.22)')
    ctx.globalAlpha = s.opacity; ctx.fillStyle = g; ctx.fill(path)
  }
  ctx.restore()

  // Border
  ctx.globalAlpha = s.opacity*.38
  ctx.strokeStyle = 'rgba(0,0,0,.4)'; ctx.lineWidth = .8
  ctx.stroke(path)

  ctx.restore()
}

function renderFrame(canvas, source, hands, s, mirror = false) {
  const W = canvas.width, H = canvas.height
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, W, H)
  if (mirror) {
    ctx.save(); ctx.scale(-1,1); ctx.drawImage(source, -W, 0, W, H); ctx.restore()
  } else {
    ctx.drawImage(source, 0, 0, W, H)
  }
  for (const hand of hands) {
    const lms = mirror ? hand.map(lm => ({ x:1-lm.x, y:lm.y, z:lm.z })) : hand
    for (const cfg of NAIL_CFG) drawNail(ctx, lms[cfg.t], lms[cfg.b], cfg.ws, s, W, H)
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function PhotoVisualizer() {
  const [phase,    setPhase]    = useState('idle')   // idle|loading|ready|camera
  const [detected, setDetected] = useState(0)
  const [error,    setError]    = useState(null)

  const [color,   setColor]   = useState('#B06EFF')
  const [shape,   setShape]   = useState('round')
  const [finish,  setFinish]  = useState('Глянцевый')
  const [design,  setDesign]  = useState('plain')
  const [opacity, setOpacity] = useState(0.9)

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

  // Redraw photo when settings change
  useEffect(() => {
    if (phase !== 'ready' || !imageRef.current || !handsRef.current.length) return
    renderFrame(canvasRef.current, imageRef.current, handsRef.current, settingsRef.current)
  }, [color, shape, finish, design, opacity, phase])

  // ── Load model lazily ────────────────────────────────────────────────────────
  const loadModel = useCallback(async (mode = 'IMAGE') => {
    if (modelRef.current) {
      if (modelRef.current._mode !== mode)
        await modelRef.current.setOptions({ runningMode: mode })
      modelRef.current._mode = mode
      return modelRef.current
    }
    const { HandLandmarker, FilesetResolver } =
      await import(/* @vite-ignore */ `${MP_CDN}/+esm`)
    const fs = await FilesetResolver.forVisionTasks(`${MP_CDN}/wasm`)
    const m = await HandLandmarker.createFromOptions(fs, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
      runningMode: mode,
      numHands: 2,
    })
    m._mode = mode
    modelRef.current = m
    return m
  }, [])

  // ── Photo upload ─────────────────────────────────────────────────────────────
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

      const canvas = canvasRef.current
      canvas.width  = img.naturalWidth
      canvas.height = img.naturalHeight
      imageRef.current = img

      const results = model.detect(img)
      handsRef.current = results.landmarks ?? []
      setDetected(handsRef.current.length)

      renderFrame(canvas, img, handsRef.current, settingsRef.current)
      setPhase('ready')
    } catch (e) {
      console.error(e)
      setError('Не удалось обработать фото. Попробуйте другое изображение.')
      setPhase('idle')
    }
  }, [loadModel])

  // ── Camera ───────────────────────────────────────────────────────────────────
  const stopCamera = useCallback(() => {
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null }
    if (videoRef.current) videoRef.current.srcObject = null
    if (phase === 'camera') setPhase('idle')
  }, [phase])

  const startCamera = useCallback(async () => {
    setError(null); setPhase('loading')
    try {
      const model = await loadModel('VIDEO')
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }
      })
      streamRef.current = stream
      const video = videoRef.current
      video.srcObject = stream
      await video.play()

      const canvas = canvasRef.current
      canvas.width  = video.videoWidth  || 1280
      canvas.height = video.videoHeight || 720
      setPhase('camera')

      let last = -1
      const loop = (now) => {
        if (video.readyState >= 2 && now !== last) {
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
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
  }, [])

  // ── Share / Download ─────────────────────────────────────────────────────────
  const download = () => {
    const a = document.createElement('a')
    a.href     = canvasRef.current.toDataURL('image/jpeg', 0.92)
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
    }, 'image/jpeg', 0.92)
  }

  const hasResult = phase === 'ready' || phase === 'camera'
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

          {/* ── Left: canvas / upload ── */}
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
                <p className={styles.dropHint}>Рука должна быть хорошо видна на фото</p>
              </label>
            )}

            <canvas ref={canvasRef}
              className={styles.canvas}
              style={{ display: showCanvas ? 'block' : 'none' }} />

            <video ref={videoRef} style={{ display:'none' }} playsInline muted />

            {phase === 'loading' && (
              <div className={styles.overlay}>
                <div className={styles.spinner} />
                <p className={styles.overlayText}>Загружаем модель и ищем ногти…</p>
                <p className={styles.overlayHint}>Первый запуск занимает ~10 сек</p>
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
              <div className={`${styles.badge} ${styles.badgeLive}`}>
                🔴 LIVE
              </div>
            )}

            {error && <div className={styles.errorBar}>{error}</div>}
          </div>

          {/* ── Right: controls ── */}
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
                <button className="btn-primary" onClick={download}>↓ Скачать</button>
                <button className="btn-outline" onClick={share}>↗ Поделиться</button>
              </div>
            )}

          </div>
        </div>
      </div>
    </div>
  )
}
