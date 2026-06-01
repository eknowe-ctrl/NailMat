import { useState, useRef, useEffect, useCallback } from 'react'
import styles from './PhotoVisualizer.module.css'
import { segmentAndRender, loadNailModel } from '../nailONNX'

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

  // nH = 0.82 * DIP-TIP.  Center at 70% toward tip so:
  //   free_edge  = center + dir*hH ≈ TIP + 11%  (covers free edge)
  //   cuticle    = center - dir*hH ≈ DIP + 29%  (sits above DIP, at actual cuticle line)
  const nH = segLen * 0.82
  const nW = nailW
  const hh = nH / 2, hw = nW / 2
  const angle = Math.atan2(dy, dx)

  const cx = tx * 0.70 + bx * 0.30
  const cy = ty * 0.70 + by * 0.30

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

// ─── Pixel-level nail edge detection ─────────────────────────────────────────
// Scans perpendicular to the finger axis and finds where brightness changes
// significantly — marking the nail/skin boundary.

function pixelBrightness(data, w, h, x, y) {
  if (x < 0 || x >= w || y < 0 || y >= h) return -1
  const i = (~~y * w + ~~x) * 4
  return 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
}

function scanHalfWidth(data, w, h, cx, cy, perpX, perpY, maxHW) {
  const maxD = Math.ceil(maxHW) + 6
  const samp = new Float32Array(maxD * 2 + 1).fill(-1)
  const C = maxD

  for (let d = -maxD; d <= maxD; d++)
    samp[C + d] = pixelBrightness(data, w, h, cx + perpX * d, cy + perpY * d)

  // Center brightness (average over ±2 px)
  let cb = 0, n = 0
  for (let d = -2; d <= 2; d++) { if (samp[C + d] >= 0) { cb += samp[C + d]; n++ } }
  cb /= Math.max(n, 1)

  // Adaptive threshold: nail/skin contrast varies across skin tones
  let lo = 255, hi = 0
  for (let d = -maxD; d <= maxD; d++) { if (samp[C + d] >= 0) { lo = Math.min(lo, samp[C + d]); hi = Math.max(hi, samp[C + d]) } }
  const thresh = Math.max(10, (hi - lo) * 0.22)

  // Walk outward from center, find first significant brightness change
  let hw = maxHW
  for (let d = 2; d <= maxD - 1; d++) {
    const bL = samp[C - d], bR = samp[C + d]
    if (bL < 0 || bR < 0) { hw = d - 1; break }
    if (Math.abs(cb - bL) > thresh || Math.abs(cb - bR) > thresh) { hw = d; break }
  }
  return Math.max(hw, 2)
}

// Returns pixel-refined nail width using 3 scanlines across the nail ROI.
// Reads from the canvas BEFORE any nail overlays are drawn.
function refineNailWidth(ctx, cx, cy, angle, estW, estH, W, H) {
  const axX = Math.cos(angle), axY = Math.sin(angle)
  const perpX = -axY,          perpY = axX

  // Bounding box for the scan strip (wider than estimated nail)
  const scanW = Math.ceil(estW * 1.5) + 8
  const scanH = Math.ceil(estH * 0.8)
  const cos = Math.abs(axX), sin = Math.abs(axY)
  const bbW  = Math.ceil(scanW * sin + scanH * cos) + 4
  const bbH  = Math.ceil(scanW * cos + scanH * sin) + 4
  const x0   = Math.max(0, Math.floor(cx - bbW / 2))
  const y0   = Math.max(0, Math.floor(cy - bbH / 2))
  const x1   = Math.min(W, x0 + bbW)
  const y1   = Math.min(H, y0 + bbH)
  if (x1 <= x0 || y1 <= y0) return estW

  const { data } = ctx.getImageData(x0, y0, x1 - x0, y1 - y0)
  const rW = x1 - x0, rH = y1 - y0

  // Scan at 3 heights: –30%, 0%, +25% along nail axis (avoids cuticle & tip)
  const offsets = [-estH * 0.28, 0, estH * 0.22]
  const hws = offsets.map(off => {
    const scx = cx + axX * off - x0
    const scy = cy + axY * off - y0
    return scanHalfWidth(data, rW, rH, scx, scy, perpX, perpY, estW * 0.62)
  }).filter(v => v > estW * 0.15)

  if (!hws.length) return estW * 0.72
  hws.sort((a, b) => a - b)
  return hws[Math.floor(hws.length / 2)] * 2  // median half-width → full width
}

// ─── Frame renderer ───────────────────────────────────────────────────────────

function renderFrame(canvas, source, hands, s, mirror = false) {
  const W = canvas.width, H = canvas.height
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, W, H)
  if (mirror) {
    ctx.save(); ctx.scale(-1, 1); ctx.drawImage(source, -W, 0, W, H); ctx.restore()
  } else {
    ctx.drawImage(source, 0, 0, W, H)
  }

  // ── Phase 1: compute all nail specs from the CLEAN source image ──
  // We read pixels BEFORE any overlay is drawn so nails don't affect each other.
  const specs = []
  for (const hand of hands) {
    const lms      = mirror ? hand.map(lm => ({ x: 1 - lm.x, y: lm.y, z: lm.z })) : hand
    const baseW    = computeNailWidths(lms, W, H)

    for (let i = 0; i < NAIL_CFG.length; i++) {
      const { t, b }  = NAIL_CFG[i]
      const tip = lms[t], dip = lms[b]
      const tx = tip.x * W, ty = tip.y * H
      const bx = dip.x * W, by = dip.y * H
      const segLen    = Math.hypot(tx - bx, ty - by)
      if (segLen < 4) continue
      const angle     = Math.atan2(ty - by, tx - bx)
      const nH        = segLen * 0.82
      const cx        = tx * 0.70 + bx * 0.30
      const cy        = ty * 0.70 + by * 0.30

      // Pixel-level refinement: find actual nail edges in the source image
      const nailW = refineNailWidth(ctx, cx, cy, angle, baseW[i], nH, W, H)
      specs.push({ tip, dip, nailW })
    }
  }

  // ── Phase 2: draw all nails using refined widths ──
  for (const { tip, dip, nailW } of specs)
    drawNail(ctx, tip, dip, nailW, s, W, H)
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function PhotoVisualizer() {
  const [phase,    setPhase]    = useState('idle')
  const [detected, setDetected] = useState(0)
  const [error,    setError]    = useState(null)
  const [loadPct,  setLoadPct]  = useState(0)

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

  // Redraw photo when settings change — re-run ONNX segmentation with new color/design
  useEffect(() => {
    if (phase !== 'ready' || !imageRef.current) return
    segmentAndRender(canvasRef.current, imageRef.current, settingsRef.current).catch(console.error)
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
      setLoadPct(0)

      const url = URL.createObjectURL(file)
      const img = new Image()
      img.src   = url
      await new Promise((res, rej) => { img.onload = res; img.onerror = rej })

      const canvas     = canvasRef.current
      canvas.width     = img.naturalWidth
      canvas.height    = img.naturalHeight
      imageRef.current = img

      // ML segmentation with download progress
      const count = await segmentAndRender(
        canvas, img, settingsRef.current,
        pct => setLoadPct(Math.round(pct * 100))
      )
      setDetected(count)

      setPhase('ready')
    } catch (e) {
      console.error(e)
      setError(`Ошибка: ${e?.message ?? String(e)}`)
      setPhase('idle')
    }
  }, [stopCamera])

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
                {loadPct > 0 && loadPct < 100 ? (
                  <>
                    <p className={styles.overlayText}>Загружаем модель… {loadPct}%</p>
                    <div className={styles.progressBar}>
                      <div className={styles.progressFill} style={{ width: `${loadPct}%` }} />
                    </div>
                    <p className={styles.overlayHint}>45 МБ · только первый раз</p>
                  </>
                ) : loadPct === 100 ? (
                  <p className={styles.overlayText}>Анализируем ногти…</p>
                ) : (
                  <p className={styles.overlayText}>Подготовка…</p>
                )}
              </div>
            )}

            {phase === 'ready' && (
              <div className={styles.badge}>
                {detected > 0
                  ? `✓ ML: ${detected} ногт${detected === 1 ? 'ь' : detected < 5 ? 'я' : 'ей'} найдено`
                  : '⚠️ Ногти не найдены — попробуйте другое фото'}
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
