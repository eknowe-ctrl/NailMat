import { useState, useRef, useEffect, useCallback } from 'react'
import styles from './PhotoVisualizer.module.css'
import { segmentAndRender, loadNailModel } from '../nailONNX'

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

  const canvasRef   = useRef(null)
  const videoRef    = useRef(null)
  const imageRef    = useRef(null)
  const rafRef      = useRef(null)
  const streamRef   = useRef(null)
  const settingsRef = useRef({})

  const settings = { color, shape, finish, design, opacity }
  useEffect(() => { settingsRef.current = settings })

  // Redraw when settings change (photo mode only)
  useEffect(() => {
    if (phase !== 'ready' || !imageRef.current) return
    segmentAndRender(canvasRef.current, imageRef.current, settingsRef.current).catch(console.error)
  }, [color, shape, finish, design, opacity, phase])

  const stopCamera = useCallback(() => {
    if (rafRef.current)    { cancelAnimationFrame(rafRef.current); rafRef.current = null }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null }
    if (videoRef.current)    videoRef.current.srcObject = null
    setPhase(p => p === 'camera' ? 'idle' : p)
  }, [])

  // ── Photo upload ──────────────────────────────────────────────────────────────
  const handleFile = useCallback(async (file) => {
    if (!file?.type.startsWith('image/')) return
    stopCamera()
    setError(null); setPhase('loading'); setLoadPct(0)
    try {
      const url = URL.createObjectURL(file)
      const img = new Image()
      img.src   = url
      await new Promise((res, rej) => { img.onload = res; img.onerror = rej })

      const canvas     = canvasRef.current
      canvas.width     = img.naturalWidth
      canvas.height    = img.naturalHeight
      imageRef.current = img

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

  // ── Camera: show live feed only, no overlay ───────────────────────────────────
  const startCamera = useCallback(async () => {
    setError(null); setPhase('loading')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } }
      })
      streamRef.current          = stream
      videoRef.current.srcObject = stream
      await videoRef.current.play()

      const canvas  = canvasRef.current
      canvas.width  = videoRef.current.videoWidth  || 1280
      canvas.height = videoRef.current.videoHeight || 720
      setPhase('camera')

      const loop = () => {
        const v = videoRef.current, c = canvasRef.current
        if (v?.readyState >= 2) {
          c.width  = v.videoWidth  || c.width
          c.height = v.videoHeight || c.height
          c.getContext('2d').drawImage(v, 0, 0, c.width, c.height)
        }
        rafRef.current = requestAnimationFrame(loop)
      }
      rafRef.current = requestAnimationFrame(loop)
    } catch (e) {
      console.error(e)
      setError('Не удалось получить доступ к камере.')
      setPhase('idle')
    }
  }, [])

  // ── Capture frame → run ONNX ─────────────────────────────────────────────────
  const takePhoto = useCallback(async () => {
    const video = videoRef.current
    if (!video) return

    // Snapshot current frame before stopping camera
    const snap = document.createElement('canvas')
    snap.width  = video.videoWidth  || 1280
    snap.height = video.videoHeight || 720
    snap.getContext('2d').drawImage(video, 0, 0, snap.width, snap.height)

    stopCamera()
    setError(null); setPhase('loading'); setLoadPct(0)

    try {
      const img = new Image()
      img.src   = snap.toDataURL('image/jpeg', 0.95)
      await new Promise(res => { img.onload = res })

      const canvas     = canvasRef.current
      canvas.width     = img.naturalWidth
      canvas.height    = img.naturalHeight
      imageRef.current = img

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

  const hasResult  = phase === 'ready'
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
                  <>
                    <p className={styles.overlayText}>Анализируем ногти…</p>
                    <p className={styles.overlayHint}>3 ракурса · ~15 сек</p>
                  </>
                ) : (
                  <p className={styles.overlayText}>Подготовка…</p>
                )}
              </div>
            )}

            {phase === 'camera' && (
              <div className={styles.captureBar}>
                <button className={styles.captureBtn} onClick={takePhoto}>
                  📸 Сфотографировать
                </button>
              </div>
            )}

            {phase === 'ready' && (
              <div className={styles.badge}>
                {detected > 0
                  ? `✓ ML: ${detected} ногт${detected === 1 ? 'ь' : detected < 5 ? 'я' : 'ей'} найдено`
                  : '⚠️ Ногти не найдены — попробуйте другое фото'}
              </div>
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
