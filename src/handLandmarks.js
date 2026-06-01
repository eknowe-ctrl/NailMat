import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

const WASM_PATH  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
const MODEL_PATH = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

let _landmarker  = null
let _failedOnce  = false

export async function loadHandLandmarker() {
  if (_landmarker)  return _landmarker
  if (_failedOnce)  return null
  try {
    const vision = await FilesetResolver.forVisionTasks(WASM_PATH)
    _landmarker  = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'CPU' },
      runningMode: 'IMAGE',
      numHands: 2,
    })
    return _landmarker
  } catch (e) {
    _failedOnce = true
    console.warn('[HandLandmarker] unavailable:', e?.message ?? e)
    return null
  }
}

// Draw hand skeleton on canvas; returns number of hands detected (0 on failure)
export async function detectAndDrawHands(canvas, source) {
  const lm = await loadHandLandmarker()
  if (!lm) return 0

  let result
  try {
    result = lm.detect(source)
  } catch (e) {
    console.warn('[HandLandmarker] detect failed:', e?.message ?? e)
    return 0
  }

  const hands = result?.landmarks
  if (!hands?.length) return 0

  const ctx = canvas.getContext('2d')
  const W = canvas.width, H = canvas.height
  const CONN = HandLandmarker.HAND_CONNECTIONS

  ctx.save()
  for (const hand of hands) {
    // Skeleton lines
    ctx.strokeStyle = 'rgba(176,110,255,.60)'
    ctx.lineWidth   = Math.max(1.5, W * 0.003)
    ctx.lineCap     = 'round'
    ctx.lineJoin    = 'round'
    ctx.beginPath()
    for (const c of CONN) {
      const a = hand[c.start], b = hand[c.end]
      ctx.moveTo(a.x * W, a.y * H)
      ctx.lineTo(b.x * W, b.y * H)
    }
    ctx.stroke()

    // Joint dots
    const dotR = Math.max(2, W * 0.005)
    for (const pt of hand) {
      ctx.beginPath()
      ctx.fillStyle = 'rgba(255,255,255,.85)'
      ctx.arc(pt.x * W, pt.y * H, dotR, 0, Math.PI * 2)
      ctx.fill()
    }
  }
  ctx.restore()

  return hands.length
}
