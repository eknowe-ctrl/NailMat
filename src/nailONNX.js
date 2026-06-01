import * as ort from 'onnxruntime-web'

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/'

const MODEL_PATH  = `${import.meta.env.BASE_URL}models/nails_seg.onnx`
const INPUT_SZ    = 320
const PROTO_SZ    = 80
const N_MASK_COEF = 32
const CONF_THRESH = 0.28
const IOU_THRESH  = 0.40

let _session = null

export async function loadNailModel(onProgress) {
  if (_session) return _session

  const resp  = await fetch(MODEL_PATH)
  const total = Number(resp.headers.get('content-length')) || 0
  const reader = resp.body.getReader()
  const chunks = []
  let received = 0

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.length
    if (onProgress && total) onProgress(received / total)
  }

  const buf = new Uint8Array(received)
  let off = 0
  for (const c of chunks) { buf.set(c, off); off += c.length }

  _session = await ort.InferenceSession.create(buf.buffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  return _session
}

// ── helpers ───────────────────────────────────────────────────────────────────

function sigmoid(x) { return 1 / (1 + Math.exp(-x)) }

function iou([ax1, ay1, ax2, ay2], [bx1, by1, bx2, by2]) {
  const ix1 = Math.max(ax1, bx1), iy1 = Math.max(ay1, by1)
  const ix2 = Math.min(ax2, bx2), iy2 = Math.min(ay2, by2)
  if (ix2 <= ix1 || iy2 <= iy1) return 0
  const inter = (ix2 - ix1) * (iy2 - iy1)
  return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)
}

function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return m ? [parseInt(m[1],16), parseInt(m[2],16), parseInt(m[3],16)] : [176,110,255]
}

// ── preprocessing ─────────────────────────────────────────────────────────────

function preprocess(source, W, H) {
  const tmp = document.createElement('canvas')
  tmp.width = W; tmp.height = H
  tmp.getContext('2d').drawImage(source, 0, 0, W, H)
  const { data } = tmp.getContext('2d').getImageData(0, 0, W, H)
  const t = new Float32Array(3 * W * H)
  for (let i = 0; i < W * H; i++) {
    t[i]         = data[i*4]   / 255
    t[W*H   + i] = data[i*4+1] / 255
    t[W*H*2 + i] = data[i*4+2] / 255
  }
  return new ort.Tensor('float32', t, [1, 3, H, W])
}

// ── post-processing ───────────────────────────────────────────────────────────

function computeMask(coefs, protos) {
  const N = PROTO_SZ * PROTO_SZ
  const out = new Float32Array(N)
  for (let i = 0; i < N; i++) {
    let v = 0
    for (let c = 0; c < N_MASK_COEF; c++) v += coefs[c] * protos[c * N + i]
    out[i] = sigmoid(v)
  }
  return out
}

function postProcess(out0, out1) {
  const N = 2100
  const dets = []
  for (let i = 0; i < N; i++) {
    const score = out0[4 * N + i]
    if (score < CONF_THRESH) continue
    const cx = out0[0*N+i], cy = out0[1*N+i]
    const bw = out0[2*N+i], bh = out0[3*N+i]
    const coefs = new Float32Array(N_MASK_COEF)
    for (let c = 0; c < N_MASK_COEF; c++) coefs[c] = out0[(5+c)*N+i]
    dets.push({ bbox:[cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2], score, coefs })
  }
  dets.sort((a,b) => b.score - a.score)
  const kept = [], seen = new Set()
  for (let i = 0; i < dets.length; i++) {
    if (seen.has(i)) continue
    kept.push(dets[i])
    for (let j = i+1; j < dets.length; j++)
      if (iou(dets[i].bbox, dets[j].bbox) > IOU_THRESH) seen.add(j)
  }
  return kept.map(d => ({ ...d, mask: computeMask(d.coefs, out1) }))
}

// ── shape clip ────────────────────────────────────────────────────────────────
// Intersects the ONNX mask with the selected shape using destination-in.
// Shapes are defined in bbox coords: tip at y=0, cuticle at y=oh.

function applyShapeClip(ctx, ow, oh, shape) {
  ctx.globalCompositeOperation = 'destination-in'
  ctx.beginPath()
  switch (shape) {
    case 'square':
      ctx.roundRect(ow*.04, 0, ow*.92, oh, [ow*.08, ow*.08, ow*.05, ow*.05])
      break
    case 'almond':
      ctx.moveTo(ow*.5, 0)
      ctx.bezierCurveTo(ow*.78, oh*.18, ow*.92, oh*.48, ow*.88, oh)
      ctx.lineTo(ow*.12, oh)
      ctx.bezierCurveTo(ow*.08, oh*.48, ow*.22, oh*.18, ow*.5, 0)
      ctx.closePath()
      break
    case 'stiletto':
      ctx.moveTo(ow*.5, 0)
      ctx.bezierCurveTo(ow*.62, oh*.1, ow*.88, oh*.38, ow*.85, oh)
      ctx.lineTo(ow*.15, oh)
      ctx.bezierCurveTo(ow*.12, oh*.38, ow*.38, oh*.1, ow*.5, 0)
      ctx.closePath()
      break
    case 'coffin':
      ctx.moveTo(ow*.2, 0)
      ctx.lineTo(ow*.8, 0)
      ctx.bezierCurveTo(ow*.9, oh*.3, ow*.9, oh*.65, ow*.87, oh)
      ctx.lineTo(ow*.13, oh)
      ctx.bezierCurveTo(ow*.1, oh*.65, ow*.1, oh*.3, ow*.2, 0)
      ctx.closePath()
      break
    default: // round
      ctx.roundRect(ow*.04, 0, ow*.92, oh, [ow*.46, ow*.46, ow*.06, ow*.06])
  }
  ctx.fill()
  ctx.globalCompositeOperation = 'source-over'
}

// ── rendering ─────────────────────────────────────────────────────────────────

function drawDetection(ctx, det, scaleX, scaleY, settings) {
  const { bbox, mask } = det
  const [ix1, iy1, ix2, iy2] = bbox

  const sx1 = ix1 * scaleX, sy1 = iy1 * scaleY
  const sw  = (ix2 - ix1) * scaleX
  const sh  = (iy2 - iy1) * scaleY
  if (sw < 4 || sh < 4) return

  // ── 1. Build clean binary mask canvas (proto-space crop) ────────────────────
  const mx1 = Math.max(0, Math.floor(ix1 * PROTO_SZ / INPUT_SZ))
  const my1 = Math.max(0, Math.floor(iy1 * PROTO_SZ / INPUT_SZ))
  const mx2 = Math.min(PROTO_SZ, Math.ceil(ix2 * PROTO_SZ / INPUT_SZ))
  const my2 = Math.min(PROTO_SZ, Math.ceil(iy2 * PROTO_SZ / INPUT_SZ))
  const mw = mx2 - mx1, mh = my2 - my1
  if (mw <= 0 || mh <= 0) return

  const mc   = document.createElement('canvas')
  mc.width   = mw; mc.height = mh
  const mCtx = mc.getContext('2d')
  const mData = mCtx.createImageData(mw, mh)
  for (let y = 0; y < mh; y++) {
    for (let x = 0; x < mw; x++) {
      if (mask[(y + my1) * PROTO_SZ + (x + mx1)] < 0.5) continue
      const ii = (y * mw + x) * 4
      mData.data[ii] = mData.data[ii+1] = mData.data[ii+2] = mData.data[ii+3] = 255
    }
  }
  mCtx.putImageData(mData, 0, 0)

  // ── 2. Offscreen canvas (screen-size bbox) ───────────────────────────────────
  const ow = Math.ceil(sw), oh = Math.ceil(sh)
  const oc   = document.createElement('canvas')
  oc.width   = ow; oc.height = oh
  const oCtx = oc.getContext('2d')

  // Scale ONNX mask to screen size — bilinear interpolation anti-aliases edges
  oCtx.drawImage(mc, 0, 0, mw, mh, 0, 0, ow, oh)

  // ── 3. Apply shape clip (intersect mask × shape) ─────────────────────────────
  applyShapeClip(oCtx, ow, oh, settings.shape)

  // ── 4. Fill nail color inside mask×shape ─────────────────────────────────────
  const [r, g, b] = hexToRgb(settings.color)
  oCtx.globalCompositeOperation = 'source-in'

  if (settings.design === 'foil') {
    const fg = oCtx.createLinearGradient(0, 0, ow, oh)
    fg.addColorStop(0,    '#FFD700')
    fg.addColorStop(0.28, '#E8E8E8')
    fg.addColorStop(0.5,  `rgb(${r},${g},${b})`)
    fg.addColorStop(0.72, 'rgba(200,160,255,1)')
    fg.addColorStop(1,    '#FFD700')
    oCtx.fillStyle = fg
  } else {
    oCtx.fillStyle = `rgb(${r},${g},${b})`
  }
  oCtx.fillRect(0, 0, ow, oh)
  oCtx.globalCompositeOperation = 'source-over'

  // ── 5. Design overlays (source-atop = clipped to nail pixels) ────────────────
  oCtx.globalCompositeOperation = 'source-atop'

  if (settings.design === 'french') {
    const tipH = oh * 0.30
    const tG = oCtx.createLinearGradient(0, 0, 0, tipH * 1.2)
    tG.addColorStop(0,    'rgba(255,252,245,.96)')
    tG.addColorStop(0.65, 'rgba(255,252,245,.86)')
    tG.addColorStop(1,    'rgba(255,252,245,0)')
    oCtx.fillStyle = tG
    oCtx.fillRect(0, 0, ow, tipH * 1.2)
  }

  if (settings.design === 'luna') {
    oCtx.fillStyle = 'rgba(255,255,255,.40)'
    oCtx.beginPath()
    oCtx.ellipse(ow*.5, oh*.82, ow*.32, oh*.13, 0, 0, Math.PI*2)
    oCtx.fill()
  }

  // ── 6. 3D edge shading ────────────────────────────────────────────────────────
  const sideSh = oCtx.createLinearGradient(0, 0, ow, 0)
  sideSh.addColorStop(0,   'rgba(0,0,0,.22)')
  sideSh.addColorStop(.18, 'rgba(0,0,0,0)')
  sideSh.addColorStop(.82, 'rgba(0,0,0,0)')
  sideSh.addColorStop(1,   'rgba(0,0,0,.22)')
  oCtx.fillStyle = sideSh
  oCtx.fillRect(0, 0, ow, oh)

  const cutFade = oCtx.createLinearGradient(0, oh*.55, 0, oh)
  cutFade.addColorStop(0, 'rgba(0,0,0,0)')
  cutFade.addColorStop(1, 'rgba(0,0,0,.28)')
  oCtx.fillStyle = cutFade
  oCtx.fillRect(0, 0, ow, oh)

  // ── 7. Finish ─────────────────────────────────────────────────────────────────
  if (settings.finish === 'Глянцевый') {
    const gloss = oCtx.createRadialGradient(ow*.22, oh*.1, 0, ow*.22, oh*.1, ow*.85)
    gloss.addColorStop(0,   'rgba(255,255,255,.48)')
    gloss.addColorStop(.38, 'rgba(255,255,255,.14)')
    gloss.addColorStop(1,   'rgba(255,255,255,0)')
    oCtx.fillStyle = gloss
    oCtx.fillRect(0, 0, ow, oh)
  } else {
    const matte = oCtx.createRadialGradient(ow*.5, oh*.5, ow*.1, ow*.5, oh*.5, ow*.85)
    matte.addColorStop(0, 'rgba(0,0,0,0)')
    matte.addColorStop(1, 'rgba(0,0,0,.18)')
    oCtx.fillStyle = matte
    oCtx.fillRect(0, 0, ow, oh)
  }

  oCtx.globalCompositeOperation = 'source-over'

  // ── 8. Composite onto main canvas ────────────────────────────────────────────
  ctx.save()
  ctx.globalAlpha = settings.opacity
  ctx.drawImage(oc, sx1, sy1)
  ctx.restore()
}

// ── public API ────────────────────────────────────────────────────────────────

export async function segmentAndRender(canvas, source, settings, onProgress) {
  const session = await loadNailModel(onProgress)
  const W = canvas.width, H = canvas.height
  const ctx = canvas.getContext('2d')

  ctx.clearRect(0, 0, W, H)
  ctx.drawImage(source, 0, 0, W, H)

  const inputTensor = preprocess(source, INPUT_SZ, INPUT_SZ)
  const outputs = await session.run({ images: inputTensor })
  const keys = Object.keys(outputs)
  const out0 = outputs[keys[0]].data
  const out1 = outputs[keys[1]].data

  const dets = postProcess(out0, out1)
  const scaleX = W / INPUT_SZ
  const scaleY = H / INPUT_SZ

  for (const det of dets) drawDetection(ctx, det, scaleX, scaleY, settings)

  return dets.length
}
