import * as ort from 'onnxruntime-web'

ort.env.wasm.wasmPaths  = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/'
ort.env.wasm.numThreads = 1   // GitHub Pages lacks COOP/COEP headers → no SharedArrayBuffer

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

// ── mask refinement ───────────────────────────────────────────────────────────
// Builds an ow×oh alpha canvas for one detection.
// Uses a steepened sigmoid (not hard binary) at proto resolution, bilinear
// upscale, then color-guided boundary correction: boundary pixels are nudged
// toward "inside" or "outside" based on similarity to the average nail vs skin
// colour sampled from the source photo. This snaps ambiguous mask edges to
// real nail contours without needing a higher-res model.

function buildRefinedMaskCanvas(mask, source, ix1, iy1, ix2, iy2, ow, oh) {
  const px1 = Math.max(0, Math.floor(ix1 * PROTO_SZ / INPUT_SZ))
  const py1 = Math.max(0, Math.floor(iy1 * PROTO_SZ / INPUT_SZ))
  const px2 = Math.min(PROTO_SZ, Math.ceil(ix2 * PROTO_SZ / INPUT_SZ))
  const py2 = Math.min(PROTO_SZ, Math.ceil(iy2 * PROTO_SZ / INPUT_SZ))
  const pw = px2 - px1, ph = py2 - py1
  if (pw <= 0 || ph <= 0) return null

  // 1. Smooth proto-crop canvas via steepened sigmoid (v=0.5→128, v=0.65→250)
  const rawMc   = document.createElement('canvas')
  rawMc.width   = pw; rawMc.height = ph
  const rawCtx  = rawMc.getContext('2d')
  const rawImg  = rawCtx.createImageData(pw, ph)
  for (let y = 0; y < ph; y++) {
    for (let x = 0; x < pw; x++) {
      const v  = mask[(y + py1) * PROTO_SZ + (x + px1)]
      const s  = 1 / (1 + Math.exp(-(v - 0.5) * 16))
      const a  = Math.round(s * 255)
      const ii = (y * pw + x) * 4
      rawImg.data[ii] = rawImg.data[ii+1] = rawImg.data[ii+2] = a
      rawImg.data[ii+3] = a
    }
  }
  rawCtx.putImageData(rawImg, 0, 0)

  // 2. Bilinear upscale to output size → anti-aliased boundary
  const mc    = document.createElement('canvas')
  mc.width    = ow; mc.height = oh
  const mCtx  = mc.getContext('2d', { willReadFrequently: true })
  mCtx.drawImage(rawMc, 0, 0, pw, ph, 0, 0, ow, oh)

  // 3. Sample source photo in bbox region at output size
  const sW     = source.naturalWidth  || source.width  || ow
  const sH     = source.naturalHeight || source.height || oh
  const cropC  = document.createElement('canvas')
  cropC.width  = ow; cropC.height = oh
  const cropCtx = cropC.getContext('2d', { willReadFrequently: true })
  cropCtx.drawImage(source,
    ix1 / INPUT_SZ * sW, iy1 / INPUT_SZ * sH,
    (ix2 - ix1) / INPUT_SZ * sW, (iy2 - iy1) / INPUT_SZ * sH,
    0, 0, ow, oh
  )
  const srcPx  = cropCtx.getImageData(0, 0, ow, oh).data
  const upData = mCtx.getImageData(0, 0, ow, oh)
  const ap     = upData.data

  // 4. Accumulate average colour of definite inside / outside regions
  let iR=0,iG=0,iB=0,iN=0, oR=0,oG=0,oB=0,oN=0
  for (let i = 0; i < ow * oh; i++) {
    const a = ap[i*4+3]
    if (a > 210) { iR+=srcPx[i*4]; iG+=srcPx[i*4+1]; iB+=srcPx[i*4+2]; iN++ }
    else if (a < 45) { oR+=srcPx[i*4]; oG+=srcPx[i*4+1]; oB+=srcPx[i*4+2]; oN++ }
  }

  // 5. Color-guided boundary nudge (only if we have enough samples from both sides)
  if (iN > 20 && oN > 20) {
    iR/=iN; iG/=iN; iB/=iN
    oR/=oN; oG/=oN; oB/=oN
    for (let i = 0; i < ow * oh; i++) {
      const a = ap[i*4+3]
      if (a <= 45 || a >= 210) continue
      const r = srcPx[i*4], g = srcPx[i*4+1], b = srcPx[i*4+2]
      const dI    = (r-iR)**2 + (g-iG)**2 + (b-iB)**2
      const dO    = (r-oR)**2 + (g-oG)**2 + (b-oB)**2
      const nudge = dI <= dO ? 0.4 : -0.4
      const fa    = Math.min(255, Math.max(0, Math.round(a + nudge * 255)))
      ap[i*4] = ap[i*4+1] = ap[i*4+2] = fa
      ap[i*4+3] = fa
    }
    mCtx.putImageData(upData, 0, 0)
  }

  return mc
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

function drawDetection(ctx, det, scaleX, scaleY, settings, source) {
  const { bbox, mask } = det
  const [ix1, iy1, ix2, iy2] = bbox

  const sx1 = ix1 * scaleX, sy1 = iy1 * scaleY
  const sw  = (ix2 - ix1) * scaleX
  const sh  = (iy2 - iy1) * scaleY
  if (sw < 4 || sh < 4) return

  const ow = Math.ceil(sw), oh = Math.ceil(sh)

  // ── 1. Build edge-refined mask canvas (ow×oh) ─────────────────────────────────
  const mc = buildRefinedMaskCanvas(mask, source, ix1, iy1, ix2, iy2, ow, oh)
  if (!mc) return

  // ── 2. Offscreen canvas (screen-size bbox) ────────────────────────────────────
  const oc   = document.createElement('canvas')
  oc.width   = ow; oc.height = oh
  const oCtx = oc.getContext('2d')

  oCtx.drawImage(mc, 0, 0)   // mc already at ow×oh with refined alpha

  // ── 3. Apply shape clip (intersect mask × shape) ──────────────────────────────
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

  // ── 6. Base shadows ───────────────────────────────────────────────────────────

  // Left / right edge darkening — nail surface curves like a cylinder
  const sideSh = oCtx.createLinearGradient(0, 0, ow, 0)
  sideSh.addColorStop(0,    'rgba(0,0,0,.30)')
  sideSh.addColorStop(.22,  'rgba(0,0,0,0)')
  sideSh.addColorStop(.78,  'rgba(0,0,0,0)')
  sideSh.addColorStop(1,    'rgba(0,0,0,.30)')
  oCtx.fillStyle = sideSh
  oCtx.fillRect(0, 0, ow, oh)

  // Cuticle fold shadow — nail plate disappears under skin at the base
  const cutFade = oCtx.createLinearGradient(0, oh * .56, 0, oh)
  cutFade.addColorStop(0, 'rgba(0,0,0,0)')
  cutFade.addColorStop(1, 'rgba(0,0,0,.40)')
  oCtx.fillStyle = cutFade
  oCtx.fillRect(0, oh * .56, ow, oh * .44)

  // Subsurface scattering — warm glow where finger skin shows through thin nail
  const sss = oCtx.createRadialGradient(ow*.5, oh*.64, 0, ow*.5, oh*.64, ow*.55)
  sss.addColorStop(0, 'rgba(255,155,120,.08)')
  sss.addColorStop(1, 'rgba(255,155,120,0)')
  oCtx.fillStyle = sss
  oCtx.fillRect(0, oh * .36, ow, oh * .64)

  // ── 7. Finish ─────────────────────────────────────────────────────────────────
  if (settings.finish === 'Глянцевый') {
    // Primary specular — narrow angled ellipse, like a studio lamp reflection
    // Real glossy nails show a bright elongated streak, NOT a round blob
    const hx = ow * .36, hy = oh * .20
    const specG = oCtx.createRadialGradient(hx, hy, 0, hx, hy + oh*.05, ow*.24)
    specG.addColorStop(0,   'rgba(255,255,255,.92)')
    specG.addColorStop(.22, 'rgba(255,255,255,.60)')
    specG.addColorStop(.55, 'rgba(255,255,255,.15)')
    specG.addColorStop(1,   'rgba(255,255,255,0)')
    oCtx.fillStyle = specG
    oCtx.beginPath()
    // Ellipse: narrow (ow*0.16 wide, oh*0.065 tall), rotated -11°
    oCtx.ellipse(hx, hy, ow*.16, oh*.065, -.18, 0, Math.PI*2)
    oCtx.fill()

    // Secondary broad ambient — faint center glow (wet-look)
    const ambG = oCtx.createRadialGradient(ow*.5, oh*.30, 0, ow*.5, oh*.36, ow*.58)
    ambG.addColorStop(0, 'rgba(255,255,255,.13)')
    ambG.addColorStop(1, 'rgba(255,255,255,0)')
    oCtx.fillStyle = ambG
    oCtx.fillRect(0, 0, ow, oh * .68)

    // Rim light — thin bright line at free edge (tip grazes the light source)
    const rimG = oCtx.createLinearGradient(0, 0, 0, oh * .055)
    rimG.addColorStop(0, 'rgba(255,255,255,.28)')
    rimG.addColorStop(1, 'rgba(255,255,255,0)')
    oCtx.fillStyle = rimG
    oCtx.fillRect(0, 0, ow, oh * .055)

  } else { // Матовый
    // No specular — matte lacquer scatters light uniformly
    const diffG = oCtx.createRadialGradient(ow*.5, oh*.28, 0, ow*.5, oh*.35, ow*.62)
    diffG.addColorStop(0, 'rgba(255,255,255,.07)')
    diffG.addColorStop(1, 'rgba(255,255,255,0)')
    oCtx.fillStyle = diffG
    oCtx.fillRect(0, 0, ow, oh * .65)

    // Velvety edge darkening — characteristic of matte finish
    const velvet = oCtx.createRadialGradient(ow*.5, oh*.44, ow*.14, ow*.5, oh*.44, ow*.88)
    velvet.addColorStop(0, 'rgba(0,0,0,0)')
    velvet.addColorStop(1, 'rgba(0,0,0,.16)')
    oCtx.fillStyle = velvet
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

  for (const det of dets) drawDetection(ctx, det, scaleX, scaleY, settings, source)

  return dets.length
}
