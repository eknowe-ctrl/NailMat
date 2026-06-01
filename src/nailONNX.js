import * as ort from 'onnxruntime-web'

// Use CDN WASM so we don't bundle it
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/'

const MODEL_PATH   = 'https://github.com/eknowe-ctrl/NailMat/releases/download/v1.0-model/nails_seg.onnx'
const INPUT_SZ     = 320
const PROTO_SZ     = 80   // mask prototype resolution (INPUT_SZ / 4)
const N_MASK_COEF  = 32
const CONF_THRESH  = 0.28
const IOU_THRESH   = 0.40

let _session = null

export async function loadNailModel() {
  if (_session) return _session
  _session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  return _session
}

// ── helpers ──────────────────────────────────────────────────────────────────

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
    t[i]           = data[i*4]   / 255
    t[W*H   + i]   = data[i*4+1] / 255
    t[W*H*2 + i]   = data[i*4+2] / 255
  }
  return new ort.Tensor('float32', t, [1, 3, H, W])
}

// ── post-processing ───────────────────────────────────────────────────────────
// output0: [1, 37, 2100]  (4 bbox + 1 cls + 32 mask coefs)
// output1: [1, 32, 80, 80] prototype masks

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

  // NMS
  dets.sort((a,b) => b.score - a.score)
  const kept = [], seen = new Set()
  for (let i = 0; i < dets.length; i++) {
    if (seen.has(i)) continue
    kept.push(dets[i])
    for (let j = i+1; j < dets.length; j++)
      if (iou(dets[i].bbox, dets[j].bbox) > IOU_THRESH) seen.add(j)
  }

  // Compute masks only for kept detections (avoids waste)
  return kept.map(d => ({ ...d, mask: computeMask(d.coefs, out1) }))
}

// ── rendering ─────────────────────────────────────────────────────────────────

function applyGloss(ctx, x1, y1, sw, sh) {
  const g = ctx.createRadialGradient(x1+sw*.22, y1+sh*.18, 0, x1+sw*.22, y1+sh*.18, sw*.85)
  g.addColorStop(0,   'rgba(255,255,255,.38)')
  g.addColorStop(.4,  'rgba(255,255,255,.1)')
  g.addColorStop(1,   'rgba(255,255,255,0)')
  ctx.fillStyle = g
  ctx.fillRect(x1, y1, sw, sh)
}

function applyFrench(ctx, maskCanvas, x1, y1, sw, sh) {
  // White tip: top 22% of the mask bbox
  ctx.save()
  ctx.drawImage(maskCanvas, x1, y1, sw, sh) // redraw mask as clip proxy
  const tipH = sh * 0.22
  const g = ctx.createLinearGradient(x1, y1, x1, y1+tipH)
  g.addColorStop(0,   'rgba(255,252,245,.92)')
  g.addColorStop(.7,  'rgba(255,252,245,.88)')
  g.addColorStop(1,   'rgba(255,252,245,0)')
  ctx.globalAlpha = .92
  ctx.fillStyle = g
  ctx.fillRect(x1, y1, sw, tipH*1.1)
  ctx.restore()
}

function drawDetection(ctx, det, scaleX, scaleY, settings) {
  const { bbox, mask } = det
  const [ix1, iy1, ix2, iy2] = bbox

  // Screen coords
  const sx1 = ix1 * scaleX, sy1 = iy1 * scaleY
  const sw  = (ix2 - ix1) * scaleX
  const sh  = (iy2 - iy1) * scaleY
  if (sw <= 0 || sh <= 0) return

  // Mask region in proto coords
  const mx1 = Math.max(0, Math.floor(ix1 * PROTO_SZ / INPUT_SZ))
  const my1 = Math.max(0, Math.floor(iy1 * PROTO_SZ / INPUT_SZ))
  const mx2 = Math.min(PROTO_SZ, Math.ceil(ix2 * PROTO_SZ / INPUT_SZ))
  const my2 = Math.min(PROTO_SZ, Math.ceil(iy2 * PROTO_SZ / INPUT_SZ))
  const mw = mx2 - mx1, mh = my2 - my1
  if (mw <= 0 || mh <= 0) return

  // Build mask canvas (cropped to bbox)
  const mc = document.createElement('canvas')
  mc.width = mw; mc.height = mh
  const mCtx = mc.getContext('2d')
  const mData = mCtx.createImageData(mw, mh)
  const [r, g, b] = hexToRgb(settings.color)

  // Design: foil gradient fills
  let fr = r, fg = g, fb = b
  for (let my = 0; my < mh; my++) {
    for (let mx = 0; mx < mw; mx++) {
      const pi    = (my + my1) * PROTO_SZ + (mx + mx1)
      const prob  = mask[pi]
      if (prob < 0.25) continue
      const alpha = Math.round(prob * settings.opacity * 255)
      const ii    = (my * mw + mx) * 4

      if (settings.design === 'foil') {
        // Diagonal gradient through the nail
        const t = (mx / mw + my / mh) / 2
        fr = Math.round(255*t*0.8 + 192*(1-t)*0.8)
        fg = Math.round(215*t*0.5 + 192*(1-t)*0.5)
        fb = Math.round(0  *t     + 176*(1-t))
      } else {
        fr = r; fg = g; fb = b
      }

      mData.data[ii]   = fr
      mData.data[ii+1] = fg
      mData.data[ii+2] = fb
      mData.data[ii+3] = alpha
    }
  }
  mCtx.putImageData(mData, 0, 0)

  // Composite onto main canvas (bilinear scaling built-in to drawImage)
  ctx.save()
  ctx.drawImage(mc, sx1, sy1, sw, sh)

  // Luna: lighter half-moon near cuticle (bottom 20%)
  if (settings.design === 'luna') {
    ctx.globalAlpha = settings.opacity * .45
    ctx.fillStyle   = 'rgba(255,255,255,.5)'
    ctx.beginPath()
    ctx.ellipse(sx1 + sw*.5, sy1 + sh*.8, sw*.32, sh*.14, 0, 0, Math.PI*2)
    ctx.fill()
  }

  // French: white tip
  if (settings.design === 'french') {
    ctx.globalAlpha = settings.opacity * .92
    const tG = ctx.createLinearGradient(sx1, sy1, sx1, sy1 + sh*.28)
    tG.addColorStop(0,   'rgba(255,252,245,.93)')
    tG.addColorStop(.65, 'rgba(255,252,245,.85)')
    tG.addColorStop(1,   'rgba(255,252,245,0)')
    ctx.fillStyle = tG
    ctx.drawImage(mc, sx1, sy1, sw, sh*.28)  // draw only top portion
  }

  // Gloss
  if (settings.finish === 'Глянцевый') applyGloss(ctx, sx1, sy1, sw, sh)

  // Matte vignette
  if (settings.finish === 'Матовый') {
    const vm = ctx.createRadialGradient(sx1+sw*.5, sy1+sh*.5, sw*.12, sx1+sw*.5, sy1+sh*.5, sw*.7)
    vm.addColorStop(0, 'rgba(0,0,0,0)')
    vm.addColorStop(1, 'rgba(0,0,0,.18)')
    ctx.globalAlpha = settings.opacity
    ctx.fillStyle   = vm
    ctx.fillRect(sx1, sy1, sw, sh)
  }

  ctx.restore()
}

// ── public API ────────────────────────────────────────────────────────────────

export async function segmentAndRender(canvas, source, settings) {
  const session = await loadNailModel()
  const W = canvas.width, H = canvas.height
  const ctx = canvas.getContext('2d')

  ctx.clearRect(0, 0, W, H)
  ctx.drawImage(source, 0, 0, W, H)

  const inputTensor = preprocess(source, INPUT_SZ, INPUT_SZ)
  const outputs = await session.run({ images: inputTensor })
  const keys = Object.keys(outputs)
  const out0 = outputs[keys[0]].data   // [37 * 2100]
  const out1 = outputs[keys[1]].data   // [32 * 80 * 80]

  const dets = postProcess(out0, out1)

  const scaleX = W / INPUT_SZ
  const scaleY = H / INPUT_SZ

  for (const det of dets) drawDetection(ctx, det, scaleX, scaleY, settings)

  return dets.length
}
