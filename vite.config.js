import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Remove ORT WASM files from the build output — they are loaded from CDN at runtime
const excludeOrtWasm = {
  name: 'exclude-ort-wasm',
  generateBundle(_opts, bundle) {
    for (const key of Object.keys(bundle)) {
      if (key.endsWith('.wasm')) delete bundle[key]
    }
  },
}

export default defineConfig({
  plugins: [react(), excludeOrtWasm],
  base: '/NailMat/',
  build: {
    chunkSizeWarningLimit: 700,
  },
})
