// WASM loader -- loads the BO WASM module (singleton, lazy init)

let wasmModule = null
let loading = null

async function loadBO(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    // SINGLE_FILE=1: .wasm is embedded in the .js file, no locateFile needed
    const createBO = require('../wasm/bo.js')
    wasmModule = await createBO(options)
    return wasmModule
  })()

  return loading
}

function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadBO() first')
  return wasmModule
}

module.exports = { loadBO, getWasm }
