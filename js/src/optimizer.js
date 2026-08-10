/**
 * optimizer.js -- BayesianOptimizer (thin C wrapper)
 *
 * Manages WASM lifecycle for a single BO instance (one search space).
 * All BO policy lives in C; this class handles:
 * - WASM heap allocation/deallocation
 * - SearchSpace compilation (JS objects <-> flat numeric arrays)
 * - Memory management via FinalizationRegistry
 */

const { loadBO, getWasm } = require('./wasm.js')
const { compileSpace, encodeParams, decodeParams } = require('./compile.js')

const SIZEOF_DOUBLE = 8

// Kernel enum values matching C
const KERNELS = { matern52: 0, matern32: 1, se: 2 }
// Acquisition enum values matching C
const ACQ_FNS = { ei: 0, ucb: 1, pi: 2 }

const registry = new FinalizationRegistry(pointers => {
  try {
    const wasm = getWasm()
    if (pointers.state) wasm._wl_bo_free(pointers.state)
    if (pointers.space) wasm._wl_bo_space_free(pointers.space)
  } catch (_) {
    // WASM may already be torn down
  }
})

class BayesianOptimizer {
  #statePtr = null
  #spacePtr = null
  #compiled = null
  #disposed = false
  #nDims = 0

  /**
   * Create a BayesianOptimizer from a SearchSpace.
   *
   * @param {object} searchSpace - { paramName: SearchParam, ... }
   * @param {object} opts
   * @param {string} opts.kernel - 'matern52' | 'matern32' | 'se'
   * @param {string} opts.acquisitionFn - 'ei' | 'ucb' | 'pi'
   * @param {number} opts.kappa - UCB exploration weight (default 2.0)
   * @param {number} opts.xi - EI/PI exploration jitter (default 0.01)
   * @param {number} opts.nRestartsHyper - L-BFGS restarts for hyperparameters (default 5)
   * @param {number} opts.nRestartsAcq - L-BFGS restarts for acquisition (default 10)
   * @param {number} opts.maxObs - observation reservoir cap (default 500)
   * @param {number} opts.minObsForGp - min observations per context for GP (default 3)
   * @param {number} opts.seed
   */
  static async create(searchSpace, opts = {}) {
    await loadBO()
    const wasm = getWasm()
    const compiled = compileSpace(searchSpace)
    const { nDims, paramTypes, lows, highs, logFlags, nValues, condParents, condValues } = compiled

    // Build space in C
    const spacePtr = wasm._wl_bo_space_create(nDims)
    if (!spacePtr) {
      throw new Error('BayesianOptimizer: failed to create space: ' + getError())
    }

    for (let i = 0; i < nDims; i++) {
      let rc
      switch (paramTypes[i]) {
        case 0: // continuous
          rc = wasm._wl_bo_space_add_continuous(spacePtr, i, lows[i], highs[i], logFlags[i])
          break
        case 1: // integer
          rc = wasm._wl_bo_space_add_integer(spacePtr, i, lows[i], highs[i], logFlags[i])
          break
        case 2: // categorical
          rc = wasm._wl_bo_space_add_categorical(spacePtr, i, nValues[i])
          break
      }
      if (rc !== 0) {
        wasm._wl_bo_space_free(spacePtr)
        throw new Error('BayesianOptimizer: failed to add dim ' + i + ': ' + getError())
      }

      // Add condition if present
      if (condParents[i] >= 0) {
        rc = wasm._wl_bo_space_add_condition(spacePtr, i, condParents[i], condValues[i])
        if (rc !== 0) {
          wasm._wl_bo_space_free(spacePtr)
          throw new Error('BayesianOptimizer: failed to add condition for dim ' + i + ': ' + getError())
        }
      }
    }

    wasm._wl_bo_space_finalize(spacePtr)

    // Create optimizer
    const kernel = KERNELS[opts.kernel || 'matern52']
    const acq = ACQ_FNS[opts.acquisitionFn || 'ei']
    if (kernel === undefined) throw new Error('Unknown kernel: ' + opts.kernel)
    if (acq === undefined) throw new Error('Unknown acquisitionFn: ' + opts.acquisitionFn)

    const statePtr = wasm._wl_bo_create(
      spacePtr,
      kernel,
      acq,
      opts.kappa ?? 2.0,
      opts.xi ?? 0.01,
      opts.nRestartsHyper ?? 5,
      opts.nRestartsAcq ?? 10,
      opts.maxObs ?? 500,
      opts.minObsForGp ?? 3,
      opts.seed ?? 42
    )

    if (!statePtr) {
      wasm._wl_bo_space_free(spacePtr)
      throw new Error('BayesianOptimizer: failed to create: ' + getError())
    }

    const optimizer = new BayesianOptimizer(statePtr, spacePtr, compiled, nDims)
    registry.register(optimizer, { state: statePtr, space: spacePtr })
    return optimizer
  }

  constructor(statePtr, spacePtr, compiled, nDims) {
    this.#statePtr = statePtr
    this.#spacePtr = spacePtr
    this.#compiled = compiled
    this.#nDims = nDims
  }

  /**
   * Observe a (params, score) pair.
   * @param {object} params - { paramName: value, ... }
   * @param {number} score - objective value (higher is better)
   */
  observe(params, score) {
    this.#checkDisposed()
    if (typeof score !== 'number' || !isFinite(score)) {
      throw new Error('BayesianOptimizer.observe: score must be a finite number, got ' + score)
    }
    const wasm = getWasm()
    const encoded = encodeParams(this.#compiled, params)

    // Copy to WASM heap
    const ptr = wasm._malloc(this.#nDims * SIZEOF_DOUBLE)
    if (!ptr) throw new Error('BayesianOptimizer.observe: malloc failed')
    try {
      const heap = new Float64Array(wasm.HEAPF64.buffer, ptr, this.#nDims)
      heap.set(encoded)
      const rc = wasm._wl_bo_observe(this.#statePtr, ptr, score)
      if (rc !== 0) throw new Error('BayesianOptimizer.observe: ' + getError())
    } finally {
      wasm._free(ptr)
    }
  }

  /**
   * Suggest next params to evaluate.
   * @returns {object} params - { paramName: value, ... }
   */
  suggest() {
    this.#checkDisposed()
    const wasm = getWasm()

    const ptr = wasm._malloc(this.#nDims * SIZEOF_DOUBLE)
    if (!ptr) throw new Error('BayesianOptimizer.suggest: malloc failed')
    try {
      const rc = wasm._wl_bo_suggest(this.#statePtr, ptr)
      if (rc !== 0) throw new Error('BayesianOptimizer.suggest: ' + getError())
      const doubles = new Float64Array(wasm.HEAPF64.buffer, ptr, this.#nDims)
      return decodeParams(this.#compiled, doubles)
    } finally {
      wasm._free(ptr)
    }
  }

  /**
   * Suggest with constant liar for batch BO.
   * @param {number} hallucinatedY - score to hallucinate for pending suggestions
   * @returns {object} params
   */
  suggestLiar(hallucinatedY) {
    this.#checkDisposed()
    const wasm = getWasm()

    const ptr = wasm._malloc(this.#nDims * SIZEOF_DOUBLE)
    if (!ptr) throw new Error('BayesianOptimizer.suggestLiar: malloc failed')
    try {
      const rc = wasm._wl_bo_suggest_liar(this.#statePtr, ptr, hallucinatedY)
      if (rc !== 0) throw new Error('BayesianOptimizer.suggestLiar: ' + getError())
      const doubles = new Float64Array(wasm.HEAPF64.buffer, ptr, this.#nDims)
      return decodeParams(this.#compiled, doubles)
    } finally {
      wasm._free(ptr)
    }
  }

  /**
   * Suggest a batch of n points using constant liar.
   * @param {number} n
   * @returns {object[]} array of params objects
   */
  suggestBatch(n) {
    if (n <= 0) return []
    const results = [this.suggest()]
    const bestScore = this.bestScore
    for (let i = 1; i < n; i++) {
      results.push(this.suggestLiar(bestScore))
    }
    return results
  }

  get nObs() {
    this.#checkDisposed()
    return getWasm()._wl_bo_get_n_obs(this.#statePtr)
  }

  get bestScore() {
    this.#checkDisposed()
    return getWasm()._wl_bo_get_best_score(this.#statePtr)
  }

  get nContexts() {
    this.#checkDisposed()
    return getWasm()._wl_bo_get_n_contexts(this.#statePtr)
  }

  get compiled() {
    return this.#compiled
  }

  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    const wasm = getWasm()
    if (this.#statePtr) {
      wasm._wl_bo_free(this.#statePtr)
      this.#statePtr = null
    }
    if (this.#spacePtr) {
      wasm._wl_bo_space_free(this.#spacePtr)
      this.#spacePtr = null
    }
  }

  #checkDisposed() {
    if (this.#disposed) throw new Error('BayesianOptimizer: already disposed')
  }
}

function getError() {
  try {
    const wasm = getWasm()
    const ptr = wasm._wl_bo_get_last_error()
    if (!ptr) return '(unknown error)'
    return wasm.UTF8ToString(ptr)
  } catch (_) {
    return '(error retrieval failed)'
  }
}

module.exports = { BayesianOptimizer }
