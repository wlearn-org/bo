/**
 * strategy.js -- BayesianStrategy (automl Strategy interface)
 *
 * Implements the Strategy contract consumed by Executor.runStrategy():
 *   - next() -> { candidateId, cls, params } | null
 *   - report(result)
 *   - isDone()
 *
 * Each model family gets its own BayesianOptimizer. During warm-up,
 * random configs are yielded. After warm-up, BO-guided suggestions.
 */

const { makeLCG } = require('@wlearn/core')
const { sampleConfig, makeCandidateId } = require('@wlearn/automl')
const { BayesianOptimizer } = require('./optimizer.js')
const { countFreeParams } = require('./compile.js')

const { ceil, min, max } = Math

class BayesianStrategy {
  #models
  #nIter
  #seed
  #opts
  #optimizers = new Map()  // modelName -> BayesianOptimizer
  #warmupQueues = new Map() // modelName -> [{ candidateId, cls, params }]
  #warmupCounts = new Map() // modelName -> number of warm-up needed
  #reported = new Map()     // modelName -> number of reports received
  #yielded = 0
  #total = 0
  #modelCycle = []
  #cycleIdx = 0
  #initialized = false
  #disposed = false

  /**
   * @param {Array<{ name: string, cls: object, searchSpace?: object, params?: object }>} models
   * @param {object} opts
   * @param {number} opts.nIter - candidates per model (default 30)
   * @param {number?} opts.nInitial - warm-up count override (null = auto)
   * @param {number} opts.seed
   * @param {string} opts.acquisitionFn - 'ei' | 'ucb' | 'pi'
   * @param {number} opts.kappa - UCB exploration weight
   * @param {number} opts.xi - EI/PI jitter
   * @param {string} opts.kernel - 'matern52' | 'matern32' | 'se'
   * @param {string} opts.task - 'classification' | 'regression' (passed to defaultSearchSpace)
   */
  constructor(models, opts = {}) {
    this.#models = models
    this.#nIter = opts.nIter ?? 30
    this.#seed = opts.seed ?? 42
    this.#opts = opts
    this.#total = models.length * this.#nIter
    this.#modelCycle = models.map(m => m.name)
  }

  async init() {
    if (this.#initialized) return
    this.#initialized = true

    const rng = makeLCG(this.#seed)

    for (const model of this.#models) {
      const task = this.#opts.task || model.task || null
      const space = model.searchSpace || model.cls.defaultSearchSpace?.(task) || {}
      // Remove fixed params from space
      const effectiveSpace = { ...space }
      if (model.params) {
        for (const key of Object.keys(model.params)) {
          delete effectiveSpace[key]
        }
      }

      // Warm-up count: min(ceil(nIter * 0.4), max(3, nFreeParams + 1))
      const nFree = countFreeParams(effectiveSpace)
      const autoWarmup = min(ceil(this.#nIter * 0.4), max(3, nFree + 1))
      const nInitial = this.#opts.nInitial ?? autoWarmup

      this.#warmupCounts.set(model.name, nInitial)
      this.#reported.set(model.name, 0)

      // Generate warm-up configs (random)
      const configRng = makeLCG((rng() * 0x7fffffff) | 0)
      const queue = []
      for (let i = 0; i < nInitial; i++) {
        const config = sampleConfig(effectiveSpace, configRng)
        const params = { ...config, ...(model.params || {}) }
        const candidateId = makeCandidateId(model.name, params)
        queue.push({ candidateId, cls: model.cls, params })
      }
      this.#warmupQueues.set(model.name, queue)

      // Create BO optimizer
      const optSeed = (rng() * 0x7fffffff) | 0
      const optimizer = await BayesianOptimizer.create(effectiveSpace, {
        kernel: this.#opts.kernel || 'matern52',
        acquisitionFn: this.#opts.acquisitionFn || 'ei',
        kappa: this.#opts.kappa ?? 2.0,
        xi: this.#opts.xi ?? 0.01,
        seed: optSeed,
      })
      this.#optimizers.set(model.name, optimizer)
    }
  }

  /**
   * Return next candidate, or null when done.
   */
  next() {
    if (this.#yielded >= this.#total) return null

    // Round-robin across models
    for (let attempts = 0; attempts < this.#modelCycle.length; attempts++) {
      const modelName = this.#modelCycle[this.#cycleIdx]
      this.#cycleIdx = (this.#cycleIdx + 1) % this.#modelCycle.length

      const model = this.#models.find(m => m.name === modelName)
      const reported = this.#reported.get(modelName)
      const warmupCount = this.#warmupCounts.get(modelName)

      // Still in warm-up phase? Yield from queue
      const queue = this.#warmupQueues.get(modelName)
      if (queue.length > 0) {
        this.#yielded++
        return queue.shift()
      }

      // BO phase: need all warm-up reported before suggesting
      if (reported < warmupCount) continue

      // Use BO to suggest
      const optimizer = this.#optimizers.get(modelName)
      const effectiveSpace = this.#getEffectiveSpace(model)
      const suggested = optimizer.suggest()
      const params = { ...suggested, ...(model.params || {}) }
      const candidateId = makeCandidateId(modelName, params)
      this.#yielded++
      return { candidateId, cls: model.cls, params }
    }

    // All models waiting for warm-up reports
    return null
  }

  /**
   * Report evaluation result. Feeds observation to the model's optimizer.
   */
  report(result) {
    const { candidateId, meanScore } = result
    // Extract model name from candidateId (format: "modelName:{params}")
    const colonIdx = candidateId.indexOf(':')
    if (colonIdx === -1) return
    const modelName = candidateId.substring(0, colonIdx)

    const optimizer = this.#optimizers.get(modelName)
    if (!optimizer) return

    const model = this.#models.find(m => m.name === modelName)
    if (!model) return

    // Use result.params if available, otherwise parse from candidateId
    const params = result.params || JSON.parse(candidateId.substring(colonIdx + 1))

    // Remove fixed params before observing (optimizer only knows the search space)
    const observeParams = { ...params }
    if (model.params) {
      for (const key of Object.keys(model.params)) {
        delete observeParams[key]
      }
    }

    // Guard: only feed finite scores to the optimizer
    if (typeof meanScore === 'number' && isFinite(meanScore)) {
      try {
        optimizer.observe(observeParams, meanScore)
      } catch (_) {
        // Observation may fail for degenerate params; ignore
      }
    }

    this.#reported.set(modelName, (this.#reported.get(modelName) || 0) + 1)
  }

  isDone() {
    return this.#yielded >= this.#total
  }

  dispose() {
    if (this.#disposed) return
    this.#disposed = true
    for (const opt of this.#optimizers.values()) {
      opt.dispose()
    }
    this.#optimizers.clear()
  }

  #getEffectiveSpace(model) {
    const task = this.#opts.task || model.task || null
    const space = model.searchSpace || model.cls.defaultSearchSpace?.(task) || {}
    const eff = { ...space }
    if (model.params) {
      for (const key of Object.keys(model.params)) {
        delete eff[key]
      }
    }
    return eff
  }
}

module.exports = { BayesianStrategy }
