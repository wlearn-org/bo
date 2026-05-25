/**
 * search.js -- BayesianSearch (automl Search wrapper)
 *
 * Same pattern as RandomSearch: creates folds, Executor, BayesianStrategy;
 * runs; disposes. This is what automl's auto-fit.js instantiates.
 */

const { stratifiedKFold, kFold, normalizeX, normalizeY,
  ValidationError } = require('@wlearn/core')
const { Executor, detectTask, scorerGreaterIsBetter } = require('@wlearn/automl')
const { BayesianStrategy } = require('./strategy.js')

class BayesianSearch {
  #models
  #opts
  #leaderboard = null
  #bestResult = null

  /**
   * @param {Array<{ name: string, cls: object, searchSpace?: object, params?: object }>} models
   * @param {object} opts
   */
  constructor(models, opts = {}) {
    if (!models || models.length === 0) {
      throw new ValidationError('BayesianSearch: at least one model is required')
    }
    this.#models = models
    this.#opts = {
      scoring: null,
      cv: 5,
      seed: 42,
      task: null,
      nIter: 30,
      maxTimeMs: 0,
      onProgress: null,
      acquisitionFn: 'ei',
      kappa: 2.0,
      xi: 0.01,
      kernel: 'matern52',
      nInitial: null,
      ...opts,
    }
  }

  async fit(X, y) {
    const Xn = normalizeX(X)
    const yn = normalizeY(y)
    const task = this.#opts.task || detectTask(yn)
    const scoring = this.#opts.scoring || (task === 'classification' ? 'accuracy' : 'r2')
    const { cv, seed, nIter, maxTimeMs, onProgress,
      acquisitionFn, kappa, xi, kernel, nInitial } = this.#opts

    const folds = task === 'classification'
      ? stratifiedKFold(yn, cv, { shuffle: true, seed })
      : kFold(yn.length, cv, { shuffle: true, seed })

    const executor = new Executor({
      folds,
      scoring,
      X: Xn,
      y: yn,
      timeLimitMs: maxTimeMs,
      seed,
      onProgress,
    })

    const strategy = new BayesianStrategy(this.#models, {
      nIter,
      seed,
      acquisitionFn,
      kappa,
      xi,
      kernel,
      nInitial,
      task,
    })

    await strategy.init()

    let result
    try {
      result = await executor.runStrategy(strategy)
    } finally {
      strategy.dispose()
    }

    const { leaderboard } = result

    if (leaderboard.length === 0) {
      throw new ValidationError('BayesianSearch: no candidates were evaluated')
    }

    this.#leaderboard = leaderboard
    this.#bestResult = leaderboard.best()
    return { leaderboard, bestResult: this.#bestResult }
  }

  async refitBest(X, y) {
    if (!this.#bestResult) {
      throw new ValidationError('BayesianSearch: must call fit() first')
    }
    const best = this.#bestResult
    const model = this.#models.find(m => m.name === best.modelName)
    const instance = await model.cls.create(best.params)
    const Xn = normalizeX(X)
    const yn = normalizeY(y)
    instance.fit(Xn, yn)
    return instance
  }

  get leaderboard() { return this.#leaderboard }
  get bestResult() { return this.#bestResult }
}

module.exports = { BayesianSearch }
