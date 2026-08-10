const { BayesianOptimizer } = require('./optimizer.js')
const { compileSpace, encodeParams, decodeParams, countFreeParams } = require('./compile.js')
const { loadBO } = require('./wasm.js')

module.exports = {
  BayesianOptimizer,
  compileSpace,
  encodeParams,
  decodeParams,
  countFreeParams,
  loadBO,
}
