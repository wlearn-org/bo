/**
 * compile.js -- SearchSpace -> numeric IR compiler
 *
 * Translates a JS SearchSpace object into the flat numeric representation
 * that the C BO engine expects. Handles:
 * - Mapping param names to integer IDs
 * - Mapping categorical values to integer indices
 * - Determining log_scale from param type
 * - Resolving conditions to (parent_id, parent_value_index) pairs
 * - Multi-parent conditions: first categorical parent goes to C,
 *   remaining conditions evaluated in JS during decode
 */

const { floor, round, log, exp } = Math
const { valueEqual, conditionOrder, conditionSatisfied } = require('./conditions.js')

/**
 * Look up a value in a categorical param's values array.
 * Uses portable value equality (object key order is irrelevant).
 * Returns index or -1.
 */
function findValueIndex(values, target) {
  return values.findIndex(value => valueEqual(value, target))
}

/**
 * Compile a SearchSpace into a numeric IR for the C engine.
 *
 * @param {object} searchSpace - { paramName: SearchParam, ... }
 * @returns {object} compiled IR
 */
function compileSpace(searchSpace) {
  const evaluationNames = conditionOrder(searchSpace)
  const paramNames = Object.keys(searchSpace)
  const n = paramNames.length
  const nameToId = new Map()
  for (let i = 0; i < n; i++) nameToId.set(paramNames[i], i)

  // Per-param arrays for C
  const paramTypes = new Int32Array(n)  // BO_DIM_*
  const lows = new Float64Array(n)
  const highs = new Float64Array(n)
  const logFlags = new Int32Array(n)
  const nValues = new Int32Array(n)
  const condParents = new Int32Array(n)
  const condValues = new Int32Array(n)

  // Full condition maps for JS-side evaluation (handles multi-parent)
  // conditions[i] = null | { parentName: requiredValue, ... }
  const conditions = new Array(n).fill(null)

  // Categorical value maps: paramName -> array of original values
  const valueMaps = new Map()

  for (let i = 0; i < n; i++) {
    const name = paramNames[i]
    const param = searchSpace[name]
    condParents[i] = -1
    condValues[i] = -1

    switch (param.type) {
      case 'categorical': {
        paramTypes[i] = 2 // BO_DIM_CATEGORICAL
        const vals = param.values
        nValues[i] = vals.length
        valueMaps.set(name, vals)
        break
      }
      case 'uniform':
        paramTypes[i] = 0 // BO_DIM_CONTINUOUS
        lows[i] = param.low
        highs[i] = param.high
        break
      case 'log_uniform':
        paramTypes[i] = 0 // BO_DIM_CONTINUOUS
        lows[i] = param.low
        highs[i] = param.high
        logFlags[i] = 1
        break
      case 'int_uniform':
        paramTypes[i] = 1 // BO_DIM_INTEGER
        lows[i] = param.low
        highs[i] = param.high
        break
      case 'int_log_uniform':
        paramTypes[i] = 1 // BO_DIM_INTEGER
        lows[i] = param.low
        highs[i] = param.high
        logFlags[i] = 1
        break
      default:
        throw new Error(`compileSpace: unknown param type "${param.type}"`)
    }

    // Resolve conditions
    if (param.condition) {
      const entries = Object.entries(param.condition)
      if (entries.length === 0) continue

      // Store full condition for JS-side decode evaluation
      conditions[i] = param.condition

      // Find first categorical parent to register with C engine.
      // Non-categorical parents are valid (evaluated in JS during decode)
      // but cannot be registered with C for context splitting.
      let cParentSet = false
      for (const [parentName, requiredValue] of entries) {
        const parentId = nameToId.get(parentName)
        if (parentId === undefined) {
          throw new Error(
            `compileSpace: param "${name}" has condition on unknown param "${parentName}"`
          )
        }
        const parentParam = searchSpace[parentName]
        // Only categorical parents can be registered with C
        if (parentParam.type !== 'categorical') continue
        const parentValueIndex = findValueIndex(parentParam.values, requiredValue)
        if (parentValueIndex === -1) {
          throw new Error(
            `compileSpace: condition value ${JSON.stringify(requiredValue)} not found in ` +
            `parent "${parentName}" values`
          )
        }
        // Register the first categorical parent with C (single-parent limit)
        if (!cParentSet) {
          condParents[i] = parentId
          condValues[i] = parentValueIndex
          cParentSet = true
        }
      }
    }
  }

  return {
    paramNames,
    evaluationOrder: evaluationNames.map(name => nameToId.get(name)),
    nameToId,
    paramTypes,
    lows,
    highs,
    logFlags,
    nValues,
    condParents,
    condValues,
    conditions,
    valueMaps,
    nDims: n,
  }
}

/**
 * Encode a JS params object into a flat double[] for the C engine.
 * Categorical values -> integer index; continuous/integer -> raw double.
 *
 * @param {object} compiled - from compileSpace()
 * @param {object} params - { paramName: value, ... }
 * @returns {Float64Array} encoded values (length = nDims)
 */
function encodeParams(compiled, params) {
  const { paramNames, paramTypes, valueMaps, nDims } = compiled
  const out = new Float64Array(nDims)

  for (let i = 0; i < nDims; i++) {
    const name = paramNames[i]
    const value = params[name]
    if (value === undefined) {
      // Inactive conditional param -- C ignores it, pass 0
      out[i] = 0
      continue
    }
    if (paramTypes[i] === 2) {
      // Categorical: map value to index
      const vals = valueMaps.get(name)
      const idx = findValueIndex(vals, value)
      if (idx === -1) {
        throw new Error(
          `encodeParams: value ${JSON.stringify(value)} not found in ` +
          `param "${name}" values`
        )
      }
      out[i] = idx
    } else {
      out[i] = value
    }
  }

  return out
}

/**
 * Decode a flat double[] from the C engine into a JS params object.
 * Integer indices -> original categorical values; integer dims -> Math.round().
 * Evaluates full multi-parent conditions in JS.
 *
 * @param {object} compiled - from compileSpace()
 * @param {Float64Array} doubles - from C (length = nDims)
 * @returns {object} decoded params { paramName: value, ... }
 */
function decodeParams(compiled, doubles) {
  const { paramNames, paramTypes, valueMaps, conditions } = compiled
  const params = {}

  // Evaluation order is separate from the persisted C coordinate order.
  for (const i of compiled.evaluationOrder) {
    if (!conditionSatisfied(conditions[i], params)) continue
    let value = doubles[i]
    if (paramTypes[i] === 2) {
      value = valueMaps.get(paramNames[i])[round(value)]
    } else if (paramTypes[i] === 1) {
      value = round(value)
    }
    params[paramNames[i]] = value
  }

  return params
}

/**
 * Count the number of free (non-fixed) params in a search space.
 * Used for warm-up sizing.
 */
function countFreeParams(searchSpace) {
  return Object.keys(searchSpace).length
}

module.exports = { compileSpace, encodeParams, decodeParams, countFreeParams }
