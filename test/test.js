let passed = 0
let failed = 0

async function test(name, fn) {
  try {
    await fn()
    console.log(`  PASS: ${name}`)
    passed++
  } catch (err) {
    console.log(`  FAIL: ${name}`)
    console.log(`        ${err.message}`)
    if (err.stack) {
      const lines = err.stack.split('\n').slice(1, 4)
      for (const l of lines) console.log(`        ${l.trim()}`)
    }
    failed++
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'assertion failed')
}

function assertClose(a, b, tol, msg) {
  const diff = Math.abs(a - b)
  if (diff > tol) throw new Error(msg || `expected ${a} ~ ${b} (diff=${diff}, tol=${tol})`)
}

const { ceil, min, max, sin, cos, round, abs } = Math

// ---------- Tests ----------

async function main() {
  console.log('\n=== @wlearn/bo tests ===\n')

  // ---- compile.js tests ----

  const { compileSpace, encodeParams, decodeParams, countFreeParams } = require('../src/compile.js')

  await test('compileSpace: continuous params', async () => {
    const space = {
      lr: { type: 'uniform', low: 0.001, high: 1.0 },
      depth: { type: 'int_uniform', low: 1, high: 10 },
    }
    const c = compileSpace(space)
    assert(c.nDims === 2)
    assert(c.paramTypes[0] === 0, 'lr should be continuous')
    assert(c.paramTypes[1] === 1, 'depth should be integer')
    assertClose(c.lows[0], 0.001, 1e-9)
    assertClose(c.highs[0], 1.0, 1e-9)
  })

  await test('compileSpace: log scale', async () => {
    const space = {
      lr: { type: 'log_uniform', low: 1e-5, high: 1.0 },
      n: { type: 'int_log_uniform', low: 10, high: 1000 },
    }
    const c = compileSpace(space)
    assert(c.logFlags[0] === 1, 'log_uniform should set logFlag')
    assert(c.logFlags[1] === 1, 'int_log_uniform should set logFlag')
  })

  await test('compileSpace: categorical', async () => {
    const space = {
      kernel: { type: 'categorical', values: ['rbf', 'linear', 'poly'] },
    }
    const c = compileSpace(space)
    assert(c.paramTypes[0] === 2, 'should be categorical')
    assert(c.nValues[0] === 3)
    const vals = c.valueMaps.get('kernel')
    assert(vals[0] === 'rbf')
    assert(vals[2] === 'poly')
  })

  await test('compileSpace: conditional params', async () => {
    const space = {
      algo: { type: 'categorical', values: ['svm', 'rf'] },
      C: { type: 'log_uniform', low: 0.01, high: 100, condition: { algo: 'svm' } },
      nTrees: { type: 'int_uniform', low: 10, high: 500, condition: { algo: 'rf' } },
    }
    const c = compileSpace(space)
    assert(c.condParents[0] === -1, 'algo is unconditional')
    assert(c.condParents[1] === 0, 'C conditioned on algo')
    assert(c.condValues[1] === 0, 'C requires algo=svm (index 0)')
    assert(c.condParents[2] === 0, 'nTrees conditioned on algo')
    assert(c.condValues[2] === 1, 'nTrees requires algo=rf (index 1)')
  })

  await test('compileSpace: complex categorical values', async () => {
    const space = {
      layers: { type: 'categorical', values: [[64], [64, 64], [128, 64]] },
    }
    const c = compileSpace(space)
    assert(c.nValues[0] === 3)
    const vals = c.valueMaps.get('layers')
    assert(JSON.stringify(vals[1]) === '[64,64]')
  })

  await test('encodeParams: round-trip continuous', async () => {
    const space = {
      lr: { type: 'uniform', low: 0.001, high: 1.0 },
      depth: { type: 'int_uniform', low: 1, high: 10 },
    }
    const c = compileSpace(space)
    const params = { lr: 0.5, depth: 7 }
    const encoded = encodeParams(c, params)
    assertClose(encoded[0], 0.5, 1e-9)
    assertClose(encoded[1], 7, 1e-9)
    const decoded = decodeParams(c, encoded)
    assertClose(decoded.lr, 0.5, 1e-9)
    assert(decoded.depth === 7)
  })

  await test('encodeParams: round-trip categorical', async () => {
    const space = {
      kernel: { type: 'categorical', values: ['rbf', 'linear', 'poly'] },
      C: { type: 'uniform', low: 0.1, high: 10.0 },
    }
    const c = compileSpace(space)
    const params = { kernel: 'linear', C: 5.0 }
    const encoded = encodeParams(c, params)
    assert(encoded[0] === 1, 'linear should be index 1')
    const decoded = decodeParams(c, encoded)
    assert(decoded.kernel === 'linear')
    assertClose(decoded.C, 5.0, 1e-9)
  })

  await test('encodeParams: complex categorical round-trip', async () => {
    const space = {
      layers: { type: 'categorical', values: [[64], [64, 64], [128, 64]] },
    }
    const c = compileSpace(space)
    const params = { layers: [64, 64] }
    const encoded = encodeParams(c, params)
    assert(encoded[0] === 1, '[64,64] should be index 1')
    const decoded = decodeParams(c, encoded)
    assert(JSON.stringify(decoded.layers) === '[64,64]')
  })

  await test('decodeParams: conditional exclusion', async () => {
    const space = {
      algo: { type: 'categorical', values: ['svm', 'rf'] },
      C: { type: 'uniform', low: 0.01, high: 100, condition: { algo: 'svm' } },
      nTrees: { type: 'int_uniform', low: 10, high: 500, condition: { algo: 'rf' } },
    }
    const c = compileSpace(space)
    // algo=rf (index 1), C should be excluded
    const doubles = new Float64Array([1, 50, 200])
    const decoded = decodeParams(c, doubles)
    assert(decoded.algo === 'rf')
    assert(decoded.nTrees === 200)
    assert(decoded.C === undefined, 'C should be excluded when algo=rf')
  })

  await test('countFreeParams', async () => {
    const space = {
      a: { type: 'uniform', low: 0, high: 1 },
      b: { type: 'categorical', values: [1, 2, 3] },
      c: { type: 'int_uniform', low: 0, high: 10, condition: { b: 1 } },
    }
    assert(countFreeParams(space) === 3)
  })

  // ---- WASM + optimizer tests ----

  const { BayesianOptimizer } = require('../src/optimizer.js')
  const { loadBO } = require('../src/wasm.js')

  await test('WASM loads', async () => {
    const wasm = await loadBO()
    assert(wasm, 'WASM module should exist')
    assert(typeof wasm._wl_bo_space_create === 'function')
  })

  await test('BayesianOptimizer: create and dispose', async () => {
    const space = { x: { type: 'uniform', low: 0, high: 1 } }
    const opt = await BayesianOptimizer.create(space, { seed: 1 })
    assert(opt.nObs === 0)
    opt.dispose()
    // Double dispose should not throw
    opt.dispose()
  })

  await test('BayesianOptimizer: suggest returns valid params', async () => {
    const space = {
      x: { type: 'uniform', low: -5, high: 5 },
      y: { type: 'int_uniform', low: 0, high: 10 },
    }
    const opt = await BayesianOptimizer.create(space, { seed: 42 })
    const params = opt.suggest()
    assert(typeof params.x === 'number', 'x should be a number')
    assert(params.x >= -5 && params.x <= 5, `x=${params.x} out of range`)
    assert(typeof params.y === 'number', 'y should be a number')
    assert(params.y === round(params.y), 'y should be integer')
    assert(params.y >= 0 && params.y <= 10, `y=${params.y} out of range`)
    opt.dispose()
  })

  await test('BayesianOptimizer: observe and suggest', async () => {
    const space = { x: { type: 'uniform', low: 0, high: 10 } }
    const opt = await BayesianOptimizer.create(space, { seed: 42 })

    // Observe a few points: f(x) = -(x-5)^2
    for (let i = 0; i <= 10; i += 2) {
      opt.observe({ x: i }, -(i - 5) * (i - 5))
    }
    assert(opt.nObs === 6)

    const suggested = opt.suggest()
    assert(typeof suggested.x === 'number')
    assert(suggested.x >= 0 && suggested.x <= 10)
    opt.dispose()
  })

  await test('BayesianOptimizer: categorical params', async () => {
    const space = {
      algo: { type: 'categorical', values: ['a', 'b', 'c'] },
      lr: { type: 'uniform', low: 0.01, high: 1.0 },
    }
    const opt = await BayesianOptimizer.create(space, { seed: 7 })

    // Observe: 'b' consistently better
    for (let i = 0; i < 10; i++) {
      const lr = 0.01 + Math.random() * 0.99
      opt.observe({ algo: 'a', lr }, 0.5)
      opt.observe({ algo: 'b', lr }, 0.9)
      opt.observe({ algo: 'c', lr }, 0.3)
    }
    assert(opt.nObs === 30)
    assert(opt.nContexts > 0, 'should have at least 1 context')

    const suggested = opt.suggest()
    assert(['a', 'b', 'c'].includes(suggested.algo), 'suggested algo should be valid')
    assert(suggested.lr >= 0.01 && suggested.lr <= 1.0)
    opt.dispose()
  })

  await test('BayesianOptimizer: conditional params', async () => {
    const space = {
      algo: { type: 'categorical', values: ['svm', 'rf'] },
      C: { type: 'log_uniform', low: 0.01, high: 100, condition: { algo: 'svm' } },
      nTrees: { type: 'int_uniform', low: 10, high: 500, condition: { algo: 'rf' } },
    }
    const opt = await BayesianOptimizer.create(space, { seed: 42 })
    opt.observe({ algo: 'svm', C: 1.0 }, 0.8)
    opt.observe({ algo: 'rf', nTrees: 100 }, 0.85)
    opt.observe({ algo: 'svm', C: 10.0 }, 0.75)
    assert(opt.nObs === 3)
    const s = opt.suggest()
    assert(['svm', 'rf'].includes(s.algo))
    opt.dispose()
  })

  await test('BayesianOptimizer: 1D optimization converges', async () => {
    // Minimize -(sin(x) + sin(3x)) on [0, 6]
    // Has global max around x~1.2 (sin(1.2)+sin(3.6) ~ 0.93+(-0.44) ~ 0.49)
    // and x~4.7 region
    const space = { x: { type: 'uniform', low: 0, high: 6 } }
    const opt = await BayesianOptimizer.create(space, {
      seed: 42,
      nRestartsHyper: 3,
      nRestartsAcq: 5,
    })

    let bestScore = -Infinity
    for (let i = 0; i < 25; i++) {
      const params = opt.suggest()
      const x = params.x
      const score = sin(x) + sin(3 * x)
      opt.observe(params, score)
      if (score > bestScore) bestScore = score
    }

    // sin(x)+sin(3x) max is ~1.56 near x=0.72
    assert(bestScore > 0.5, `1D BO should find decent score, got ${bestScore}`)
    opt.dispose()
  })

  await test('BayesianOptimizer: constant liar batch', async () => {
    const space = {
      x: { type: 'uniform', low: 0, high: 10 },
      y: { type: 'uniform', low: 0, high: 10 },
    }
    const opt = await BayesianOptimizer.create(space, { seed: 42 })

    // Need some observations first for GP
    for (let i = 0; i < 5; i++) {
      opt.observe({ x: i * 2, y: i * 2 }, -(i - 2.5) * (i - 2.5))
    }

    const batch = opt.suggestBatch(3)
    assert(batch.length === 3, 'should return 3 suggestions')
    // Suggestions should be distinct (not identical)
    const strs = batch.map(p => JSON.stringify(p))
    const unique = new Set(strs)
    assert(unique.size >= 2, 'batch suggestions should be mostly distinct')
    opt.dispose()
  })

  await test('BayesianOptimizer: determinism', async () => {
    const space = {
      x: { type: 'uniform', low: 0, high: 10 },
      cat: { type: 'categorical', values: ['a', 'b'] },
    }

    const run = async (seed) => {
      const opt = await BayesianOptimizer.create(space, { seed })
      const results = []
      for (let i = 0; i < 5; i++) {
        const p = opt.suggest()
        results.push(JSON.stringify(p))
        opt.observe(p, sin(p.x || 0))
      }
      opt.dispose()
      return results
    }

    const r1 = await run(123)
    const r2 = await run(123)
    for (let i = 0; i < r1.length; i++) {
      assert(r1[i] === r2[i], `run diverged at step ${i}: ${r1[i]} vs ${r2[i]}`)
    }
  })

  await test('BayesianOptimizer: degenerate data (constant y)', async () => {
    const space = { x: { type: 'uniform', low: 0, high: 1 } }
    const opt = await BayesianOptimizer.create(space, { seed: 42 })
    for (let i = 0; i < 5; i++) {
      opt.observe({ x: i * 0.2 }, 1.0) // all same score
    }
    // Should not crash, should return valid suggestion
    const p = opt.suggest()
    assert(typeof p.x === 'number' && p.x >= 0 && p.x <= 1)
    opt.dispose()
  })

  await test('BayesianOptimizer: observe rejects non-finite score', async () => {
    const space = { x: { type: 'uniform', low: 0, high: 1 } }
    const opt = await BayesianOptimizer.create(space, { seed: 1 })
    let threw = false
    try { opt.observe({ x: 0.5 }, undefined) } catch (_) { threw = true }
    assert(threw, 'should reject undefined score')
    threw = false
    try { opt.observe({ x: 0.5 }, NaN) } catch (_) { threw = true }
    assert(threw, 'should reject NaN score')
    threw = false
    try { opt.observe({ x: 0.5 }, Infinity) } catch (_) { threw = true }
    assert(threw, 'should reject Infinity score')
    // Valid score should work
    opt.observe({ x: 0.5 }, 1.0)
    assert(opt.nObs === 1)
    opt.dispose()
  })

  await test('BayesianOptimizer: disposed throws', async () => {
    const space = { x: { type: 'uniform', low: 0, high: 1 } }
    const opt = await BayesianOptimizer.create(space, { seed: 1 })
    opt.dispose()
    let threw = false
    try { opt.suggest() } catch (_) { threw = true }
    assert(threw, 'suggest on disposed optimizer should throw')
  })

  await test('BayesianOptimizer: best score tracking', async () => {
    const space = { x: { type: 'uniform', low: 0, high: 10 } }
    const opt = await BayesianOptimizer.create(space, { seed: 1 })
    opt.observe({ x: 2 }, 0.5)
    opt.observe({ x: 5 }, 0.9)
    opt.observe({ x: 8 }, 0.3)
    assertClose(opt.bestScore, 0.9, 1e-9, 'best score should be 0.9')
    opt.dispose()
  })

  await test('BayesianOptimizer: integer params rounded', async () => {
    const space = { n: { type: 'int_uniform', low: 1, high: 100 } }
    const opt = await BayesianOptimizer.create(space, { seed: 42 })
    for (let i = 0; i < 10; i++) {
      const p = opt.suggest()
      assert(p.n === round(p.n), `n=${p.n} should be integer`)
      assert(p.n >= 1 && p.n <= 100, `n=${p.n} out of range`)
      opt.observe(p, -abs(p.n - 50))
    }
    opt.dispose()
  })

  await test('BayesianOptimizer: all 5 param types', async () => {
    const space = {
      a: { type: 'uniform', low: 0, high: 1 },
      b: { type: 'log_uniform', low: 0.001, high: 100 },
      c: { type: 'int_uniform', low: 1, high: 50 },
      d: { type: 'int_log_uniform', low: 10, high: 1000 },
      e: { type: 'categorical', values: ['x', 'y', 'z'] },
    }
    const opt = await BayesianOptimizer.create(space, { seed: 42 })
    const p = opt.suggest()
    assert(typeof p.a === 'number' && p.a >= 0 && p.a <= 1)
    assert(typeof p.b === 'number' && p.b >= 0.001 && p.b <= 100)
    assert(typeof p.c === 'number' && p.c === round(p.c) && p.c >= 1 && p.c <= 50)
    assert(typeof p.d === 'number' && p.d === round(p.d) && p.d >= 10 && p.d <= 1000)
    assert(['x', 'y', 'z'].includes(p.e))
    opt.dispose()
  })

  await test('BayesianOptimizer: reservoir handles many observations', async () => {
    const space = { x: { type: 'uniform', low: 0, high: 1 } }
    const opt = await BayesianOptimizer.create(space, { seed: 42, maxObs: 50 })
    // Add more observations than maxObs
    for (let i = 0; i < 100; i++) {
      opt.observe({ x: i / 100 }, sin(i / 100 * 6.28))
    }
    // Should still work, nObs may be capped
    assert(opt.nObs <= 100, 'nObs should be manageable')
    const p = opt.suggest()
    assert(typeof p.x === 'number')
    opt.dispose()
  })

  // ---- compile edge cases ----

  await test('compileSpace: multi-parent conditions', async () => {
    const space = {
      a: { type: 'categorical', values: [1, 2] },
      b: { type: 'categorical', values: [3, 4] },
      c: { type: 'uniform', low: 0, high: 1, condition: { a: 1, b: 3 } },
    }
    const c = compileSpace(space)
    assert(c.nDims === 3)
    // First categorical parent should be registered with C
    assert(c.condParents[2] >= 0, 'should have a C-level condition parent')
    // Full condition stored for JS-side evaluation
    assert(c.conditions[2] !== null, 'should store full condition')
    assert(Object.keys(c.conditions[2]).length === 2, 'condition should have 2 keys')

    // Decode: c should be included only when both a=1 AND b=3
    const doubles_ok = new Float64Array([0, 0, 0.5])  // a=1, b=3
    const decoded_ok = decodeParams(c, doubles_ok)
    assertClose(decoded_ok.c, 0.5, 1e-9, 'c should be decoded when both conditions met')

    // Decode: c excluded when a=1 but b=4
    const doubles_bad = new Float64Array([0, 1, 0.5])  // a=1, b=4
    const decoded_bad = decodeParams(c, doubles_bad)
    assert(decoded_bad.c === undefined, 'c should be excluded when second condition fails')
  })

  await test('compileSpace: non-categorical condition parent', async () => {
    // Non-categorical parents are valid in the SearchSpace IR.
    // They can't be registered with C for context splitting, but
    // are evaluated in JS during decodeParams().
    const space = {
      a: { type: 'uniform', low: 0, high: 1 },
      b: { type: 'uniform', low: 0, high: 1, condition: { a: 0.5 } },
    }
    const c = compileSpace(space)
    assert(c.nDims === 2)
    // No C-level condition (parent is not categorical)
    assert(c.condParents[1] === -1, 'non-categorical parent should not register with C')
    // Full condition stored for JS-side evaluation
    assert(c.conditions[1] !== null, 'should store condition for JS eval')

    // Decode: b included only when a === 0.5
    const d1 = decodeParams(c, new Float64Array([0.5, 0.7]))
    assertClose(d1.b, 0.7, 1e-9, 'b included when a=0.5')
    const d2 = decodeParams(c, new Float64Array([0.3, 0.7]))
    assert(d2.b === undefined, 'b excluded when a!=0.5')
  })

  // ---- Summary ----

  console.log(`\n${passed} passed, ${failed} failed\n`)
  if (failed > 0) process.exit(1)
}

main().catch(err => {
  console.error('Test runner error:', err)
  process.exit(1)
})
