/**
 * bench.js -- Convergence benchmarks for @wlearn/bo
 *
 * Synthetic functions with known optima. Deterministic, reports environment.
 */

const { BayesianOptimizer, loadBO } = require('../src/index.js')

const { sin, cos, exp, abs, PI, sqrt } = Math

// ---------- Synthetic objectives ----------

// Branin 2D: f(x1, x2) where x1 in [-5, 10], x2 in [0, 15]
// Global min ~0.397887 at (pi, 2.275), (-pi, 12.275), (9.4248, 2.475)
function branin(x1, x2) {
  const a = 1, b = 5.1 / (4 * PI * PI), c = 5 / PI
  const r = 6, s = 10, t = 1 / (8 * PI)
  return a * (x2 - b * x1 * x1 + c * x1 - r) ** 2 + s * (1 - t) * cos(x1) + s
}

// Rosenbrock 2D: min 0 at (1,1), x in [-5, 10]
function rosenbrock(x1, x2) {
  return (1 - x1) ** 2 + 100 * (x2 - x1 * x1) ** 2
}

// Hartmann 6D: min ~-3.3224 at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
// x in [0, 1]^6
function hartmann6(...x) {
  const alpha = [1.0, 1.2, 3.0, 3.2]
  const A = [
    [10, 3, 17, 3.5, 1.7, 8],
    [0.05, 10, 17, 0.1, 8, 14],
    [3, 3.5, 1.7, 10, 17, 8],
    [17, 8, 0.05, 10, 0.1, 14],
  ]
  const P = [
    [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
    [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
    [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
    [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
  ]
  let result = 0
  for (let i = 0; i < 4; i++) {
    let inner = 0
    for (let j = 0; j < 6; j++) {
      inner += A[i][j] * (x[j] - P[i][j]) ** 2
    }
    result -= alpha[i] * exp(-inner)
  }
  return result
}

// ---------- Benchmark runner ----------

async function runBench(name, space, objective, encodeForObj, nEvals, knownOptimum) {
  const t0 = performance.now()
  const opt = await BayesianOptimizer.create(space, {
    seed: 42,
    nRestartsHyper: 3,
    nRestartsAcq: 10,
  })

  let bestScore = -Infinity
  let evalsToTarget = nEvals // never reached
  const target = knownOptimum !== null ? knownOptimum * 0.99 : null // within 1%

  for (let i = 0; i < nEvals; i++) {
    const params = opt.suggest()
    const score = -objective(encodeForObj(params)) // negate: BO maximizes, objectives are to minimize
    opt.observe(params, score)
    if (score > bestScore) {
      bestScore = score
      if (target !== null && -bestScore <= -target && evalsToTarget === nEvals) {
        evalsToTarget = i + 1
      }
    }
  }

  const elapsed = performance.now() - t0
  opt.dispose()

  const result = {
    name,
    nEvals,
    bestFound: -bestScore,
    knownOptimum,
    evalsToTarget: evalsToTarget < nEvals ? evalsToTarget : null,
    wallMs: elapsed.toFixed(1),
    msPerEval: (elapsed / nEvals).toFixed(1),
  }

  console.log(`  ${name}:`)
  console.log(`    best found:     ${result.bestFound.toFixed(6)}`)
  if (knownOptimum !== null) {
    console.log(`    known optimum:  ${knownOptimum.toFixed(6)}`)
  }
  if (result.evalsToTarget !== null) {
    console.log(`    evals to 1%:    ${result.evalsToTarget}`)
  }
  console.log(`    wall time:      ${result.wallMs} ms (${result.msPerEval} ms/eval)`)

  return result
}

async function main() {
  console.log('\n=== @wlearn/bo benchmarks ===\n')
  console.log(`  node: ${process.version}`)
  console.log(`  platform: ${process.platform} ${process.arch}`)
  console.log()

  await loadBO()

  // Branin 2D
  await runBench(
    'Branin 2D (50 evals)',
    {
      x1: { type: 'uniform', low: -5, high: 10 },
      x2: { type: 'uniform', low: 0, high: 15 },
    },
    (args) => branin(args[0], args[1]),
    (p) => [p.x1, p.x2],
    50,
    0.397887
  )

  console.log()

  // Rosenbrock 2D
  await runBench(
    'Rosenbrock 2D (80 evals)',
    {
      x1: { type: 'uniform', low: -5, high: 10 },
      x2: { type: 'uniform', low: -5, high: 10 },
    },
    (args) => rosenbrock(args[0], args[1]),
    (p) => [p.x1, p.x2],
    80,
    0.0
  )

  console.log()

  // Hartmann 6D
  await runBench(
    'Hartmann 6D (100 evals)',
    {
      x0: { type: 'uniform', low: 0, high: 1 },
      x1: { type: 'uniform', low: 0, high: 1 },
      x2: { type: 'uniform', low: 0, high: 1 },
      x3: { type: 'uniform', low: 0, high: 1 },
      x4: { type: 'uniform', low: 0, high: 1 },
      x5: { type: 'uniform', low: 0, high: 1 },
    },
    (args) => hartmann6(...args),
    (p) => [p.x0, p.x1, p.x2, p.x3, p.x4, p.x5],
    100,
    -3.3224
  )

  console.log()

  // Mixed: 2 continuous + 1 categorical (3 values)
  await runBench(
    'Mixed 2D+cat (50 evals)',
    {
      x: { type: 'uniform', low: 0, high: 6.28 },
      y: { type: 'uniform', low: 0, high: 6.28 },
      mode: { type: 'categorical', values: ['sin', 'cos', 'sincos'] },
    },
    (args) => {
      const [x, y, mode] = args
      switch (mode) {
        case 'sin': return -(sin(x) + sin(y))
        case 'cos': return -(cos(x) + cos(y))
        case 'sincos': return -(sin(x) + cos(y))
      }
    },
    (p) => [p.x, p.y, p.mode],
    50,
    -2.0 // max of sin+sin or cos+cos is 2
  )

  console.log('\n=== done ===\n')
}

main().catch(err => {
  console.error('Benchmark error:', err)
  process.exit(1)
})
