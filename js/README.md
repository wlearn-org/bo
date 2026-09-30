# @wlearn/bo

Bayesian optimization with Gaussian processes for hyperparameter tuning. Part of the [wlearn](https://github.com/wlearn-org/wlearn) ecosystem.

C11 core compiled to WebAssembly. All BO policy (GP fitting, acquisition optimization, categorical Thompson sampling, per-context GPs) runs in native code. JS wrapper is a thin translation layer.

## Package Layout

- `src/` is the JavaScript wrapper.
- `csrc/` is generated from repository root `src/` before build/pack.
- `wasm/` and `dist/` are generated package artifacts.
- `WLEARN_PYTHON` is honored by package lifecycle scripts when Emscripten needs Python.

## Install

```bash
npm install @wlearn/bo
```

## Usage

### Standalone optimizer

```js
const { BayesianOptimizer, loadBO } = require('@wlearn/bo')

await loadBO()

const optimizer = await BayesianOptimizer.create({
  learningRate: { type: 'log_uniform', low: 1e-4, high: 1.0 },
  nLayers: { type: 'int_uniform', low: 1, high: 5 },
  activation: { type: 'categorical', values: ['relu', 'tanh', 'gelu'] },
}, { seed: 42 })

for (let i = 0; i < 50; i++) {
  const params = optimizer.suggest()
  const score = evaluate(params)  // your objective function
  optimizer.observe(params, score)
}

console.log('Best score:', optimizer.bestScore)
```

### With AutoML

`@wlearn/automl` depends on `@wlearn/bo` and owns the AutoML search wrapper. Use BO directly for standalone objective optimization; use AutoML for model-selection workflows.

```js
const { autoFit } = require('@wlearn/automl')
const { LinearModel } = require('@wlearn/liblinear')
const models = [{ name: 'linear', classId: 'wlearn.liblinear.classifier@1', cls: LinearModel }]

const result = await autoFit(models, X, y, {
  strategy: 'bayesian',
  nIter: 30,
})
```

## Search space types

| Type | Description | Example |
|------|-------------|---------|
| `uniform` | Continuous uniform | `{ type: 'uniform', low: 0, high: 1 }` |
| `log_uniform` | Log-uniform continuous | `{ type: 'log_uniform', low: 1e-5, high: 1 }` |
| `int_uniform` | Integer uniform | `{ type: 'int_uniform', low: 1, high: 100 }` |
| `int_log_uniform` | Log-uniform integer | `{ type: 'int_log_uniform', low: 10, high: 1000 }` |
| `categorical` | Categorical choice | `{ type: 'categorical', values: ['a', 'b'] }` |

Conditional parameters are supported via `condition`:

```js
const space = {
  algo: { type: 'categorical', values: ['svm', 'rf'] },
  C: { type: 'log_uniform', low: 0.01, high: 100, condition: { algo: 'svm' } },
  nTrees: { type: 'int_uniform', low: 10, high: 500, condition: { algo: 'rf' } },
}
```

## API

Conditions may refer to parents declared later and may have multiple parents.
All parents must be active and match. Missing parents and cycles are rejected;
an empty condition is unconditional. Object values compare independently of key
order, while booleans remain distinct from numbers. Decoding evaluates dependencies
in order without changing the compiled coordinate order.


- `loadBO()` initializes the WASM module.
- `BayesianOptimizer.create(searchSpace, opts)` creates an optimizer.
- `suggest()` returns one parameter object.
- `suggestLiar(score)` returns one constant-liar suggestion for batch BO.
- `suggestBatch(n)` returns an array of suggestions.
- `observe(params, score)` records a finite score; larger is better.
- `nObs`, `bestScore`, `nContexts`, `compiled` expose optimizer state.
- `dispose()` releases WASM memory immediately in long-running optimization services.
- `compileSpace`, `encodeParams`, `decodeParams`, `countFreeParams` expose the search-space IR helpers.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `kernel` | `'matern52'` | GP kernel: `'matern52'`, `'matern32'`, `'se'` |
| `acquisitionFn` | `'ei'` | Acquisition function: `'ei'`, `'ucb'`, `'pi'` |
| `kappa` | `2.0` | UCB exploration weight |
| `xi` | `0.01` | EI/PI exploration jitter |
| `nRestartsHyper` | `5` | Hyperparameter optimization restarts |
| `nRestartsAcq` | `10` | Acquisition optimization restarts |
| `seed` | `42` | Random seed for reproducibility |
| `maxObs` | `500` | Observation reservoir cap |
| `minObsForGp` | `3` | Minimum observations before GP fitting |

BO state is ephemeral. Warm-start by replaying `observe()` calls.

## License

Apache-2.0
