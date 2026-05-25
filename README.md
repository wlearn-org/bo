# @wlearn/bo

Bayesian optimization with Gaussian processes for hyperparameter tuning. Part of the [wlearn](https://github.com/wlearn-org/wlearn) ecosystem.

C11 core compiled to WebAssembly. All BO policy (GP fitting, acquisition optimization, categorical Thompson sampling, per-context GPs) runs in native code. JS wrapper is a thin translation layer.

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
optimizer.dispose()
```

### With AutoML

```js
const { autoFit } = require('@wlearn/automl')

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
{
  algo: { type: 'categorical', values: ['svm', 'rf'] },
  C: { type: 'log_uniform', low: 0.01, high: 100, condition: { algo: 'svm' } },
  nTrees: { type: 'int_uniform', low: 10, high: 500, condition: { algo: 'rf' } },
}
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `kernel` | `'matern52'` | GP kernel: `'matern52'`, `'matern32'`, `'se'` |
| `acquisitionFn` | `'ei'` | Acquisition function: `'ei'`, `'ucb'`, `'pi'` |
| `kappa` | `2.0` | UCB exploration weight |
| `xi` | `0.01` | EI/PI exploration jitter |
| `seed` | `42` | Random seed for reproducibility |
| `maxObs` | `500` | Observation reservoir cap |

## License

Apache-2.0
