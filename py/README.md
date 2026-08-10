# wlearn-bo Python

Python wrapper for the wlearn C11 Bayesian optimization engine.

## Install

```bash
pip install wlearn-bo
```

## Example

```python
from wlearn_bo import BayesianOptimizer

space = {
    "lr": {"type": "log_uniform", "low": 1e-4, "high": 1.0},
    "depth": {"type": "int_uniform", "low": 2, "high": 12},
    "booster": {"type": "categorical", "values": ["gbtree", "dart"]},
    "dropout": {
        "type": "uniform",
        "low": 0.0,
        "high": 0.5,
        "condition": {"booster": "dart"},
    },
}

opt = BayesianOptimizer(space, seed=42)
for _ in range(30):
    params = opt.suggest()
    score = evaluate(params)
    opt.observe(params, score)

print(opt.best_score, opt.n_obs)
```

## API

- `BayesianOptimizer(search_space, **opts)` creates an optimizer.
- `suggest()` returns one parameter dict.
- `suggest_liar(score)` returns one suggestion using constant-liar batch BO.
- `suggest_batch(n)` returns `n` suggestions.
- `observe(params, score)` records a finite objective value; larger is better.
- `n_obs`, `best_score`, `n_contexts` expose optimizer state.
- `dispose()` releases native memory early in long-running optimization services.

Search parameter types: `uniform`, `log_uniform`, `int_uniform`,
`int_log_uniform`, `categorical`. Conditional parameters use
`condition: {parent: value}`.

Options: `kernel` (`matern52`, `matern32`, `se`), `acquisition_fn` (`ei`,
`ucb`, `pi`), `kappa`, `xi`, `n_restarts_hyper`, `n_restarts_acq`, `max_obs`,
`min_obs_for_gp`, `seed`.

BO state is intentionally ephemeral; warm-start by replaying `observe()` calls.

## Development

The canonical native source is repository root `src/`; `py/csrc/` is generated
for Python builds. Use `WLEARN_PYTHON=/path/to/python make test-py` from the
repository root to force a specific Python environment.
