# Changelog

## [0.1.0] - 2026-03-13

### Added

- Bayesian optimization engine in C11 compiled to WebAssembly
- GP surrogate with ARD Matern 5/2, Matern 3/2, and Squared Exponential kernels
- Cholesky-based GP with adaptive jitter for numerical stability
- L-BFGS bounded optimizer for hyperparameters and acquisition functions
- EI, UCB, PI acquisition functions with multi-start optimization
- Thompson sampling (Beta distributions) for categorical parameter selection
- Per-context GPs: separate GP per categorical combination
- Conditional parameter support with multi-parent conditions
- Observation reservoir for memory-bounded operation
- SearchSpace compiler: all 5 param types (uniform, log_uniform, int_uniform, int_log_uniform, categorical)
- BayesianOptimizer: thin JS wrapper with WASM heap management
- BayesianStrategy: automl Strategy interface with warm-up phase
- BayesianSearch: automl Search wrapper with CV
- Constant liar batch suggestion via suggestLiar/suggestBatch
- Deterministic execution with seed control
- Browser bundles (IIFE + ESM via esbuild)
