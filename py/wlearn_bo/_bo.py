"""
BayesianOptimizer -- thin Python wrapper around the C11 BO engine.

Handles:
- SearchSpace compilation (Python dicts -> flat numeric arrays)
- ctypes memory management
- Categorical value mapping (complex values via JSON equality)
- Multi-parent condition handling (first categorical parent -> C, rest -> Python)
"""

import ctypes
import json
import math

from wlearn_bo._ffi import get_lib

# Kernel enum values matching C
KERNELS = {'matern52': 0, 'matern32': 1, 'se': 2}

# Acquisition function enum values matching C
ACQ_FNS = {'ei': 0, 'ucb': 1, 'pi': 2}

# Dimension type enum values matching C
DIM_CONTINUOUS = 0
DIM_INTEGER = 1
DIM_CATEGORICAL = 2


def _find_value_index(values, target):
    """Find target in values list. Uses == first, falls back to JSON equality."""
    for i, v in enumerate(values):
        if v == target:
            return i
    target_str = json.dumps(target, sort_keys=True)
    for i, v in enumerate(values):
        if json.dumps(v, sort_keys=True) == target_str:
            return i
    return -1


def _compile_space(search_space):
    """
    Compile a SearchSpace dict into numeric IR for the C engine.

    search_space: { param_name: { type, low, high, values, condition, ... }, ... }
    Returns a dict with all arrays and mappings needed for encode/decode.
    """
    param_names = list(search_space.keys())
    n = len(param_names)
    name_to_id = {name: i for i, name in enumerate(param_names)}

    param_types = [0] * n
    lows = [0.0] * n
    highs = [0.0] * n
    log_flags = [0] * n
    n_values = [0] * n
    cond_parents = [-1] * n
    cond_values = [-1] * n
    conditions = [None] * n
    value_maps = {}

    for i, name in enumerate(param_names):
        param = search_space[name]
        ptype = param['type']

        if ptype == 'categorical':
            param_types[i] = DIM_CATEGORICAL
            vals = param['values']
            n_values[i] = len(vals)
            value_maps[name] = vals
        elif ptype == 'uniform':
            param_types[i] = DIM_CONTINUOUS
            lows[i] = param['low']
            highs[i] = param['high']
        elif ptype == 'log_uniform':
            param_types[i] = DIM_CONTINUOUS
            lows[i] = param['low']
            highs[i] = param['high']
            log_flags[i] = 1
        elif ptype == 'int_uniform':
            param_types[i] = DIM_INTEGER
            lows[i] = param['low']
            highs[i] = param['high']
        elif ptype == 'int_log_uniform':
            param_types[i] = DIM_INTEGER
            lows[i] = param['low']
            highs[i] = param['high']
            log_flags[i] = 1
        else:
            raise ValueError(f'compile_space: unknown param type "{ptype}"')

        # Resolve conditions
        condition = param.get('condition')
        if condition:
            entries = list(condition.items())
            if not entries:
                continue
            conditions[i] = condition

            c_parent_set = False
            for parent_name, required_value in entries:
                parent_id = name_to_id.get(parent_name)
                if parent_id is None:
                    raise ValueError(
                        f'compile_space: param "{name}" has condition on '
                        f'unknown param "{parent_name}"'
                    )
                parent_param = search_space[parent_name]
                if parent_param['type'] != 'categorical':
                    continue
                parent_value_index = _find_value_index(
                    parent_param['values'], required_value
                )
                if parent_value_index == -1:
                    raise ValueError(
                        f'compile_space: condition value {json.dumps(required_value)} '
                        f'not found in parent "{parent_name}" values'
                    )
                if not c_parent_set:
                    cond_parents[i] = parent_id
                    cond_values[i] = parent_value_index
                    c_parent_set = True

    return {
        'param_names': param_names,
        'name_to_id': name_to_id,
        'param_types': param_types,
        'lows': lows,
        'highs': highs,
        'log_flags': log_flags,
        'n_values': n_values,
        'cond_parents': cond_parents,
        'cond_values': cond_values,
        'conditions': conditions,
        'value_maps': value_maps,
        'n_dims': n,
    }


def _encode_params(compiled, params):
    """Encode a Python params dict into a flat double array for C."""
    n = compiled['n_dims']
    param_names = compiled['param_names']
    param_types = compiled['param_types']
    value_maps = compiled['value_maps']
    out = [0.0] * n

    for i in range(n):
        name = param_names[i]
        value = params.get(name)
        if value is None:
            out[i] = 0.0
            continue
        if param_types[i] == DIM_CATEGORICAL:
            vals = value_maps[name]
            idx = _find_value_index(vals, value)
            if idx == -1:
                raise ValueError(
                    f'encode_params: value {json.dumps(value)} not found '
                    f'in param "{name}" values'
                )
            out[i] = float(idx)
        else:
            out[i] = float(value)
    return out


def _is_condition_satisfied(condition, decoded_params):
    """Check if all condition keys are satisfied (conjunctive)."""
    for key, required_value in condition.items():
        actual = decoded_params.get(key)
        if actual == required_value:
            continue
        if json.dumps(actual, sort_keys=True) != json.dumps(required_value, sort_keys=True):
            return False
    return True


def _decode_params(compiled, doubles):
    """Decode a flat double array from C into a Python params dict."""
    n = compiled['n_dims']
    param_names = compiled['param_names']
    param_types = compiled['param_types']
    value_maps = compiled['value_maps']
    conditions = compiled['conditions']
    params = {}

    # First pass: decode categoricals (needed for condition evaluation)
    for i in range(n):
        if param_types[i] == DIM_CATEGORICAL:
            idx = round(doubles[i])
            vals = value_maps[param_names[i]]
            params[param_names[i]] = vals[idx]

    # Second pass: decode continuous/integer, respecting conditions
    for i in range(n):
        if param_types[i] == DIM_CATEGORICAL:
            continue
        if conditions[i] is not None:
            if not _is_condition_satisfied(conditions[i], params):
                continue
        value = doubles[i]
        if param_types[i] == DIM_INTEGER:
            value = round(value)
        params[param_names[i]] = value

    return params


class BayesianOptimizer:
    """
    Bayesian optimization with Gaussian processes.

    Thin wrapper around the C11 BO engine. All optimization policy
    (GP fitting, acquisition optimization, Thompson sampling for
    categoricals, per-context GPs) runs in native code.

    Usage:
        opt = BayesianOptimizer({
            'learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1.0},
            'n_layers': {'type': 'int_uniform', 'low': 1, 'high': 5},
            'activation': {'type': 'categorical', 'values': ['relu', 'tanh']},
        }, seed=42)

        for _ in range(50):
            params = opt.suggest()
            score = evaluate(params)
            opt.observe(params, score)

        print('Best:', opt.best_score)
        opt.dispose()
    """

    def __init__(self, search_space, kernel='matern52', acquisition_fn='ei',
                 kappa=2.0, xi=0.01, n_restarts_hyper=5, n_restarts_acq=10,
                 max_obs=500, min_obs_for_gp=3, seed=42):
        lib = get_lib()

        kernel_id = KERNELS.get(kernel)
        if kernel_id is None:
            raise ValueError(f'Unknown kernel: {kernel}')
        acq_id = ACQ_FNS.get(acquisition_fn)
        if acq_id is None:
            raise ValueError(f'Unknown acquisition_fn: {acquisition_fn}')

        self._compiled = _compile_space(search_space)
        n = self._compiled['n_dims']

        # Build space in C
        self._space_ptr = lib.wl_bo_space_create(n)
        if not self._space_ptr:
            raise RuntimeError('Failed to create space: ' + _get_error())

        try:
            self._build_space(lib, n)
        except Exception:
            lib.wl_bo_space_free(self._space_ptr)
            self._space_ptr = None
            raise

        lib.wl_bo_space_finalize(self._space_ptr)

        self._state_ptr = lib.wl_bo_create(
            self._space_ptr, kernel_id, acq_id,
            kappa, xi, n_restarts_hyper, n_restarts_acq,
            max_obs, min_obs_for_gp, seed,
        )
        if not self._state_ptr:
            lib.wl_bo_space_free(self._space_ptr)
            self._space_ptr = None
            raise RuntimeError('Failed to create optimizer: ' + _get_error())

        self._disposed = False
        self._lib = lib

    def _build_space(self, lib, n):
        """Add all dimensions and conditions to the C space."""
        c = self._compiled
        for i in range(n):
            ptype = c['param_types'][i]
            if ptype == DIM_CONTINUOUS:
                rc = lib.wl_bo_space_add_continuous(
                    self._space_ptr, i, c['lows'][i], c['highs'][i], c['log_flags'][i]
                )
            elif ptype == DIM_INTEGER:
                rc = lib.wl_bo_space_add_integer(
                    self._space_ptr, i, c['lows'][i], c['highs'][i], c['log_flags'][i]
                )
            elif ptype == DIM_CATEGORICAL:
                rc = lib.wl_bo_space_add_categorical(
                    self._space_ptr, i, c['n_values'][i]
                )
            else:
                raise ValueError(f'Unknown param type id: {ptype}')

            if rc != 0:
                raise RuntimeError(f'Failed to add dim {i}: ' + _get_error())

            if c['cond_parents'][i] >= 0:
                rc = lib.wl_bo_space_add_condition(
                    self._space_ptr, i, c['cond_parents'][i], c['cond_values'][i]
                )
                if rc != 0:
                    raise RuntimeError(
                        f'Failed to add condition for dim {i}: ' + _get_error()
                    )

    def observe(self, params, score):
        """
        Observe a (params, score) pair.

        Args:
            params: dict mapping param names to values
            score: objective value (higher is better), must be finite
        """
        self._check_disposed()
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(
                f'observe: score must be a finite number, got {score}'
            )
        encoded = _encode_params(self._compiled, params)
        n = self._compiled['n_dims']
        arr = (ctypes.c_double * n)(*encoded)
        rc = self._lib.wl_bo_observe(self._state_ptr, arr, float(score))
        if rc != 0:
            raise RuntimeError('observe failed: ' + _get_error())

    def suggest(self):
        """
        Suggest next params to evaluate.

        Returns:
            dict mapping param names to values
        """
        self._check_disposed()
        n = self._compiled['n_dims']
        arr = (ctypes.c_double * n)()
        rc = self._lib.wl_bo_suggest(self._state_ptr, arr)
        if rc != 0:
            raise RuntimeError('suggest failed: ' + _get_error())
        return _decode_params(self._compiled, list(arr))

    def suggest_liar(self, hallucinated_y):
        """
        Suggest with constant liar for batch BO.

        Args:
            hallucinated_y: score to hallucinate for pending suggestions

        Returns:
            dict mapping param names to values
        """
        self._check_disposed()
        n = self._compiled['n_dims']
        arr = (ctypes.c_double * n)()
        rc = self._lib.wl_bo_suggest_liar(
            self._state_ptr, arr, float(hallucinated_y)
        )
        if rc != 0:
            raise RuntimeError('suggest_liar failed: ' + _get_error())
        return _decode_params(self._compiled, list(arr))

    def suggest_batch(self, n):
        """
        Suggest a batch of n points using constant liar.

        Returns:
            list of param dicts
        """
        if n <= 0:
            return []
        results = [self.suggest()]
        best = self.best_score
        for _ in range(n - 1):
            results.append(self.suggest_liar(best))
        return results

    @property
    def n_obs(self):
        """Number of observations recorded."""
        self._check_disposed()
        return self._lib.wl_bo_get_n_obs(self._state_ptr)

    @property
    def best_score(self):
        """Best observed score."""
        self._check_disposed()
        return self._lib.wl_bo_get_best_score(self._state_ptr)

    @property
    def n_contexts(self):
        """Number of categorical contexts created."""
        self._check_disposed()
        return self._lib.wl_bo_get_n_contexts(self._state_ptr)

    def dispose(self):
        """Free C resources. Safe to call multiple times."""
        if self._disposed:
            return
        self._disposed = True
        if self._state_ptr:
            self._lib.wl_bo_free(self._state_ptr)
            self._state_ptr = None
        if self._space_ptr:
            self._lib.wl_bo_space_free(self._space_ptr)
            self._space_ptr = None

    def __del__(self):
        self.dispose()

    def _check_disposed(self):
        if self._disposed:
            raise RuntimeError('BayesianOptimizer: already disposed')


def _get_error():
    """Get last error message from C engine."""
    try:
        lib = get_lib()
        msg = lib.wl_bo_get_last_error()
        if msg:
            return msg.decode('utf-8')
        return '(unknown error)'
    except Exception:
        return '(error retrieval failed)'
