"""
test_python.py -- Python tests for wlearn_bo (Bayesian optimization)
"""
import sys
import os
import math
import json

# Add the Python package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'python'))

from wlearn_bo import BayesianOptimizer
from wlearn_bo._bo import (
    _compile_space, _encode_params, _decode_params, _find_value_index,
    _is_condition_satisfied,
)

tests_run = 0
tests_passed = 0


def test(name, fn):
    global tests_run, tests_passed
    tests_run += 1
    try:
        fn()
        print(f'  PASS: {name}')
        tests_passed += 1
    except Exception as e:
        print(f'  FAIL: {name}')
        print(f'        {e}')


def assert_close(a, b, tol, msg=''):
    diff = abs(a - b)
    if diff > tol:
        raise AssertionError(msg or f'expected {a} ~ {b} (diff={diff}, tol={tol})')


# === compile tests ===
print('\n=== Compile tests ===')


def test_compile_continuous():
    space = {
        'lr': {'type': 'uniform', 'low': 0.001, 'high': 1.0},
        'depth': {'type': 'int_uniform', 'low': 1, 'high': 10},
    }
    c = _compile_space(space)
    assert c['n_dims'] == 2
    assert c['param_types'][0] == 0, 'lr should be continuous'
    assert c['param_types'][1] == 1, 'depth should be integer'
    assert_close(c['lows'][0], 0.001, 1e-9)
    assert_close(c['highs'][0], 1.0, 1e-9)

test('compile: continuous params', test_compile_continuous)


def test_compile_log_scale():
    space = {
        'lr': {'type': 'log_uniform', 'low': 1e-5, 'high': 1.0},
        'n': {'type': 'int_log_uniform', 'low': 10, 'high': 1000},
    }
    c = _compile_space(space)
    assert c['log_flags'][0] == 1, 'log_uniform should set log_flag'
    assert c['log_flags'][1] == 1, 'int_log_uniform should set log_flag'

test('compile: log scale', test_compile_log_scale)


def test_compile_categorical():
    space = {
        'kernel': {'type': 'categorical', 'values': ['rbf', 'linear', 'poly']},
    }
    c = _compile_space(space)
    assert c['param_types'][0] == 2, 'should be categorical'
    assert c['n_values'][0] == 3
    vals = c['value_maps']['kernel']
    assert vals[0] == 'rbf'
    assert vals[2] == 'poly'

test('compile: categorical', test_compile_categorical)


def test_compile_conditional():
    space = {
        'algo': {'type': 'categorical', 'values': ['svm', 'rf']},
        'C': {'type': 'log_uniform', 'low': 0.01, 'high': 100,
               'condition': {'algo': 'svm'}},
        'n_trees': {'type': 'int_uniform', 'low': 10, 'high': 500,
                    'condition': {'algo': 'rf'}},
    }
    c = _compile_space(space)
    assert c['cond_parents'][0] == -1, 'algo is unconditional'
    assert c['cond_parents'][1] == 0, 'C conditioned on algo'
    assert c['cond_values'][1] == 0, 'C requires algo=svm (index 0)'
    assert c['cond_parents'][2] == 0, 'n_trees conditioned on algo'
    assert c['cond_values'][2] == 1, 'n_trees requires algo=rf (index 1)'

test('compile: conditional params', test_compile_conditional)


def test_compile_complex_categorical():
    space = {
        'layers': {'type': 'categorical', 'values': [[64], [64, 64], [128, 64]]},
    }
    c = _compile_space(space)
    assert c['n_values'][0] == 3
    vals = c['value_maps']['layers']
    assert vals[1] == [64, 64]

test('compile: complex categorical values', test_compile_complex_categorical)


# === encode/decode tests ===
print('\n=== Encode/decode tests ===')


def test_encode_continuous():
    space = {
        'lr': {'type': 'uniform', 'low': 0.001, 'high': 1.0},
        'depth': {'type': 'int_uniform', 'low': 1, 'high': 10},
    }
    c = _compile_space(space)
    params = {'lr': 0.5, 'depth': 7}
    encoded = _encode_params(c, params)
    assert_close(encoded[0], 0.5, 1e-9)
    assert_close(encoded[1], 7, 1e-9)
    decoded = _decode_params(c, encoded)
    assert_close(decoded['lr'], 0.5, 1e-9)
    assert decoded['depth'] == 7

test('encode/decode: continuous round-trip', test_encode_continuous)


def test_encode_categorical():
    space = {
        'kernel': {'type': 'categorical', 'values': ['rbf', 'linear', 'poly']},
        'C': {'type': 'uniform', 'low': 0.1, 'high': 10.0},
    }
    c = _compile_space(space)
    params = {'kernel': 'linear', 'C': 5.0}
    encoded = _encode_params(c, params)
    assert encoded[0] == 1, 'linear should be index 1'
    decoded = _decode_params(c, encoded)
    assert decoded['kernel'] == 'linear'
    assert_close(decoded['C'], 5.0, 1e-9)

test('encode/decode: categorical round-trip', test_encode_categorical)


def test_encode_complex_categorical():
    space = {
        'layers': {'type': 'categorical', 'values': [[64], [64, 64], [128, 64]]},
    }
    c = _compile_space(space)
    params = {'layers': [64, 64]}
    encoded = _encode_params(c, params)
    assert encoded[0] == 1, '[64,64] should be index 1'
    decoded = _decode_params(c, encoded)
    assert decoded['layers'] == [64, 64]

test('encode/decode: complex categorical round-trip', test_encode_complex_categorical)


def test_decode_conditional_exclusion():
    space = {
        'algo': {'type': 'categorical', 'values': ['svm', 'rf']},
        'C': {'type': 'uniform', 'low': 0.01, 'high': 100,
               'condition': {'algo': 'svm'}},
        'n_trees': {'type': 'int_uniform', 'low': 10, 'high': 500,
                    'condition': {'algo': 'rf'}},
    }
    c = _compile_space(space)
    # algo=rf (index 1), C should be excluded
    doubles = [1.0, 50.0, 200.0]
    decoded = _decode_params(c, doubles)
    assert decoded['algo'] == 'rf'
    assert decoded['n_trees'] == 200
    assert 'C' not in decoded, 'C should be excluded when algo=rf'

test('decode: conditional exclusion', test_decode_conditional_exclusion)


def test_compile_multi_parent():
    space = {
        'a': {'type': 'categorical', 'values': [1, 2]},
        'b': {'type': 'categorical', 'values': [3, 4]},
        'c': {'type': 'uniform', 'low': 0, 'high': 1,
               'condition': {'a': 1, 'b': 3}},
    }
    c = _compile_space(space)
    assert c['n_dims'] == 3
    assert c['cond_parents'][2] >= 0, 'should have a C-level condition parent'
    assert c['conditions'][2] is not None
    assert len(c['conditions'][2]) == 2

    # Decode: c included only when both a=1 AND b=3
    decoded_ok = _decode_params(c, [0.0, 0.0, 0.5])  # a=1, b=3
    assert_close(decoded_ok['c'], 0.5, 1e-9, 'c should be included')

    # Decode: c excluded when a=1 but b=4
    decoded_bad = _decode_params(c, [0.0, 1.0, 0.5])  # a=1, b=4
    assert 'c' not in decoded_bad, 'c should be excluded'

test('compile: multi-parent conditions', test_compile_multi_parent)


def test_compile_non_categorical_parent():
    space = {
        'a': {'type': 'uniform', 'low': 0, 'high': 1},
        'b': {'type': 'uniform', 'low': 0, 'high': 1,
               'condition': {'a': 0.5}},
    }
    c = _compile_space(space)
    assert c['n_dims'] == 2
    assert c['cond_parents'][1] == -1, 'non-categorical parent not registered with C'
    assert c['conditions'][1] is not None

    d1 = _decode_params(c, [0.5, 0.7])
    assert_close(d1['b'], 0.7, 1e-9, 'b included when a=0.5')
    d2 = _decode_params(c, [0.3, 0.7])
    assert 'b' not in d2, 'b excluded when a!=0.5'

test('compile: non-categorical condition parent', test_compile_non_categorical_parent)


def test_find_value_index():
    assert _find_value_index(['a', 'b', 'c'], 'b') == 1
    assert _find_value_index(['a', 'b', 'c'], 'd') == -1
    assert _find_value_index([[1, 2], [3, 4]], [3, 4]) == 1

test('_find_value_index', test_find_value_index)


# === Optimizer tests (require libbo.so) ===
print('\n=== Optimizer tests ===')


def test_create_dispose():
    space = {'x': {'type': 'uniform', 'low': 0, 'high': 1}}
    opt = BayesianOptimizer(space, seed=1)
    assert opt.n_obs == 0
    opt.dispose()
    opt.dispose()  # double dispose should not throw

test('create and dispose', test_create_dispose)


def test_suggest_valid():
    space = {
        'x': {'type': 'uniform', 'low': -5, 'high': 5},
        'y': {'type': 'int_uniform', 'low': 0, 'high': 10},
    }
    opt = BayesianOptimizer(space, seed=42)
    params = opt.suggest()
    assert isinstance(params['x'], float), 'x should be float'
    assert -5 <= params['x'] <= 5, f'x={params["x"]} out of range'
    assert isinstance(params['y'], int), 'y should be int'
    assert 0 <= params['y'] <= 10, f'y={params["y"]} out of range'
    opt.dispose()

test('suggest returns valid params', test_suggest_valid)


def test_observe_and_suggest():
    space = {'x': {'type': 'uniform', 'low': 0, 'high': 10}}
    opt = BayesianOptimizer(space, seed=42)
    for i in range(0, 11, 2):
        opt.observe({'x': float(i)}, -(i - 5) ** 2)
    assert opt.n_obs == 6
    s = opt.suggest()
    assert isinstance(s['x'], float)
    assert 0 <= s['x'] <= 10
    opt.dispose()

test('observe and suggest', test_observe_and_suggest)


def test_categorical():
    space = {
        'algo': {'type': 'categorical', 'values': ['a', 'b', 'c']},
        'lr': {'type': 'uniform', 'low': 0.01, 'high': 1.0},
    }
    opt = BayesianOptimizer(space, seed=7)
    import random
    rng = random.Random(42)
    for i in range(10):
        lr = 0.01 + rng.random() * 0.99
        opt.observe({'algo': 'a', 'lr': lr}, 0.5)
        opt.observe({'algo': 'b', 'lr': lr}, 0.9)
        opt.observe({'algo': 'c', 'lr': lr}, 0.3)
    assert opt.n_obs == 30
    assert opt.n_contexts > 0
    s = opt.suggest()
    assert s['algo'] in ['a', 'b', 'c']
    assert 0.01 <= s['lr'] <= 1.0
    opt.dispose()

test('categorical params', test_categorical)


def test_conditional():
    space = {
        'algo': {'type': 'categorical', 'values': ['svm', 'rf']},
        'C': {'type': 'log_uniform', 'low': 0.01, 'high': 100,
               'condition': {'algo': 'svm'}},
        'n_trees': {'type': 'int_uniform', 'low': 10, 'high': 500,
                    'condition': {'algo': 'rf'}},
    }
    opt = BayesianOptimizer(space, seed=42)
    opt.observe({'algo': 'svm', 'C': 1.0}, 0.8)
    opt.observe({'algo': 'rf', 'n_trees': 100}, 0.85)
    opt.observe({'algo': 'svm', 'C': 10.0}, 0.75)
    assert opt.n_obs == 3
    s = opt.suggest()
    assert s['algo'] in ['svm', 'rf']
    opt.dispose()

test('conditional params', test_conditional)


def test_1d_convergence():
    space = {'x': {'type': 'uniform', 'low': 0, 'high': 6}}
    opt = BayesianOptimizer(space, seed=42, n_restarts_hyper=3, n_restarts_acq=5)
    best_score = -math.inf
    for _ in range(25):
        params = opt.suggest()
        x = params['x']
        score = math.sin(x) + math.sin(3 * x)
        opt.observe(params, score)
        if score > best_score:
            best_score = score
    assert best_score > 0.5, f'1D BO should converge, got {best_score}'
    opt.dispose()

test('1D optimization converges', test_1d_convergence)


def test_constant_liar_batch():
    space = {
        'x': {'type': 'uniform', 'low': 0, 'high': 10},
        'y': {'type': 'uniform', 'low': 0, 'high': 10},
    }
    opt = BayesianOptimizer(space, seed=42)
    for i in range(5):
        opt.observe({'x': float(i * 2), 'y': float(i * 2)}, -(i - 2.5) ** 2)
    batch = opt.suggest_batch(3)
    assert len(batch) == 3
    strs = [json.dumps(p, sort_keys=True) for p in batch]
    unique = set(strs)
    assert len(unique) >= 2, 'batch suggestions should be mostly distinct'
    opt.dispose()

test('constant liar batch', test_constant_liar_batch)


def test_determinism():
    space = {
        'x': {'type': 'uniform', 'low': 0, 'high': 10},
        'cat': {'type': 'categorical', 'values': ['a', 'b']},
    }

    def run(seed):
        opt = BayesianOptimizer(space, seed=seed)
        results = []
        for _ in range(5):
            p = opt.suggest()
            results.append(json.dumps(p, sort_keys=True))
            opt.observe(p, math.sin(p.get('x', 0)))
        opt.dispose()
        return results

    r1 = run(123)
    r2 = run(123)
    for i in range(len(r1)):
        assert r1[i] == r2[i], f'diverged at step {i}: {r1[i]} vs {r2[i]}'

test('determinism', test_determinism)


def test_degenerate_constant_y():
    space = {'x': {'type': 'uniform', 'low': 0, 'high': 1}}
    opt = BayesianOptimizer(space, seed=42)
    for i in range(5):
        opt.observe({'x': i * 0.2}, 1.0)
    p = opt.suggest()
    assert isinstance(p['x'], float) and 0 <= p['x'] <= 1
    opt.dispose()

test('degenerate data (constant y)', test_degenerate_constant_y)


def test_non_finite_score_rejected():
    space = {'x': {'type': 'uniform', 'low': 0, 'high': 1}}
    opt = BayesianOptimizer(space, seed=1)

    for bad_score in [None, float('nan'), float('inf'), float('-inf')]:
        threw = False
        try:
            opt.observe({'x': 0.5}, bad_score)
        except (ValueError, TypeError):
            threw = True
        assert threw, f'should reject score={bad_score}'

    opt.observe({'x': 0.5}, 1.0)
    assert opt.n_obs == 1
    opt.dispose()

test('observe rejects non-finite score', test_non_finite_score_rejected)


def test_disposed_throws():
    space = {'x': {'type': 'uniform', 'low': 0, 'high': 1}}
    opt = BayesianOptimizer(space, seed=1)
    opt.dispose()
    threw = False
    try:
        opt.suggest()
    except RuntimeError:
        threw = True
    assert threw, 'suggest on disposed optimizer should throw'

test('disposed optimizer throws', test_disposed_throws)


def test_best_score_tracking():
    space = {'x': {'type': 'uniform', 'low': 0, 'high': 10}}
    opt = BayesianOptimizer(space, seed=1)
    opt.observe({'x': 2.0}, 0.5)
    opt.observe({'x': 5.0}, 0.9)
    opt.observe({'x': 8.0}, 0.3)
    assert_close(opt.best_score, 0.9, 1e-9, 'best score should be 0.9')
    opt.dispose()

test('best score tracking', test_best_score_tracking)


def test_integer_params_rounded():
    space = {'n': {'type': 'int_uniform', 'low': 1, 'high': 100}}
    opt = BayesianOptimizer(space, seed=42)
    for _ in range(10):
        p = opt.suggest()
        assert p['n'] == round(p['n']), f'n={p["n"]} should be integer'
        assert 1 <= p['n'] <= 100, f'n={p["n"]} out of range'
        opt.observe(p, -abs(p['n'] - 50))
    opt.dispose()

test('integer params rounded', test_integer_params_rounded)


def test_all_5_param_types():
    space = {
        'a': {'type': 'uniform', 'low': 0, 'high': 1},
        'b': {'type': 'log_uniform', 'low': 0.001, 'high': 100},
        'c': {'type': 'int_uniform', 'low': 1, 'high': 50},
        'd': {'type': 'int_log_uniform', 'low': 10, 'high': 1000},
        'e': {'type': 'categorical', 'values': ['x', 'y', 'z']},
    }
    opt = BayesianOptimizer(space, seed=42)
    p = opt.suggest()
    assert isinstance(p['a'], float) and 0 <= p['a'] <= 1
    assert isinstance(p['b'], float) and 0.001 <= p['b'] <= 100
    assert isinstance(p['c'], int) and 1 <= p['c'] <= 50
    assert isinstance(p['d'], int) and 10 <= p['d'] <= 1000
    assert p['e'] in ['x', 'y', 'z']
    opt.dispose()

test('all 5 param types', test_all_5_param_types)


def test_reservoir():
    space = {'x': {'type': 'uniform', 'low': 0, 'high': 1}}
    opt = BayesianOptimizer(space, seed=42, max_obs=50)
    for i in range(100):
        opt.observe({'x': i / 100.0}, math.sin(i / 100.0 * 6.28))
    assert opt.n_obs <= 100
    p = opt.suggest()
    assert isinstance(p['x'], float)
    opt.dispose()

test('reservoir handles many observations', test_reservoir)


# === JS-Python parity ===
print('\n=== Parity tests ===')


def test_parity_deterministic():
    """Same seed and space should produce the same sequence in Python as JS."""
    space = {'x': {'type': 'uniform', 'low': 0, 'high': 10}}
    opt = BayesianOptimizer(space, seed=42)
    first = opt.suggest()
    # Just verify it's valid and deterministic (actual parity test
    # requires running both JS and Python together)
    opt2 = BayesianOptimizer(space, seed=42)
    first2 = opt2.suggest()
    assert_close(first['x'], first2['x'], 1e-9, 'same seed should produce same suggestion')
    opt.dispose()
    opt2.dispose()

test('deterministic parity', test_parity_deterministic)


# === Summary ===
print(f'\n{tests_run} tests: {tests_passed} passed, {tests_run - tests_passed} failed\n')
sys.exit(0 if tests_passed == tests_run else 1)
