"""
wlearn_bo -- Bayesian optimization with Gaussian processes.

Thin Python wrapper around the C11 BO engine via ctypes.
No persistence (BO state is ephemeral). Warm-starting via observe() calls.
"""

from wlearn_bo._bo import BayesianOptimizer

__all__ = ['BayesianOptimizer']
