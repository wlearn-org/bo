"""Portable SearchSpace condition semantics, independent of AutoML."""


def value_equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(value_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(value_equal(x, y) for x, y in zip(a, b))
    return a == b


def condition_satisfied(condition, config):
    # Presence is separate from value equality: a missing parent is not null.
    return all(key in config and value_equal(config[key], value)
               for key, value in (condition or {}).items())


def condition_order(space):
    parents = {}
    for key, param in space.items():
        condition = param.get('condition')
        if condition is not None and not isinstance(condition, dict):
            raise ValueError(f'Condition for "{key}" must be an object')
        parents[key] = list(condition or {})
        for name in parents[key]:
            if name not in space:
                raise ValueError(f'Unknown condition parent "{name}" for "{key}"')
    order, done = [], set()
    # Stable dependency layers preserve unconditional seeded sampling order.
    while len(order) < len(space):
        ready = [key for key in space if key not in done
                 and all(name in done for name in parents[key])]
        if not ready:
            raise ValueError('Cyclic SearchSpace conditions')
        order.extend(ready)
        done.update(ready)
    return order
