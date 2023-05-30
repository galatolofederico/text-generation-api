from collections.abc import MutableMapping as Map

def nested_update(d, v):
    for key in v:
        if key in d and isinstance(d[key], Map) and isinstance(v[key], Map):
            nested_update(d[key], v[key])
        else:
            d[key] = v[key]