from functools import wraps
import warnings

def deprecated(func):
    """
    This function is a decorator, which diplays a deprecation warning.
    """
    @wraps(func)
    def __inner(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn("{}".format(func.__name__), DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)
        return func(*args, **kwargs)
    return __inner

