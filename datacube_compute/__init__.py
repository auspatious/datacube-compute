from .backend import _geomedian


def geomedian(in_array, **kwargs):

    kwargs.setdefault("num_threads", 1)
    kwargs.setdefault("eps", 1e-6)
    kwargs.setdefault("maxiters", 1e-6)

    return _geomedian(in_array, kwargs["maxiters"], kwargs["eps"], kwargs["num_threads"])


__all__ = ("geomedian")