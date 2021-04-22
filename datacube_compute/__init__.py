from .backend import _geomedian, _geomedian_int16, _geomedian_uint16
import numpy as np


def geomedian(in_array, **kwargs):

    kwargs.setdefault("num_threads", 1)
    kwargs.setdefault("eps", 1e-6)
    kwargs.setdefault("maxiters", 1000)
    kwargs.setdefault("scale", 1.0)
    kwargs.setdefault("offset", 0.0)

    if len(in_array.shape) != 4:
        raise ValueError(
            f"in_array: expected array to have 4 dimensions in format (y, x, band, time), found {len(in_array.shape)} dimensions."
        )

    if in_array.dtype == np.float32:
        kwargs.setdefault("nodata", np.nan)
        return _geomedian(in_array, kwargs["maxiters"], kwargs["eps"], kwargs["num_threads"], kwargs["scale"], kwargs["offset"])
    elif in_array.dtype == np.int16:
        kwargs.setdefault("nodata", -1)
        return _geomedian_int16(in_array, kwargs["maxiters"], kwargs["eps"], kwargs["num_threads"], kwargs["nodata"], kwargs["scale"], kwargs["offset"])
    elif in_array.dtype == np.uint16:
        kwargs.setdefault("nodata", 0)
        return _geomedian_uint16(in_array, kwargs["maxiters"], kwargs["eps"], kwargs["num_threads"], kwargs["nodata"], kwargs["scale"], kwargs["offset"])
    else:
        raise TypeError(f"in_array: expected dtype to be one of {np.float32}, {np.int16}, {np.uint16}, found {in_array.dtype}.")

    return func

__all__ = ("geomedian")