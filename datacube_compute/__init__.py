from collections.abc import Iterable
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr

from .backend import (
    _geomedian,
    _geomedian_int16,
    _geomedian_uint16,
    _percentile_f32,
    _percentile_f64,
    _percentile_int8,
    _percentile_int16,
    _percentile_uint8,
    _percentile_uint16,
)


def geomedian(in_array, **kwargs):
    kwargs.setdefault("num_threads", 1)
    kwargs.setdefault("eps", 1e-6)
    kwargs.setdefault("maxiters", 1000)
    kwargs.setdefault("scale", 1.0)
    kwargs.setdefault("offset", 0.0)

    if len(in_array.shape) != 4:
        raise ValueError(
            (
                "in_array: expected array to have 4 dimensions in format"
                f"(y, x, band, time), found {len(in_array.shape)} dimensions."
            )
        )

    if in_array.dtype == np.float32:
        kwargs.setdefault("nodata", np.nan)
        return _geomedian(
            in_array,
            kwargs["maxiters"],
            kwargs["eps"],
            kwargs["num_threads"],
            kwargs["scale"],
            kwargs["offset"],
        )
    elif in_array.dtype == np.int16:
        kwargs.setdefault("nodata", -1)
        return _geomedian_int16(
            in_array,
            kwargs["maxiters"],
            kwargs["eps"],
            kwargs["num_threads"],
            kwargs["nodata"],
            kwargs["scale"],
            kwargs["offset"],
        )
    elif in_array.dtype == np.uint16:
        kwargs.setdefault("nodata", 0)
        return _geomedian_uint16(
            in_array,
            kwargs["maxiters"],
            kwargs["eps"],
            kwargs["num_threads"],
            kwargs["nodata"],
            kwargs["scale"],
            kwargs["offset"],
        )
    else:
        raise TypeError(
            (
                "in_array: expected dtype to be one of "
                "{np.float32}, {np.int16}, {np.uint16}, found {in_array.dtype}."
            )
        )


def percentile(in_array, percentiles, nodata=None):
    """
    Calculates the percentiles of the input data along the first axis.

    It accepts an array with shape (t, *other dims) and returns an array with shape
    (len(percentiles), *other dims) where the first index of the output array correspond
    to the percentiles.
    e.g. `out[i, :]` corresponds to the ith percentile

    :param in_array: a numpy array

    :param percentiles: A sequence of percentiles or singular percentile in the
    [0.0, 1.0] range

    :param nodata: The `nodata` value - this must have the same type as in_array.dtype
        and must be provided for integer datatypes. For float types this value is
        ignored and nodata is assumed to be NaN.

    """

    if isinstance(percentiles, Iterable):
        percentiles = np.array(list(percentiles))
    else:
        percentiles = np.array([percentiles])

    shape = in_array.shape
    in_array = in_array.reshape((shape[0], -1))

    if in_array.dtype == np.uint16:
        out_array = _percentile_uint16(in_array, percentiles, nodata)
    elif in_array.dtype == np.int16:
        out_array = _percentile_int16(in_array, percentiles, nodata)
    elif in_array.dtype == np.uint8:
        out_array = _percentile_uint8(in_array, percentiles, nodata)
    elif in_array.dtype == np.int8:
        out_array = _percentile_int8(in_array, percentiles, nodata)
    elif in_array.dtype == np.float32:
        out_array = _percentile_f32(in_array, percentiles)
    elif in_array.dtype == np.float64:
        out_array = _percentile_f64(in_array, percentiles)
    else:
        raise NotImplementedError

    return out_array.reshape((len(percentiles),) + shape[1:])


__all__ = ("geomedian", "percentile")


# From here: https://github.com/opendatacube/odc-algo/blob/add-rust-geomedian-impl/odc/algo/_geomedian.py
def geomedian_block_processor(
    input: xr.Dataset,
    nodata=None,
    scale=1,
    offset=0,
    eps=1e-6,
    maxiters=1000,
    num_threads=1,
    dim="time",
    is_float=True,
):
    array = input.to_array(dim="band").transpose("y", "x", "band", dim)

    if nodata is None:
        nodata = input.attrs.get("nodata", None)

    if nodata is None:
        # Grab the nodata value from our input array
        nodata_vals = set(
            dv.attrs.get("nodata", None) for dv in input.data_vars.values()
        )
        if len(nodata_vals) > 1:
            raise ValueError(
                "Data arrays have more than 1 nodata value across them", nodata_vals
            )
        elif len(nodata_vals) == 1:
            nodata = nodata_vals.pop()

    if nodata is None:
        if is_float:
            nodata = np.nan
        else:
            nodata = 0

    gm_data, mads = geomedian(
        array.data,
        nodata=nodata,
        num_threads=num_threads,
        eps=eps,
        maxiters=maxiters,
        scale=scale,
        offset=offset,
    )

    dims = ("y", "x", "band")
    coords = {k: array.coords[k] for k in dims}
    result = xr.DataArray(
        data=gm_data, dims=dims, coords=coords, attrs=array.attrs
    ).to_dataset("band")

    emad = mads[:, :, 0]
    smad = mads[:, :, 1]
    bcmad = mads[:, :, 2]

    result["emad"] = xr.DataArray(data=emad, dims=dims[:2], coords=result.coords)
    result["smad"] = xr.DataArray(data=smad, dims=dims[:2], coords=result.coords)
    result["bcmad"] = xr.DataArray(data=bcmad, dims=dims[:2], coords=result.coords)

    if np.isnan(nodata):
        count_good = np.all(~np.isnan(array.data), axis=2).sum(axis=2)
    else:
        count_good = np.all(array.data != nodata, axis=2).sum(axis=2)

    result["count"] = xr.DataArray(
        data=count_good, dims=dims[:2], coords=result.coords
    ).astype("uint16")

    # This is required to ensure that nodata is set per-band
    for band_name in result.data_vars:
        band = result[band_name]
        # Don't think these should all be nan
        if band_name in ["emad", "smad", "bcmad"]:
            band.attrs = dict(nodata=np.nan)
        elif band_name == "count":
            band.attrs = dict(nodata=9999)
        else:
            band.attrs = dict(nodata=nodata)

    return result


def geomedian_with_mads(
    src: Union[xr.Dataset, xr.DataArray],
    scale: float = 1.0,
    offset: float = 0.0,
    eps: Optional[float] = None,
    maxiters: int = 1000,
    num_threads: int = 1,
    work_chunks: Tuple[int, int] = (100, 100),
    is_float: bool = True,
) -> xr.Dataset:
    """
    Compute Geomedian on Dask backed Dataset.

    :param src: xr.Dataset or a single array in YXBT order, bands can be either
                float or integer with `nodata` values to indicate gaps in data.

    :param compute_mads: Whether to compute smad,emad,bcmad statistics

    :param compute_count: Whether to compute count statistic (number of
                          contributing observations per output pixels)

    :param out_chunks: Advanced option, allows to rechunk output internally,
                       order is ``(ny, nx, nband)``

    :param reshape_strategy: One of ``mem`` (default) or ``yxbt``. This is only
    applicable when supplying Dataset object. It controls how Dataset is
    reshaped into DataArray in the format expected by Geomedian code. If you
    have enough RAM and use single-worker Dask cluster, then use ``mem``, it
    should be the most efficient. If there is not enough RAM to load entire
    input you can try ``yxbt`` mode, but you might still run out of RAM anyway.
    If using multi-worker Dask cluster you have to use ``yxbt`` strategy.

    :param scale, offset: Only used when input contains integer values, actual
                          Geomedian will run on scaled values
                          ``scale*X+offset``. Only affects internal
                          computation, final result is scaled back to the
                          original value range.

    :param eps: Termination criteria passed on to geomedian algorithm

    :param maxiters: Maximum number of iterations done per output pixel

    :param num_threads: Configure internal concurrency of the Geomedian
                        computation. Default is 1 as we assume that Dask will
                        run a bunch of those concurrently.

    :param work_chunks: Default is ``(100, 100)``, only applicable when input
                        is Dataset.
    """
    ny, nx = work_chunks

    dim = "time"
    if dim not in src.dims:
        if "spec" in src.dims:
            dim = "spec"
        else:
            raise ValueError("Input dataset must have a 'time' or 'spec' dimension")

    chunked = src.chunk({"y": ny, "x": nx, dim: -1})

    # Check the dtype of the first data variable
    is_float = next(iter(src.dtypes.values())) == "f"

    if eps is None:
        # Is it sensible having a float for an int array?
        eps = 1e-4 if is_float else 0.1

    _gm_with_mads = chunked.map_blocks(
        geomedian_block_processor,
        kwargs=dict(
            scale=scale,
            offset=offset,
            eps=eps,
            maxiters=maxiters,
            num_threads=num_threads,
            dim=dim,
            is_float=is_float,
        ),
    )

    return _gm_with_mads
