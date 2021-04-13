import time
import numpy as np
import pytest
from datacube_compute import geomedian
from hdstats import nangeomedian_pcm


@pytest.fixture
def arr():
    xx = np.random.random((100, 100, 10, 50)).astype(np.float32)
    mask = np.random.random(50) > 0.4  # 40% of time slices invalid
    xx[:, :, :, mask] = np.nan
    return xx

@pytest.fixture
def kwargs(): 
    return {"maxiters": 1000, "eps": 1e-5, "num_threads": 1}


@pytest.fixture
def kwargs_parallel(): 
    return {"maxiters": 1000, "eps": 1e-5, "num_threads": 4}


def test_benchmark(benchmark, arr, kwargs):
    benchmark(geomedian, arr, **kwargs)


def test_benchmark_hdstats(benchmark, arr, kwargs):
    benchmark(nangeomedian_pcm, arr, **kwargs, nocheck=True)


def test_benchmark_parallel(benchmark, arr, kwargs_parallel):
    benchmark(geomedian, arr, **kwargs_parallel)


def test_benchmark_hdstats_parallel(benchmark, arr, kwargs_parallel):
    benchmark(nangeomedian_pcm, arr, **kwargs_parallel, nocheck=True)


def test_accuracy(arr, kwargs):

    gm_1 = geomedian(arr, **kwargs)
    gm_2 = nangeomedian_pcm(arr, **kwargs)

    dist = np.linalg.norm(gm_1 - gm_2)
    assert dist < 100 * kwargs['eps']
