# Datacube Compute

A library with fast implementations of algorithms for processing satellite images.

## Install

First install rust if not already installed:

``` bash
    curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
    export PATH="$HOME/.cargo/bin:$PATH"
```

Then clone the repo and use pip to install the package:

``` bash
    git clone https://github.com/auspatious/datacube-compute.git
    pip install datacube-compute
```

## Tests

To run the tests install `pytest-benchmark` and to do performance comparisons, install `hdstats`.


## Versioning

To release a new version, update the [Cargo.toml](Cargo.toml) file, and then create a release
on GitHub with the same version number.

## References

Some references for the algorithms include:

* For the Geomedian: D. Roberts, N. Mueller and A. Mcintyre, "High-Dimensional Pixel Composites From Earth Observation Time Series," in IEEE Transactions on Geoscience and Remote Sensing, vol. 55, no. 11, pp. 6254-6264, Nov. 2017, doi: 10.1109/TGRS.2017.2723896.
* For the Median Absolute Deviations: D. Roberts, B. Dunn and N. Mueller, "Open Data Cube Products Using High-Dimensional Statistics of Time Series," IGARSS 2018 - 2018 IEEE International Geoscience and Remote Sensing Symposium, Valencia, Spain, 2018, pp. 8647-8650, doi: 10.1109/IGARSS.2018.8518312.
