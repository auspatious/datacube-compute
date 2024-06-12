# datacube-compute

A library with fast implementations of algorithms for processing satellite images.

## Install

First install rust if not already installed:

``` bash
    curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
    export PATH="$HOME/.cargo/bin:$PATH"
```

Then clone the repo and use pip to install the package:

``` bash
    git clone https://github.com/opendatacube/datacube-compute.git
    pip install datacube-compute
```

## Tests

To run the tests install `pytest-benchmark` and `hdstats`.
