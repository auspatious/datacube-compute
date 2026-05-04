use numpy::{
    PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4, ToPyArray,
};
use pyo3::prelude::*;

mod geomedian;
mod mad;
mod percentile;

#[pyfunction]
fn _geomedian<'py>(
    py: Python<'py>,
    in_array: PyReadonlyArray4<'py, f32>,
    maxiters: usize,
    eps: f32,
    num_threads: usize,
    scale: f32,
    offset: f32,
) -> (Bound<'py, PyArray3<f32>>, Bound<'py, PyArray3<f32>>) {
    let in_array = in_array.as_array();
    let (gm, mads) = py.detach(|| {
        geomedian::geomedian(in_array, maxiters, eps, num_threads, scale, offset)
    });
    (gm.to_pyarray(py), mads.to_pyarray(py))
}

#[pyfunction]
fn _geomedian_int16<'py>(
    py: Python<'py>,
    in_array: PyReadonlyArray4<'py, i16>,
    maxiters: usize,
    eps: f32,
    num_threads: usize,
    nodata: i16,
    scale: f32,
    offset: f32,
) -> (Bound<'py, PyArray3<i16>>, Bound<'py, PyArray3<f32>>) {
    let in_array = in_array.as_array();
    let (gm, mads) = py.detach(|| {
        geomedian::geomedian_int(in_array, maxiters, eps, num_threads, nodata, scale, offset)
    });
    (gm.to_pyarray(py), mads.to_pyarray(py))
}

#[pyfunction]
fn _geomedian_uint16<'py>(
    py: Python<'py>,
    in_array: PyReadonlyArray4<'py, u16>,
    maxiters: usize,
    eps: f32,
    num_threads: usize,
    nodata: u16,
    scale: f32,
    offset: f32,
) -> (Bound<'py, PyArray3<u16>>, Bound<'py, PyArray3<f32>>) {
    let in_array = in_array.as_array();
    let (gm, mads) = py.detach(|| {
        geomedian::geomedian_int(in_array, maxiters, eps, num_threads, nodata, scale, offset)
    });
    (gm.to_pyarray(py), mads.to_pyarray(py))
}

#[pyfunction]
fn _percentile_uint16<'py>(
    py: Python<'py>,
    in_array: PyReadonlyArray2<'py, u16>,
    percentiles: PyReadonlyArray1<'py, f64>,
    nodata: u16,
) -> Bound<'py, PyArray2<u16>> {
    let in_array = in_array.as_array();
    let percentiles = percentiles.as_array();
    let out = py.detach(|| percentile::percentile(in_array, percentiles, nodata));
    out.to_pyarray(py)
}

#[pyfunction]
fn _percentile_int16<'py>(
    py: Python<'py>,
    in_array: PyReadonlyArray2<'py, i16>,
    percentiles: PyReadonlyArray1<'py, f64>,
    nodata: i16,
) -> Bound<'py, PyArray2<i16>> {
    let in_array = in_array.as_array();
    let percentiles = percentiles.as_array();
    let out = py.detach(|| percentile::percentile(in_array, percentiles, nodata));
    out.to_pyarray(py)
}

#[pyfunction]
fn _percentile_uint8<'py>(
    py: Python<'py>,
    in_array: PyReadonlyArray2<'py, u8>,
    percentiles: PyReadonlyArray1<'py, f64>,
    nodata: u8,
) -> Bound<'py, PyArray2<u8>> {
    let in_array = in_array.as_array();
    let percentiles = percentiles.as_array();
    let out = py.detach(|| percentile::percentile(in_array, percentiles, nodata));
    out.to_pyarray(py)
}

#[pyfunction]
fn _percentile_int8<'py>(
    py: Python<'py>,
    in_array: PyReadonlyArray2<'py, i8>,
    percentiles: PyReadonlyArray1<'py, f64>,
    nodata: i8,
) -> Bound<'py, PyArray2<i8>> {
    let in_array = in_array.as_array();
    let percentiles = percentiles.as_array();
    let out = py.detach(|| percentile::percentile(in_array, percentiles, nodata));
    out.to_pyarray(py)
}

#[pyfunction]
fn _percentile_f32<'py>(
    py: Python<'py>,
    in_array: PyReadonlyArray2<'py, f32>,
    percentiles: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray2<f32>> {
    let in_array = in_array.as_array();
    let percentiles = percentiles.as_array();
    let out = py.detach(|| percentile::percentile(in_array, percentiles, f32::NAN));
    out.to_pyarray(py)
}

#[pyfunction]
fn _percentile_f64<'py>(
    py: Python<'py>,
    in_array: PyReadonlyArray2<'py, f64>,
    percentiles: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray2<f64>> {
    let in_array = in_array.as_array();
    let percentiles = percentiles.as_array();
    let out = py.detach(|| percentile::percentile(in_array, percentiles, f64::NAN));
    out.to_pyarray(py)
}

#[pymodule]
fn backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_geomedian, m)?)?;
    m.add_function(wrap_pyfunction!(_geomedian_int16, m)?)?;
    m.add_function(wrap_pyfunction!(_geomedian_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(_percentile_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(_percentile_int16, m)?)?;
    m.add_function(wrap_pyfunction!(_percentile_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(_percentile_int8, m)?)?;
    m.add_function(wrap_pyfunction!(_percentile_f32, m)?)?;
    m.add_function(wrap_pyfunction!(_percentile_f64, m)?)?;
    Ok(())
}
