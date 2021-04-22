use numpy::{PyArray3, PyArray4, ToPyArray};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

mod geomedian;
mod mad;

#[pymodule]
fn backend(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "_geomedian")]
    fn py_geomedian<'a>(
        py: Python<'a>,
        in_array: &'a PyArray4<f32>,
        maxiters: usize,
        eps: f32,
        num_threads: usize,
        scale: f32,
        offset: f32,
    ) -> (&'a PyArray3<f32>, &'a PyArray3<f32>) {
        let in_array = in_array.readonly();
        let in_array = in_array.as_array();

        // release GIL for call to geomedian
        let (gm, mads) = py
            .allow_threads(|| geomedian::geomedian(in_array, maxiters, eps, num_threads, scale, offset));

        (gm.to_pyarray(py), mads.to_pyarray(py))
    }

    #[pyfn(m, "_geomedian_int16")]
    fn py_geomedian_int16<'a>(
        py: Python<'a>,
        in_array: &'a PyArray4<i16>,
        maxiters: usize,
        eps: f32,
        num_threads: usize,
        nodata: i16,
        scale: f32,
        offset: f32,
    ) -> (&'a PyArray3<i16>, &'a PyArray3<f32>) {
        let in_array = in_array.readonly();
        let in_array = in_array.as_array();

        // release GIL for call to geomedian
        let (gm, mads) =
            py.allow_threads(|| geomedian::geomedian_int(in_array, maxiters, eps, num_threads, nodata, scale, offset));

        (gm.to_pyarray(py), mads.to_pyarray(py))
    }

    #[pyfn(m, "_geomedian_uint16")]
    fn py_geomedian_uint16<'a>(
        py: Python<'a>,
        in_array: &'a PyArray4<u16>,
        maxiters: usize,
        eps: f32,
        num_threads: usize,
        nodata: u16,
        scale: f32,
        offset: f32,
    ) -> (&'a PyArray3<u16>, &'a PyArray3<f32>) {
        let in_array = in_array.readonly();
        let in_array = in_array.as_array();

        // release GIL for call to geomedian
        let (gm, mads) =
            py.allow_threads(|| geomedian::geomedian_int(in_array, maxiters, eps, num_threads, nodata, scale, offset));

        (gm.to_pyarray(py), mads.to_pyarray(py))
    }

    Ok(())
}
