use ndarray::parallel::prelude::*;
use ndarray::{Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Ix1, Ix2, Ix3, Ix4, s, Zip};
use numpy::{PyArray3, PyArray4, ToPyArray};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use rayon;

fn geomedian<'a>(
    in_array: ArrayView<'a, f32, Ix4>,
    maxiters: usize,
    eps: f32,
    num_threads: usize,
) -> Array<f32, Ix3> {
    let rows = in_array.shape()[0];
    let columns = in_array.shape()[1];
    let bands = in_array.shape()[2];
    let mut out_array: Array<f32, Ix3> = ArrayBase::zeros([rows, columns, bands]);

    let iter = Zip::from(out_array.axis_iter_mut(Axis(0))).and(in_array.axis_iter(Axis(0)));

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .unwrap();
    pool.install(|| {
        iter.into_par_iter()
            .for_each(|(out_arr, in_arr)| geomedian_column(in_arr, out_arr, maxiters, eps))
    });

    out_array
}

fn geomedian_column<'a>(
    in_array: ArrayView<'a, f32, Ix3>,
    mut out_array: ArrayViewMut<f32, Ix2>,
    maxiters: usize,
    eps: f32,
) {
    let shape = in_array.shape();
    let columns = shape[0];
    let bands = shape[1];
    let time_steps = shape[2];

    let mut data: Array<f32, Ix2> = ArrayBase::zeros([time_steps, bands]);

    for column in 0..columns {
        let in_array = in_array.index_axis(Axis(0), column);
        let mut out_array = out_array.index_axis_mut(Axis(0), column);
        let data: ArrayView<f32, Ix2> = get_valid_data(in_array, &mut data);

        // seed initialization
        out_array.assign(&data.mean_axis(Axis(0)).unwrap());

        geomedian_pixel(data, out_array, maxiters, eps);
    }
}

fn get_valid_data<'a>(in_array: ArrayView<f32, Ix2>, data: &'a mut Array<f32, Ix2>) -> ArrayView<'a, f32, Ix2> {
    // copies the valid data for each (row, column) data from `in_array` to `valid_data
    // and tranposes the band and time dimensions for better cache perfomance

    let bands = in_array.shape()[0];
    let time_steps = in_array.shape()[1];
    let mut idx: usize = 0;

    for t in 0..time_steps {
        if !in_array[[0, t]].is_nan() {
            for band in 0..bands {
                data[[idx, band]] = in_array[[band, t]];
            }

            idx += 1;
        }
    }

    data.slice(s![0..idx, ..])
}

fn geomedian_pixel(
    data: ArrayView<f32, Ix2>,
    mut out_array: ArrayViewMut<f32, Ix1>,
    maxiters: usize,
    eps: f32,
) { 
    let time_steps = data.shape()[0];
    let bands = data.shape()[1];
    let mut temp_median: Vec<f32> = vec![0.0; bands];

    // main loop
    for _ in 0..maxiters {
        temp_median.iter_mut().for_each(|x| *x = 0.0);
        let mut inv_dist_sum: f32 = 0.0;

        for t in 0..time_steps {
            let mut dist: f32 = 0.0;
            for band in 0..bands {
                dist += (out_array[[band]] - data[[t, band]]).powi(2);
            }

            dist = dist.sqrt();

            let mut inv_dist: f32 = 0.0;
            if dist > 0.0 {
                inv_dist = 1.0 / dist;
            }

            inv_dist_sum += inv_dist;

            for band in 0..bands {
                temp_median[band] += data[[t, band]] * inv_dist;
            }
        }

        // check improvement between iterations
        // exit if smaller than tolerance
        let mut change: f32 = 0.0;
        for band in 0..bands {
            temp_median[band] /= inv_dist_sum;
            change += (&temp_median[band] - out_array[[band]]).powi(2);
            out_array[[band]] = temp_median[band];
        }

        if change.sqrt() < eps {
            break;
        }
    }
}

#[pymodule]
fn backend(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "_geomedian")]
    fn py_geomedian<'a>(
        py: Python<'a>,
        in_array: &'a PyArray4<f32>,
        maxiters: usize,
        eps: f32,
        num_threads: usize,
    ) -> &'a PyArray3<f32> {
        let in_array = in_array.readonly();
        let in_array = in_array.as_array();

        // release GIL for call to geomedian
        py.allow_threads(|| geomedian(in_array, maxiters, eps, num_threads))
            .to_pyarray(py)
    }
    Ok(())
}
