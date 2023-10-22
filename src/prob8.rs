#![feature(float_next_up_down)]
use std::error::Error;
use std::ops::MulAssign;
use std::{
    fs::{self, File},
    io::Write,
};

use image::{DynamicImage::ImageRgb8, ImageBuffer, Rgb, RgbImage};
use nalgebra::{
    allocator::Allocator, ComplexField, DMatrix, DefaultAllocator, Dim, DimMin, DimMinimum, Matrix,
    RawStorage, Storage, SVD,
};
use num_traits::AsPrimitive;
use show_image::create_window;

#[show_image::main]
fn main() -> Result<(), Box<dyn Error>> {
    // Open original .pbm image
    let img = image::open("hokiebirdwithacat.jpg")?;
    // Display image
    let window = create_window("Original Image", Default::default())?;
    window.set_image("Figure 1", img.clone())?;

    // Assert image is 8-bit Rgb
    let ImageRgb8(img) = img else { unreachable!() };
    // Convert image to 3 matrices of doubles
    let a1 = rgb_to_mat(&img, 0);
    let a2 = rgb_to_mat(&img, 1);
    let a3 = rgb_to_mat(&img, 2);

    // Save size of $\(\mat{A}\)$ for later
    let (m, n) = a1.shape();

    // check matrix is still the same image
    mat_to_img_show(&a1, &a2, &a3, "RGB Image Conversion Check")?;

    // Compute SVD of $\(\mat{A}\)$
    let svd1 = SVD::new(a1, true, true);
    eprintln!("svd1 computed");
    let svd2 = SVD::new(a2.clone(), true, true);
    eprintln!("svd2 computed");
    let svd3 = SVD::new(a3.clone(), true, true);
    eprintln!("svd3 computed");

    // Create output file
    fs::create_dir_all("./out")?;
    let mut out = File::create("./out/all_singular_values8.dat")?;
    // Print normalized singular values to output file
    writeln!(out, "{:8} {:8} {:8}", "A1", "A2", "A3")?;
    // s1 is singular values of A1, s2 is singular values of A2, s3 is singular values of A3
    for ((s1, s2), s3) in svd1
        .singular_values
        .iter()
        .zip(&svd2.singular_values)
        .zip(&svd3.singular_values)
    {
        writeln!(
            out,
            "{:8.6} {:8.6} {:8.6}",
            s1 / svd1.singular_values[0],
            s2 / svd2.singular_values[0],
            s3 / svd3.singular_values[0]
        )?;
    }
    // notify Terminal of completed SVD printing
    eprintln!("finished printing");

    // Compute optimal rank-$\(k\)$ approximations for specified max errors, display them, and save them
    let mut approx1 = DMatrix::zeros(m, n);
    let mut prev_k1 = 0;
    let mut approx2 = DMatrix::zeros(m, n);
    let mut prev_k2 = 0;
    let mut approx3 = DMatrix::zeros(m, n);
    let mut prev_k3 = 0;
    for err in [0.10, 0.05, 0.01] {
        (prev_k1, approx1) = rel_err_approx(&svd1, err, &approx1, prev_k1);
        (prev_k2, approx2) = rel_err_approx(&svd2, err, &approx2, prev_k2);
        (prev_k3, approx3) = rel_err_approx(&svd3, err, &approx3, prev_k3);
        mat_to_img_show(&approx1, &approx2, &approx3, format!("err_all_{err}_Approximation"))?;
        // Print relative errors
        println!(
            "Goal error = {err}\n
            Smallest k for layer 1 = {:3}\tActual error for layer 1 = {}\n
            Smallest k for layer 2 = {:3}\tActual error for layer 2 = {}\n
            Smallest k for layer 3 = {:3}\tActual error for layer 3 = {}\n",
            prev_k1 -1,
            svd1.singular_values[(prev_k1 + 1) - 1] / svd1.singular_values[1 - 1],
            prev_k2 -1,
            svd2.singular_values[(prev_k2 + 1) - 1] / svd2.singular_values[1 - 1],
            prev_k3 -1,
            svd3.singular_values[(prev_k3 + 1) - 1] / svd3.singular_values[1 - 1],
        );
    }

    // keep images up until original window is closed
    for _event in window.event_channel()? {}

    Ok(())
}

/// Converts image buffer reference to matrix of f64 for given channel `c`
fn rgb_to_mat(img: &ImageBuffer<Rgb<u8>, Vec<u8>>, c: usize) -> DMatrix<f64> {
    DMatrix::from_row_iterator(
        img.height() as usize,
        img.width() as usize,
        img.enumerate_pixels().map(|(_, _, x)| x.0[c] as f64),
    )
}

/// Compute optimal approximation with relative error less than err.
/// Calls `rank_k_approx` after computing required k for error, then returns that rank and the approximation.
fn rel_err_approx<T: ComplexField, R: DimMin<C>, C: Dim>(
    svd: &SVD<T, R, C>,
    mut err: T::RealField,
    prev: &Matrix<T, R, C, impl Storage<T, R, C>>,
    prev_k: usize,
) -> (
    usize,
    Matrix<
        T,
        R,
        C,
        <nalgebra::DefaultAllocator as nalgebra::allocator::Allocator<T, R, C>>::Buffer,
    >,
)
where
    DefaultAllocator: Allocator<T, DimMinimum<R, C>, C>
        + Allocator<T, R, DimMinimum<R, C>>
        + Allocator<T::RealField, DimMinimum<R, C>>
        + Allocator<T, R, C>,
    T::RealField: for<'a> MulAssign<&'a T::RealField>,
{
    err *= &svd.singular_values[0];

    let k = match svd
        .singular_values
        .as_slice()
        .binary_search_by(|x| err.partial_cmp(x).unwrap())
    {
        Ok(i) | Err(i) => i,
    };

    (k, rank_k_approx(svd, k, prev, prev_k))
}

/// Calculates the optimal rank `k` approximation of a matrix represented by its singular value decomosition.
/// Uses previously-computed optimal approximation `prev`, which is rank `prev_k`, which must be less than `k`.
/// For new approximation, pass zero matrix for `prev` and 0 for `prev_k`
fn rank_k_approx<T: ComplexField, R: DimMin<C>, C: Dim>(
    svd: &SVD<T, R, C>,
    k: usize,
    prev: &Matrix<T, R, C, impl Storage<T, R, C>>,
    prev_k: usize,
) -> Matrix<T, R, C, <nalgebra::DefaultAllocator as nalgebra::allocator::Allocator<T, R, C>>::Buffer>
where
    DefaultAllocator: Allocator<T, DimMinimum<R, C>, C>
        + Allocator<T, R, DimMinimum<R, C>>
        + Allocator<T::RealField, DimMinimum<R, C>>
        + Allocator<T, R, C>,
{
    let mut u = svd.u.as_ref().unwrap().clone();
    // Multiply $\(\sigma_iu_i\)$ for $\(i\in[1,k]\)$
    for i in prev_k..k {
        let val = svd.singular_values[i].clone();
        u.column_mut(i).scale_mut(val);
    }
    // Multiply $\(\displaystyle\sum_{i=1}^k (\sigma_iu_i )v^{\top}_i \)$
    prev + u.columns(prev_k, k - prev_k) * svd.v_t.as_ref().unwrap().rows(prev_k, k - prev_k)
}

/// Displays and saves the image stored in mat to the file "./out/<`wind_name`>.png"
fn mat_to_img_show<T: AsPrimitive<u8>, R: Dim, C: Dim, S: RawStorage<T, R, C>>(
    mat_r: &Matrix<T, R, C, S>,
    mat_g: &Matrix<T, R, C, S>,
    mat_b: &Matrix<T, R, C, S>,
    wind_name: impl AsRef<str>,
) -> Result<(), Box<dyn Error>> {
    // Convert matrix to 8-bit rgb image
    // Annoyingly, image crate and matrix crate use different size types, so converting is required
    let im2 = RgbImage::from_fn(mat_r.ncols() as u32, mat_r.nrows() as u32, |c, r| {
        let i = (r as usize, c as usize);
        image::Rgb([mat_r[i].as_(), mat_g[i].as_(), mat_b[i].as_()])
    });

    // Create window and display image
    let window2 = create_window(wind_name.as_ref(), Default::default())?;
    window2.set_image("f", im2.clone())?;

    // Save image as png in output folder
    im2.save_with_format(
        format!("./out/{}.png", wind_name.as_ref()),
        image::ImageFormat::Png,
    )?;

    Ok(())
}