#![feature(float_next_up_down)]
#![feature(array_methods)]
#![feature(array_try_map)]
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
    let a = [0, 1, 2].map(|c| rgb_to_mat(&img, c));

    // Save size of $\(\mat{A}\)$ for later
    let (m, n) = a[0].shape();

    // check matrix is still the same image
    mat_to_img_show(a.each_ref(), "RGB Image Conversion Check")?;

    // Compute SVD of $\(\mat{A}\)$
    let svd = a.map(|a| SVD::new(a, true, true));

    // Create output file
    fs::create_dir_all("./out")?;
    let mut out = File::create("./out/all_singular_values8.dat")?;
    // Print normalized singular values to output file
    writeln!(out, "{:8} {:8} {:8}", "A1", "A2", "A3")?;
    // Write normalized singular values out
    for sigma in zip_array(svd.each_ref().map(|svd| svd.singular_values.iter())) {
        for (sigma, svd) in sigma.into_iter().zip(&svd) {
            write!(out, "{:8.6} ", *sigma / svd.singular_values[0])?;
        }
        writeln!(out)?;
    }
    // notify Terminal of completed SVD printing
    eprintln!("finished printing");

    // Optimal rank-0 approximations of the channels. Used for computing iteratively higher rank approximations
    let mut approx = [
        (0, DMatrix::zeros(m, n)),
        (0, DMatrix::zeros(m, n)),
        (0, DMatrix::zeros(m, n)),
    ];

    for err in [0.10, 0.05, 0.01] {
        // Compute the next optimal approximation for each channel based on the target error, as well as the previous approximation and its rank
        for ((prev_k, approx), svd) in approx.iter_mut().zip(&svd) {
            (*prev_k, *approx) = rel_err_approx(svd, err, approx, *prev_k);
        }

        // Show and save approximation of image
        mat_to_img_show(
            approx.each_ref().map(|(_, approx)| approx),
            format!("err_all_{err}_Approximation"),
        )?;

        // Print rank and error information
        println!("Goal error = {err}");
        for (i, ((k, _), svd)) in approx.iter().zip(&svd).enumerate() {
            println!(
                "Smallest k for layer {i} = {k:3}  Actual error = {:8.6}  Error for k-1: {:8.6}",
                svd.singular_values[(k + 1) - 1] / svd.singular_values[1 - 1],
                svd.singular_values[(k) - 1] / svd.singular_values[1 - 1],
            );
        }
    }

    // 1% error approximations were already previously computed for each channel
    let per_1_approx = approx;
    // Compute 50% error approximations for each channel
    let per_50_approx = svd.map(|svd| rel_err_approx(&svd, 0.5, &DMatrix::zeros(m, n), 0));

    // Display all combinations of 50,1% error approximations where exactly a single channel uses its 50% error approximation,
    // and other channels use their 1% error approximation
    mat_to_img_show(
        [&per_50_approx[0].1, &per_1_approx[1].1, &per_1_approx[2].1],
        "50-err-red",
    )?;
    mat_to_img_show(
        [&per_1_approx[0].1, &per_50_approx[1].1, &per_1_approx[2].1],
        "50-err-green",
    )?;
    mat_to_img_show(
        [&per_1_approx[0].1, &per_1_approx[1].1, &per_50_approx[2].1],
        "50-err-blue",
    )?;

    // Print rank information for the 50% error approximations
    for (i, (k, _)) in per_50_approx.iter().enumerate() {
        println!("Err 50 approximation for channel {} k = {k}", i + 1)
    }

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
    Matrix<T, R, C, <DefaultAllocator as Allocator<T, R, C>>::Buffer>,
)
where
    DefaultAllocator: Allocator<T, DimMinimum<R, C>, C>
        + Allocator<T, R, DimMinimum<R, C>>
        + Allocator<T::RealField, DimMinimum<R, C>>
        + Allocator<T, R, C>,
    T::RealField: for<'a> MulAssign<&'a T::RealField>,
{
    // Calculate the absolute error we're interested in finding
    err *= &svd.singular_values[0];

    // The singular values are already sorted, so we can binary search through them to find the one we're looking for.
    // We know that our previous approximation had less error than the one we're currently looking for, so we don't need to search through the first prev_k singular values.
    // We therefore exclude those from the search
    let k = match svd.singular_values.as_slice()[(prev_k + 1)..]
        // f64 isn't totally ordered, so we can't use a normal binary search.
        // Instead, it's partially ordered, so we use partial_cmp(...).unwrap() to assert that we are in a totally ordered subset (no NaNs).
        // As well, the singular values are sorted in descending order, so we reverse the comparison to err.partial_cmp(x).
        .binary_search_by(|x| err.partial_cmp(x).unwrap())
    {
        // Binary search can return two things:
        // Ok - it found a singular value which is exactly err. Then i is the index of that singular value.
        // Err - it didn't find a singular value which is exactly err. Then i is the index in the list where we could insert err and it would still be sorted.
        // In both cases, i is the choice of k we want, since it means that sigma_i < err and sigma_{i - 1} >= err - and the vector is 0-indexed.
        // i in this specific case is the index from (prev_k + 1) so we need to add that.
        Ok(i) | Err(i) => i + prev_k + 1,
    };

    // Calculate the approximation based on the k we chose above. Then return this along with that k.
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

/// Displays and saves the image stored in the three matrices (representing rgb channels) to the file "./out/<`wind_name`>.png"
fn mat_to_img_show<T: AsPrimitive<u8>, R: Dim, C: Dim, S: RawStorage<T, R, C>>(
    mats: [&Matrix<T, R, C, S>; 3],
    wind_name: impl AsRef<str>,
) -> Result<(), Box<dyn Error>> {
    // Convert matrix to 8-bit rgb image
    // Annoyingly, image crate and matrix crate use different size types, so converting is required
    let im2 = RgbImage::from_fn(mats[0].ncols() as u32, mats[0].nrows() as u32, |c, r| {
        let i = (r as usize, c as usize);
        image::Rgb(mats.map(|m| m[i].as_()))
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

// Some helper code. Allows us to zip an array of iterators into an iterator over arrays
pub struct ZipArray<T, const N: usize> {
    array: [T; N],
}

pub fn zip_array<T: Iterator, const N: usize>(array: [T; N]) -> ZipArray<T, N> {
    ZipArray { array }
}

impl<T: Iterator, const N: usize> Iterator for ZipArray<T, N> {
    type Item = [T::Item; N];

    fn next(&mut self) -> Option<Self::Item> {
        self.array.each_mut().try_map(|i| i.next())
    }
}
