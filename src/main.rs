#![feature(float_next_up_down)]
use image::{DynamicImage::ImageLuma8, GrayImage};
use nalgebra::{
    allocator::Allocator, ComplexField, DMatrix, DefaultAllocator, Dim, DimMin, DimMinimum, Matrix,
    RawStorage, SVD,
};
use num_traits::AsPrimitive;
use show_image::create_window;
use std::{
    fs::{self, File},
    io::Write,
};

#[show_image::main]
fn main() {
    // Open original .pbm image
    let img = image::open("HW4_Prob6_fingerprint.pbm").unwrap();
    // Display image
    let window = create_window("Original Image", Default::default()).unwrap();
    window.set_image("Figure 1", img.clone()).unwrap();

    // Assert image is 8-bit monochrome
    let ImageLuma8(img) = img else { unreachable!() };
    // Convert image to matrix of doubles
    let a = DMatrix::from_row_iterator(
        img.height() as usize,
        img.width() as usize,
        img.as_raw().iter().map(|x| *x as f64),
    );

    // Save size of $\(\mat{A}\)$ for later
    let (m, n) = a.shape();

    // check matrix is still the same image
    mat_to_img_show(&a, "Original matrix as image check");

    // Compute SVD of $\(\mat{A}\)$
    let svd = SVD::new(a, true, true);

    // Create output file
    fs::create_dir_all("./out").unwrap();
    let mut out = File::create("./out/all_singular_values7.dat").unwrap();
    // Print normalized singular values to output file
    writeln!(out, "#all singular values of A").unwrap();
    for s in &svd.singular_values {
        writeln!(out, "{}", s / svd.singular_values[0]).unwrap();
    }
    // notify Terminal of completed SVD printing
    eprintln!("finished printing");

    // The rank() function in matlab uses a default tolerance value. The nalgebra version doesn't - it needs to be provided. So we replicate matlab's default tolerance.
    // tol from MatLab = max(size(A))*eps(norm(A))
    // From matlab's documentation, eps(x) is the positive distance between |x| and the next largest float.
    // Rust's f64::next_up(x) returns that next largest float. As well, we are using the already computed largest singular value as norm(A).
    let tol = (m.max(n) as f64) * (f64::next_up(svd.singular_values[0]) - svd.singular_values[0]);

    // Compute the Rank of $\(\mat{A}\)$ with the computed tolerance
    println!("Computed Rank of A = {}", svd.rank(tol));

    // Compute optimal rank-$\(k\)$ approximations, display them, and save them
    for k in [1, 10, 50] {
        mat_to_img_show(&rank_k_approx(&svd, k), format!("Rank_{k}_Approximation"));
        // Print relative errors
        println!(
            "Relative error for A_{k} = {}",
            // Rust is 0 indexed, need to subtract 1
            svd.singular_values[(k + 1) - 1] / svd.singular_values[1 - 1]
        );
    }

    // keep images up until original window is closed
    for _event in window.event_channel().unwrap() {}
}

/// Calculates the optimal rank `k` approximation of a matrix represented by its singular value decomosition.
fn rank_k_approx<T: ComplexField, R: DimMin<C>, C: Dim>(
    svd: &SVD<T, R, C>,
    k: usize,
) -> Matrix<T, R, C, <nalgebra::DefaultAllocator as nalgebra::allocator::Allocator<T, R, C>>::Buffer>
where
    DefaultAllocator: Allocator<T, DimMinimum<R, C>, C>
        + Allocator<T, R, DimMinimum<R, C>>
        + Allocator<T::RealField, DimMinimum<R, C>>
        + Allocator<T, R, C>,
{
    let mut u = svd.u.as_ref().unwrap().clone();
    // Multiply $\(\sigma_iu_i\)$ for $\(i\in[1,k]\)$
    for i in 0..k {
        let val = svd.singular_values[i].clone();
        u.column_mut(i).scale_mut(val);
    }
    // Multiply $\(\displaystyle\sum_{i=1}^k (\sigma_iu_i )v^{\top}_i \)$
    u.columns(0, k) * svd.v_t.as_ref().unwrap().rows(0, k)
}

/// Displays and saves the image stored in mat to the file "./out/<`wind_name`>.png"
fn mat_to_img_show<T: AsPrimitive<u8>, R: Dim, C: Dim, S: RawStorage<T, R, C>>(
    mat: &Matrix<T, R, C, S>,
    wind_name: impl AsRef<str>,
) {
    // Convert matrix to 8-bit monochrome image
    // Annoyingly, image crate and matrix crate use different size types, so converting is required
    let im2 = GrayImage::from_fn(mat.ncols() as u32, mat.nrows() as u32, |c, r| {
        image::Luma([mat[(r as usize, c as usize)].as_()])
    });

    // Create window and display image
    let window2 = create_window(wind_name.as_ref(), Default::default()).unwrap();
    window2.set_image("f", im2.clone()).unwrap();

    // Save image as png in output folder
    im2.save_with_format(
        format!("./out/{}.png", wind_name.as_ref()),
        image::ImageFormat::Png,
    )
    .unwrap();
}
