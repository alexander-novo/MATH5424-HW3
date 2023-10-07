use image::DynamicImage::ImageLuma8;
use image::GrayImage;
use nalgebra::DMatrix;
use show_image::create_window;
use show_image::event;

#[show_image::main]
fn main() {
    let img = image::open("HW4_Prob6_fingerprint.pbm").unwrap();
    let window = create_window("Original Image", Default::default()).unwrap();
    window.set_image("Figure 1", img.clone()).unwrap();

    // convert img to matrix of doubles
    let ImageLuma8(img) = img else { unreachable!() };
    let A = DMatrix::from_row_iterator(
        img.height() as usize,
        img.width() as usize,
        img.as_raw().iter().copied(),
    );

    println!("{A}");

    // check matrix is still the same image
    let im2 = GrayImage::from_fn(A.ncols() as u32, A.nrows() as u32, |c, r| {
        return image::Luma([A[(r as usize, c as usize)]]);
    });
    let window2 = create_window("Image from Matrix", Default::default()).unwrap();
    window2.set_image("Figure 1.5", im2).unwrap();

    // keep images up until closed
    for event in window.event_channel().unwrap() {}
}
