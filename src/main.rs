use show_image::create_window;

#[show_image::main]
fn main() {
    let img = image::open("HW4_Prob6_fingerprint.pbm").unwrap();
    let window = create_window("Original Image", Default::default()).unwrap();
    window.set_image("Figure 1", img).unwrap();
    println!("Hello, world!");
}
