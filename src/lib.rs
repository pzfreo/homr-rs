//! Prototype Rust port of homr (https://github.com/liebharc/homr).
#![cfg_attr(
    all(target_arch = "wasm32", feature = "encoder-chunk-bench"),
    feature(stdarch_wasm_atomic_wait)
)]
pub mod braces;
pub mod contours;
pub mod decoder;
pub mod ellipse;
pub mod encoder;
pub mod geom;
pub mod musicxml;
pub mod npmath;
pub mod onnx;
pub mod page;
pub mod polygon;
pub mod preprocess;
pub mod raster;
pub mod rational;
pub mod score;
pub mod sintable;
pub mod staff;
pub mod staff_image;
pub mod symbols;
pub mod tokens;
pub mod triplets;
pub mod warp;
