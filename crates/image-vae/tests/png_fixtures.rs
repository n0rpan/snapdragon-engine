//! PNG tests: python3-generated fixtures (zlib level 0/1/6/9, filters 0-4, RGB and
//! RGBA, split IDAT) and writer round trips. Model-free.
//!
//! Fixture pixel formula (from tests/fixtures generator, mirrored here):
//!   RGB channel c at (x, y): (x*37 + y*91 + c*53 + (x*y) % 17) % 256
//!   alpha (RGBA only) is ignored on read.

use image_vae::png::{decode_png, read_png, write_png};
use image_vae::Rgb;

fn pix(x: usize, y: usize, c: usize) -> u8 {
    ((x * 37 + y * 91 + c * 53 + (x * y) % 17) % 256) as u8
}

fn expected_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                v.push(pix(x, y, c));
            }
        }
    }
    v
}

fn check(name: &str, bytes: &[u8], w: usize, h: usize) {
    let img = decode_png(bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!((img.width, img.height), (w, h), "{name}: dims");
    assert_eq!(img.data, expected_rgb(w, h), "{name}: pixels");
}

#[test]
fn stored_deflate_filter0() {
    check("rgb_stored_f0", include_bytes!("fixtures/rgb_stored_f0.png"), 40, 12);
}

#[test]
fn dynamic_huffman_all_filters() {
    check("rgb_mixed_dyn", include_bytes!("fixtures/rgb_mixed_dyn.png"), 120, 64);
}

#[test]
fn fixed_or_dynamic_paeth_only() {
    check("rgb_paeth_fixed", include_bytes!("fixtures/rgb_paeth_fixed.png"), 16, 8);
}

#[test]
fn multiple_idat_chunks() {
    check("rgb_split_idat", include_bytes!("fixtures/rgb_split_idat.png"), 64, 32);
}

#[test]
fn tiny_all_filters_fixed_block() {
    check("rgb_tiny_all_filters", include_bytes!("fixtures/rgb_tiny_all_filters.png"), 5, 5);
}

#[test]
fn rgba_drops_alpha() {
    let bytes = include_bytes!("fixtures/rgba_mixed.png");
    let img = decode_png(bytes).unwrap();
    assert_eq!((img.width, img.height), (48, 24));
    assert_eq!(img.data, expected_rgb(48, 24));
    assert_eq!(img.data.len(), 48 * 24 * 3);
}

#[test]
fn rejects_truncated_and_corrupt() {
    let bytes = include_bytes!("fixtures/rgb_stored_f0.png");
    assert!(decode_png(&bytes[..bytes.len() - 20]).is_err());
    let mut bad = bytes.to_vec();
    bad[40] ^= 0x55; // inside IHDR/IDAT payload: CRC must fail
    assert!(decode_png(&bad).is_err());
    assert!(decode_png(b"not a png at all").is_err());
}

#[test]
fn write_then_read_round_trip() {
    let dir = std::env::temp_dir().join(format!("image-vae-png-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("roundtrip.png");
    let (w, h) = (300usize, 70usize); // > 65535 bytes raw, exercises several stored blocks
    let data: Vec<u8> = (0..w * h * 3).map(|i| ((i * 2654435761usize) >> 7) as u8).collect();
    let img = Rgb {
        width: w,
        height: h,
        data,
    };
    write_png(&path, &img).unwrap();
    let back = read_png(&path).unwrap();
    assert_eq!(back, img);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn write_rejects_bad_length() {
    let img = Rgb {
        width: 2,
        height: 2,
        data: vec![0; 11],
    };
    let path = std::env::temp_dir().join("image-vae-never-written.png");
    assert!(write_png(&path, &img).is_err());
}
