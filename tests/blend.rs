use image::{Rgb, RgbImage, Rgba, RgbaImage};
use poissonfusion::{BlendConfig, blend};

fn constant_background(width: u32, height: u32, value: u8) -> RgbImage {
    RgbImage::from_pixel(width, height, Rgb([value, value, value]))
}

fn opaque_foreground(width: u32, height: u32, value: u8) -> RgbaImage {
    RgbaImage::from_pixel(width, height, Rgba([value, value, value, 255]))
}

#[test]
fn leaves_pixels_outside_overlap_unchanged() {
    let mut background = constant_background(80, 60, 90);
    for y in 0..60 {
        for x in 0..80 {
            background.put_pixel(x, y, Rgb([(x * 3) as u8, (y * 4) as u8, 90]));
        }
    }
    let foreground = opaque_foreground(24, 20, 180);
    let result = blend(
        &background,
        &foreground,
        (27, 19),
        BlendConfig {
            appearance_sigma: 2.0,
            integration_length: 8.0,
            ..BlendConfig::default()
        },
    )
    .unwrap();
    assert_eq!(result.image.dimensions(), background.dimensions());
    for y in 0..60 {
        for x in 0..80 {
            if !(27..51).contains(&x) || !(19..39).contains(&y) {
                assert_eq!(
                    result.image.get_pixel(x, y),
                    background.get_pixel(x, y),
                    "destination changed outside the pasted rectangle at ({x}, {y})"
                );
            }
        }
    }
}

#[test]
fn integration_length_has_the_expected_limit_direction() {
    let background = constant_background(96, 96, 220);
    let foreground = opaque_foreground(64, 64, 45);
    let base = BlendConfig {
        appearance_sigma: 2.5,
        tolerance: 1.0e-6,
        max_iterations: 1200,
        ..BlendConfig::default()
    };
    let faithful = blend(
        &background,
        &foreground,
        (16, 16),
        BlendConfig {
            integration_length: 2.0,
            ..base
        },
    )
    .unwrap();
    let poisson_like = blend(
        &background,
        &foreground,
        (16, 16),
        BlendConfig {
            integration_length: 200.0,
            ..base
        },
    )
    .unwrap();
    let center_faithful = faithful.image.get_pixel(48, 48)[0];
    let center_poisson = poisson_like.image.get_pixel(48, 48)[0];
    assert!(center_faithful < center_poisson);
    assert!(
        center_faithful < 100,
        "small ℓ should preserve source level"
    );
    assert!(
        center_poisson > 190,
        "large ℓ should approach boundary level"
    );
}

#[test]
fn transparent_margin_and_clipping_are_safe() {
    let background = constant_background(70, 50, 120);
    let mut foreground = RgbaImage::from_pixel(50, 44, Rgba([0, 0, 0, 0]));
    for y in 10..34 {
        for x in 12..38 {
            foreground.put_pixel(x, y, Rgba([210, 70, 45, 255]));
        }
    }
    let result = blend(
        &background,
        &foreground,
        (-8, 12),
        BlendConfig {
            appearance_sigma: 2.0,
            integration_length: 7.0,
            ..BlendConfig::default()
        },
    )
    .unwrap();
    assert_eq!(result.image.dimensions(), background.dimensions());
    assert!(result.stats.domain_pixels > 0);
    assert!(
        result
            .stats
            .relative_residual
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
fn deterministic_for_fixed_inputs() {
    let background = constant_background(64, 64, 150);
    let mut foreground = opaque_foreground(32, 32, 70);
    for y in 8..24 {
        for x in 8..24 {
            foreground.put_pixel(x, y, Rgba([220, 80, 35, 255]));
        }
    }
    let config = BlendConfig {
        appearance_sigma: 2.0,
        integration_length: 9.0,
        ..BlendConfig::default()
    };
    let first = blend(&background, &foreground, (16, 16), config).unwrap();
    let second = blend(&background, &foreground, (16, 16), config).unwrap();
    assert_eq!(first.image.as_raw(), second.image.as_raw());
    assert_eq!(first.stats.iterations, second.stats.iterations);
}
