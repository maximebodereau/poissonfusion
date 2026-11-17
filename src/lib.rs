//! PoissonFusion: an identity-preserving Poisson compositor.
//!
//! Instead of reconstructing the pasted object, the method solves only for a
//! smooth low-frequency correction field `c`:
//!
//! ```text
//! c* = argmin  1/2 ∫Ω |∇c|² + 1/(2ℓ²) ∫Ω |c|²
//!      s.t.    c|∂Ω = lowpass(target) - normalized_lowpass(source)
//! ```
//!
//! The Euler equation is the modified Helmholtz equation
//! `(-Δ + 1/ℓ²)c = 0`.  The final foreground is `source + c`; therefore source
//! texture is passed through instead of being reconstructed from boundary
//! conditions.  Positive screening makes the discrete system symmetric
//! positive definite and gives it a unique minimizer.

use image::{Rgb, RgbImage, RgbaImage};
use serde::Serialize;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug)]
pub struct BlendConfig {
    /// Gaussian scale used to estimate the low-frequency boundary mismatch.
    pub appearance_sigma: f32,
    /// Propagation length ℓ, in pixels. Small values protect source identity;
    /// large values approach classic Poisson's harmonic correction.
    pub integration_length: f32,
    /// Alpha value used to define the PDE domain.
    pub alpha_threshold: f32,
    /// Relative residual tolerance for preconditioned conjugate gradients.
    pub tolerance: f32,
    /// Maximum PCG iterations per colour channel.
    pub max_iterations: usize,
}

impl Default for BlendConfig {
    fn default() -> Self {
        Self {
            appearance_sigma: 3.0,
            integration_length: 6.0,
            alpha_threshold: 0.05,
            tolerance: 1.0e-5,
            max_iterations: 800,
        }
    }
}

impl BlendConfig {
    pub fn faithful() -> Self {
        Self {
            integration_length: 2.0,
            ..Self::default()
        }
    }

    pub fn seamless() -> Self {
        Self {
            integration_length: 18.0,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct BlendStats {
    pub roi: Roi,
    pub domain_pixels: usize,
    pub boundary_pixels: usize,
    pub unknowns: usize,
    pub iterations: [usize; 3],
    pub relative_residual: [f32; 3],
    pub screening: f32,
}

#[derive(Debug)]
pub struct BlendResult {
    pub image: RgbImage,
    pub stats: BlendStats,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlendError {
    InvalidConfig(&'static str),
    NoOverlap,
    EmptyMask,
}

impl Display for BlendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid configuration: {message}"),
            Self::NoOverlap => write!(f, "foreground does not overlap the background"),
            Self::EmptyMask => write!(f, "foreground alpha is empty after clipping"),
        }
    }
}

impl Error for BlendError {}

#[inline]
fn srgb_to_linear(value: u8) -> f32 {
    let value = value as f32 / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[inline]
fn linear_to_srgb(value: f32) -> u8 {
    let value = value.clamp(0.0, 1.0);
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round().clamp(0.0, 255.0) as u8
}

fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    if sigma <= 0.01 {
        return vec![1.0];
    }
    let radius = (3.0 * sigma).ceil() as i32;
    let mut kernel = Vec::with_capacity((radius * 2 + 1) as usize);
    let mut sum = 0.0;
    for offset in -radius..=radius {
        let x = offset as f32;
        let value = (-0.5 * x * x / (sigma * sigma)).exp();
        kernel.push(value);
        sum += value;
    }
    for value in &mut kernel {
        *value /= sum;
    }
    kernel
}

fn blur_scalar(input: &[f32], width: usize, height: usize, kernel: &[f32]) -> Vec<f32> {
    let radius = (kernel.len() / 2) as isize;
    let mut horizontal = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (k, weight) in kernel.iter().enumerate() {
                let sample_x =
                    (x as isize + k as isize - radius).clamp(0, width as isize - 1) as usize;
                sum += *weight * input[y * width + sample_x];
            }
            horizontal[y * width + x] = sum;
        }
    }
    let mut output = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (k, weight) in kernel.iter().enumerate() {
                let sample_y =
                    (y as isize + k as isize - radius).clamp(0, height as isize - 1) as usize;
                sum += *weight * horizontal[sample_y * width + x];
            }
            output[y * width + x] = sum;
        }
    }
    output
}

fn blur_rgb(input: &[[f32; 3]], width: usize, height: usize, kernel: &[f32]) -> Vec<[f32; 3]> {
    let radius = (kernel.len() / 2) as isize;
    let mut horizontal = vec![[0.0; 3]; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = [0.0; 3];
            for (k, weight) in kernel.iter().enumerate() {
                let sample_x =
                    (x as isize + k as isize - radius).clamp(0, width as isize - 1) as usize;
                let sample = input[y * width + sample_x];
                for channel in 0..3 {
                    sum[channel] += *weight * sample[channel];
                }
            }
            horizontal[y * width + x] = sum;
        }
    }
    let mut output = vec![[0.0; 3]; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = [0.0; 3];
            for (k, weight) in kernel.iter().enumerate() {
                let sample_y =
                    (y as isize + k as isize - radius).clamp(0, height as isize - 1) as usize;
                let sample = horizontal[sample_y * width + x];
                for channel in 0..3 {
                    sum[channel] += *weight * sample[channel];
                }
            }
            output[y * width + x] = sum;
        }
    }
    output
}

#[derive(Clone)]
struct SparseStencil {
    diagonal: Vec<f32>,
    neighbours: Vec<[i32; 4]>,
}

impl SparseStencil {
    fn apply(&self, input: &[f32], output: &mut [f32]) {
        for index in 0..input.len() {
            let mut value = self.diagonal[index] * input[index];
            for neighbour in self.neighbours[index] {
                if neighbour >= 0 {
                    value -= input[neighbour as usize];
                }
            }
            output[index] = value;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(left, right)| *left as f64 * *right as f64)
        .sum()
}

fn pcg(
    matrix: &SparseStencil,
    rhs: &[f32],
    tolerance: f32,
    max_iterations: usize,
) -> (Vec<f32>, usize, f32) {
    if rhs.is_empty() {
        return (Vec::new(), 0, 0.0);
    }
    let rhs_norm = dot(rhs, rhs).sqrt();
    if rhs_norm <= 1.0e-14 {
        return (vec![0.0; rhs.len()], 0, 0.0);
    }

    let mut x = vec![0.0; rhs.len()];
    let mut residual = rhs.to_vec();
    let mut z: Vec<f32> = residual
        .iter()
        .zip(&matrix.diagonal)
        .map(|(value, diagonal)| value / diagonal)
        .collect();
    let mut direction = z.clone();
    let mut matrix_direction = vec![0.0; rhs.len()];
    let mut rz = dot(&residual, &z);
    let mut relative = 1.0_f32;

    for iteration in 0..max_iterations {
        matrix.apply(&direction, &mut matrix_direction);
        let denominator = dot(&direction, &matrix_direction);
        if denominator.abs() <= 1.0e-30 {
            return (x, iteration, relative);
        }
        let alpha = rz / denominator;
        for index in 0..x.len() {
            x[index] += (alpha as f32) * direction[index];
            residual[index] -= (alpha as f32) * matrix_direction[index];
        }
        relative = (dot(&residual, &residual).sqrt() / rhs_norm) as f32;
        if relative <= tolerance {
            return (x, iteration + 1, relative);
        }
        for index in 0..z.len() {
            z[index] = residual[index] / matrix.diagonal[index];
        }
        let next_rz = dot(&residual, &z);
        let beta = next_rz / rz;
        for index in 0..direction.len() {
            direction[index] = z[index] + (beta as f32) * direction[index];
        }
        rz = next_rz;
    }
    (x, max_iterations, relative)
}

fn neighbours(index: usize, width: usize, height: usize) -> [Option<usize>; 4] {
    let x = index % width;
    let y = index / width;
    [
        (x > 0).then_some(index - 1),
        (x + 1 < width).then_some(index + 1),
        (y > 0).then_some(index - width),
        (y + 1 < height).then_some(index + width),
    ]
}

/// Blend an RGBA foreground into an RGB background at a top-left position.
///
/// The returned image has the same dimensions as `background`. Pixels outside
/// the foreground overlap are byte-for-byte unchanged.
pub fn blend(
    background: &RgbImage,
    foreground: &RgbaImage,
    position: (i32, i32),
    config: BlendConfig,
) -> Result<BlendResult, BlendError> {
    if !config.appearance_sigma.is_finite() || config.appearance_sigma <= 0.0 {
        return Err(BlendError::InvalidConfig(
            "appearance_sigma must be positive and finite",
        ));
    }
    if !config.integration_length.is_finite() || config.integration_length <= 0.0 {
        return Err(BlendError::InvalidConfig(
            "integration_length must be positive and finite",
        ));
    }
    if !(0.0..1.0).contains(&config.alpha_threshold) {
        return Err(BlendError::InvalidConfig(
            "alpha_threshold must be between 0 and 1",
        ));
    }
    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err(BlendError::InvalidConfig(
            "tolerance must be positive and finite",
        ));
    }
    if config.max_iterations == 0 {
        return Err(BlendError::InvalidConfig("max_iterations must be non-zero"));
    }

    let (background_width, background_height) = background.dimensions();
    let (foreground_width, foreground_height) = foreground.dimensions();
    let (offset_x, offset_y) = position;
    let overlap_x0 = offset_x.max(0);
    let overlap_y0 = offset_y.max(0);
    let overlap_x1 = (offset_x + foreground_width as i32).min(background_width as i32);
    let overlap_y1 = (offset_y + foreground_height as i32).min(background_height as i32);
    if overlap_x1 <= overlap_x0 || overlap_y1 <= overlap_y0 {
        return Err(BlendError::NoOverlap);
    }

    let mut active_min_x = i32::MAX;
    let mut active_min_y = i32::MAX;
    let mut active_max_x = i32::MIN;
    let mut active_max_y = i32::MIN;
    for global_y in overlap_y0..overlap_y1 {
        for global_x in overlap_x0..overlap_x1 {
            let source_x = (global_x - offset_x) as u32;
            let source_y = (global_y - offset_y) as u32;
            let alpha = foreground.get_pixel(source_x, source_y)[3] as f32 / 255.0;
            if alpha >= config.alpha_threshold {
                active_min_x = active_min_x.min(global_x);
                active_min_y = active_min_y.min(global_y);
                active_max_x = active_max_x.max(global_x);
                active_max_y = active_max_y.max(global_y);
            }
        }
    }
    if active_min_x == i32::MAX {
        return Err(BlendError::EmptyMask);
    }

    let kernel = gaussian_kernel(config.appearance_sigma);
    let padding = (kernel.len() / 2 + 2) as i32;
    let roi_x0 = (active_min_x - padding).max(0);
    let roi_y0 = (active_min_y - padding).max(0);
    let roi_x1 = (active_max_x + padding + 1).min(background_width as i32);
    let roi_y1 = (active_max_y + padding + 1).min(background_height as i32);
    let roi_width = (roi_x1 - roi_x0) as usize;
    let roi_height = (roi_y1 - roi_y0) as usize;
    let area = roi_width * roi_height;

    let mut target = vec![[0.0; 3]; area];
    let mut source = vec![[0.0; 3]; area];
    let mut alpha = vec![0.0; area];
    for local_y in 0..roi_height {
        for local_x in 0..roi_width {
            let index = local_y * roi_width + local_x;
            let global_x = roi_x0 + local_x as i32;
            let global_y = roi_y0 + local_y as i32;
            let target_pixel = background.get_pixel(global_x as u32, global_y as u32);
            for channel in 0..3 {
                target[index][channel] = srgb_to_linear(target_pixel[channel]);
            }
            let source_x = global_x - offset_x;
            let source_y = global_y - offset_y;
            if source_x >= 0
                && source_y >= 0
                && source_x < foreground_width as i32
                && source_y < foreground_height as i32
            {
                let source_pixel = foreground.get_pixel(source_x as u32, source_y as u32);
                alpha[index] = source_pixel[3] as f32 / 255.0;
                for channel in 0..3 {
                    source[index][channel] = srgb_to_linear(source_pixel[channel]);
                }
            }
        }
    }

    let target_low = blur_rgb(&target, roi_width, roi_height, &kernel);
    let mut weighted_source = vec![[0.0; 3]; area];
    for index in 0..area {
        for channel in 0..3 {
            weighted_source[index][channel] = alpha[index] * source[index][channel];
        }
    }
    let blurred_weight = blur_scalar(&alpha, roi_width, roi_height, &kernel);
    let blurred_source = blur_rgb(&weighted_source, roi_width, roi_height, &kernel);
    let mut source_low = vec![[0.0; 3]; area];
    for index in 0..area {
        if blurred_weight[index] > 1.0e-6 {
            for channel in 0..3 {
                source_low[index][channel] = blurred_source[index][channel] / blurred_weight[index];
            }
        } else {
            source_low[index] = target_low[index];
        }
    }

    let domain: Vec<bool> = alpha
        .iter()
        .map(|value| *value >= config.alpha_threshold)
        .collect();
    let mut boundary = vec![false; area];
    let mut domain_pixels = 0;
    let mut boundary_pixels = 0;
    for index in 0..area {
        if !domain[index] {
            continue;
        }
        domain_pixels += 1;
        let is_boundary = neighbours(index, roi_width, roi_height)
            .into_iter()
            .any(|neighbour| neighbour.is_none_or(|other| !domain[other]));
        boundary[index] = is_boundary;
        if is_boundary {
            boundary_pixels += 1;
        }
    }

    let mut boundary_correction = vec![[0.0; 3]; area];
    for index in 0..area {
        if boundary[index] {
            for channel in 0..3 {
                boundary_correction[index][channel] =
                    target_low[index][channel] - source_low[index][channel];
            }
        }
    }

    let mut mapping = vec![-1_i32; area];
    let mut unknown_pixels = Vec::new();
    for index in 0..area {
        if domain[index] && !boundary[index] {
            mapping[index] = unknown_pixels.len() as i32;
            unknown_pixels.push(index);
        }
    }
    let screening = 1.0 / (config.integration_length * config.integration_length);
    let mut stencil = SparseStencil {
        diagonal: vec![4.0 + screening; unknown_pixels.len()],
        neighbours: vec![[-1; 4]; unknown_pixels.len()],
    };
    let mut rhs = [
        vec![0.0; unknown_pixels.len()],
        vec![0.0; unknown_pixels.len()],
        vec![0.0; unknown_pixels.len()],
    ];
    for (unknown_index, pixel_index) in unknown_pixels.iter().copied().enumerate() {
        for (slot, neighbour) in neighbours(pixel_index, roi_width, roi_height)
            .into_iter()
            .enumerate()
        {
            if let Some(other) = neighbour {
                let mapped = mapping[other];
                if mapped >= 0 {
                    stencil.neighbours[unknown_index][slot] = mapped;
                } else if boundary[other] {
                    for channel in 0..3 {
                        rhs[channel][unknown_index] += boundary_correction[other][channel];
                    }
                }
            }
        }
    }

    let mut correction = boundary_correction;
    let mut iterations = [0; 3];
    let mut relative_residual = [0.0; 3];
    for channel in 0..3 {
        let (solution, used, residual) = pcg(
            &stencil,
            &rhs[channel],
            config.tolerance,
            config.max_iterations,
        );
        iterations[channel] = used;
        relative_residual[channel] = residual;
        for (unknown_index, pixel_index) in unknown_pixels.iter().copied().enumerate() {
            correction[pixel_index][channel] = solution[unknown_index];
        }
    }

    let mut output = background.clone();
    for global_y in overlap_y0..overlap_y1 {
        for global_x in overlap_x0..overlap_x1 {
            let source_x = (global_x - offset_x) as u32;
            let source_y = (global_y - offset_y) as u32;
            let source_pixel = foreground.get_pixel(source_x, source_y);
            let pixel_alpha = source_pixel[3] as f32 / 255.0;
            if pixel_alpha <= 0.0 {
                continue;
            }
            let local_x = (global_x - roi_x0) as usize;
            let local_y = (global_y - roi_y0) as usize;
            let index = local_y * roi_width + local_x;
            let target_pixel = target[index];
            let mut encoded = [0_u8; 3];
            for channel in 0..3 {
                let corrected_source = source[index][channel] + correction[index][channel];
                let composited =
                    pixel_alpha * corrected_source + (1.0 - pixel_alpha) * target_pixel[channel];
                encoded[channel] = linear_to_srgb(composited);
            }
            output.put_pixel(global_x as u32, global_y as u32, Rgb(encoded));
        }
    }

    Ok(BlendResult {
        image: output,
        stats: BlendStats {
            roi: Roi {
                x: roi_x0 as u32,
                y: roi_y0 as u32,
                width: roi_width as u32,
                height: roi_height as u32,
            },
            domain_pixels,
            boundary_pixels,
            unknowns: unknown_pixels.len(),
            iterations,
            relative_residual,
            screening,
        },
    })
}
