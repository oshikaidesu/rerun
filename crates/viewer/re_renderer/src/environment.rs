//! Image-based environment: an equirectangular radiance map that lights meshes and fills the background.
//!
//! The embedder owns the textures; `re_renderer` only binds them for the frame.
//! Diffuse lighting uses a pre-convolved irradiance map (Ramamoorthi & Hanrahan 2001 style cosine lobe),
//! computed on the CPU by [`convolve_irradiance`] since mipmaps aren't available yet.

use crate::resource_managers::GpuTexture2D;

/// Equirectangular environment for one view.
#[derive(Clone, Debug)]
pub struct Environment {
    /// Linear radiance, equirectangular, sampled by the background.
    pub radiance: GpuTexture2D,

    /// Cosine-convolved irradiance divided by π, equirectangular (typically small), sampled by lit meshes.
    /// A uniform radiance `L` convolves to `L` so an unlit white surface renders at `L`.
    pub irradiance: GpuTexture2D,

    /// Glossy radiance for [`SPECULAR_LEVELS`] roughness levels `k / SPECULAR_LEVELS` (k = 1..), stacked
    /// vertically as one equirectangular atlas (see [`prefilter_specular`]). Roughness 0 samples `radiance`.
    pub specular: GpuTexture2D,

    /// Rotation applied to a world direction before the equirectangular lookup.
    pub environment_from_world: glam::Mat3,

    /// Multiplier on both maps.
    pub strength: f32,
}

/// Number of roughness levels in [`Environment::specular`]. Mirrors `SPECULAR_LEVELS` in `shader/utils/lighting.wgsl`.
pub const SPECULAR_LEVELS: usize = 5;

/// Blinn-Phong exponent standing in for the GGX lobe of a roughness (Karis 2013, "Real Shading in Unreal Engine 4":
/// `α = roughness²`, `n = 2 / α² - 2`).
pub fn phong_exponent_for_roughness(roughness: f32) -> f32 {
    let alpha = roughness.max(0.05).powi(2);
    (2.0 / (alpha * alpha) - 2.0).max(0.0)
}

/// Convolves an equirectangular RGB radiance map with a glossy lobe for each roughness level `k / SPECULAR_LEVELS`
/// (k = 1..=SPECULAR_LEVELS) and stacks the results vertically: the output is `out_width * (out_height * SPECULAR_LEVELS) * 3`.
///
/// Uses the normalized Phong lobe `max(0, R·L)^n` as the prefilter kernel (the split-sum first term, Karis 2013).
pub fn prefilter_specular(
    rgb: &[f32],
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<f32> {
    assert_eq!(rgb.len(), width * height * 3);
    let texels = weighted_texels(rgb, width, height);
    let mut out = Vec::with_capacity(out_width * out_height * SPECULAR_LEVELS * 3);
    for level in 1..=SPECULAR_LEVELS {
        let n = phong_exponent_for_roughness(level as f32 / SPECULAR_LEVELS as f32);
        for y in 0..out_height {
            for x in 0..out_width {
                let r = direction_from_equirect_uv(glam::vec2(
                    (x as f32 + 0.5) / out_width as f32,
                    (y as f32 + 0.5) / out_height as f32,
                ));
                let mut sum = glam::Vec3::ZERO;
                let mut weight = 0.0f32;
                for (dir, radiance, solid_angle) in &texels {
                    let cos = r.dot(*dir);
                    if cos > 0.0 {
                        let w = cos.powf(n) * solid_angle;
                        sum += *radiance * w;
                        weight += w;
                    }
                }
                let e = if weight > 0.0 { sum / weight } else { glam::Vec3::ZERO };
                out.extend_from_slice(&[e.x, e.y, e.z]);
            }
        }
    }
    out
}

/// (direction, radiance, solid angle) per texel of an equirectangular map.
fn weighted_texels(rgb: &[f32], width: usize, height: usize) -> Vec<(glam::Vec3, glam::Vec3, f32)> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| {
            let uv = glam::vec2(
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            );
            let theta = uv.y * std::f32::consts::PI;
            let solid_angle =
                theta.sin() * (std::f32::consts::PI / height as f32) * (std::f32::consts::TAU / width as f32);
            let i = (y * width + x) * 3;
            (direction_from_equirect_uv(uv), glam::vec3(rgb[i], rgb[i + 1], rgb[i + 2]), solid_angle)
        })
        .collect()
}

/// Equirectangular texture coordinate for a direction (+Y up, -Z at the image center, +X to the right half).
///
/// Mirrors `equirect_uv_from_direction` in `shader/utils/lighting.wgsl`.
pub fn equirect_uv_from_direction(dir: glam::Vec3) -> glam::Vec2 {
    let dir = dir.normalize_or_zero();
    let u = 0.5 + dir.x.atan2(-dir.z) / std::f32::consts::TAU;
    let v = dir.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
    glam::vec2(u, v)
}

/// Direction for the center of an equirectangular texel. Inverse of [`equirect_uv_from_direction`].
pub fn direction_from_equirect_uv(uv: glam::Vec2) -> glam::Vec3 {
    let phi = (uv.x - 0.5) * std::f32::consts::TAU;
    let theta = uv.y * std::f32::consts::PI;
    let (sin_theta, cos_theta) = theta.sin_cos();
    glam::vec3(sin_theta * phi.sin(), cos_theta, -sin_theta * phi.cos())
}

/// Convolves an equirectangular RGB radiance map with the cosine lobe, divided by π.
///
/// `rgb` is tightly packed `width * height * 3` linear floats. Returns `out_width * out_height * 3`.
/// Cost is `width * height * out_width * out_height`, so downsample the input first.
pub fn convolve_irradiance(
    rgb: &[f32],
    width: usize,
    height: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<f32> {
    assert_eq!(rgb.len(), width * height * 3);
    let texels: Vec<(glam::Vec3, glam::Vec3)> = weighted_texels(rgb, width, height)
        .into_iter()
        .map(|(dir, radiance, solid_angle)| (dir, radiance * solid_angle))
        .collect();

    let mut out = Vec::with_capacity(out_width * out_height * 3);
    for y in 0..out_height {
        for x in 0..out_width {
            let normal = direction_from_equirect_uv(glam::vec2(
                (x as f32 + 0.5) / out_width as f32,
                (y as f32 + 0.5) / out_height as f32,
            ));
            let mut sum = glam::Vec3::ZERO;
            for (dir, weighted) in &texels {
                let cos = normal.dot(*dir);
                if cos > 0.0 {
                    sum += *weighted * cos;
                }
            }
            let e = sum / std::f32::consts::PI;
            out.extend_from_slice(&[e.x, e.y, e.z]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uv_round_trips_through_direction() {
        for (u, v) in [(0.5, 0.5), (0.25, 0.3), (0.8, 0.9), (0.1, 0.05)] {
            let uv = glam::vec2(u, v);
            let back = equirect_uv_from_direction(direction_from_equirect_uv(uv));
            assert!((back - uv).length() < 1e-4, "{uv} -> {back}");
        }
        assert!((direction_from_equirect_uv(glam::vec2(0.5, 0.5)) - glam::Vec3::NEG_Z).length() < 1e-5);
        assert!((direction_from_equirect_uv(glam::vec2(0.5, 0.0)) - glam::Vec3::Y).length() < 1e-5);
    }

    #[test]
    fn uniform_radiance_convolves_to_itself() {
        let (w, h) = (64, 32);
        let rgb: Vec<f32> = (0..w * h).flat_map(|_| [2.0, 0.5, 1.0]).collect();
        let out = convolve_irradiance(&rgb, w, h, 4, 2);
        for px in out.chunks(3) {
            assert!((px[0] - 2.0).abs() < 0.02 && (px[1] - 0.5).abs() < 0.01 && (px[2] - 1.0).abs() < 0.02, "{px:?}");
        }
    }

    #[test]
    fn specular_levels_go_from_mirror_to_matte() {
        let (w, h) = (64, 32);
        let rgb: Vec<f32> = (0..h)
            .flat_map(|y| (0..w).map(move |_| if y < h / 2 { 1.0 } else { 0.0 }))
            .flat_map(|l| [l, l, l])
            .collect();
        let (ow, oh) = (8, 8);
        let out = prefilter_specular(&rgb, w, h, ow, oh);
        assert_eq!(out.len(), ow * oh * SPECULAR_LEVELS * 3);
        let at = |level: usize, y: usize| out[((level * oh + y) * ow) * 3];
        // Straight up stays white at every level; near the horizon the sharp level keeps the edge and the matte level blurs it.
        assert!(at(0, 0) > 0.95 && at(SPECULAR_LEVELS - 1, 0) > 0.8, "{} {}", at(0, 0), at(SPECULAR_LEVELS - 1, 0));
        let just_above = oh / 2 - 1;
        assert!(at(0, just_above) > at(SPECULAR_LEVELS - 1, just_above) + 0.1, "{} vs {}", at(0, just_above), at(SPECULAR_LEVELS - 1, just_above));
        let uniform: Vec<f32> = (0..w * h).flat_map(|_| [0.5, 0.25, 1.0]).collect();
        for px in prefilter_specular(&uniform, w, h, 4, 2).chunks(3) {
            assert!((px[0] - 0.5).abs() < 1e-3 && (px[2] - 1.0).abs() < 1e-3, "{px:?}");
        }
    }

    #[test]
    fn sky_from_above_lights_upward_normals_more() {
        let (w, h) = (64, 32);
        let rgb: Vec<f32> = (0..h)
            .flat_map(|y| (0..w).map(move |_| if y < h / 2 { 1.0 } else { 0.0 }))
            .flat_map(|l| [l, l, l])
            .collect();
        let out = convolve_irradiance(&rgb, w, h, 1, 8);
        let up = out[0];
        let down = out[(8 - 1) * 3];
        assert!(up > 0.9 && down < 0.1, "up {up} down {down}");
    }
}
