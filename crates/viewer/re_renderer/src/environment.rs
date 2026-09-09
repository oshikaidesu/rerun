//! Image-based environment: an equirectangular radiance map that lights meshes and fills the background.
//!
//! The embedder owns the textures; `re_renderer` only binds them for the frame.
//! Diffuse lighting uses a pre-convolved irradiance map (Ramamoorthi & Hanrahan 2001 style cosine lobe,
//! low frequency so a tiny CPU convolution suffices). Glossy reflections read the radiance map's mip chain
//! at a level chosen by roughness ([`roughness_to_lod`]), so the radiance texture should be created with
//! `TextureManager2D::create_with_mipmaps`.

use crate::resource_managers::GpuTexture2D;

/// Equirectangular environment for one view.
#[derive(Clone, Debug)]
pub struct Environment {
    /// Linear radiance, equirectangular, with a mip chain: level 0 is the background and mirror reflections,
    /// rougher surfaces read higher levels.
    pub radiance: GpuTexture2D,

    /// Cosine-convolved irradiance divided by π, equirectangular (typically small), sampled by lit meshes.
    /// A uniform radiance `L` convolves to `L` so an unlit white surface renders at `L`.
    pub irradiance: GpuTexture2D,

    /// Rotation applied to a world direction before the equirectangular lookup.
    pub environment_from_world: glam::Mat3,

    /// Multiplier on both maps.
    pub strength: f32,
}

/// Shared local reflection capture. Up to two sets of six faces (+X, -X, +Y, -Y, +Z, -Z) in a 3x4 atlas.
/// RGB is linear radiance and alpha is coverage; uncovered directions use the environment.
#[derive(Clone, Debug)]
pub struct SceneReflection {
    pub atlas: GpuTexture2D,
    pub origins: [glam::Vec3; 2],
    pub count: u32,
    pub bounds_min: glam::Vec3,
    pub bounds_max: glam::Vec3,
}

/// Mip level of the radiance map for a roughness in [0, 1], given the chain's level count.
/// Mirrors `environment_specular_along` in `shader/utils/lighting.wgsl`. Borrowed from Karis 2013
/// (`ComputeReflectionCaptureMipFromRoughness`): a GGX lobe of roughness `r` wants the level
/// `1 - 1.2·log2(r)` above 1×1 of a cube face; an equirectangular map is four faces wide, so two
/// more levels. Clamped to the chain, so roughness 0 is the full-resolution level.
pub fn roughness_to_lod(roughness: f32, levels: u32) -> f32 {
    let top = (levels.max(1) - 1) as f32;
    let r = roughness.clamp(0.0, 1.0).max(0.0001);
    let above_1x1 = 1.0 - 1.2 * r.log2() + 2.0;
    (top - above_1x1).clamp(0.0, top)
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
            let solid_angle = theta.sin()
                * (std::f32::consts::PI / height as f32)
                * (std::f32::consts::TAU / width as f32);
            let i = (y * width + x) * 3;
            (
                direction_from_equirect_uv(uv),
                glam::vec3(rgb[i], rgb[i + 1], rgb[i + 2]),
                solid_angle,
            )
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
        assert!(
            (direction_from_equirect_uv(glam::vec2(0.5, 0.5)) - glam::Vec3::NEG_Z).length() < 1e-5
        );
        assert!((direction_from_equirect_uv(glam::vec2(0.5, 0.0)) - glam::Vec3::Y).length() < 1e-5);
    }

    #[test]
    fn uniform_radiance_convolves_to_itself() {
        let (w, h) = (64, 32);
        let rgb: Vec<f32> = (0..w * h).flat_map(|_| [2.0, 0.5, 1.0]).collect();
        let out = convolve_irradiance(&rgb, w, h, 4, 2);
        for px in out.chunks(3) {
            assert!(
                (px[0] - 2.0).abs() < 0.02
                    && (px[1] - 0.5).abs() < 0.01
                    && (px[2] - 1.0).abs() < 0.02,
                "{px:?}"
            );
        }
    }

    #[test]
    fn roughness_picks_levels_like_a_reflection_capture() {
        // 4096-wide chain: 13 levels. Mirror stays on the base level; 0.3 lands near a 34 px wide map.
        assert_eq!(roughness_to_lod(0.0, 13), 0.0);
        let glossy = roughness_to_lod(0.3, 13);
        assert!((glossy - 6.9).abs() < 0.1, "{glossy}");
        assert!(roughness_to_lod(0.02, 13) < glossy && glossy < roughness_to_lod(0.6, 13));
        assert!(roughness_to_lod(1.0, 13) <= 12.0);
        assert_eq!(roughness_to_lod(0.5, 1), 0.0);
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
