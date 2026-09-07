#import <../global_bindings.wgsl>

/// Simple directional lighting: two fixed lights + ambient.
///
/// TODO(#1800): We should implement proper scene lighting and material properties.
fn simple_lighting(normal: vec3f) -> f32 {
    var shading = 0.2;
    shading += 1.0 * clamp(dot(normalize(vec3f(1.0, 2.0, 3.0)), normal), 0.0, 1.0);
    shading += 0.5 * clamp(dot(normalize(vec3f(-1.0, -3.0, -5.0)), normal), 0.0, 1.0);
    return clamp(shading, 0.0, 1.0);
}

/// Equirectangular texture coordinate for a direction (+Y up, -Z at the image center).
/// Mirrors `equirect_uv_from_direction` in `environment.rs`.
fn equirect_uv_from_direction(dir: vec3f) -> vec2f {
    let u = 0.5 + atan2(dir.x, -dir.z) / 6.283185307179586;
    let v = acos(clamp(dir.y, -1.0, 1.0)) / 3.141592653589793;
    return vec2f(u, v);
}

fn environment_direction(world_dir: vec3f) -> vec3f {
    return normalize(frame.environment_from_world * world_dir);
}

/// Background radiance seen along a world direction.
fn environment_radiance_along(world_dir: vec3f) -> vec3f {
    let uv = equirect_uv_from_direction(environment_direction(world_dir));
    return textureSampleLevel(environment_radiance, trilinear_sampler_clamped, uv, 0.0).rgb * frame.environment_strength;
}

/// Diffuse shading (radiance per unit albedo) for a world normal: the bound environment when present,
/// otherwise the fixed lights of `simple_lighting`.
fn diffuse_shading(normal: vec3f) -> vec3f {
    if frame.environment_present == 1u {
        let uv = equirect_uv_from_direction(environment_direction(normal));
        return textureSampleLevel(environment_irradiance, trilinear_sampler_clamped, uv, 0.0).rgb * frame.environment_strength;
    }
    return vec3f(simple_lighting(normal));
}
