#import <../global_bindings.wgsl>
#import <./camera.wgsl>

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
    return textureSampleLevel(environment_radiance, equirect_sampler, uv, 0.0).rgb * frame.environment_strength;
}

/// Diffuse shading (radiance per unit albedo) for a world normal: the bound environment when present,
/// otherwise the fixed lights of `simple_lighting`.
fn diffuse_shading(normal: vec3f) -> vec3f {
    if frame.environment_present == 1u {
        let uv = equirect_uv_from_direction(environment_direction(normal));
        return textureSampleLevel(environment_irradiance, equirect_sampler, uv, 0.0).rgb * frame.environment_strength;
    }
    return vec3f(simple_lighting(normal));
}

/// Glossy radiance along a world direction for a roughness in [0, 1]: the radiance map's mip chain,
/// level picked by roughness. Mirrors `roughness_to_lod` in `environment.rs` (Karis 2013,
/// `ComputeReflectionCaptureMipFromRoughness`: `1 - 1.2 log2(r)` levels above 1x1 for a cube face;
/// an equirectangular map is four faces wide, hence two more levels).
fn environment_specular_along(world_dir: vec3f, roughness: f32) -> vec3f {
    let uv = equirect_uv_from_direction(environment_direction(world_dir));
    let levels = f32(textureNumLevels(environment_radiance));
    let r = clamp(roughness, 0.0, 1.0);
    let above_1x1 = 1.0 - 1.2 * log2(max(r, 0.0001)) + 2.0;
    let lod = clamp(levels - 1.0 - above_1x1, 0.0, levels - 1.0);
    return textureSampleLevel(environment_radiance, equirect_sampler, uv, lod).rgb * frame.environment_strength;
}

fn view_direction_to_camera(world_position: vec3f) -> vec3f {
    if is_camera_orthographic() {
        return -frame.camera_forward;
    }
    return normalize(frame.camera_position - world_position);
}

/// Split-sum environment BRDF, analytic fit (Karis 2014, "Physically Based Shading on Mobile").
fn env_brdf_approx(f0: vec3f, roughness: f32, n_dot_v: f32) -> vec3f {
    let c0 = vec4f(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4f(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let ab = vec2f(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

/// Radiance leaving a surface: Lambert diffuse from the irradiance map, glossy reflection from the
/// radiance mip chain (split-sum), and refracted see-through of the environment for transmissive surfaces.
/// `surface` = (roughness, metallic, transmission, ior). Without an environment, the fixed lights apply.
fn shade_surface(albedo: vec3f, normal: vec3f, view_dir: vec3f, surface: vec4f) -> vec3f {
    let roughness = clamp(surface.x, 0.0, 1.0);
    let metallic = clamp(surface.y, 0.0, 1.0);
    let transmission = clamp(surface.z, 0.0, 1.0);
    let ior = max(surface.w, 1.0);
    let n_dot_v = clamp(dot(normal, view_dir), 1e-4, 1.0);
    let f0_dielectric = pow((ior - 1.0) / (ior + 1.0), 2.0);
    let f0 = mix(vec3f(f0_dielectric), albedo, metallic);
    let diffuse_weight = (1.0 - metallic) * (1.0 - transmission);

    if frame.environment_present != 1u {
        return albedo * simple_lighting(normal);
    }

    let diffuse = albedo * diffuse_weight * diffuse_shading(normal);
    let reflected = reflect(-view_dir, normal);
    let specular = environment_specular_along(reflected, roughness) * env_brdf_approx(f0, roughness, n_dot_v);
    var transmitted = vec3f(0.0);
    if transmission > 0.0 {
        let fresnel = f0_dielectric + (1.0 - f0_dielectric) * pow(1.0 - n_dot_v, 5.0);
        let refracted = refract(-view_dir, normal, 1.0 / ior);
        let through = select(reflected, refracted, any(refracted != vec3f(0.0)));
        transmitted = environment_specular_along(through, roughness) * albedo * transmission * (1.0 - metallic) * (1.0 - fresnel);
    }
    return diffuse + specular + transmitted;
}
