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

// Box-projected local reflection, as used by reflection probes (Arm / Happy Elements).
fn local_reflection(position: vec3f, direction: vec3f, roughness: f32, origin: vec3f, probe: u32) -> vec4f {
    var ray = direction;
    if all(position >= frame.reflection_min.xyz) && all(position <= frame.reflection_max.xyz) {
        let safe_dir = select(select(vec3f(-1e-6), vec3f(1e-6), direction >= vec3f(0.0)), direction, abs(direction) > vec3f(1e-6));
        let far = max((frame.reflection_min.xyz - position) / safe_dir, (frame.reflection_max.xyz - position) / safe_dir);
        let distance = min(far.x, min(far.y, far.z));
        ray = position + direction * distance - origin;
    }
    let a = abs(ray);
    var face = 0u;
    if a.x >= a.y && a.x >= a.z { face = select(1u, 0u, ray.x >= 0.0); }
    else if a.y >= a.z { face = select(3u, 2u, ray.y >= 0.0); }
    else { face = select(5u, 4u, ray.z >= 0.0); }
    let directions = array<vec3f, 6>(vec3f(1,0,0),vec3f(-1,0,0),vec3f(0,1,0),vec3f(0,-1,0),vec3f(0,0,1),vec3f(0,0,-1));
    let ups = array<vec3f, 6>(vec3f(0,-1,0),vec3f(0,-1,0),vec3f(0,0,1),vec3f(0,0,-1),vec3f(0,-1,0),vec3f(0,-1,0));
    let right = cross(directions[face], ups[face]);
    let uv = vec2f(dot(ray, right), -dot(ray, ups[face])) / max(dot(ray,directions[face]),1e-6) * 0.5 + 0.5;
    // Keep each face at least 4x4 to avoid cross-face mip leakage.
    let face_size = f32(textureDimensions(scene_reflection).x) / 3.0;
    let lod = clamp(roughness,0.0,1.0) * max(log2(face_size) - 2.0, 0.0);
    let margin = min(0.49, exp2(ceil(lod)) / face_size);
    let local = clamp(uv,vec2f(margin),vec2f(1.0-margin));
    let atlas_uv = (local + vec2f(f32(face % 3u),f32(face / 3u) + f32(probe)*2.0)) / vec2f(3,4);
    let captured = textureSampleLevel(scene_reflection, screen_sampler, atlas_uv, lod);
    return captured;
}

fn scene_specular(position: vec3f, direction: vec3f, roughness: f32) -> vec3f {
    let fallback = environment_specular_along(direction, roughness);
    if frame.reflection_origin.w == 0.0 { return fallback; }
    var captured = local_reflection(position,direction,roughness,frame.reflection_origin.xyz,0u);
    if frame.reflection_origin.w > 1.0 {
        let d0 = position-frame.reflection_origin.xyz;
        let d1 = position-frame.reflection_origin_second.xyz;
        let weight = smoothstep(0.0,1.0,dot(d0,d0)/max(dot(d0,d0)+dot(d1,d1),1e-6));
        captured = mix(captured,local_reflection(position,direction,roughness,frame.reflection_origin_second.xyz,1u),weight);
    }
    return captured.rgb + (1.0-captured.a)*fallback;
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
/// radiance mip chain (split-sum), and for transmissive surfaces the refracted see-through of what
/// is drawn behind (the backdrop, read where the refracted ray leaves a slab of `thickness`; vgpu's
/// transmission example) with the environment where nothing is drawn.
/// `surface` = (roughness, metallic, transmission, ior). Without an environment, the fixed lights apply.
fn shade_surface(albedo: vec3f, normal: vec3f, view_dir: vec3f, world_position: vec3f, thickness: f32, surface: vec4f) -> vec3f {
    let roughness = clamp(surface.x, 0.0, 1.0);
    let metallic = clamp(surface.y, 0.0, 1.0);
    let transmission = clamp(surface.z, 0.0, 1.0);
    let ior = max(surface.w, 1.0);
    let n_dot_v = clamp(dot(normal, view_dir), 1e-4, 1.0);
    let f0_dielectric = pow((ior - 1.0) / (ior + 1.0), 2.0);
    let f0 = mix(vec3f(f0_dielectric), albedo, metallic);
    let diffuse_weight = (1.0 - metallic) * (1.0 - transmission);

    if frame.environment_present != 1u && frame.reflection_origin.w == 0.0 {
        return albedo * simple_lighting(normal);
    }

    let diffuse = albedo * diffuse_weight * diffuse_shading(normal);
    let reflected = reflect(-view_dir, normal);
    let specular = scene_specular(world_position, reflected, roughness) * env_brdf_approx(f0, roughness, n_dot_v);
    var transmitted = vec3f(0.0);
    if transmission > 0.0 {
        let fresnel = f0_dielectric + (1.0 - f0_dielectric) * pow(1.0 - n_dot_v, 5.0);
        let refracted = refract(-view_dir, normal, 1.0 / ior);
        let through = select(reflected, refracted, any(refracted != vec3f(0.0)));
        let exit = world_position + through * thickness;
        let clip = frame.projection_from_world * vec4f(exit, 1.0);
        let uv = clamp(vec2f(clip.x, -clip.y) / max(clip.w, 1e-4) * 0.5 + 0.5, vec2f(0.0), vec2f(1.0));
        let levels = f32(textureNumLevels(backdrop_texture));
        let lod = pow(roughness, 0.8) * max(levels - 1.0, 0.0) * 0.55;
        let behind = textureSampleLevel(backdrop_texture, screen_sampler, uv, lod);
        let seen = behind.rgb + (1.0 - behind.a) * environment_specular_along(through, roughness);
        transmitted = seen * albedo * transmission * (1.0 - metallic) * (1.0 - fresnel);
    }
    return diffuse + specular + transmitted;
}
