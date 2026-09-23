const FILTER_SURFACE_FOOTPRINT: bool = false;
var<private> surface_position_dx: vec3f;
var<private> surface_position_dy: vec3f;
var<private> surface_normal_reference: vec3f;
var<private> surface_normal_dx: vec3f;
var<private> surface_normal_dy: vec3f;
var<private> surface_ray_dx: vec3f;
var<private> surface_ray_dy: vec3f;

fn prepare_surface_footprint(position: vec3f, normal: vec3f) {
    surface_normal_reference = normal;
    surface_position_dx = dpdx(position);
    surface_position_dy = dpdy(position);
    surface_normal_dx = dpdx(normal);
    surface_normal_dy = dpdy(normal);
}

fn footprint_neighbor_normal(normal: vec3f, delta: vec3f) -> vec3f {
    let orientation = select(-1.0,1.0,dot(normal,surface_normal_reference) >= 0.0);
    return normalize(normal + orientation * delta);
}

fn footprint_lod(dx: vec2f, dy: vec2f, size: vec2f) -> f32 {
    return max(0.0, log2(max(max(length(dx * size), length(dy * size)), 1.0)));
}

fn projected_surface_uv(position: vec3f) -> vec2f {
    let clip = frame.projection_from_world * vec4f(position, 1.0);
    return vec2f(clip.x, -clip.y) / max(clip.w, 1e-4) * 0.5 + 0.5;
}

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
    var lod = clamp(levels - 1.0 - above_1x1, 0.0, levels - 1.0);
    if FILTER_SURFACE_FOOTPRINT {
        var dx = equirect_uv_from_direction(environment_direction(world_dir + surface_ray_dx)) - uv;
        var dy = equirect_uv_from_direction(environment_direction(world_dir + surface_ray_dy)) - uv;
        dx.x -= round(dx.x); dy.x -= round(dy.x);
        lod = min(levels-1.0, max(lod, footprint_lod(dx,dy,vec2f(textureDimensions(environment_radiance)))));
    }
    if FILTER_SURFACE_FOOTPRINT {
        // Bright HDR texels must not dominate a footprint before display clamping (Karis / Lottes).
        let offsets = array<vec2f,4>(vec2f(-0.125,-0.375),vec2f(0.375,-0.125),vec2f(-0.375,0.125),vec2f(0.125,0.375));
        var weighted = vec3f(0.0); var weights = 0.0;
        for (var i=0u; i<4u; i+=1u) {
            let ray = world_dir + surface_ray_dx * offsets[i].x + surface_ray_dy * offsets[i].y;
            let sample_uv = equirect_uv_from_direction(environment_direction(ray));
            let radiance = textureSampleLevel(environment_radiance,equirect_sampler,sample_uv,lod).rgb * frame.environment_strength;
            let weight = 1.0 / (1.0 + max(0.0,max(radiance.x,max(radiance.y,radiance.z))));
            weighted += radiance * weight; weights += weight;
        }
        return weighted / max(weights,1e-6);
    }
    return textureSampleLevel(environment_radiance, equirect_sampler, uv, lod).rgb * frame.environment_strength;
}

fn view_direction_to_camera(world_position: vec3f) -> vec3f {
    if is_camera_orthographic() {
        return -frame.camera_forward;
    }
    return normalize(frame.camera_position - world_position);
}
