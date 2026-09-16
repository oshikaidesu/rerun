struct FrameUniformBuffer {
    view_from_world: mat4x3f,
    projection_from_view: mat4x4f,
    projection_from_world: mat4x4f,

    /// Camera position in world space.
    camera_position: vec3f,

    /// For perspective: Multiply this with a camera distance to get a measure of how wide a pixel is in world units.
    /// For orthographic: This is the world size value, independent of distance.
    pixel_world_size_from_camera_distance: f32,

    /// Camera direction in world space.
    /// Same as -vec3f(view_from_world[0].z, view_from_world[1].z, view_from_world[2].z)
    camera_forward: vec3f,

    /// How many pixels there are per point.
    /// I.e. the UI zoom factor.
    pixels_from_point: f32,

    /// (tan(fov_y / 2) * aspect_ratio, tan(fov_y /2)), i.e. half ratio of screen dimension to screen distance in x & y.
    /// Both values are set to f32max for orthographic projection
    tan_half_fov: vec2f,

    /// re_renderer defined device tier.
    device_tier: u32,

    /// boolean (0/1): set to true for snapshot tests to minimize
    /// GPU/driver-specific stuff like alpha-to-coverage.
    deterministic_rendering: u32,

    /// Screen resolution in pixels.
    framebuffer_resolution: vec2f,

    /// Multiplier on the environment maps.
    environment_strength: f32,

    /// boolean (0/1): whether an environment is bound.
    environment_present: u32,

    /// Rotation applied to world directions before the equirectangular lookup.
    environment_from_world: mat3x3f,
    reflection_origin: vec4f,
    reflection_origin_second: vec4f,
    reflection_min: vec4f,
    reflection_max: vec4f,
    /// xyz: world direction toward the sun (the environment's brightest texel); w: the share of the
    /// diffuse light that comes from it. 0 = no sun bound, nothing casts a shadow.
    sun_direction: vec4f,
    /// rgb: the sun's tint. w = 1 while the light cookie is captured: surfaces then write what they let through.
    sun_color: vec4f,
    /// World → light cookie uv (orthographic, looking along the sun).
    light_uv_from_world: mat4x4f,
    /// x: camera-forward depth below which world geometry starts to fade (0 = never); it is gone at x / 3.
    near_fade: vec4f,
};

@group(0) @binding(0)
var<uniform> frame: FrameUniformBuffer;

@group(0) @binding(1)
var nearest_sampler_repeat: sampler;
@group(0) @binding(2)
var nearest_sampler_clamped: sampler;
@group(0) @binding(3)
var trilinear_sampler_repeat: sampler;
@group(0) @binding(4)
var environment_radiance: texture_2d<f32>;
@group(0) @binding(5)
var environment_irradiance: texture_2d<f32>;
@group(0) @binding(6)
var equirect_sampler: sampler;
@group(0) @binding(7)
var backdrop_texture: texture_2d<f32>;
@group(0) @binding(8)
var screen_sampler: sampler;

@group(0) @binding(9)
var scene_reflection: texture_2d<f32>;

/// What the sun's light meets on its way: premultiplied tint × coverage of the blockers, seen from the sun.
@group(0) @binding(10)
var light_cookie: texture_2d<f32>;

/// Per-object world motion written on the GPU by the embedder: three vec4 per object —
/// `(offset.xyz, turn)`, `(centre.xyz, _)`, `(axis.xyz, _)`. The object whose last param is `n`
/// reads entry `n - 1`; 0 = not moved. A vertex is turned by `turn` radians about `axis` through
/// `centre`, then offset.
@group(0) @binding(11)
var<storage, read> motion: array<vec4f>;

fn motion_offset(slot: f32, world_position: vec3f) -> vec3f {
    let n = u32(max(slot, 0.0) + 0.5);
    if n == 0u || n * 3u > arrayLength(&motion) {
        return vec3f(0.0);
    }
    let base = (n - 1u) * 3u;
    let move_turn = motion[base];
    var turned = vec3f(0.0);
    if move_turn.w != 0.0 {
        let r = world_position - motion[base + 1u].xyz;
        let axis = motion[base + 2u].xyz;
        turned = r * cos(move_turn.w) + cross(axis, r) * sin(move_turn.w) + axis * dot(axis, r) * (1.0 - cos(move_turn.w)) - r;
    }
    return move_turn.xyz + turned;
}

// See config.rs#DeviceTier
const DEVICE_TIER_GLES = 0u;
const DEVICE_TIER_WEBGPU = 1u;

/// How much of a fragment survives the camera's near fade (Unity Camera Fading, Godot Distance Fade).
fn near_fade(world_position: vec3f) -> f32 {
    let start = frame.near_fade.x;
    if start <= 0.0 {
        return 1.0;
    }
    let depth = dot(world_position - frame.camera_position, frame.camera_forward);
    return smoothstep(start / 3.0, start, depth);
}
