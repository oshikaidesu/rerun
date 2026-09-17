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

/// Per-object motion written on the GPU by the embedder: four vec4 per object —
/// `(offset.xyz, turn)`, `(centre.xyz, scale)`, `(axis.xyz, kind)`, `(tint.rgb, opacity)`. Kind 0 is a thing;
/// kind 1 is a connector between two things (see `motion_offset`). The object whose
/// last param is `n` reads entry `n - 1`; 0 = not moved. A vertex is scaled by `scale` and turned by
/// `turn` radians about `axis` through `centre`, then offset. A fragment is multiplied by `tint` and
/// `opacity` (premultiplied, so opacity scales colour and coverage alike).
@group(0) @binding(11)
var<storage, read> motion: array<vec4f>;

const MOTION_STRIDE = 4u;

fn motion_entry(slot: f32) -> u32 {
    let n = u32(max(slot, 0.0) + 0.5);
    if n == 0u || n * MOTION_STRIDE > arrayLength(&motion) {
        return 0u;
    }
    return n;
}

fn motion_offset(slot: f32, world_position: vec3f) -> vec3f {
    let n = motion_entry(slot);
    if n == 0u {
        return vec3f(0.0);
    }
    let base = (n - 1u) * MOTION_STRIDE;
    let move_turn = motion[base];
    let centre_scale = motion[base + 1u];
    let axis_kind = motion[base + 2u];
    // A connector entry (kind 1): `(offset at A, _)`, `(A, _)`, `(B - A, 1)`, `(offset at B, _)`.
    // Each vertex takes the offset of the end it is nearer to, blended along A→B, so a line between two
    // moved things keeps touching both.
    if axis_kind.w == 1.0 {
        let ab = axis_kind.xyz;
        let t = clamp(dot(world_position - centre_scale.xyz, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
        return mix(move_turn.xyz, motion[base + 3u].xyz, t);
    }
    let scale = select(centre_scale.w, 1.0, centre_scale.w == 0.0);
    var placed = vec3f(0.0);
    if move_turn.w != 0.0 || scale != 1.0 {
        let r = (world_position - centre_scale.xyz) * scale;
        let axis = axis_kind.xyz;
        let c = cos(move_turn.w);
        let s = sin(move_turn.w);
        placed = r * c + cross(axis, r) * s + axis * dot(axis, r) * (1.0 - c) - (world_position - centre_scale.xyz);
    }
    return move_turn.xyz + placed;
}

/// `(tint.rgb, opacity)` of the object, `vec4f(1.0)` when it carries no motion entry.
fn motion_tint(slot: f32) -> vec4f {
    let n = motion_entry(slot);
    if n == 0u || motion[(n - 1u) * MOTION_STRIDE + 2u].w == 1.0 {
        return vec4f(1.0);
    }
    return motion[(n - 1u) * MOTION_STRIDE + 3u];
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
