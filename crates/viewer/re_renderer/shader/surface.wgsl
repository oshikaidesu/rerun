#import <./utils/lighting.wgsl>

/// Fragment hook input. `normal` already faces the camera (two-sided).
struct SurfaceIn {
    albedo: vec3f,
    normal: vec3f,
    view_dir: vec3f,
    world_position: vec3f,
    /// Instance scale (length of one axis of world_from_mesh): the slab a refracted ray crosses.
    thickness: f32,
    params: array<vec4f, 3>,
    uv: vec2f,
    coverage: f32,
};

