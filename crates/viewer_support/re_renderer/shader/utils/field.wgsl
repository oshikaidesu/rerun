/// Vertex/sample hook shared by meshes (instanced_mesh_base.wgsl) and rectangles (rectangle_fragment.wgsl).
/// `frame_position` is the point in the instance frame (rotation and scale, no translation), so a field
/// travels with what it moves. `params` are the 24 floats the embedder attached.
struct FieldIn {
    frame_position: vec3f,
    normal: vec3f,
    params: array<vec4f, 6>,
};

struct FieldOut {
    /// Added to the world position.
    offset: vec3f,
    /// Replaces the world normal.
    normal: vec3f,
};
