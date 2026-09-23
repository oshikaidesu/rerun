#import <./rectangle.wgsl>
#import <./utils/depth_offset.wgsl>
#import <./utils/field.wgsl>

/// Vertex stage for rectangles that carry a surface program.
///
/// The picture is a grid of `field_grid²` cells (TriangleList, 6 vertices per cell) so the field can
/// move it the way it moves a mesh vertex: forward, in the instance frame. That is what lets the
/// silhouette leave the rectangle. `field_grid <= 1` degenerates to the plain two triangles.
@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32) -> VertexOut {
    let n = max(u32(rect_info.field_grid), 1u);
    let cell = v_idx / 6u;
    var corner = array<vec2f, 6>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
        vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
    );
    let texcoord = (vec2f(f32(cell % n), f32(cell / n)) + corner[v_idx % 6u]) / f32(n);

    let frame_position = texcoord.x * rect_info.extent_u + texcoord.y * rect_info.extent_v;
    let cross_normal = cross(rect_info.extent_u, rect_info.extent_v);
    let normal = cross_normal / max(length(cross_normal), 1e-6);
    let field = program_field(FieldIn(frame_position, normal, rect_info.surface_params));
    let placed = rect_info.top_left_corner_position + frame_position + field.offset;
    let pos = placed + program_motion(rect_info.surface_params[5].w, placed);

    var out: VertexOut;
    out.position = apply_depth_offset(frame.projection_from_world * vec4f(pos, 1.0), rect_info.depth_offset);
    out.texcoord = texcoord;
    out.world_position = pos;
    if rect_info.sample_type == SAMPLE_TYPE_NV12 {
        out.texcoord.y /= 1.5;
    }
    if rect_info.sample_type == SAMPLE_TYPE_YUY2 {
        out.texcoord.x /= 2.0;
    }

    return out;
}
