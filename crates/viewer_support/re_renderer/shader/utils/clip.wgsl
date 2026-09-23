/// World-space cut shared by rectangles, point clouds and meshes (mirrors `ClipPlane` in clip.rs):
/// outside when a plane is set and the point lies past it.
fn clip_outside(plane: vec4f, world_position: vec3f) -> bool {
    return dot(plane.xyz, plane.xyz) > 0.0 && dot(plane.xyz, world_position) > plane.w;
}
