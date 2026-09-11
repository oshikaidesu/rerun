#import <./rectangle_fragment.wgsl>

fn motolii_field(in: FieldIn) -> FieldOut { return FieldOut(vec3f(0.0), in.normal); }

fn motolii_surface(in: SurfaceIn) -> vec3f {
    if frame.sun_color.w > 0.0 {
        return vec3f(0.0); // an unlit picture is an opaque blocker in the light cookie
    }
    return in.albedo * sun_shade(in.world_position, in.normal, 0.5);
}
