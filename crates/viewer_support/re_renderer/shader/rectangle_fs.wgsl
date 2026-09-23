#import <./rectangle_fragment.wgsl>

// A rectangle without a surface program: its picture as it is.
fn program_field(in: FieldIn) -> FieldOut { return FieldOut(vec3f(0.0), in.normal); }
fn program_tint(slot: f32) -> vec4f { return vec4f(1.0); }
fn program_surface(in: SurfaceIn) -> vec3f { return in.albedo; }
