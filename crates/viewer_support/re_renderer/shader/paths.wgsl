#import <./types.wgsl>
#import <./global_bindings.wgsl>
#import <./utils/srgb.wgsl>

struct VertexIn {
    @location(0) position: vec2f,
    @location(1) color: vec4f, // straight sRGB, unorm8x4
};

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f, // linear, premultiplied
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.position = frame.projection_from_world * vec4f(in.position, 0.0, 1.0);
    let linear = linear_from_srgba_unmultiplied(in.color);
    out.color = vec4f(linear.rgb * linear.a, linear.a);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    return in.color;
}
