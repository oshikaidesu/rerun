#import <./types.wgsl>
#import <./global_bindings.wgsl>
#import <./mesh_vertex.wgsl>
#import <./utils/srgb.wgsl>
#import <./utils/lighting.wgsl>
#import <./utils/noise.wgsl>
#import <./utils/clip.wgsl>

// This file is never compiled alone: `MeshProgram` appends the two hooks below
// (defaults or the embedder's own) and compiles the result. See mesh_program.rs.
//
//   fn motolii_field(in: FieldIn) -> FieldOut        moves a vertex in the instance frame
//   fn motolii_surface(in: SurfaceIn) -> vec3f       radiance leaving a shaded fragment

/// Vertex hook input. `frame_position` is the vertex in the instance frame (rotation and scale of
/// world_from_mesh, no translation), so a field travels with its mesh. `params` are the instance's 12 floats.
struct FieldIn {
    frame_position: vec3f,
    normal: vec3f,
    params: array<vec4f, 3>,
};

struct FieldOut {
    /// Added to the world position.
    offset: vec3f,
    /// Replaces the world normal.
    normal: vec3f,
};

#import <./surface.wgsl>

@group(1) @binding(0)
var albedo_texture: texture_2d<f32>;

// Keep in sync with `gpu_data::TextureFormat` in mesh.rs
const FORMAT_RGBA: u32 = 0;
const FORMAT_GRAYSCALE: u32 = 1;

// Keep in sync with `gpu_data::MaterialUniformBuffer` in mesh.rs
struct MaterialUniformBuffer {
    albedo_factor: vec4f,
    texture_format: u32,
};

@group(1) @binding(1)
var<uniform> material: MaterialUniformBuffer;

// Keep in sync with `clip_gpu_data::ClipUniformBuffer` in mesh_renderer.rs
struct ClipUniformBuffer {
    plane: vec4f,
    cap: u32,
};

@group(2) @binding(0)
var<uniform> clip: ClipUniformBuffer;

struct VertexOut {
    @builtin(position)
    position: vec4f,

    @location(0)
    color: vec3f, // 0-1 linear space with unmultiplied/separate alpha

    @location(1)
    texcoord: vec2f,

    @location(2)
    normal_world_space: vec3f,

    @location(3) @interpolate(flat)
    additive_tint_rgba: vec4f, // 0-1 linear space with unmultiplied/separate alpha

    @location(4) @interpolate(flat)
    outline_mask_ids: vec2u,

    @location(5) @interpolate(flat)
    picking_layer_id: vec4u,

    @location(6)
    world_position: vec3f,

    @location(7) @interpolate(flat)
    params0: vec4f,
    @location(8) @interpolate(flat)
    params1: vec4f,
    @location(9) @interpolate(flat)
    params2: vec4f,
    @location(10) @interpolate(flat)
    thickness: f32,
};

@vertex
fn vs_main(in_vertex: VertexIn, in_instance: InstanceIn) -> VertexOut {
    // Instance frame: rotation and scale of world_from_mesh, no translation. The field travels with the mesh.
    let frame_position = vec3f(
        dot(in_instance.world_from_mesh_row_0.xyz, in_vertex.position),
        dot(in_instance.world_from_mesh_row_1.xyz, in_vertex.position),
        dot(in_instance.world_from_mesh_row_2.xyz, in_vertex.position),
    );
    let translation = vec3f(in_instance.world_from_mesh_row_0.w, in_instance.world_from_mesh_row_1.w, in_instance.world_from_mesh_row_2.w);
    var world_normal = vec3f(
        dot(in_instance.world_from_mesh_normal_row_0.xyz, in_vertex.normal),
        dot(in_instance.world_from_mesh_normal_row_1.xyz, in_vertex.normal),
        dot(in_instance.world_from_mesh_normal_row_2.xyz, in_vertex.normal),
    );
    var world_position = frame_position + translation;
    let params = array<vec4f, 3>(in_instance.params0, in_instance.params1, in_instance.params2);
    let field = motolii_field(FieldIn(frame_position, world_normal, params));
    world_position += field.offset;
    world_normal = field.normal;

    var out: VertexOut;
    out.position = frame.projection_from_world * vec4f(world_position, 1.0);
    out.color = linear_from_srgb(in_vertex.color.rgb);
    out.texcoord = in_vertex.texcoord;
    out.normal_world_space = world_normal;
    // Instance encoded is with pre-multiplied alpha in sRGB.
    out.additive_tint_rgba = vec4f(linear_from_srgb(in_instance.additive_tint_srgba.rgb / in_instance.additive_tint_srgba.a),
                                    in_instance.additive_tint_srgba.a);
    out.outline_mask_ids = in_instance.outline_mask_ids;
    out.picking_layer_id = in_instance.picking_layer_id;
    out.world_position = world_position;
    out.params0 = in_instance.params0;
    out.params1 = in_instance.params1;
    out.params2 = in_instance.params2;
    out.thickness = length(in_instance.world_from_mesh_row_0.xyz);

    return out;
}

@fragment
fn fs_main_shaded(in: VertexOut) -> @location(0) vec4f {
    if clip_outside(clip.plane, in.world_position) {
        discard;
    }
    let sample = textureSample(albedo_texture, trilinear_sampler_repeat, in.texcoord);
    var texture: vec3f;
    switch material.texture_format {
        case FORMAT_RGBA: { texture = linear_from_srgb(sample.rgb); }
        case FORMAT_GRAYSCALE: { texture = linear_from_srgb(sample.rrr); }
        default: { texture = vec3f(0.0); }
    }

    // TODO(andreas): We could just pass on vertex & texture alpha here and make use of it.
    // However, we currently don't have the detection code on the CPU side to flag such meshes as transparent.
    // Therefore, using alpha here would mean that you get it surprise-enabled once you change the tint & albedo factor.
    // To avoid that, we simply ignore it for now.
    var albedo = vec4f(texture * in.color, 1.0) * material.albedo_factor;

    // The additive tint linear space with unmultiplied/separate (!!) alpha.
    albedo += vec4f(in.additive_tint_rgba.rgb, 0.0);
    albedo *= in.additive_tint_rgba.a;

    if all(in.normal_world_space == vec3f(0.0, 0.0, 0.0)) {
        // no normal, no shading
        return albedo;
    }
    let view_dir = view_direction_to_camera(in.world_position);
    var normal = normalize(in.normal_world_space);
    let back_side = dot(normal, view_dir) < 0.0;
    if back_side {
        normal = -normal; // two-sided
    }
    // Inside faces seen through the cut (their normal points away from the viewer) read as a flat cap:
    // shade them with the plane's normal. Winding is not used: meshes are drawn two-sided.
    if clip.cap == 1u && back_side && dot(clip.plane.xyz, clip.plane.xyz) > 0.0 {
        normal = normalize(clip.plane.xyz);
    }
    let params = array<vec4f, 3>(in.params0, in.params1, in.params2);
    let coverage = albedo.a;
    if coverage <= 0.0 {
        return vec4f(0.0);
    }
    let radiance = motolii_surface(SurfaceIn(albedo.rgb / coverage, normal, view_dir, in.world_position, in.thickness, params, in.texcoord, coverage));
    return vec4f(radiance * coverage, coverage);
}

@fragment
fn fs_main_picking_layer(in: VertexOut) -> @location(0) vec4u {
    if clip_outside(clip.plane, in.world_position) {
        discard;
    }
    return in.picking_layer_id;
}

@fragment
fn fs_main_outline_mask(in: VertexOut) -> @location(0) vec2u {
    return in.outline_mask_ids;
}
