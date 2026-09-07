#import <./types.wgsl>
#import <./global_bindings.wgsl>
#import <./mesh_vertex.wgsl>
#import <./utils/srgb.wgsl>
#import <./utils/lighting.wgsl>
#import <./utils/noise.wgsl>

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
    surface: vec4f, // roughness, metallic, transmission, ior
};

/// Field coordinate for a point in the instance frame: shifted, scaled to feature size, moved by evolution.
fn displace_coordinate(frame_position: vec3f, displace: vec4f, offset: vec4f) -> vec3f {
    let size = max(displace.y, 1e-3);
    return (frame_position + offset.xyz) / size + displace.z * vec3f(0.53, 0.71, 0.89);
}

/// Scalar field and its gradient (central differences) at a frame position.
fn displace_scalar(q: vec3f, octaves: u32) -> f32 {
    return fbm3(q, octaves);
}

fn displace_vector(q: vec3f, octaves: u32) -> vec3f {
    return vec3f(fbm3(q, octaves), fbm3(q + vec3f(31.7, 0.0, 0.0), octaves), fbm3(q + vec3f(0.0, 47.3, 0.0), octaves));
}

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

    let amount = in_instance.displace.x;
    if amount != 0.0 {
        let octaves = u32(clamp(in_instance.displace.w, 1.0, 8.0));
        let q = displace_coordinate(frame_position, in_instance.displace, in_instance.displace_offset);
        if in_instance.displace_offset.w < 0.5 && any(world_normal != vec3f(0.0)) {
            let n = normalize(world_normal);
            let h = displace_scalar(q, octaves);
            world_position += n * amount * h;
            // Bend the normal by the field's gradient (bump mapping): n' = n - amount * (∇h - n(n·∇h)) / size.
            let e = 0.01;
            let grad = vec3f(
                displace_scalar(q + vec3f(e, 0.0, 0.0), octaves) - displace_scalar(q - vec3f(e, 0.0, 0.0), octaves),
                displace_scalar(q + vec3f(0.0, e, 0.0), octaves) - displace_scalar(q - vec3f(0.0, e, 0.0), octaves),
                displace_scalar(q + vec3f(0.0, 0.0, e), octaves) - displace_scalar(q - vec3f(0.0, 0.0, e), octaves),
            ) / (2.0 * e * max(in_instance.displace.y, 1e-3));
            let tangent_grad = grad - n * dot(n, grad);
            world_normal = normalize(n - amount * tangent_grad);
        } else {
            world_position += amount * displace_vector(q, octaves);
        }
    }

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
    out.surface = in_instance.surface;

    return out;
}

@fragment
fn fs_main_shaded(in: VertexOut) -> @location(0) vec4f {
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
    if dot(normal, view_dir) < 0.0 {
        normal = -normal; // two-sided
    }
    let radiance = shade_surface(albedo.rgb, normal, view_dir, in.surface);
    return vec4f(radiance, albedo.a);
}

@fragment
fn fs_main_picking_layer(in: VertexOut) -> @location(0) vec4u {
    return in.picking_layer_id;
}

@fragment
fn fs_main_outline_mask(in: VertexOut) -> @location(0) vec2u {
    return in.outline_mask_ids;
}
