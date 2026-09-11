#import <./surface.wgsl>
#import <./colormap.wgsl>
#import <./rectangle.wgsl>
#import <./utils/srgb.wgsl>
#import <./utils/interpolation.wgsl>
#import <./utils/noise.wgsl>
#import <./utils/field.wgsl>

fn is_magnifying(pixel_coord: vec2f) -> bool {
    return fwidth(pixel_coord.x) < 1.0;
}

fn tex_filter(pixel_coord: vec2f) -> u32 {
    if is_magnifying(pixel_coord) {
        return rect_info.magnification_filter;
    } else {
        return rect_info.minification_filter;
    }
}

fn normalize_range(sampled_value: vec4f) -> vec4f {
    let range = rect_info.range_min_max;
    return (sampled_value - range.x) / (range.y - range.x);
}

fn decode_color(sampled_value: vec4f) -> vec4f {
    // Normalize the value first, otherwise premultiplying alpha and linear space conversion won't make sense.
    var rgba = normalize_range(sampled_value);

    // BGR(A) -> RGB(A)
    if rect_info.bgra_to_rgba != 0u {
        rgba = rgba.bgra;
    }

    // Convert to linear space
    if rect_info.decode_srgb != 0u {
        if all(vec3f(0.0) <= rgba.rgb) && all(rgba.rgb <= vec3f(1.0)) {
            rgba = linear_from_srgba_unmultiplied(rgba);
        } else {
            rgba = ERROR_RGBA; // out of range
        }
    }

    if rect_info.texture_alpha == TEXTURE_ALPHA_OPAQUE {
        rgba.a = 1.0; // ignore the alpha in the texture
    } else if rect_info.texture_alpha == TEXTURE_ALPHA_SEPARATE_ALPHA {
        // Premultiply alpha.
        rgba = vec4f(rgba.xyz * rgba.a, rgba.a);
    } else if rect_info.texture_alpha == TEXTURE_ALPHA_ALREADY_PREMULTIPLIED {
        // All good
    } else {
        rgba = ERROR_RGBA; // unknown enum
    }

    return rgba;
}

/// Takes a floating point texel coordinate and outputs a integer texel coordinate
/// on the neighrest neighbor, clamped to the texture edge.
fn clamp_to_edge_nearest_neighbor(coord: vec2f, texture_dimension: vec2f) -> vec2i {
    return vec2i(clamp(floor(coord), vec2f(0.0), texture_dimension - vec2f(1.0)));
}

/// Load a texel at the given integer coordinate, regardless of sample type.
fn load_texel(tc: vec2i) -> vec4f {
    if rect_info.sample_type == SAMPLE_TYPE_FLOAT {
        return textureLoad(texture_float, tc, 0);
    } else if rect_info.sample_type == SAMPLE_TYPE_SINT {
        return vec4f(textureLoad(texture_sint, tc, 0));
    } else if rect_info.sample_type == SAMPLE_TYPE_UINT {
        return vec4f(textureLoad(texture_uint, tc, 0));
    }
    return vec4f(0.0);
}

/// Load and decode a texel at the given floating-point coordinate (clamped to edge).
fn sample_and_decode(coord: vec2f, texture_dimensions: vec2f) -> vec4f {
    return decode_color(load_texel(clamp_to_edge_nearest_neighbor(coord, texture_dimensions)));
}

/// Apply bicubic filtering using Catmull-Rom spline interpolation on a 4x4 grid of decoded colors.
/// The grid is stored as a flat array of 16 elements in row-major order (index = row * 4 + col)
/// to avoid array-of-arrays, which is not supported in WebGL/GLSL ES.
fn filter_bicubic(colors: array<vec4f, 16>, wx: vec4f, wy: vec4f) -> vec4f {
    var result = vec4f(0.0);
    for (var row = 0u; row < 4u; row++) {
        var row_color = vec4f(0.0);
        for (var col = 0u; col < 4u; col++) {
            row_color += colors[row * 4u + col] * wx[col];
        }
        result += row_color * wy[row];
    }
    return result;
}

struct FieldSample {
    texcoord: vec2f,
    normal: vec3f,
};

/// The field moves the picture's surface; a flat picture shows that as a shift of where it is sampled.
/// In-plane offset shifts the sample directly; an offset along the normal is seen through the view
/// direction (parallax mapping, Kaneko et al. 2001), so a bump leans away with the camera.
/// The frame is the rectangle's own: top-left corner as origin, extent_u/extent_v as axes.
fn sample_field(in: VertexOut) -> FieldSample {
    let eu = rect_info.extent_u;
    let ev = rect_info.extent_v;
    let frame_position = in.texcoord.x * eu + in.texcoord.y * ev;
    let cross_normal = cross(eu, ev);
    let n = cross_normal / max(length(cross_normal), 1e-6);
    let field = motolii_field(FieldIn(frame_position, n, rect_info.surface_params));
    let o = field.offset;
    let v = view_direction_to_camera(in.world_position);
    let uu = max(dot(eu, eu), 1e-12);
    let vv = max(dot(ev, ev), 1e-12);
    let v_n = dot(v, n);
    // Limit the parallax at grazing angles, as parallax mapping customarily does.
    let lean = vec2f(dot(v, eu) / uu, dot(v, ev) / vv) / select(v_n, sign(v_n) * 0.1, abs(v_n) < 0.1);
    let shift = vec2f(dot(o, eu) / uu, dot(o, ev) / vv) - dot(o, n) * lean;
    return FieldSample(in.texcoord - shift, field.normal);
}

fn texture_size() -> vec2f {
    if rect_info.sample_type == SAMPLE_TYPE_FLOAT {
        return vec2f(textureDimensions(texture_float).xy);
    } else if rect_info.sample_type == SAMPLE_TYPE_SINT {
        return vec2f(textureDimensions(texture_sint).xy);
    } else if rect_info.sample_type == SAMPLE_TYPE_UINT {
        return vec2f(textureDimensions(texture_uint).xy);
    }
    return vec2f(0.0);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    if FILTER_SURFACE_FOOTPRINT {
        let cross_normal = cross(rect_info.extent_u,rect_info.extent_v);
        var n = cross_normal / max(length(cross_normal),1e-6);
        prepare_surface_footprint(in.world_position,n);
    }
    if clip_outside(rect_info.clip_plane, in.world_position) {
        discard;
    }
    // Sample the main texture where the field says the surface is:
    var normalized_value: vec4f;
    let texture_dimensions = texture_size();
    let sampled = sample_field(in);
    let coord = sampled.texcoord * texture_dimensions;
    let active_filter = tex_filter(coord);

    switch active_filter {
        case FILTER_NEAREST {
            normalized_value = sample_and_decode(coord, texture_dimensions);
        }
        case FILTER_BILINEAR {
            // Bilinear filtering: weighted sum of 2x2 decoded texels.
            let center = coord - vec2f(0.5);
            let base = floor(center);
            let f = center - base;

            let c00 = sample_and_decode(base, texture_dimensions);
            let c10 = sample_and_decode(base + vec2f(1.0, 0.0), texture_dimensions);
            let c01 = sample_and_decode(base + vec2f(0.0, 1.0), texture_dimensions);
            let c11 = sample_and_decode(base + vec2f(1.0, 1.0), texture_dimensions);

            let top = mix(c00, c10, f.x);
            let bottom = mix(c01, c11, f.x);
            normalized_value = mix(top, bottom, f.y);
        }
        case FILTER_BICUBIC {
            // Bicubic (Catmull-Rom) filtering: weighted sum of 4x4 decoded texels.
            let center = coord - vec2f(0.5);
            let base = floor(center);
            let f = center - base;
            let wx = catmull_rom_weights(f.x);
            let wy = catmull_rom_weights(f.y);

            var colors: array<vec4f, 16>;
            for (var row = 0u; row < 4u; row++) {
                for (var col = 0u; col < 4u; col++) {
                    colors[row * 4u + col] = sample_and_decode(
                        base + vec2f(f32(col), f32(row)) - vec2f(0.5),
                        texture_dimensions
                    );
                }
            }
            normalized_value = filter_bicubic(colors, wx, wy);

            // Catmull-Rom filtering can overshoot due to negative weights!
            normalized_value = clamp(normalized_value, vec4f(0.0), vec4f(1.0));
        }
        default {
            normalized_value = ERROR_RGBA;
        }
    }

    // Apply gamma:
    normalized_value = vec4f(pow(normalized_value.rgb, vec3f(rect_info.gamma)), normalized_value.a);

    // Apply colormap, if any:
    var texture_color: vec4f;
    if rect_info.color_mapper == COLOR_MAPPER_OFF_GRAYSCALE {
        texture_color = vec4f(normalized_value.rrr, 1.0);
    } else if rect_info.color_mapper == COLOR_MAPPER_OFF_RGB {
        texture_color = normalized_value;
    } else if rect_info.color_mapper == COLOR_MAPPER_FUNCTION {
        texture_color = colormap_linear(rect_info.colormap_function, normalized_value.r);
    } else if rect_info.color_mapper == COLOR_MAPPER_TEXTURE {
        let colormap_size = textureDimensions(colormap_texture).xy;
        let num_colors = colormap_size.x * colormap_size.y;
        let max_index = num_colors - 1u;
        let color_index = normalized_value.r * f32(max_index);
        // TODO(emilk): interpolate between neighboring colors for non-integral color indices
        // It's important to round here since otherwise numerical instability can push us to the adjacent class-id
        // See: https://github.com/rerun-io/rerun/issues/1968
        let color_index_u32 = u32(round(color_index));
        let x = color_index_u32 % colormap_size.x;
        let y = color_index_u32 / colormap_size.x;
        texture_color = textureLoad(colormap_texture, vec2u(x, y), 0);
    } else {
        return ERROR_RGBA; // unknown color mapper
    }

    // Surface radiance is unpremultiplied; coverage belongs to the source image.
    let coverage = texture_color.a;
    if coverage <= 0.0 {
        return vec4f(0.0);
    }
    let view_dir = view_direction_to_camera(in.world_position);
    var normal = normalize(sampled.normal);
    if dot(normal, view_dir) < 0.0 {
        normal = -normal;
    }
    let surface = SurfaceIn(texture_color.rgb / coverage, normal, view_dir,
        in.world_position, rect_info.surface_thickness, rect_info.surface_params, sampled.texcoord, coverage);
    return vec4f(motolii_surface(surface) * coverage, coverage) * rect_info.multiplicative_tint;
}

@fragment
fn fs_main_picking_layer(in: VertexOut) -> @location(0) vec4u {
    return vec4u(0u, 0u, 0u, 0u); // TODO(andreas): Implement picking layer id pass-through.
}

@fragment
fn fs_main_outline_mask(in: VertexOut) -> @location(0) vec2u {
    // The outline follows the picture, not the quad: transparent texels leave no mask,
    // so a glyph or a cut-out gets its own silhouette instead of the rectangle's.
    if rect_info.texture_alpha != TEXTURE_ALPHA_OPAQUE {
        let texture_dimensions = texture_size();
        let coverage = sample_and_decode(sample_field(in).texcoord * texture_dimensions, texture_dimensions).a;
        if coverage < 0.5 {
            discard;
        }
    }
    return rect_info.outline_mask;
}
