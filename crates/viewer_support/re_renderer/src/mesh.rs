use re_span::Span;
use std::mem::size_of;

use ecolor::Rgba;
use smallvec::{SmallVec, smallvec};

use crate::allocator::create_and_fill_uniform_buffer_batch;
use crate::label::Label;
use crate::renderer::MeshRenderer;
use crate::resource_managers::GpuTexture2D;
use crate::wgpu_resources::{BindGroupDesc, BindGroupEntry, BufferDesc, GpuBindGroup, GpuBuffer};
use crate::{RenderContext, Rgba32Unmul};

/// Defines how mesh vertices are built.
pub mod mesh_vertices {
    use crate::wgpu_resources::VertexBufferLayout;

    /// Vertex buffer layouts describing how vertex data should be laid out.
    ///
    /// Needs to be kept in sync with `mesh_vertex.wgsl`.
    pub fn vertex_buffer_layouts() -> smallvec::SmallVec<[VertexBufferLayout; 4]> {
        // TODO(andreas): Compress normals. Afaik Octahedral Mapping is the best by far, see https://jcgt.org/published/0003/02/01/
        VertexBufferLayout::from_formats(
            [
                wgpu::VertexFormat::Float32x3, // position
                wgpu::VertexFormat::Unorm8x4,  // RGBA
                wgpu::VertexFormat::Float32x3, // normal
                wgpu::VertexFormat::Float32x2, // texcoord
            ]
            .into_iter(),
        )
    }

    /// Next vertex attribute index that can be used for another vertex buffer.
    pub fn next_free_shader_location() -> u32 {
        vertex_buffer_layouts()
            .iter()
            .flat_map(|layout| layout.attributes.iter())
            .max_by(|a1, a2| a1.shader_location.cmp(&a2.shader_location))
            .unwrap()
            .shader_location
            + 1
    }
}

#[derive(Clone)]
pub struct CpuMesh {
    pub label: Label,

    /// Non-empty array of vertex triangle indices.
    ///
    /// The length has to be a multiple of 3.
    pub triangle_indices: Vec<glam::UVec3>,

    /// Non-empty array of vertex positions.
    pub vertex_positions: Vec<glam::Vec3>,

    /// Per-vertex albedo color.
    /// Must be equal in length to [`Self::vertex_positions`].
    pub vertex_colors: Vec<Rgba32Unmul>,

    /// Must be equal in length to [`Self::vertex_positions`].
    /// Use ZERO for unshaded.
    pub vertex_normals: Vec<glam::Vec3>,

    /// Must be equal in length to [`Self::vertex_positions`].
    pub vertex_texcoords: Vec<glam::Vec2>,

    pub materials: SmallVec<[Material; 1]>,

    /// Object space bounding box.
    pub bbox: macaw::BoundingBox,
}

impl CpuMesh {
    #[track_caller]
    pub fn sanity_check(&self) -> Result<(), MeshError> {
        re_tracing::profile_function!();

        let Self {
            label: _,
            triangle_indices,
            vertex_positions,
            vertex_colors,
            vertex_normals,
            vertex_texcoords,
            materials: _,
            bbox,
        } = self;

        let num_pos = vertex_positions.len();
        let num_color = vertex_colors.len();
        let num_normals = vertex_normals.len();
        let num_texcoords = vertex_texcoords.len();

        if num_pos != num_color {
            return Err(MeshError::WrongNumberOfColors { num_pos, num_color });
        }
        if num_pos != num_normals {
            return Err(MeshError::WrongNumberOfNormals {
                num_pos,
                num_normals,
            });
        }
        if num_pos != num_texcoords {
            return Err(MeshError::WrongNumberOfTexcoord {
                num_pos,
                num_texcoords,
            });
        }
        if self.vertex_positions.is_empty() {
            return Err(MeshError::ZeroVertices);
        }

        if self.triangle_indices.is_empty() {
            return Err(MeshError::ZeroIndices);
        }

        if bbox.is_nan() || !bbox.is_finite() || bbox.is_nothing() {
            return Err(MeshError::InvalidBbox(*bbox));
        }

        for indices in triangle_indices {
            let max_index = indices.max_element();
            if num_pos <= max_index as usize {
                return Err(MeshError::IndexOutOfBounds {
                    num_pos,
                    index: max_index,
                });
            }
        }

        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum MeshError {
    #[error(
        "Number of vertex positions {num_pos} differed from the number of vertex colors {num_color}"
    )]
    WrongNumberOfColors { num_pos: usize, num_color: usize },

    #[error(
        "Number of vertex positions {num_pos} differed from the number of vertex normals {num_normals}"
    )]
    WrongNumberOfNormals { num_pos: usize, num_normals: usize },

    #[error(
        "Number of vertex positions {num_pos} differed from the number of vertex tex-coords {num_texcoords}"
    )]
    WrongNumberOfTexcoord {
        num_pos: usize,
        num_texcoords: usize,
    },

    #[error("Mesh has no vertices.")]
    ZeroVertices,

    #[error("Mesh has no triangle indices.")]
    ZeroIndices,

    #[error("Mesh has an invalid bounding box {0:?}")]
    InvalidBbox(macaw::BoundingBox),

    #[error("Index {index} was out of bounds for {num_pos} vertex positions")]
    IndexOutOfBounds { num_pos: usize, index: u32 },

    #[error(transparent)]
    CpuWriteGpuReadError(#[from] crate::allocator::CpuWriteGpuReadError),

    #[error(transparent)]
    Renderer(#[from] crate::RendererRegistrationError),
}

const _: () = assert!(
    std::mem::size_of::<MeshError>() <= 64,
    "Error type is too large. Try to reduce its size by boxing some of its variants.",
);

#[derive(Clone)]
pub struct Material {
    pub label: Label,

    /// Index range within the owning [`CpuMesh`] that should be rendered with this material.
    pub index_range: Span<u32>,

    /// Base color texture, also known as albedo.
    /// (not optional, needs to be at least a 1pix texture with a color!)
    pub albedo: GpuTexture2D,

    /// Factor applied to the decoded albedo color.
    pub albedo_factor: Rgba,
    /// The sampled texture contains linear premultiplied RGBA, including coverage.
    pub albedo_is_premultiplied: bool,
    /// The premultiplied texture's coverage is the surface's silhouette (a cutout): what it covers
    /// is drawn opaque — un-premultiplied colour, full coverage, depth-tested.
    pub albedo_is_cutout: bool,
    /// The vertex field (`program_field`) is evaluated at the vertex's texcoord (x, y, 0) instead of
    /// its position, e.g. a stroke whose texcoord is its centreline point keeps its width under the
    /// field.
    pub field_at_texcoord: bool,
    /// The texcoords that span the surface's picture, as `(origin, size)`, when they are in some
    /// other unit (a path's texcoords are its points): a surface program's `SurfaceIn::uv` is
    /// `(texcoord - origin) / size`, 0..1 across the picture. `None`: the texcoords are the uv.
    pub texcoord_frame: Option<(glam::Vec2, glam::Vec2)>,
    /// Coverage comes from these curves, evaluated per fragment, instead of the triangles' edges:
    /// the triangles only need to cover the curves' bounds. Exact at any magnification.
    pub curves: Option<std::sync::Arc<CurveFill>>,
}

/// Quadratic Bézier outlines (p0, p1, p2 in mesh units) whose winding decides coverage.
#[derive(Clone, Debug, PartialEq)]
pub struct CurveFill {
    pub curves: Vec<[glam::Vec2; 3]>,
    pub even_odd: bool,
    /// Paint varying over the fill; `None` paints the vertex colour. Shared by the pieces of one fill.
    pub gradient: Option<std::sync::Arc<CurveGradient>>,
}

/// A [`CurveFill`] as the fragment shader reads it: the curves bucketed into bands along each axis
/// so a fragment only visits the curves that can cross its ray.
///
/// A curve contributes to the ray along +x only where its control points straddle the ray's y
/// (`curve_ray_coverage` returns exactly 0 otherwise), so a fragment at `p` only needs the curves
/// whose y range contains `p.y`; the same along x for the ray along +y. The fill's bounds are cut
/// into `rows` bands of y and `columns` bands of x; a band lists, in the fill's curve order, every
/// curve whose range touches it. The sum over a band's curves is the sum over all curves with the
/// zero terms left out, so the coverage is the same to the bit.
///
/// Texel layout (`Rgba32Float`): `rows + columns` band headers `(first texel, curve count, 0, 0)`,
/// rows first, then each band's curves as two texels each, `(p0, p1)`, `(p2, 0)`. Column bands hold
/// their curves with x and y swapped, so the shader runs the one ray routine on `p.yx`.
pub(crate) struct CurveBands {
    texels: Vec<[f32; 4]>,
    lo: glam::Vec2,
    hi: glam::Vec2,
    /// Bands per unit along each axis; 0 where the bounds are flat.
    bands_per_unit: glam::Vec2,
    rows: u32,
    columns: u32,
}

impl CurveBands {
    /// The band table for `curves`; empty bounds (no curves) make every fragment skip.
    fn build(curves: &[[glam::Vec2; 3]]) -> Self {
        // Fragments outside the band's own span can still land in it through rounding when the
        // shader turns `(p - lo) * bands_per_unit` into a band index. That error is under
        // `bands * 2^-22` bands; overlap the bands by well more than that.
        const OVERLAP: f64 = 1e-4;
        // Two bands per curve: measured flat from there (30 ellipses of ~40 curves at 512²:
        // n/2 bands 3.8 ns/px, n 3.5, 2n 3.2, 4n 3.2); memory grows with it, so it is capped.
        const MAX_BANDS: usize = 128;

        let (lo, hi) = curves.iter().flatten().fold(
            (glam::Vec2::splat(f32::MAX), glam::Vec2::splat(f32::MIN)),
            |(lo, hi), p| (lo.min(*p), hi.max(*p)),
        );
        if curves.is_empty() {
            return Self {
                texels: Vec::new(),
                lo: glam::Vec2::ZERO,
                hi: glam::Vec2::ZERO,
                bands_per_unit: glam::Vec2::ZERO,
                rows: 0,
                columns: 0,
            };
        }
        let extent = (hi - lo).as_dvec2();
        let bands = (2 * curves.len()).clamp(1, MAX_BANDS);
        let bands_along = |axis: usize| if extent[axis] > 0.0 { bands } else { 1 };
        let (rows, columns) = (bands_along(1), bands_along(0));
        let bands_per_unit = glam::dvec2(
            if extent.x > 0.0 {
                columns as f64 / extent.x
            } else {
                0.0
            },
            if extent.y > 0.0 {
                rows as f64 / extent.y
            } else {
                0.0
            },
        );

        // (min, max) of each curve's control points along `axis`.
        let spans = |axis: usize| -> Vec<(f64, f64)> {
            curves
                .iter()
                .map(|c| {
                    let v = [c[0][axis] as f64, c[1][axis] as f64, c[2][axis] as f64];
                    (v[0].min(v[1]).min(v[2]), v[0].max(v[1]).max(v[2]))
                })
                .collect()
        };
        let members = |axis: usize, count: usize| -> Vec<Vec<usize>> {
            let spans = spans(axis);
            let step = if count > 1 {
                extent[axis] / count as f64
            } else {
                extent[axis]
            };
            let slack = step * OVERLAP;
            (0..count)
                .map(|band| {
                    let start = lo[axis] as f64 + band as f64 * step - slack;
                    let end = lo[axis] as f64 + (band + 1) as f64 * step + slack;
                    spans
                        .iter()
                        .enumerate()
                        .filter(|(_, (min, max))| *min <= end && *max >= start)
                        .map(|(i, _)| i)
                        .collect()
                })
                .collect()
        };
        let row_members = members(1, rows);
        let column_members = members(0, columns);

        let mut texels = Vec::with_capacity(
            rows + columns
                + 2 * (row_members
                    .iter()
                    .chain(&column_members)
                    .map(Vec::len)
                    .sum::<usize>()),
        );
        texels.resize(rows + columns, [0.0; 4]);
        for (band, (list, swap)) in row_members
            .iter()
            .map(|m| (m, false))
            .chain(column_members.iter().map(|m| (m, true)))
            .enumerate()
        {
            texels[band] = [texels.len() as f32, list.len() as f32, 0.0, 0.0];
            for &i in list {
                let [a, b, c] = curves[i];
                let (a, b, c) = if swap {
                    (
                        glam::vec2(a.y, a.x),
                        glam::vec2(b.y, b.x),
                        glam::vec2(c.y, c.x),
                    )
                } else {
                    (a, b, c)
                };
                texels.push([a.x, a.y, b.x, b.y]);
                texels.push([c.x, c.y, 0.0, 0.0]);
            }
        }
        Self {
            texels,
            lo,
            hi,
            bands_per_unit: bands_per_unit.as_vec2(),
            rows: rows as u32,
            columns: columns as u32,
        }
    }
}

/// Where along a gradient a point lies (`t` in 0..=1), evaluated per fragment; the colour at `t`
/// comes from `ramp`, which the embedder samples from its own colour model.
#[derive(Clone, Debug, PartialEq)]
pub struct CurveGradient {
    pub kind: CurveGradientKind,
    /// `start`, `end` are in the gradient's own space: `mesh = space_origin + gradient * space_scale`
    /// (an object-bounding-box gradient stretches with its object).
    pub space_origin: glam::Vec2,
    pub space_scale: glam::Vec2,
    pub start: glam::Vec2,
    pub end: glam::Vec2,
    /// Straight sRGB with alpha, evenly spaced over t = 0..=1.
    pub ramp: Vec<crate::Rgba32Unmul>,
}

/// Keep in sync with `GRADIENT_` in `instanced_mesh_base.wgsl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurveGradientKind {
    Linear = 1,
    Radial = 2,
    Angular = 3,
    Diamond = 4,
}

#[derive(Clone)]
pub struct GpuMesh {
    // It would be desirable to put both vertex and index buffer into the same buffer, BUT
    // WebGL doesn't allow us to do so! (see https://github.com/gfx-rs/wgpu/pull/3157)
    pub index_buffer: GpuBuffer,

    /// Buffer for all vertex data, subdivided in several sections for different vertex buffer bindings.
    /// See [`mesh_vertices`]
    pub vertex_buffer_combined: GpuBuffer,
    pub vertex_buffer_positions_range: Span<u64>,
    pub vertex_buffer_colors_range: Span<u64>,
    pub vertex_buffer_normals_range: Span<u64>,
    pub vertex_buffer_texcoord_range: Span<u64>,

    pub index_buffer_range: Span<u64>,

    /// Every mesh has at least one material.
    pub materials: SmallVec<[GpuMaterial; 1]>,

    /// Object space bounding box.
    ///
    /// Needed for distance sorting.
    pub bbox: macaw::BoundingBox,
}

impl GpuMesh {
    /// Returns the byte size this `GpuMesh` uses in total.
    pub fn gpu_byte_size(&self) -> u64 {
        self.index_buffer.inner.size() + self.vertex_buffer_combined.size()
    }
}

#[derive(Clone)]
pub struct GpuMaterial {
    /// Index range within the owning [`CpuMesh`] that should be rendered with this material.
    pub index_range: Span<u32>,

    pub bind_group: GpuBindGroup,

    /// Whether there's any transparency in this material.
    pub has_transparency: bool,
}

pub(crate) mod gpu_data {
    use crate::wgpu_buffer_types;

    /// Internally supported texture formats for our textures.
    ///
    /// Keep in sync with the `FORMAT_` constants in `instanced_mesh.wgsl`
    #[repr(u32)]
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum TextureFormat {
        Rgba = 0,
        Grayscale = 1,
        PremultipliedRgba = 2,
        Curves = 3,
        OpaquePremultipliedRgba = 4,
    }

    /// Keep in sync with [`MaterialUniformBuffer`] in `instanced_mesh.wgsl`
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct MaterialUniformBuffer {
        albedo_factor: ecolor::Rgba,
        texture_format: wgpu_buffer_types::U32RowPadded,
        field_at_texcoord: wgpu_buffer_types::U32RowPadded,
        curve_count: wgpu_buffer_types::U32RowPadded,
        even_odd: wgpu_buffer_types::U32RowPadded,
        gradient_kind: wgpu_buffer_types::U32RowPadded,
        gradient_line: wgpu_buffer_types::Vec4,
        gradient_space: wgpu_buffer_types::Vec4,
        uv_frame: wgpu_buffer_types::Vec4,
        /// The curves' bounds: `lo.xy`, `hi.xy`.
        curve_bounds: wgpu_buffer_types::Vec4,
        /// Bands per unit along x and y, in `xy`.
        curve_band_scale: wgpu_buffer_types::Vec4,
        /// Row bands (along y) and column bands (along x), see `CurveBands`.
        curve_bands: wgpu_buffer_types::UVec2RowPadded,
        end_padding: [wgpu_buffer_types::PaddingRow; 16 - 12],
    }

    impl MaterialUniformBuffer {
        pub fn new(
            albedo_factor: ecolor::Rgba,
            texture_format: TextureFormat,
            field_at_texcoord: bool,
            texcoord_frame: Option<(glam::Vec2, glam::Vec2)>,
            curves: Option<(&super::CurveFill, &super::CurveBands)>,
        ) -> Self {
            let gradient = curves.and_then(|(fill, _)| fill.gradient.as_ref());
            let bands = curves.map(|(_, bands)| bands);
            Self {
                albedo_factor,
                texture_format: (texture_format as u32).into(),
                field_at_texcoord: u32::from(field_at_texcoord).into(),
                curve_count: curves
                    .map_or(0, |(fill, _)| fill.curves.len() as u32)
                    .into(),
                even_odd: u32::from(curves.is_some_and(|(fill, _)| fill.even_odd)).into(),
                gradient_kind: gradient.map_or(0, |g| g.kind as u32).into(),
                gradient_line: gradient
                    .map_or(glam::Vec4::ZERO, |g| {
                        glam::vec4(g.start.x, g.start.y, g.end.x, g.end.y)
                    })
                    .into(),
                gradient_space: gradient
                    .map_or(glam::vec4(0.0, 0.0, 1.0, 1.0), |g| {
                        glam::vec4(
                            g.space_origin.x,
                            g.space_origin.y,
                            g.space_scale.x,
                            g.space_scale.y,
                        )
                    })
                    .into(),
                uv_frame: texcoord_frame
                    .map_or(glam::vec4(0.0, 0.0, 1.0, 1.0), |(origin, size)| {
                        glam::vec4(origin.x, origin.y, size.x, size.y)
                    })
                    .into(),
                curve_bounds: bands
                    .map_or(glam::Vec4::ZERO, |b| {
                        glam::vec4(b.lo.x, b.lo.y, b.hi.x, b.hi.y)
                    })
                    .into(),
                curve_band_scale: bands
                    .map_or(glam::Vec4::ZERO, |b| {
                        b.bands_per_unit.extend(0.0).extend(0.0)
                    })
                    .into(),
                curve_bands: bands
                    .map_or(glam::UVec2::ZERO, |b| glam::uvec2(b.rows, b.columns))
                    .into(),
                end_padding: Default::default(),
            }
        }
    }
}

impl GpuMesh {
    // TODO(andreas): Take read-only context here and make uploads happen on staging belt.
    pub fn new(ctx: &RenderContext, data: &CpuMesh) -> Result<Self, MeshError> {
        re_tracing::profile_function!();

        data.sanity_check()?;

        re_log::trace!(
            "uploading new mesh named {:?} with {} vertices and {} triangles",
            data.label.get(),
            data.vertex_positions.len(),
            data.triangle_indices.len(),
        );

        // TODO(andreas): Have a variant that gets this from a stack allocator.
        let vb_positions_size = (data.vertex_positions.len() * size_of::<glam::Vec3>()) as u64;
        let vb_color_size = (data.vertex_colors.len() * size_of::<Rgba32Unmul>()) as u64;
        let vb_normals_size = (data.vertex_normals.len() * size_of::<glam::Vec3>()) as u64;
        let vb_texcoords_size = (data.vertex_texcoords.len() * size_of::<glam::Vec2>()) as u64;

        let vb_combined_size =
            vb_positions_size + vb_color_size + vb_normals_size + vb_texcoords_size;

        let pools = &ctx.gpu_resources;
        let device = &ctx.device;

        let vertex_buffer_combined = {
            let vertex_buffer_combined = pools.buffers.alloc(
                device,
                &BufferDesc {
                    label: format!("{} - vertices", data.label).into(),
                    size: vb_combined_size,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            );

            let mut staging_buffer = ctx.cpu_write_gpu_read_belt.lock().allocate::<u8>(
                &ctx.device,
                &ctx.gpu_resources.buffers,
                vb_combined_size as _,
            )?;
            staging_buffer.extend_from_slice(bytemuck::cast_slice(&data.vertex_positions))?;
            staging_buffer.extend_from_slice(bytemuck::cast_slice(&data.vertex_colors))?;
            staging_buffer.extend_from_slice(bytemuck::cast_slice(&data.vertex_normals))?;
            staging_buffer.extend_from_slice(bytemuck::cast_slice(&data.vertex_texcoords))?;
            staging_buffer.copy_to_buffer(
                ctx.active_frame.before_view_builder_encoder.lock().get(),
                &vertex_buffer_combined,
                0,
            )?;
            vertex_buffer_combined
        };

        let index_buffer_size = (size_of::<glam::UVec3>() * data.triangle_indices.len()) as u64;
        let index_buffer = {
            let index_buffer = pools.buffers.alloc(
                device,
                &BufferDesc {
                    label: format!("{} - indices", data.label).into(),
                    size: index_buffer_size,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            );

            let mut staging_buffer = ctx.cpu_write_gpu_read_belt.lock().allocate::<glam::UVec3>(
                &ctx.device,
                &ctx.gpu_resources.buffers,
                data.triangle_indices.len(),
            )?;
            staging_buffer.extend_from_slice(bytemuck::cast_slice(&data.triangle_indices))?;
            staging_buffer.copy_to_buffer(
                ctx.active_frame.before_view_builder_encoder.lock().get(),
                &index_buffer,
                0,
            )?;
            index_buffer
        };

        let materials = {
            let curve_bands: Vec<Option<CurveBands>> = data
                .materials
                .iter()
                .map(|material| {
                    material
                        .curves
                        .as_deref()
                        .map(|fill| CurveBands::build(&fill.curves))
                })
                .collect();
            let uniform_buffer_bindings = create_and_fill_uniform_buffer_batch(
                ctx,
                format!("{} - material uniforms", data.label).into(),
                std::iter::zip(&data.materials, &curve_bands).map(|(material, bands)| {
                    gpu_data::MaterialUniformBuffer::new(
                        material.albedo_factor,
                        if material.curves.is_some() {
                            gpu_data::TextureFormat::Curves
                        } else if material.albedo_is_premultiplied && material.albedo_is_cutout {
                            gpu_data::TextureFormat::OpaquePremultipliedRgba
                        } else if material.albedo_is_premultiplied {
                            gpu_data::TextureFormat::PremultipliedRgba
                        } else if material.albedo.texture.format().components() == 1 {
                            gpu_data::TextureFormat::Grayscale
                        } else {
                            gpu_data::TextureFormat::Rgba
                        },
                        material.field_at_texcoord,
                        material.texcoord_frame,
                        material.curves.as_deref().zip(bands.as_ref()),
                    )
                }),
            );

            let mut materials = SmallVec::with_capacity(data.materials.len());

            // The bind group layout must be in sync with the mesh renderer.
            let mesh_bind_group_layout = ctx.renderer::<MeshRenderer>()?.bind_group_layout;

            for ((material, uniform_buffer_binding), bands) in std::iter::zip(
                std::iter::zip(&data.materials, uniform_buffer_bindings),
                &curve_bands,
            ) {
                // A data texture (see `CurveBands`), as other renderers keep per-element data.
                let curves = match bands {
                    Some(bands) if !bands.texels.is_empty() => {
                        let mut source = crate::DataTextureSource::<[f32; 4]>::new(ctx);
                        if let Err(err) = source.extend_from_slice(&bands.texels) {
                            re_log::error_once!(
                                "Failed to write the curves of {}: {err}",
                                material.label
                            );
                        }
                        source
                            .finish(
                                wgpu::TextureFormat::Rgba32Float,
                                format!("{} - curves", material.label),
                            )?
                            .handle
                    }
                    _ => ctx.texture_manager_2d.zeroed_texture_float().handle,
                };
                let bind_group = pools.bind_groups.alloc(
                    device,
                    pools,
                    &BindGroupDesc {
                        label: material.label.clone(),
                        entries: smallvec![
                            BindGroupEntry::DefaultTextureView(material.albedo.handle()),
                            uniform_buffer_binding,
                            BindGroupEntry::DefaultTextureView(curves),
                        ],
                        layout: mesh_bind_group_layout,
                    },
                );

                // TODO(#12223): handle texture transparency
                let is_transparent = material.curves.is_some()
                    || (material.albedo_is_premultiplied && !material.albedo_is_cutout)
                    || material.albedo_factor.a() < 1.0
                    || data.vertex_colors.iter().any(|color| color.0[3] < 255);

                materials.push(GpuMaterial {
                    index_range: material.index_range,
                    bind_group,
                    has_transparency: is_transparent,
                });
            }
            materials
        };

        let vb_colors_start = vb_positions_size;
        let vb_normals_start = vb_colors_start + vb_color_size;
        let vb_texcoord_start = vb_normals_start + vb_normals_size;

        Ok(Self {
            index_buffer,
            vertex_buffer_combined,
            vertex_buffer_positions_range: Span::from_start_end(0, vb_positions_size),
            vertex_buffer_colors_range: Span::from_start_end(vb_colors_start, vb_normals_start),
            vertex_buffer_normals_range: Span::from_start_end(vb_normals_start, vb_texcoord_start),
            vertex_buffer_texcoord_range: Span::from_start_end(vb_texcoord_start, vb_combined_size),
            index_buffer_range: Span::from_start_end(0, index_buffer_size),
            materials,
            bbox: data.bbox,
        })
    }
}

#[cfg(test)]
mod curve_band_tests {
    use super::CurveBands;

    fn curve(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> [glam::Vec2; 3] {
        [
            glam::vec2(a.0, a.1),
            glam::vec2(b.0, b.1),
            glam::vec2(c.0, c.1),
        ]
    }

    /// The curves of band `index`, unswizzled, as (curve, is_column_band).
    fn band(bands: &CurveBands, index: usize) -> Vec<[glam::Vec2; 3]> {
        let header = bands.texels[index];
        let (first, count) = (header[0] as usize, header[1] as usize);
        (0..count)
            .map(|i| {
                let a = bands.texels[first + 2 * i];
                let b = bands.texels[first + 2 * i + 1];
                [
                    glam::vec2(a[0], a[1]),
                    glam::vec2(a[2], a[3]),
                    glam::vec2(b[0], b[1]),
                ]
            })
            .collect()
    }

    /// Every curve whose y range holds a sample of the band is in that band, in the fill's order,
    /// and column bands carry the same curves with x and y swapped.
    #[test]
    fn bands_hold_every_curve_that_can_cross_a_ray_from_them() {
        // 12 curves around an ellipse-ish outline: two bands per curve along each axis.
        let n = 12;
        let curves: Vec<[glam::Vec2; 3]> = (0..n)
            .map(|i| {
                let at = |k: usize| {
                    let t = k as f32 / n as f32 * std::f32::consts::TAU;
                    glam::vec2(100.0 + 80.0 * t.cos(), 50.0 + 30.0 * t.sin())
                };
                let (a, c) = (at(i), at(i + 1));
                [a, (a + c) * 0.5 + glam::vec2(1.0, -1.0), c]
            })
            .collect();
        let bands = CurveBands::build(&curves);
        assert_eq!((bands.rows, bands.columns), (24, 24));
        assert_eq!(bands.lo, glam::vec2(20.0, 20.0));
        assert!(bands.hi.x > 179.0 && bands.hi.y > 79.0);

        let (rows, columns) = (bands.rows as usize, bands.columns as usize);
        let step = (bands.hi - bands.lo) / glam::vec2(columns as f32, rows as f32);
        for row in 0..rows {
            let listed = band(&bands, row);
            let indices: Vec<usize> = listed
                .iter()
                .map(|c| curves.iter().position(|d| d == c).expect("a fill curve"))
                .collect();
            assert!(
                indices.windows(2).all(|w| w[0] < w[1]),
                "fill order kept: {indices:?}"
            );
            for sample in 0..50 {
                let y = bands.lo.y + step.y * (row as f32 + sample as f32 / 49.0);
                for (i, c) in curves.iter().enumerate() {
                    let (min, max) = (
                        c[0].y.min(c[1].y).min(c[2].y),
                        c[0].y.max(c[1].y).max(c[2].y),
                    );
                    if min <= y && y < max {
                        assert!(indices.contains(&i), "curve {i} spans y = {y} of row {row}");
                    }
                }
            }
        }
        for column in 0..columns {
            let listed = band(&bands, rows + column);
            for c in &listed {
                let unswapped = [
                    glam::vec2(c[0].y, c[0].x),
                    glam::vec2(c[1].y, c[1].x),
                    glam::vec2(c[2].y, c[2].x),
                ];
                assert!(
                    curves.contains(&unswapped),
                    "column bands are swizzled fill curves"
                );
            }
            for sample in 0..50 {
                let x = bands.lo.x + step.x * (column as f32 + sample as f32 / 49.0);
                for c in &curves {
                    let (min, max) = (
                        c[0].x.min(c[1].x).min(c[2].x),
                        c[0].x.max(c[1].x).max(c[2].x),
                    );
                    if min <= x && x < max {
                        let swapped = [
                            glam::vec2(c[0].y, c[0].x),
                            glam::vec2(c[1].y, c[1].x),
                            glam::vec2(c[2].y, c[2].x),
                        ];
                        assert!(listed.contains(&swapped), "column {column} at x = {x}");
                    }
                }
            }
        }
        // Bands are a real cut, not everything everywhere.
        assert!((0..rows + columns).all(|b| band(&bands, b).len() < n));
    }

    /// Flat bounds along an axis make one band there; no curves make no bands and empty bounds.
    #[test]
    fn degenerate_fills_stay_well_formed() {
        let flat = CurveBands::build(&[
            curve((0.0, 5.0), (5.0, 5.0), (10.0, 5.0)),
            curve((10.0, 5.0), (5.0, 5.0), (0.0, 5.0)),
        ]);
        assert_eq!((flat.rows, flat.columns), (1, 4));
        assert_eq!(flat.bands_per_unit.y, 0.0);
        assert_eq!(flat.lo.y, flat.hi.y);
        assert_eq!(band(&flat, 0).len(), 2);
        assert_eq!(
            band(&flat, 1).len(),
            2,
            "every column of a flat fill sees both curves"
        );

        let none = CurveBands::build(&[]);
        assert_eq!((none.rows, none.columns), (0, 0));
        assert!(none.texels.is_empty());
        assert_eq!(none.lo, none.hi);
    }
}
