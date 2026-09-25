//! A compiled variant of the mesh and rectangle shaders with the embedder's hooks appended.
//!
//! `instanced_mesh_base.wgsl` and `rectangle_fragment.wgsl` / `rectangle_vertex.wgsl` call functions
//! they do not define: `program_field` and `program_motion` (vertex), `program_tint` and
//! `program_surface` (fragment). A [`SurfaceProgram`] appends either the defaults or the embedder's
//! own WGSL, writes the composed file next to the base shader (or a temp dir when shaders load from
//! disk), and builds the full pipeline set. Instances point at a program; the renderer batches by it.
//!
//! The hooks read the frame's resources through the global bindings (the environment, the backdrop,
//! the per-object `motion` storage, ...) and the 24 floats the instance carries; what those mean is
//! the embedder's.

use std::hash::{Hash as _, Hasher as _};
use std::path::PathBuf;

use smallvec::smallvec;

use crate::draw_phases::{OutlineMaskProcessor, PickingLayerProcessor};
use crate::mesh::mesh_vertices;
use crate::renderer::mesh_renderer::gpu_data;
use crate::view_builder::ViewBuilder;
use crate::wgpu_resources::{
    GpuPipelineLayoutHandle, GpuRenderPipelineHandle, RenderPipelineDesc, ShaderModuleDesc,
};
use crate::{Label, RenderContext, include_file};

/// Hook sources. `None` keeps the default: no displacement, no motion, and the renderer's own shading
/// (lit meshes, pictures as they are). The hooks reach meshes and rectangles alike: a rectangle
/// shows the field as a shift of where its picture is sampled.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SurfaceProgramDesc {
    pub label: String,
    /// WGSL the hooks share (functions, constants), included once in every variant.
    pub prelude: Option<String>,
    /// WGSL defining `fn program_field(in: FieldIn) -> FieldOut`.
    pub field: Option<String>,
    /// WGSL defining `fn program_motion(slot: f32, world_position: vec3f) -> vec3f` and
    /// `fn program_tint(slot: f32) -> vec4f`: what an instance's slot (its last param) does to a
    /// placed vertex and to a fragment.
    pub motion: Option<String>,
    /// WGSL defining `fn program_surface(in: SurfaceIn) -> vec3f` for meshes (and rectangles, unless
    /// `rectangle_surface` is given).
    pub surface: Option<String>,
    /// WGSL defining `fn program_surface(in: SurfaceIn) -> vec3f` for rectangles.
    pub rectangle_surface: Option<String>,
    /// Shade once per pixel (centroid) even where the context shades per sample: for surfaces whose
    /// colour does not vary inside a pixel (unlit pictures, analytic curve coverage). MSAA still
    /// resolves their geometric edges from coverage.
    pub pixel_rate: bool,
}

pub const DEFAULT_FIELD: &str =
    "fn program_field(in: FieldIn) -> FieldOut { return FieldOut(vec3f(0.0), in.normal); }";
pub const DEFAULT_MOTION: &str = "fn program_motion(slot: f32, world_position: vec3f) -> vec3f { return vec3f(0.0); }\nfn program_tint(slot: f32) -> vec4f { return vec4f(1.0); }";
pub const DEFAULT_SURFACE: &str =
    "fn program_surface(in: SurfaceIn) -> vec3f { return in.albedo * diffuse_shading(in.normal); }";
pub const DEFAULT_RECTANGLE_SURFACE: &str =
    "fn program_surface(in: SurfaceIn) -> vec3f { return in.albedo; }";

pub struct SurfaceProgram {
    pub(crate) rectangle_pipelines: Option<[GpuRenderPipelineHandle; 5]>,
    pub(crate) desc: SurfaceProgramDesc,

    /// Opaque meshes: depth first, then the material once per sample at the surviving depth.
    pub(crate) rp_depth_prepass: [GpuRenderPipelineHandle; 3],
    pub(crate) rp_shaded_at_depth: [GpuRenderPipelineHandle; 3],

    pub(crate) rp_shaded_alpha_blended_cull_back: GpuRenderPipelineHandle,
    pub(crate) rp_shaded_alpha_blended_cull_front: GpuRenderPipelineHandle,

    pub(crate) rp_picking_layer: GpuRenderPipelineHandle,
    pub(crate) rp_picking_layer_cull_back: GpuRenderPipelineHandle,
    pub(crate) rp_picking_layer_cull_front: GpuRenderPipelineHandle,

    pub(crate) rp_outline_mask: GpuRenderPipelineHandle,
    pub(crate) rp_outline_mask_cull_back: GpuRenderPipelineHandle,
    pub(crate) rp_outline_mask_cull_front: GpuRenderPipelineHandle,
}

impl std::fmt::Debug for SurfaceProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceProgram")
            .field("desc", &self.desc)
            .finish_non_exhaustive()
    }
}

fn base_path() -> PathBuf {
    include_file!("../../shader/instanced_mesh_base.wgsl")
}

/// Full WGSL of a variant, for validation by the embedder before creating pipelines.
pub fn compose_source(desc: &SurfaceProgramDesc) -> String {
    let import = if cfg!(load_shaders_from_disk) {
        base_path().display().to_string()
    } else {
        "./instanced_mesh_base.wgsl".to_owned()
    };
    format!(
        "#import <{import}>\n\n{}\n\n{}\n\n{}\n\n{}\n",
        desc.prelude.as_deref().unwrap_or(""),
        desc.field.as_deref().unwrap_or(DEFAULT_FIELD),
        desc.motion.as_deref().unwrap_or(DEFAULT_MOTION),
        desc.surface.as_deref().unwrap_or(DEFAULT_SURFACE),
    )
}

fn variant_path(desc: &SurfaceProgramDesc) -> PathBuf {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    desc.hash(&mut hasher);
    let name = format!("program_mesh_{:016x}.wgsl", hasher.finish());
    if cfg!(load_shaders_from_disk) {
        std::env::temp_dir()
            .join(format!("re_renderer-mesh-programs-{}", std::process::id()))
            .join(name)
    } else {
        base_path().with_file_name(name)
    }
}

fn write_variant(path: &std::path::Path, text: &str) -> anyhow::Result<()> {
    #[cfg(load_shaders_from_disk)]
    {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, text)?;
        Ok(())
    }
    #[cfg(not(load_shaders_from_disk))]
    {
        use crate::FileSystem as _;
        match crate::get_filesystem().create_file(path, text.to_owned().into()) {
            Ok(()) => Ok(()),
            // Same desc, same text: an existing file is fine.
            Err(err) if crate::get_filesystem().read_to_string(path).is_ok() => {
                let _ = err;
                Ok(())
            }
            Err(err) => Err(err),
        }
    }
}

impl SurfaceProgram {
    /// A variant sharing the mesh renderer's bind group and pipeline layouts.
    pub fn new(ctx: &RenderContext, desc: SurfaceProgramDesc) -> anyhow::Result<Self> {
        let pipeline_layout = ctx
            .renderer::<super::mesh_renderer::MeshRenderer>()?
            .pipeline_layout;
        let mut program = Self::with_layout(ctx, pipeline_layout, desc)?;
        let bases = ctx
            .renderer::<super::rectangles::RectangleRenderer>()?
            .surface_pipeline_descs
            .clone();
        let path = variant_path(&program.desc).with_extension("rectangle.wgsl");
        let import_of = |name: &str| {
            if cfg!(load_shaders_from_disk) {
                base_path().with_file_name(name).display().to_string()
            } else {
                format!("./{name}")
            }
        };
        let import = import_of("rectangle_fragment.wgsl");
        // The vertex stage is part of the variant too: it calls the field to move the grid.
        let import_vs = import_of("rectangle_vertex.wgsl");
        let desc = &program.desc;
        let prelude = desc.prelude.as_deref().unwrap_or("");
        let field = desc.field.as_deref().unwrap_or(DEFAULT_FIELD);
        let motion = desc.motion.as_deref().unwrap_or(DEFAULT_MOTION);
        let surface = desc
            .rectangle_surface
            .as_deref()
            .or(desc.surface.as_deref())
            .unwrap_or(DEFAULT_RECTANGLE_SURFACE);
        write_variant(
            &path,
            &format!(
                "#import <{import}>\n#import <{import_vs}>\n{prelude}\n{field}\n{motion}\n{surface}\n"
            ),
        )?;
        let shader = ctx.gpu_resources.shader_modules.get_or_create(
            ctx,
            &ShaderModuleDesc {
                label: format!("SurfaceProgram::rectangle::{}", program.desc.label).into(),
                source: path,
                extra_workaround_replacements: surface_sampling_replacements(ctx, program.desc.pixel_rate),
            },
        );
        // Every phase takes the variant's vertex stage — that is where the field moves the grid —
        // while only the colour phases take its fragment stage. The grid is a list, not the plain
        // strip of four.
        let variant = |base: RenderPipelineDesc, colour: bool| RenderPipelineDesc {
            vertex_handle: shader,
            fragment_handle: if colour { shader } else { base.fragment_handle },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            ..base
        };
        let pipelines = bases
            .into_iter()
            .enumerate()
            .map(|(i, base)| {
                ctx.gpu_resources
                    .render_pipelines
                    .get_or_create(ctx, &variant(base, i < 2))
            })
            .collect::<Vec<_>>();
        program.rectangle_pipelines = Some(pipelines.try_into().expect("five phases"));
        Ok(program)
    }

    pub(crate) fn with_layout(
        ctx: &RenderContext,
        pipeline_layout: GpuPipelineLayoutHandle,
        desc: SurfaceProgramDesc,
    ) -> anyhow::Result<Self> {
        re_tracing::profile_function!();

        let path = variant_path(&desc);
        write_variant(&path, &compose_source(&desc))?;
        let shader_module = ctx.gpu_resources.shader_modules.get_or_create(
            ctx,
            &ShaderModuleDesc {
                label: Label::from(format!("SurfaceProgram::{}", desc.label)),
                source: path,
                extra_workaround_replacements: surface_sampling_replacements(ctx, desc.pixel_rate),
            },
        );
        let render_pipelines = &ctx.gpu_resources.render_pipelines;

        // We always assume counter-clockwise faces as front.
        let front_face = wgpu::FrontFace::Ccw;
        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            front_face,
            ..Default::default()
        };
        // Put instance vertex buffer on slot 0 since it doesn't change for several draws.
        let vertex_buffers: smallvec::SmallVec<[_; 4]> = std::iter::chain(
            std::iter::once(gpu_data::InstanceData::vertex_buffer_layout()),
            mesh_vertices::vertex_buffer_layouts(),
        )
        .collect();
        let label = |suffix: &str| Label::from(format!("SurfaceProgram::{}::{suffix}", desc.label));
        let cull = |base: &RenderPipelineDesc, face: Option<wgpu::Face>, suffix: &str| {
            RenderPipelineDesc {
                label: label(suffix),
                primitive: wgpu::PrimitiveState {
                    cull_mode: face,
                    ..primitive
                },
                ..base.clone()
            }
        };

        let rp_shaded_desc = RenderPipelineDesc {
            label: label("shaded"),
            pipeline_layout,
            vertex_entrypoint: "vs_main".into(),
            vertex_handle: shader_module,
            fragment_entrypoint: "fs_main_shaded".into(),
            fragment_handle: shader_module,
            vertex_buffers,
            render_targets: smallvec![Some(ViewBuilder::MAIN_TARGET_COLOR_FORMAT.into())],
            primitive,
            depth_stencil: Some(ViewBuilder::MAIN_TARGET_DEFAULT_DEPTH_STATE),
            multisample: ViewBuilder::main_target_default_msaa_state(ctx.render_config(), false),
        };
        let rp_shaded_alpha_blended_desc = RenderPipelineDesc {
            label: label("shaded_alpha_blended"),
            render_targets: smallvec![Some(wgpu::ColorTargetState {
                format: ViewBuilder::MAIN_TARGET_COLOR_FORMAT,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            depth_stencil: Some(ViewBuilder::MAIN_TARGET_DEFAULT_DEPTH_STATE_NO_WRITE),
            ..rp_shaded_desc.clone()
        };
        let rp_depth_prepass_desc = RenderPipelineDesc {
            label: label("depth_prepass"),
            fragment_entrypoint: "fs_main_depth_only".into(),
            render_targets: smallvec![Some(wgpu::ColorTargetState {
                format: ViewBuilder::MAIN_TARGET_COLOR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::empty(),
            })],
            ..rp_shaded_desc.clone()
        };
        let rp_shaded_at_depth_desc = RenderPipelineDesc {
            label: label("shaded_at_depth"),
            depth_stencil: Some(wgpu::DepthStencilState {
                depth_compare: Some(wgpu::CompareFunction::Equal),
                depth_write_enabled: Some(false),
                ..ViewBuilder::MAIN_TARGET_DEFAULT_DEPTH_STATE
            }),
            ..rp_shaded_desc.clone()
        };
        let three = |base: &RenderPipelineDesc, name: &str| -> [GpuRenderPipelineHandle; 3] {
            [None, Some(wgpu::Face::Back), Some(wgpu::Face::Front)]
                .map(|face| render_pipelines.get_or_create(ctx, &cull(base, face, name)))
        };
        let rp_depth_prepass = three(&rp_depth_prepass_desc, "depth_prepass");
        let rp_shaded_at_depth = three(&rp_shaded_at_depth_desc, "shaded_at_depth");
        let rp_picking_layer_desc = RenderPipelineDesc {
            label: label("picking_layer"),
            fragment_entrypoint: "fs_main_picking_layer".into(),
            render_targets: smallvec![Some(PickingLayerProcessor::PICKING_LAYER_FORMAT.into())],
            depth_stencil: PickingLayerProcessor::PICKING_LAYER_DEPTH_STATE,
            multisample: PickingLayerProcessor::PICKING_LAYER_MSAA_STATE,
            ..rp_shaded_desc.clone()
        };
        let rp_outline_mask_desc = RenderPipelineDesc {
            label: label("outline_mask"),
            fragment_entrypoint: "fs_main_outline_mask".into(),
            render_targets: smallvec![Some(OutlineMaskProcessor::MASK_FORMAT.into())],
            depth_stencil: OutlineMaskProcessor::MASK_DEPTH_STATE,
            multisample: OutlineMaskProcessor::mask_default_msaa_state(ctx.device_caps().tier),
            ..rp_shaded_desc.clone()
        };

        Ok(Self {
            rectangle_pipelines: None,
            rp_depth_prepass,
            rp_shaded_at_depth,
            rp_shaded_alpha_blended_cull_back: render_pipelines.get_or_create(
                ctx,
                &cull(
                    &rp_shaded_alpha_blended_desc,
                    Some(wgpu::Face::Back),
                    "shaded_alpha_blended_cull_back",
                ),
            ),
            rp_shaded_alpha_blended_cull_front: render_pipelines.get_or_create(
                ctx,
                &cull(
                    &rp_shaded_alpha_blended_desc,
                    Some(wgpu::Face::Front),
                    "shaded_alpha_blended_cull_front",
                ),
            ),
            rp_picking_layer: render_pipelines.get_or_create(ctx, &rp_picking_layer_desc),
            rp_picking_layer_cull_back: render_pipelines.get_or_create(
                ctx,
                &cull(
                    &rp_picking_layer_desc,
                    Some(wgpu::Face::Back),
                    "picking_layer_cull_back",
                ),
            ),
            rp_picking_layer_cull_front: render_pipelines.get_or_create(
                ctx,
                &cull(
                    &rp_picking_layer_desc,
                    Some(wgpu::Face::Front),
                    "picking_layer_cull_front",
                ),
            ),
            rp_outline_mask: render_pipelines.get_or_create(ctx, &rp_outline_mask_desc),
            rp_outline_mask_cull_back: render_pipelines.get_or_create(
                ctx,
                &cull(
                    &rp_outline_mask_desc,
                    Some(wgpu::Face::Back),
                    "outline_mask_cull_back",
                ),
            ),
            rp_outline_mask_cull_front: render_pipelines.get_or_create(
                ctx,
                &cull(
                    &rp_outline_mask_desc,
                    Some(wgpu::Face::Front),
                    "outline_mask_cull_front",
                ),
            ),
            desc,
        })
    }

    pub fn desc(&self) -> &SurfaceProgramDesc {
        &self.desc
    }
}

/// Compatibility names for mesh-only callers.
pub type MeshProgram = SurfaceProgram;
pub type MeshProgramDesc = SurfaceProgramDesc;

fn surface_sampling_replacements(ctx: &RenderContext, pixel_rate: bool) -> Vec<(String, String)> {
    let mut replacements = Vec::new();
    let full = ctx.device_caps().tier != crate::device_caps::DeviceCapabilityTier::Limited;
    if !full
        || pixel_rate
        || ctx.render_config().msaa_mode == crate::MsaaMode::Off
        || ctx.render_config().surface_sampling != crate::SurfaceSampling::Sample
    {
        replacements.push((
            "@interpolate(perspective, sample)".into(),
            "@interpolate(perspective, centroid)".into(),
        ));
    }
    if full && ctx.render_config().surface_sampling == crate::SurfaceSampling::FilteredPixel {
        replacements.push((
            "const FILTER_SURFACE_FOOTPRINT: bool = false;".into(),
            "const FILTER_SURFACE_FOOTPRINT: bool = true;".into(),
        ));
    }
    replacements
}
