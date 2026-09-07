//! A compiled variant of the mesh shader with the embedder's hooks appended.
//!
//! `instanced_mesh_base.wgsl` calls two functions it does not define — `motolii_field` (vertex) and
//! `motolii_surface` (fragment). A [`MeshProgram`] appends either the defaults or the embedder's own
//! WGSL, writes the composed file next to the base shader (or a temp dir when shaders load from disk),
//! and builds the full pipeline set. Instances point at a program; the renderer batches by it.

use std::hash::{Hash as _, Hasher as _};
use std::path::PathBuf;

use smallvec::smallvec;

use crate::draw_phases::{OutlineMaskProcessor, PickingLayerProcessor};
use crate::renderer::mesh_renderer::gpu_data;
use crate::view_builder::ViewBuilder;
use crate::wgpu_resources::{
    GpuPipelineLayoutHandle, GpuRenderPipelineHandle, RenderPipelineDesc, ShaderModuleDesc,
};
use crate::mesh::mesh_vertices;
use crate::{include_file, Label, RenderContext};

/// Hook sources. `None` keeps the default (no displacement / matte dielectric).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MeshProgramDesc {
    pub label: String,
    /// WGSL defining `fn motolii_field(in: FieldIn) -> FieldOut`.
    pub field: Option<String>,
    /// WGSL defining `fn motolii_surface(in: SurfaceIn) -> vec3f`.
    pub surface: Option<String>,
}

pub const DEFAULT_FIELD: &str = "fn motolii_field(in: FieldIn) -> FieldOut { return FieldOut(vec3f(0.0), in.normal); }";
pub const DEFAULT_SURFACE: &str =
    "fn motolii_surface(in: SurfaceIn) -> vec3f { return shade_surface(in.albedo, in.normal, in.view_dir, vec4f(1.0, 0.0, 0.0, 1.5)); }";

pub struct MeshProgram {
    pub(crate) desc: MeshProgramDesc,

    pub(crate) rp_shaded: GpuRenderPipelineHandle,
    pub(crate) rp_shaded_cull_back: GpuRenderPipelineHandle,
    pub(crate) rp_shaded_cull_front: GpuRenderPipelineHandle,

    pub(crate) rp_shaded_alpha_blended_cull_back: GpuRenderPipelineHandle,
    pub(crate) rp_shaded_alpha_blended_cull_front: GpuRenderPipelineHandle,

    pub(crate) rp_picking_layer: GpuRenderPipelineHandle,
    pub(crate) rp_picking_layer_cull_back: GpuRenderPipelineHandle,
    pub(crate) rp_picking_layer_cull_front: GpuRenderPipelineHandle,

    pub(crate) rp_outline_mask: GpuRenderPipelineHandle,
    pub(crate) rp_outline_mask_cull_back: GpuRenderPipelineHandle,
    pub(crate) rp_outline_mask_cull_front: GpuRenderPipelineHandle,
}

impl std::fmt::Debug for MeshProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshProgram").field("desc", &self.desc).finish_non_exhaustive()
    }
}

fn base_path() -> PathBuf {
    include_file!("../../shader/instanced_mesh_base.wgsl")
}

/// Full WGSL of a variant, for validation by the embedder before creating pipelines.
pub fn compose_source(desc: &MeshProgramDesc) -> String {
    let import = if cfg!(load_shaders_from_disk) {
        base_path().display().to_string()
    } else {
        "./instanced_mesh_base.wgsl".to_owned()
    };
    format!(
        "#import <{import}>\n\n{}\n\n{}\n",
        desc.field.as_deref().unwrap_or(DEFAULT_FIELD),
        desc.surface.as_deref().unwrap_or(DEFAULT_SURFACE),
    )
}

fn variant_path(desc: &MeshProgramDesc) -> PathBuf {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    desc.hash(&mut hasher);
    let name = format!("motolii_mesh_{:016x}.wgsl", hasher.finish());
    if cfg!(load_shaders_from_disk) {
        std::env::temp_dir().join(format!("re_renderer-mesh-programs-{}", std::process::id())).join(name)
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

impl MeshProgram {
    pub fn new(ctx: &RenderContext, pipeline_layout: GpuPipelineLayoutHandle, desc: MeshProgramDesc) -> anyhow::Result<Self> {
        re_tracing::profile_function!();

        let path = variant_path(&desc);
        write_variant(&path, &compose_source(&desc))?;
        let shader_module = ctx.gpu_resources.shader_modules.get_or_create(
            ctx,
            &ShaderModuleDesc {
                label: Label::from(format!("MeshProgram::{}", desc.label)),
                source: path,
                extra_workaround_replacements: Vec::new(),
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
        let label = |suffix: &str| Label::from(format!("MeshProgram::{}::{suffix}", desc.label));
        let cull = |base: &RenderPipelineDesc, face: Option<wgpu::Face>, suffix: &str| RenderPipelineDesc {
            label: label(suffix),
            primitive: wgpu::PrimitiveState { cull_mode: face, ..primitive },
            ..base.clone()
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
            rp_shaded: render_pipelines.get_or_create(ctx, &rp_shaded_desc),
            rp_shaded_cull_back: render_pipelines.get_or_create(ctx, &cull(&rp_shaded_desc, Some(wgpu::Face::Back), "shaded_cull_back")),
            rp_shaded_cull_front: render_pipelines.get_or_create(ctx, &cull(&rp_shaded_desc, Some(wgpu::Face::Front), "shaded_cull_front")),
            rp_shaded_alpha_blended_cull_back: render_pipelines.get_or_create(ctx, &cull(&rp_shaded_alpha_blended_desc, Some(wgpu::Face::Back), "shaded_alpha_blended_cull_back")),
            rp_shaded_alpha_blended_cull_front: render_pipelines.get_or_create(ctx, &cull(&rp_shaded_alpha_blended_desc, Some(wgpu::Face::Front), "shaded_alpha_blended_cull_front")),
            rp_picking_layer: render_pipelines.get_or_create(ctx, &rp_picking_layer_desc),
            rp_picking_layer_cull_back: render_pipelines.get_or_create(ctx, &cull(&rp_picking_layer_desc, Some(wgpu::Face::Back), "picking_layer_cull_back")),
            rp_picking_layer_cull_front: render_pipelines.get_or_create(ctx, &cull(&rp_picking_layer_desc, Some(wgpu::Face::Front), "picking_layer_cull_front")),
            rp_outline_mask: render_pipelines.get_or_create(ctx, &rp_outline_mask_desc),
            rp_outline_mask_cull_back: render_pipelines.get_or_create(ctx, &cull(&rp_outline_mask_desc, Some(wgpu::Face::Back), "outline_mask_cull_back")),
            rp_outline_mask_cull_front: render_pipelines.get_or_create(ctx, &cull(&rp_outline_mask_desc, Some(wgpu::Face::Front), "outline_mask_cull_front")),
            desc,
        })
    }

    pub fn desc(&self) -> &MeshProgramDesc {
        &self.desc
    }
}
