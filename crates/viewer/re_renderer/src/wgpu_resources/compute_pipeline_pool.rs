use super::pipeline_layout_pool::{GpuPipelineLayoutHandle, GpuPipelineLayoutPool};
use super::resource::PoolError;
use super::shader_module_pool::{GpuShaderModuleHandle, GpuShaderModulePool};
use super::static_resource_pool::{StaticResourcePool, StaticResourcePoolReadLockAccessor};
use crate::RenderContext;
use crate::label::Label;

slotmap::new_key_type! { pub struct GpuComputePipelineHandle; }

/// Compute pipeline descriptor, can be converted into [`wgpu::ComputePipeline`] (which isn't hashable or comparable).
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct ComputePipelineDesc {
    /// Debug label of the pipeline. This will show up in graphics debuggers for easy identification.
    pub label: Label,

    pub pipeline_layout: GpuPipelineLayoutHandle,

    pub entrypoint: String,
    pub shader_handle: GpuShaderModuleHandle,
}

#[derive(thiserror::Error, Debug)]
pub enum ComputePipelineCreationError {
    #[error("Referenced pipeline layout not found: {0}")]
    PipelineLayout(PoolError),

    #[error("Referenced compute shader not found: {0}")]
    ShaderNotFound(PoolError),
}

impl ComputePipelineDesc {
    fn create_compute_pipeline(
        &self,
        device: &wgpu::Device,
        pipeline_layouts: &GpuPipelineLayoutPool,
        shader_modules: &GpuShaderModulePool,
    ) -> Result<wgpu::ComputePipeline, ComputePipelineCreationError> {
        let pipeline_layouts = pipeline_layouts.resources();
        let pipeline_layout = pipeline_layouts
            .get(self.pipeline_layout)
            .map_err(ComputePipelineCreationError::PipelineLayout)?;

        let shader_modules = shader_modules.resources();
        let module = shader_modules
            .get(self.shader_handle)
            .map_err(ComputePipelineCreationError::ShaderNotFound)?;

        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(self.label.get()),
                layout: Some(pipeline_layout),
                module,
                entry_point: Some(&self.entrypoint),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }),
        )
    }
}

pub type GpuComputePipelinePoolAccessor<'a> =
    StaticResourcePoolReadLockAccessor<'a, GpuComputePipelineHandle, wgpu::ComputePipeline>;

/// Compute pipelines, created once per descriptor and recreated when their shader is reloaded:
/// the compute counterpart of [`super::GpuRenderPipelinePool`].
#[derive(Default)]
pub struct GpuComputePipelinePool {
    pool: StaticResourcePool<GpuComputePipelineHandle, ComputePipelineDesc, wgpu::ComputePipeline>,
}

impl GpuComputePipelinePool {
    pub fn get_or_create(
        &self,
        ctx: &RenderContext,
        desc: &ComputePipelineDesc,
    ) -> Result<GpuComputePipelineHandle, ComputePipelineCreationError> {
        self.pool.get_or_try_create(desc, |desc| {
            desc.create_compute_pipeline(
                &ctx.device,
                &ctx.gpu_resources.pipeline_layouts,
                &ctx.gpu_resources.shader_modules,
            )
        })
    }

    pub fn begin_frame(
        &mut self,
        device: &wgpu::Device,
        frame_index: u64,
        shader_modules: &GpuShaderModulePool,
        pipeline_layouts: &GpuPipelineLayoutPool,
    ) {
        re_tracing::profile_function!();
        self.pool.current_frame_index = frame_index;

        // Recompile compute pipelines referencing shader modules that have been recompiled this frame
        // (see `GpuRenderPipelinePool::begin_frame`).
        self.pool.recreate_resources(|desc| {
            let frame_created = shader_modules
                .resources()
                .get_statistics(desc.shader_handle)
                .map(|sm| sm.frame_created)
                .unwrap_or(0);
            if frame_created < frame_index {
                return None;
            }
            match desc.create_compute_pipeline(device, pipeline_layouts, shader_modules) {
                Ok(pipeline) => {
                    re_log::info!(label = desc.label.get(), "recompiled compute pipeline");
                    Some(pipeline)
                }
                Err(err) => {
                    re_log::error!("Failed to compile compute pipeline: {err}");
                    None
                }
            }
        });
    }

    /// Locks the resource pool for resolving handles.
    ///
    /// While it is locked, no new resources can be added.
    pub fn resources(&self) -> GpuComputePipelinePoolAccessor<'_> {
        self.pool.resources()
    }

    pub fn num_resources(&self) -> usize {
        self.pool.num_resources()
    }
}
