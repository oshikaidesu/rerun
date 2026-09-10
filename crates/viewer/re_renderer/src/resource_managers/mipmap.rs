//! GPU mip chain for 2D textures: every level is the 2×2 box average of the level above, drawn with a
//! bilinear tap at the destination texel centre. Used for equirectangular environment maps so glossy
//! reflections can pick a level by roughness instead of a CPU-convolved atlas.

use std::collections::HashMap;

const SHADER: &str = r#"
struct VsOut { @builtin(position) position: vec4f, @location(0) uv: vec2f };

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    let corners = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(-1.0, 1.0), vec2f(3.0, 1.0));
    var out: VsOut;
    out.position = vec4f(corners[index], 0.0, 1.0);
    out.uv = vec2f(corners[index].x * 0.5 + 0.5, 1.0 - (corners[index].y * 0.5 + 0.5));
    return out;
}

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var bilinear: sampler;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    return textureSampleLevel(source, bilinear, in.uv, 0.0);
}
"#;

pub struct MipmapGenerator {
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    sampler: wgpu::Sampler,
    pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

impl MipmapGenerator {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MipmapGenerator::layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MipmapGenerator::pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MipmapGenerator::shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("MipmapGenerator::sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        Self {
            layout,
            pipeline_layout,
            module,
            sampler,
            pipelines: HashMap::new(),
        }
    }

    /// Full chain down to 1×1: `floor(log2(max(width, height))) + 1`.
    pub fn mip_level_count(width: u32, height: u32) -> u32 {
        32 - width.max(height).max(1).leading_zeros()
    }

    /// Records one draw per level below the base into `encoder`. The texture must carry
    /// `RENDER_ATTACHMENT | TEXTURE_BINDING` and its levels must already be allocated.
    pub fn generate(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture) {
        self.generate_levels(device, encoder, texture, texture.mip_level_count());
    }

    /// Like `generate`, but stops after `levels` levels counting the base: levels the reader will
    /// never sample stay untouched (Unity's "blur only as far as the roughness needs" for screen copies).
    pub fn generate_levels(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture, levels: u32) {
        let format = texture.format();
        let pipeline = self.pipelines.entry(format).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("MipmapGenerator::pipeline"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        });
        for level in 1..levels.min(texture.mip_level_count()) {
            let view = |base_mip_level: u32| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            };
            let (source, target) = (view(level - 1), view(level));
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("MipmapGenerator::bind_group"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&source) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("MipmapGenerator::pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MipmapGenerator;

    #[test]
    fn level_count_reaches_one_by_one() {
        assert_eq!(MipmapGenerator::mip_level_count(1, 1), 1);
        assert_eq!(MipmapGenerator::mip_level_count(2, 1), 2);
        assert_eq!(MipmapGenerator::mip_level_count(4096, 2048), 13);
        assert_eq!(MipmapGenerator::mip_level_count(3000, 10), 12);
    }

    /// A 4×4 checker of 0 and 1 averages to 0.5 on the 1×1 level.
    #[test]
    fn levels_are_box_averages_of_the_level_above() {
        let mut ctx = crate::RenderContext::new_test();
        let mut pixels = Vec::new();
        for y in 0..4u32 {
            for x in 0..4u32 {
                let v: f32 = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                for c in [v, v, v, 1.0] {
                    pixels.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
                }
            }
        }
        let probe = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mip probe"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut readback = None;
        ctx.execute_test_frame(|ctx| {
            let texture = ctx
                .texture_manager_2d
                .create_with_mipmaps(
                    ctx,
                    crate::resource_managers::ImageDataDesc {
                        label: "checker".into(),
                        data: pixels.into(),
                        format: wgpu::TextureFormat::Rgba16Float.into(),
                        width_height: [4, 4],
                        alpha_channel_usage: crate::AlphaChannelUsage::Opaque,
                    },
                )
                .unwrap();
            assert_eq!(texture.texture.mip_level_count(), 3);
            // Same frame encoder as the upload and the mip passes, so the copy lands after them and
            // before the readback (which also records there).
            ctx.active_frame.before_view_builder_encoder.lock().get().copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo { texture: &texture.texture, mip_level: 2, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyTextureInfo { texture: &probe, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            readback = Some(crate::texture_readback::schedule_read_texture(ctx, &probe).unwrap());
            []
        });
        let mut result = None;
        for _ in 0..100 {
            if let Some(data) = crate::texture_readback::poll_read_texture(&ctx, readback.unwrap()) {
                result = Some(data);
                break;
            }
            ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        }
        let data = result.expect("readback").data;
        let r = half::f16::from_le_bytes([data[0], data[1]]).to_f32();
        assert!((r - 0.5).abs() < 0.02, "1×1 level should be the checker average, got {r}");
    }
}
