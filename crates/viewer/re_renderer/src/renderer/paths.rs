//! Path renderer: filled and stroked vector paths (cubic béziers) tessellated on the CPU
//! with lyon and drawn as one premultiplied-alpha-blended triangle list.
//!
//! This is the drawing side of vector content (shapes, text outlines, SVG). The caller only
//! describes contours and paint; flattening, fill rules, joins, caps, dashes and colour
//! conversion all live here.

use lyon_algorithms::measure::{PathMeasurements, SampleType};
use lyon_tessellation::path::{Path, math::point};
use lyon_tessellation::{
    BuffersBuilder, FillOptions, FillTessellator, FillVertex, StrokeOptions, StrokeTessellator,
    StrokeVertex, VertexBuffers,
};
use smallvec::smallvec;

use super::{
    DrawData, DrawError, DrawPhase, GpuRenderPipelinePoolAccessor, RenderContext, Renderer,
};
use crate::renderer::{DrawDataDrawable, DrawInstruction, DrawableCollectionViewInfo};
use crate::view_builder::ViewBuilder;
use crate::wgpu_resources::{
    BufferDesc, GpuBuffer, GpuRenderPipelineHandle, PipelineLayoutDesc, RenderPipelineDesc,
    VertexBufferLayout,
};
use crate::{DrawableCollector, Rgba32Unmul, include_shader_module};

/// One anchor of a cubic contour. Tangents are **relative to the point** (Lottie's `i`/`o`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathVertex {
    pub point: glam::Vec2,
    pub in_tangent: glam::Vec2,
    pub out_tangent: glam::Vec2,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PathContour {
    pub closed: bool,
    pub vertices: Vec<PathVertex>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathFillRule {
    NonZero,
    EvenOdd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathLineCap {
    Butt,
    Round,
    Square,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathLineJoin {
    Miter,
    Round,
    Bevel,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PathStroke {
    pub width: f32,
    pub cap: PathLineCap,
    pub join: PathLineJoin,
    pub miter_limit: f32,
    /// Dash lengths (on, off, …) and the phase offset, in path units.
    pub dash: Option<(Vec<f32>, f32)>,
}

/// Flattening tolerance in path units (pixels for 2D canvases).
pub const TOLERANCE: f32 = 0.05;

mod gpu_data {
    #[repr(C, packed)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct Vertex {
        pub position: [f32; 2],
        /// Straight (unmultiplied) sRGB. The shader linearises and premultiplies.
        pub color: [u8; 4],
    }

    impl Vertex {
        pub fn layout() -> super::VertexBufferLayout {
            super::VertexBufferLayout {
                array_stride: std::mem::size_of::<Self>() as _,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: super::VertexBufferLayout::attributes_from_formats(
                    0,
                    [wgpu::VertexFormat::Float32x2, wgpu::VertexFormat::Unorm8x4].into_iter(),
                ),
            }
        }
    }
}

/// Accumulates fills and strokes, in draw order, into one vertex/index list.
#[derive(Default)]
pub struct PathDrawDataBuilder {
    vertices: Vec<gpu_data::Vertex>,
    indices: Vec<u32>,
}

fn lyon_path(contours: &[PathContour]) -> Path {
    let mut b = Path::builder();
    for c in contours {
        let n = c.vertices.len();
        if n == 0 {
            continue;
        }
        let at = |v: &PathVertex| point(v.point.x, v.point.y);
        b.begin(at(&c.vertices[0]));
        let edges = if c.closed { n } else { n - 1 };
        for i in 0..edges {
            let v0 = &c.vertices[i];
            let v1 = &c.vertices[(i + 1) % n];
            if v0.out_tangent == glam::Vec2::ZERO && v1.in_tangent == glam::Vec2::ZERO {
                b.line_to(at(v1));
            } else {
                let c1 = v0.point + v0.out_tangent;
                let c2 = v1.point + v1.in_tangent;
                b.cubic_bezier_to(point(c1.x, c1.y), point(c2.x, c2.y), at(v1));
            }
        }
        b.end(c.closed);
    }
    b.build()
}

/// Cuts the path into dashes. Lengths are measured along the flattened path.
fn dashed(path: &Path, pattern: &[f32], offset: f32) -> Path {
    let pattern: Vec<f32> = pattern.iter().copied().filter(|d| d.is_finite() && *d >= 0.0).collect();
    let period: f32 = pattern.iter().sum();
    if pattern.is_empty() || period <= 0.0 {
        return path.clone();
    }
    let measurements = PathMeasurements::from_path(path, TOLERANCE);
    let mut sampler = measurements.create_sampler(path, SampleType::Distance);
    let length = sampler.length();
    let mut out = Path::builder();
    // Start one period before zero so the offset can shift dashes either way.
    let mut at = -((offset % period) + period) % period;
    let mut i = 0usize;
    while at < length {
        let d = pattern[i % pattern.len()];
        let (a, b) = (at.max(0.0), (at + d).min(length));
        if i % 2 == 0 && b > a {
            sampler.split_range(a..b, &mut out);
        }
        at += d;
        i += 1;
        if d == 0.0 && pattern.iter().all(|d| *d == 0.0) {
            break;
        }
    }
    out.build()
}

impl PathDrawDataBuilder {
    fn push(&mut self, buffers: VertexBuffers<[f32; 2], u32>, color_at: &dyn Fn(glam::Vec2) -> Rgba32Unmul) {
        let base = self.vertices.len() as u32;
        self.vertices.extend(buffers.vertices.iter().map(|p| gpu_data::Vertex {
            position: *p,
            color: color_at(glam::Vec2::from(*p)).0,
        }));
        self.indices.extend(buffers.indices.iter().map(|i| base + i));
    }

    /// Fill the contours. `color_at` is sampled per tessellated vertex, so a linear gradient
    /// is exact and anything else is interpolated across the triangles.
    pub fn fill(
        &mut self,
        contours: &[PathContour],
        rule: PathFillRule,
        color_at: &dyn Fn(glam::Vec2) -> Rgba32Unmul,
    ) {
        let path = lyon_path(contours);
        let options = FillOptions::tolerance(TOLERANCE).with_fill_rule(match rule {
            PathFillRule::NonZero => lyon_tessellation::FillRule::NonZero,
            PathFillRule::EvenOdd => lyon_tessellation::FillRule::EvenOdd,
        });
        let mut buffers: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
        let result = FillTessellator::new().tessellate_path(
            &path,
            &options,
            &mut BuffersBuilder::new(&mut buffers, |v: FillVertex<'_>| v.position().to_array()),
        );
        if result.is_ok() {
            self.push(buffers, color_at);
        }
    }

    pub fn stroke(
        &mut self,
        contours: &[PathContour],
        stroke: &PathStroke,
        color_at: &dyn Fn(glam::Vec2) -> Rgba32Unmul,
    ) {
        if !(stroke.width > 0.0) {
            return;
        }
        let mut path = lyon_path(contours);
        if let Some((pattern, offset)) = &stroke.dash {
            path = dashed(&path, pattern, *offset);
        }
        let options = StrokeOptions::tolerance(TOLERANCE)
            .with_line_width(stroke.width)
            .with_miter_limit(stroke.miter_limit.max(1.0))
            .with_line_cap(match stroke.cap {
                PathLineCap::Butt => lyon_tessellation::LineCap::Butt,
                PathLineCap::Round => lyon_tessellation::LineCap::Round,
                PathLineCap::Square => lyon_tessellation::LineCap::Square,
            })
            .with_line_join(match stroke.join {
                PathLineJoin::Miter => lyon_tessellation::LineJoin::Miter,
                PathLineJoin::Round => lyon_tessellation::LineJoin::Round,
                PathLineJoin::Bevel => lyon_tessellation::LineJoin::Bevel,
            });
        let mut buffers: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
        let result = StrokeTessellator::new().tessellate_path(
            &path,
            &options,
            &mut BuffersBuilder::new(&mut buffers, |v: StrokeVertex<'_, '_>| v.position().to_array()),
        );
        if result.is_ok() {
            self.push(buffers, color_at);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn build(self, ctx: &RenderContext, label: &str) -> PathDrawData {
        if self.indices.is_empty() {
            return PathDrawData { buffers: None };
        }
        let vertex_buffer = ctx.gpu_resources.buffers.alloc(
            &ctx.device,
            &BufferDesc {
                label: format!("{label} - path vertices").into(),
                size: (self.vertices.len() * std::mem::size_of::<gpu_data::Vertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );
        ctx.queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));
        let index_buffer = ctx.gpu_resources.buffers.alloc(
            &ctx.device,
            &BufferDesc {
                label: format!("{label} - path indices").into(),
                size: (self.indices.len() * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );
        ctx.queue.write_buffer(&index_buffer, 0, bytemuck::cast_slice(&self.indices));
        PathDrawData {
            buffers: Some(PathBuffers { vertex_buffer, index_buffer, index_count: self.indices.len() as u32 }),
        }
    }
}

#[derive(Clone)]
struct PathBuffers {
    vertex_buffer: GpuBuffer,
    index_buffer: GpuBuffer,
    index_count: u32,
}

#[derive(Clone)]
pub struct PathDrawData {
    buffers: Option<PathBuffers>,
}

impl DrawData for PathDrawData {
    type Renderer = PathRenderer;

    fn collect_drawables(
        &self,
        _view_info: &DrawableCollectionViewInfo,
        collector: &mut DrawableCollector<'_>,
    ) {
        if self.buffers.is_some() {
            collector.add_drawable(
                DrawPhase::Transparent,
                DrawDataDrawable { distance_sort_key: 0.0, secondary_sort_key: 0.0, draw_data_payload: 0 },
            );
        }
    }
}

pub struct PathRenderer {
    render_pipeline: GpuRenderPipelineHandle,
}

impl Renderer for PathRenderer {
    type RendererDrawData = PathDrawData;

    fn create_renderer(ctx: &RenderContext) -> Self {
        let shader_modules = &ctx.gpu_resources.shader_modules;
        let shader = shader_modules.get_or_create(ctx, &include_shader_module!("../../shader/paths.wgsl"));
        let render_pipeline = ctx.gpu_resources.render_pipelines.get_or_create(
            ctx,
            &RenderPipelineDesc {
                label: "PathRenderer::render_pipeline".into(),
                pipeline_layout: ctx.gpu_resources.pipeline_layouts.get_or_create(
                    ctx,
                    &PipelineLayoutDesc {
                        label: "global only".into(),
                        entries: vec![ctx.global_bindings.layout],
                    },
                ),
                vertex_entrypoint: "vs_main".into(),
                vertex_handle: shader,
                fragment_entrypoint: "fs_main".into(),
                fragment_handle: shader,
                vertex_buffers: smallvec![gpu_data::Vertex::layout()],
                render_targets: smallvec![Some(wgpu::ColorTargetState {
                    format: ViewBuilder::MAIN_TARGET_COLOR_FORMAT,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: ViewBuilder::MAIN_TARGET_DEPTH_FORMAT,
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    depth_write_enabled: Some(false),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: ViewBuilder::main_target_default_msaa_state(ctx.render_config(), false),
            },
        );
        Self { render_pipeline }
    }

    fn draw(
        &self,
        render_pipelines: &GpuRenderPipelinePoolAccessor<'_>,
        _phase: DrawPhase,
        pass: &mut wgpu::RenderPass<'_>,
        draw_instructions: &[DrawInstruction<'_, Self::RendererDrawData>],
    ) -> Result<(), DrawError> {
        let pipeline = render_pipelines.get(self.render_pipeline)?;
        pass.set_pipeline(pipeline);
        for DrawInstruction { draw_data, .. } in draw_instructions {
            let Some(buffers) = &draw_data.buffers else { continue };
            pass.set_vertex_buffer(0, buffers.vertex_buffer.slice(..));
            pass.set_index_buffer(buffers.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..buffers.index_count, 0, 0..1);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(size: f32) -> PathContour {
        let v = |x: f32, y: f32| PathVertex { point: glam::vec2(x, y), in_tangent: glam::Vec2::ZERO, out_tangent: glam::Vec2::ZERO };
        PathContour { closed: true, vertices: vec![v(0.0, 0.0), v(size, 0.0), v(size, size), v(0.0, size)] }
    }

    /// A square fills as two triangles, an even-odd ring leaves the hole, a dashed open line
    /// drops the gaps, and the colour callback sees tessellated positions.
    #[test]
    fn fills_strokes_and_dashes_tessellate() {
        let white = |_: glam::Vec2| Rgba32Unmul::WHITE;
        let mut b = PathDrawDataBuilder::default();
        b.fill(&[square(10.0)], PathFillRule::NonZero, &white);
        assert_eq!(b.triangle_count(), 2);

        let mut ring = PathDrawDataBuilder::default();
        let mut inner = square(4.0);
        for v in &mut inner.vertices { v.point += glam::vec2(3.0, 3.0); }
        ring.fill(&[square(10.0), inner], PathFillRule::EvenOdd, &white);
        assert!(ring.triangle_count() >= 8, "{}", ring.triangle_count());

        let line = PathContour { closed: false, vertices: vec![
            PathVertex { point: glam::vec2(0.0, 0.0), in_tangent: glam::Vec2::ZERO, out_tangent: glam::Vec2::ZERO },
            PathVertex { point: glam::vec2(100.0, 0.0), in_tangent: glam::Vec2::ZERO, out_tangent: glam::Vec2::ZERO },
        ] };
        let solid = PathStroke { width: 2.0, cap: PathLineCap::Butt, join: PathLineJoin::Miter, miter_limit: 4.0, dash: None };
        let mut s = PathDrawDataBuilder::default();
        s.stroke(&[line.clone()], &solid, &white);
        assert_eq!(s.triangle_count(), 2);
        let mut d = PathDrawDataBuilder::default();
        d.stroke(&[line], &PathStroke { dash: Some((vec![10.0, 10.0], 0.0)), ..solid }, &white);
        assert_eq!(d.triangle_count(), 10, "5 dashes of 2 triangles");

        let seen = std::cell::Cell::new(0usize);
        let mut g = PathDrawDataBuilder::default();
        g.fill(&[square(10.0)], PathFillRule::NonZero, &|p| { seen.set(seen.get() + 1); Rgba32Unmul([p.x as u8, 0, 0, 255]) });
        assert_eq!(seen.get(), 4);
    }
}
