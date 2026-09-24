//! Pixel oracle for exact curve fills (`PathDrawDataBuilder::fill_exact*`, `FORMAT_CURVES`).
//!
//! The shader evaluates a fill's coverage from its quadratic outline per fragment; changes to how
//! it gets there (which curves a fragment visits, how they are laid out) must not change the
//! picture. Each scene is rendered at 64² and 512² and compared pixel for pixel with a reference
//! written by an earlier revision (`tests/snapshots/exact_fill/`).
//!
//! `RE_RENDERER_EXACT_FILL_WRITE_REFERENCE=1` (re)writes the references from the current code.
//! `RE_RENDERER_EXACT_FILL_DUMP=1` writes the current renders next to them as `*.actual.png`.
//!
//! `cargo test --test exact_fill_coverage -- --ignored --nocapture bench` times 30 overlapping
//! ellipses at 512² and prints ns per covered fragment (per bounding quad pixel).

use re_renderer::device_caps;
use re_renderer::mesh::{CurveGradient, CurveGradientKind};
use re_renderer::renderer::{
    MeshDrawData, PathContour, PathDrawDataBuilder, PathFillRule, PathVertex,
};
use re_renderer::view_builder::{
    OrthographicCameraMode, Projection, RenderMode, TargetConfiguration, ViewBuilder,
};
use re_renderer::{
    BlendWithBackground, Color32, CpuModel, RenderConfig, RenderContext, Rgba, Rgba32Unmul,
    ScreenshotProcessor, ViewBuilderId,
};

/// Scenes are laid out in these units; a render maps them onto its resolution.
const CANVAS: f32 = 512.0;

fn render_context() -> RenderContext {
    let instance = wgpu::Instance::new(device_caps::testing_instance_descriptor());
    let adapter = pollster::block_on(device_caps::select_testing_adapter(&instance));
    let device_caps =
        device_caps::DeviceCaps::from_adapter(&adapter).expect("Failed to determine device caps");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&device_caps.device_descriptor()))
            .expect("Failed to request device");
    RenderContext::new(
        &adapter,
        device,
        queue,
        wgpu::TextureFormat::Rgba8Unorm,
        |_| RenderConfig::testing(),
    )
    .expect("Failed to create render context")
}

fn vertex(point: glam::Vec2, in_tangent: glam::Vec2, out_tangent: glam::Vec2) -> PathVertex {
    PathVertex {
        point,
        in_tangent,
        out_tangent,
    }
}

fn corner(point: glam::Vec2) -> PathVertex {
    vertex(point, glam::Vec2::ZERO, glam::Vec2::ZERO)
}

/// Four cubics, the usual kappa approximation; `clockwise` flips the winding.
fn ellipse(center: glam::Vec2, radii: glam::Vec2, clockwise: bool) -> PathContour {
    const KAPPA: f32 = 0.552_284_8;
    let k = radii * KAPPA;
    let (cx, cy) = (center.x, center.y);
    let mut vertices = vec![
        vertex(
            glam::vec2(cx + radii.x, cy),
            glam::vec2(0.0, -k.y),
            glam::vec2(0.0, k.y),
        ),
        vertex(
            glam::vec2(cx, cy + radii.y),
            glam::vec2(k.x, 0.0),
            glam::vec2(-k.x, 0.0),
        ),
        vertex(
            glam::vec2(cx - radii.x, cy),
            glam::vec2(0.0, k.y),
            glam::vec2(0.0, -k.y),
        ),
        vertex(
            glam::vec2(cx, cy - radii.y),
            glam::vec2(-k.x, 0.0),
            glam::vec2(k.x, 0.0),
        ),
    ];
    if clockwise {
        vertices.reverse();
        for v in &mut vertices {
            std::mem::swap(&mut v.in_tangent, &mut v.out_tangent);
        }
    }
    PathContour {
        closed: true,
        vertices,
    }
}

/// A pentagram: every edge crosses two others, so non-zero and even-odd differ in the middle.
fn pentagram(center: glam::Vec2, radius: f32) -> PathContour {
    let vertices = (0..5)
        .map(|i| {
            let angle = -std::f32::consts::FRAC_PI_2 + (i * 2) as f32 * std::f32::consts::TAU / 5.0;
            corner(center + radius * glam::vec2(angle.cos(), angle.sin()))
        })
        .collect();
    PathContour {
        closed: true,
        vertices,
    }
}

/// A lens `width` wide and `height` tall at 512²: sub-pixel at 64². Tilted so both rays see slopes.
fn sliver(center: glam::Vec2, width: f32, height: f32) -> PathContour {
    let tilt = glam::vec2(1.0, 0.04).normalize();
    let along = tilt * width * 0.5;
    let normal = glam::vec2(-tilt.y, tilt.x);
    // A cubic's peak is 3/4 of its control offset.
    let bulge = normal * (height * 0.5 / 0.75);
    let handle = tilt * width / 3.0;
    let a = center - along;
    let b = center + along;
    PathContour {
        closed: true,
        vertices: vec![
            vertex(a, -handle - bulge, handle - bulge),
            vertex(b, -handle - bulge, -handle + bulge),
        ],
    }
}

fn rgba(r: u8, g: u8, b: u8, a: u8) -> Rgba32Unmul {
    Rgba32Unmul([r, g, b, a])
}

fn gradient(kind: CurveGradientKind, start: glam::Vec2, end: glam::Vec2) -> CurveGradient {
    CurveGradient {
        kind,
        space_origin: glam::Vec2::ZERO,
        space_scale: glam::Vec2::ONE,
        start,
        end,
        ramp: (0..256)
            .map(|i| {
                let t = i as f32 / 255.0;
                rgba(
                    (255.0 * (1.0 - t)) as u8,
                    (255.0 * (0.5 - (t - 0.5).abs()) * 2.0) as u8,
                    (255.0 * t) as u8,
                    (255.0 - 96.0 * t) as u8,
                )
            })
            .collect(),
    }
}

type Scene = Box<dyn Fn(&mut PathDrawDataBuilder)>;

fn scene(f: impl Fn(&mut PathDrawDataBuilder) + 'static) -> Scene {
    Box::new(f)
}

/// The scenes and how they are built; each is one mesh.
fn scenes() -> Vec<(&'static str, Scene)> {
    let mut out: Vec<(&'static str, Scene)> = Vec::new();

    out.push((
        "ellipse",
        scene(|b| {
            b.fill_exact(
                &[ellipse(
                    glam::vec2(256.0, 256.0),
                    glam::vec2(200.0, 120.0),
                    false,
                )],
                PathFillRule::NonZero,
                rgba(220, 90, 40, 255),
            );
        }),
    ));

    out.push((
        "star_nonzero",
        scene(|b| {
            b.fill_exact(
                &[pentagram(glam::vec2(256.0, 270.0), 230.0)],
                PathFillRule::NonZero,
                rgba(240, 220, 60, 255),
            );
        }),
    ));

    out.push((
        "star_evenodd",
        scene(|b| {
            b.fill_exact(
                &[pentagram(glam::vec2(256.0, 270.0), 230.0)],
                PathFillRule::EvenOdd,
                rgba(60, 200, 240, 200),
            );
        }),
    ));

    out.push((
        "sliver_5px",
        scene(|b| {
            b.fill_exact(
                &[sliver(glam::vec2(256.0, 256.0), 400.0, 5.0)],
                PathFillRule::NonZero,
                rgba(255, 255, 255, 255),
            );
        }),
    ));

    out.push((
        "hole_evenodd",
        scene(|b| {
            b.fill_exact(
                &[
                    ellipse(glam::vec2(256.0, 256.0), glam::vec2(210.0, 170.0), false),
                    ellipse(glam::vec2(230.0, 250.0), glam::vec2(110.0, 60.0), false),
                ],
                PathFillRule::EvenOdd,
                rgba(90, 160, 90, 255),
            );
        }),
    ));

    out.push((
        "hole_nonzero_reversed",
        scene(|b| {
            b.fill_exact(
                &[
                    ellipse(glam::vec2(256.0, 256.0), glam::vec2(210.0, 170.0), false),
                    ellipse(glam::vec2(230.0, 250.0), glam::vec2(110.0, 60.0), true),
                ],
                PathFillRule::NonZero,
                rgba(160, 90, 160, 255),
            );
        }),
    ));

    out.push((
        "gradient_linear",
        scene(|b| {
            b.fill_exact_gradient(
                &[ellipse(
                    glam::vec2(256.0, 256.0),
                    glam::vec2(200.0, 150.0),
                    false,
                )],
                PathFillRule::NonZero,
                gradient(
                    CurveGradientKind::Linear,
                    glam::vec2(60.0, 100.0),
                    glam::vec2(450.0, 400.0),
                ),
            );
        }),
    ));

    out.push((
        "gradient_radial_star",
        scene(|b| {
            b.fill_exact_gradient(
                &[pentagram(glam::vec2(256.0, 270.0), 230.0)],
                PathFillRule::EvenOdd,
                gradient(
                    CurveGradientKind::Radial,
                    glam::vec2(256.0, 256.0),
                    glam::vec2(456.0, 256.0),
                ),
            );
        }),
    ));

    // Everything overlapping in one mesh: several materials, blending order.
    out.push((
        "everything",
        scene(|b| {
            b.fill_exact(
                &[ellipse(
                    glam::vec2(200.0, 220.0),
                    glam::vec2(180.0, 140.0),
                    false,
                )],
                PathFillRule::NonZero,
                rgba(220, 90, 40, 255),
            );
            b.fill_exact(
                &[pentagram(glam::vec2(300.0, 290.0), 200.0)],
                PathFillRule::EvenOdd,
                rgba(60, 200, 240, 160),
            );
            b.fill_exact(
                &[sliver(glam::vec2(256.0, 300.0), 460.0, 5.0)],
                PathFillRule::NonZero,
                rgba(255, 255, 255, 255),
            );
            b.fill_exact(
                &[
                    ellipse(glam::vec2(330.0, 200.0), glam::vec2(120.0, 100.0), false),
                    ellipse(glam::vec2(330.0, 200.0), glam::vec2(60.0, 40.0), false),
                ],
                PathFillRule::EvenOdd,
                rgba(90, 160, 90, 180),
            );
        }),
    ));

    out
}

/// Bounding quad pixels of every exact fill, at `resolution` px across the canvas: what the
/// fragment shader evaluates (the quad has a 1% + 1 unit margin, as `into_mesh` adds).
fn covered_fragments(ctx: &RenderContext, builder: PathDrawDataBuilder, resolution: u32) -> f64 {
    let mesh_units_per_px = CANVAS / resolution as f32;
    let mesh = builder.into_mesh(ctx, "count");
    let mut total = 0.0;
    for material in &mesh.materials {
        if material.curves.is_none() {
            continue;
        }
        let (lo, hi) = material
            .index_range
            .range()
            .map(|i| {
                mesh.vertex_positions
                    [mesh.triangle_indices[i as usize / 3][i as usize % 3] as usize]
                    .truncate()
            })
            .fold(
                (glam::Vec2::splat(f32::MAX), glam::Vec2::splat(f32::MIN)),
                |(lo, hi), p| (lo.min(p), hi.max(p)),
            );
        let size = (hi - lo) / mesh_units_per_px;
        total += (size.x as f64) * (size.y as f64);
    }
    total
}

fn draw_data(ctx: &RenderContext, builder: PathDrawDataBuilder) -> MeshDrawData {
    let mesh = builder.into_mesh(ctx, "exact fill oracle");
    let mut instances = CpuModel::from_single_mesh(mesh)
        .into_gpu_meshes(ctx)
        .expect("mesh upload");
    for instance in &mut instances {
        // The tint is additive and its alpha scales the fragment: opaque black leaves it alone.
        instance.additive_tint = Color32::BLACK;
    }
    MeshDrawData::new(ctx, &instances).expect("mesh draw data")
}

fn target(resolution: u32, name: &str) -> TargetConfiguration {
    TargetConfiguration {
        name: name.into(),
        render_mode: RenderMode::Deterministic,
        resolution_in_pixel: [resolution, resolution],
        view_from_world: macaw::IsoTransform::IDENTITY,
        projection_from_view: Projection::Orthographic {
            camera_mode: OrthographicCameraMode::TopLeftCornerAndExtendZ,
            vertical_world_size: CANVAS,
            far_plane_distance: 1000.0,
        },
        pixels_per_point: 1.0,
        blend_with_background: BlendWithBackground::Premultiplied,
        ..Default::default()
    }
}

fn wait(ctx: &RenderContext) {
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .expect("GPU work");
}

/// Renders one scene and returns its RGBA8 pixels.
fn render(
    ctx: &mut RenderContext,
    builder: PathDrawDataBuilder,
    resolution: u32,
    name: &str,
) -> Vec<u8> {
    const READBACK: re_renderer::GpuReadbackIdentifier = 0x0f11;
    ctx.begin_frame();
    let mut view = ViewBuilder::new(ctx, target(resolution, name), ViewBuilderId::new(0)).unwrap();
    view.queue_draw(ctx, draw_data(ctx, builder)).unwrap();
    view.schedule_screenshot(ctx, READBACK, ()).unwrap();
    let command_buffer = view.draw(ctx, Rgba::TRANSPARENT).unwrap();
    ctx.before_submit();
    ctx.queue.submit([command_buffer]);
    ctx.begin_frame();
    wait(ctx);
    let mut pixels = None;
    ScreenshotProcessor::next_readback_result(ctx, READBACK, |data, size, ()| {
        assert_eq!(size, glam::uvec2(resolution, resolution));
        pixels = Some(data.to_vec());
    })
    .expect("screenshot readback");
    pixels.unwrap()
}

fn snapshot_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("snapshots")
        .join("exact_fill")
}

fn write_png(path: &std::path::Path, pixels: &[u8], resolution: u32) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    image::save_buffer(
        path,
        pixels,
        resolution,
        resolution,
        image::ColorType::Rgba8,
    )
    .unwrap();
}

fn read_png(path: &std::path::Path) -> Option<Vec<u8>> {
    let image = image::open(path).ok()?;
    Some(image.into_rgba8().into_raw())
}

/// Every scene, at both sizes, pixel-identical to the reference. On a mismatch the message says
/// how many pixels differ and by how much at most (per channel).
#[test]
fn exact_fills_match_the_reference_pixels() {
    re_log::setup_logging();
    let write = std::env::var_os("RE_RENDERER_EXACT_FILL_WRITE_REFERENCE").is_some();
    let dump = std::env::var_os("RE_RENDERER_EXACT_FILL_DUMP").is_some();
    let mut ctx = render_context();
    let mut failures = Vec::new();
    for (name, build) in scenes() {
        for resolution in [64u32, 512] {
            let mut builder = PathDrawDataBuilder::default();
            build(&mut builder);
            let pixels = render(&mut ctx, builder, resolution, name);
            let path = snapshot_dir().join(format!("{name}_{resolution}.png"));
            if write {
                write_png(&path, &pixels, resolution);
                continue;
            }
            if dump {
                write_png(&path.with_extension("actual.png"), &pixels, resolution);
            }
            let Some(reference) = read_png(&path) else {
                failures.push(format!(
                    "{name}@{resolution}: no reference at {}",
                    path.display()
                ));
                continue;
            };
            assert_eq!(reference.len(), pixels.len());
            let mut differing_pixels = 0usize;
            let mut max_abs_diff = 0u8;
            let mut sum_abs_diff = 0u64;
            for (expected, actual) in reference.chunks_exact(4).zip(pixels.chunks_exact(4)) {
                if expected != actual {
                    differing_pixels += 1;
                    for (e, a) in expected.iter().zip(actual) {
                        let d = e.abs_diff(*a);
                        max_abs_diff = max_abs_diff.max(d);
                        sum_abs_diff += d as u64;
                    }
                }
            }
            if differing_pixels > 0 {
                failures.push(format!(
                    "{name}@{resolution}: {differing_pixels} of {} pixels differ, max abs diff {max_abs_diff}, sum abs diff {sum_abs_diff}",
                    resolution * resolution
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// 30 overlapping ellipses at 512², timed on the wall clock around submit + wait.
#[test]
#[ignore]
fn bench_overlapping_ellipses() {
    const RESOLUTION: u32 = 512;
    const FRAMES: usize = 40;
    // `RE_RENDERER_EXACT_FILL_BENCH_FILLS=1` is near the frame's floor (clear, resolve, composite);
    // an empty mesh cannot be uploaded, so 0 is not a valid setting.
    let fills: usize = std::env::var("RE_RENDERER_EXACT_FILL_BENCH_FILLS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30);
    let build = |builder: &mut PathDrawDataBuilder| {
        for i in 0..fills {
            let angle = i as f32 * std::f32::consts::TAU / 30.0;
            let center = glam::vec2(256.0, 256.0) + 70.0 * glam::vec2(angle.cos(), angle.sin());
            builder.fill_exact(
                &[ellipse(center, glam::vec2(170.0, 130.0), i % 2 == 0)],
                PathFillRule::NonZero,
                rgba(80 + (i * 5) as u8, 200 - (i * 4) as u8, 120, 200),
            );
        }
    };
    let builder = || {
        let mut b = PathDrawDataBuilder::default();
        build(&mut b);
        b
    };
    let mut ctx = render_context();
    let fragments = covered_fragments(&ctx, builder(), RESOLUTION);
    // Warm up: pipelines, mesh upload.
    let _ = render(&mut ctx, builder(), RESOLUTION, "warmup");
    let _ = render(&mut ctx, builder(), RESOLUTION, "warmup");
    ctx.begin_frame();
    let draw_data = draw_data(&ctx, builder());
    let mut views = Vec::new();
    for frame in 0..FRAMES {
        let mut view = ViewBuilder::new(
            &ctx,
            target(RESOLUTION, "bench"),
            ViewBuilderId::new(frame as u64),
        )
        .unwrap();
        view.queue_draw(&ctx, draw_data.clone()).unwrap();
        views.push(view);
    }
    let mut command_buffers = Vec::new();
    for view in &mut views {
        command_buffers.push(view.draw(&ctx, Rgba::TRANSPARENT).unwrap());
    }
    ctx.before_submit();
    wait(&ctx);
    let start = std::time::Instant::now();
    ctx.queue.submit(command_buffers);
    wait(&ctx);
    let elapsed = start.elapsed();
    let per_frame = elapsed / FRAMES as u32;
    let ns_per_fragment = per_frame.as_nanos() as f64 / fragments;
    println!(
        "{fills} ellipses @ {RESOLUTION}²: {per_frame:?} per frame over {FRAMES} frames, {fragments:.0} bounding quad px/frame -> {ns_per_fragment:.2} ns/px"
    );
    ctx.begin_frame();
}
