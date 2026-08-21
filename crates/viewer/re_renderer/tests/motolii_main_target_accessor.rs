//! Motolii seam: proves `ViewBuilder::main_target()` (added in this fork) is a live read
//! seat onto the resolved main target, not a dead accessor.
//!
//! See `docs/reviews/2026-08-21-blend-fork-accessor-decision.md` and BL1b in Motolii for why
//! this exists: `ViewBuilder::composite()` is the only other public way to read a view's
//! result, and it always round-trips through `composite.wgsl`'s gamma encode into a fixed,
//! non-sRGB-tagged output format. `main_target()` gives direct read access to the sRGB-tagged
//! (linear-mixed) intermediate instead, before that round-trip.

use re_renderer::device_caps;
use re_renderer::view_builder::{Projection, RenderMode, TargetConfiguration, ViewBuilder};
use re_renderer::{RenderConfig, RenderContext, Rgba, ViewBuilderId};

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

/// The accessor must return a texture that actually matches what this view builder was
/// configured and drawn with: right resolution, right (sRGB-tagged) format, and it must stay
/// reachable after `draw()` has recorded a command buffer against it.
#[test]
fn main_target_reflects_the_view_that_was_drawn() {
    let ctx = render_context();

    let resolution_in_pixel = [37, 51];

    let mut view_builder = ViewBuilder::new(
        &ctx,
        TargetConfiguration {
            name: "motolii-main-target-accessor-test".into(),
            render_mode: RenderMode::Deterministic,
            resolution_in_pixel,
            projection_from_view: Projection::Perspective {
                vertical_fov: 70.0 * std::f32::consts::TAU / 360.0,
                near_plane_distance: 0.01,
                aspect_ratio: resolution_in_pixel[0] as f32 / resolution_in_pixel[1] as f32,
            },
            ..Default::default()
        },
        ViewBuilderId::new(1),
    )
    .expect("failed to build view");

    // Before drawing, the accessor already points at an allocated texture of the configured
    // size and format -- it is not something `draw()` conjures up.
    {
        let main_target = view_builder.main_target();
        assert_eq!(
            [main_target.texture.width(), main_target.texture.height()],
            resolution_in_pixel,
            "main_target() resolution should match the view's configured resolution"
        );
        assert_eq!(
            main_target.texture.format(),
            ViewBuilder::MAIN_TARGET_COLOR_FORMAT,
            "main_target() must stay sRGB-tagged -- that's the whole point of the seam"
        );
    }

    let _command_buffer = view_builder
        .draw(&ctx, Rgba::TRANSPARENT)
        .expect("draw should succeed");

    // After `draw()` recorded its render pass into this exact texture, the accessor still
    // resolves to the same GPU resource (same handle behind the `GpuTexture` Arc) -- proving
    // it's the real target `draw()` wrote to, not a stale or unrelated copy.
    let main_target_after_draw = view_builder.main_target();
    assert_eq!(
        [
            main_target_after_draw.texture.width(),
            main_target_after_draw.texture.height()
        ],
        resolution_in_pixel,
        "main_target() must still resolve after draw() has recorded its pass"
    );
}
