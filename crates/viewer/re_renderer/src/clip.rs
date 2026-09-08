//! A world-space cut every drawable obeys: rectangles, point clouds and meshes discard the half-space
//! `dot(normal, p) > distance`. One rule in one place (`shader/utils/clip.wgsl` mirrors it), so a flat
//! image at z = 0 and a mesh are cut by the same plane.

/// World-space clip plane. `normal == 0` means no clipping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipPlane {
    pub normal: glam::Vec3,
    /// Points with `dot(normal, p) > distance` are discarded.
    pub distance: f32,
    /// Meshes shade the inside faces seen through the cut as a flat cap.
    pub cap: bool,
}

impl ClipPlane {
    pub const NONE: Self = Self { normal: glam::Vec3::ZERO, distance: 0.0, cap: false };

    /// `(normal, distance)` as the shader reads it.
    pub fn gpu(&self) -> glam::Vec4 {
        self.normal.extend(self.distance)
    }
}

impl Default for ClipPlane {
    fn default() -> Self {
        Self::NONE
    }
}
