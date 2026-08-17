//! Motolii seam — 埋め込み側 (embedder) が明示的に置くカメラ。
//!
//! **このファイルは Rerun 上流に無い追加ファイルである。** カメラ注入に関わる型と
//! 変換をここ1枚へ寄せてあるのは、上流を rebase したときに conflict しうる面を
//! 最小にするためである。上流 file 側へ入れた改変は次の2点だけに留めてある:
//!
//! - `eye.rs` — [`EyeState`](crate::eye::EyeState) の欄1つと、`EyeState::update` の
//!   先頭で「その欄を見るだけ」のフック1ブロック
//! - `lib.rs` — この module の宣言と再公開
//!
//! 公開署名は plain な数学型 (`[f32; 3]` と `f32`) だけで書いてある。`Eye` や
//! `IsoTransform` といった Rerun 内部型との接触は [`StageCamera::to_eye`] の内側に
//! 閉じてあるので、上流が内部型を変えても直すのはその関数の数行で済む。
//!
//! 経緯と台帳: `docs/reviews/2026-08-18-rerun-fork-seam-ledger.md` (Motolii 側)。

use glam::Vec3;
use macaw::IsoTransform;

use crate::eye::Eye;

/// 埋め込み側が明示的に指定するカメラ姿勢。world 座標で与える。
///
/// 置かれている間、その view のカメラはブループリント由来の既定値ではなく
/// この値になる。指定は sticky で、次に別の値を置くか取り消すまで有効である。
///
/// 座標系は Rerun の world 座標そのままで、`up` は「画面の上」を向けたい方向。
/// `position` と `look_target` が同一点になった場合でも `position` は守られる。
#[derive(Clone, Copy, Debug, PartialEq, re_byte_size::SizeBytes)]
pub struct StageCamera {
    /// カメラ位置。
    pub position: [f32; 3],

    /// 注視点。
    pub look_target: [f32; 3],

    /// 上方向。`look_target - position` と平行だと姿勢が決まらないので避ける。
    pub up: [f32; 3],

    /// 垂直画角 (radian)。`None` なら Rerun の既定値を使う。
    pub fov_y_radians: Option<f32>,
}

impl StageCamera {
    /// 位置・注視点・上方向からカメラを作る。画角は Rerun の既定値。
    pub fn new(position: [f32; 3], look_target: [f32; 3], up: [f32; 3]) -> Self {
        Self {
            position,
            look_target,
            up,
            fov_y_radians: None,
        }
    }

    /// 垂直画角 (radian) を指定する。
    #[must_use]
    pub fn with_fov_y_radians(mut self, fov_y_radians: f32) -> Self {
        self.fov_y_radians = Some(fov_y_radians);
        self
    }

    /// Rerun 内部のカメラ表現へ変換する。
    ///
    /// **seam の境界はここである。** 上流の `Eye` の形が変わったら、直すのはこの関数。
    /// 組み立て方は `EyeController::get_eye` (`eye.rs`) に合わせてある。
    pub(crate) fn to_eye(self) -> Eye {
        let position = Vec3::from(self.position);
        let look_target = Vec3::from(self.look_target);
        let up = Vec3::from(self.up);

        Eye {
            // `look_at_rh` は view_from_world を返すので、world_from_rub_view は逆。
            // 注視点が縮退している等で作れないときは、位置だけでも守る。
            world_from_rub_view: IsoTransform::look_at_rh(position, look_target, up)
                .unwrap_or_else(|| IsoTransform::from_translation(position))
                .inverse(),
            fov_y: Some(self.fov_y_radians.unwrap_or(Eye::DEFAULT_FOV_Y)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 上流の `Eye` の組み立てが変わったら、まずここが落ちる。
    #[test]
    fn to_eye_preserves_position_and_forward() {
        let eye = StageCamera::new([0.0, 0.0, 2.5], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]).to_eye();

        let position = eye.pos_in_world();
        assert!(
            (position - Vec3::new(0.0, 0.0, 2.5)).length() < 1e-5,
            "position was {position:?}"
        );

        // +z から原点を見るので、前方は -z。
        let forward = eye.forward_in_world();
        assert!(
            (forward - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-5,
            "forward was {forward:?}"
        );
    }

    #[test]
    fn fov_falls_back_to_the_rerun_default() {
        let camera = StageCamera::new([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert_eq!(camera.to_eye().fov_y, Some(Eye::DEFAULT_FOV_Y));
        assert_eq!(camera.with_fov_y_radians(0.5).to_eye().fov_y, Some(0.5));
    }

    #[test]
    fn degenerate_look_target_still_yields_the_requested_position() {
        // position == look_target では姿勢が決まらないが、位置は守られる。
        let camera = StageCamera::new([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 1.0, 0.0]);
        let position = camera.to_eye().pos_in_world();
        assert!(
            (position - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5,
            "position was {position:?}"
        );
    }
}
