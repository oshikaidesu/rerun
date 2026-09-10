use crate::decode::{Chunk, FrameResult};
use crate::{Sender, VideoDataDescription};

/// Blocking decoder of video chunks.
pub trait SyncDecoder {
    /// Submit some work and read the results.
    ///
    /// Stop early if `should_stop` is `true` or turns `true`.
    fn submit_chunk(
        &mut self,
        should_stop: &std::sync::atomic::AtomicBool,
        chunk: Chunk,
        output_sender: &Sender<FrameResult>,
    );

    /// Clear and reset everything
    fn reset(&mut self, video_data_description: &VideoDataDescription);

    /// See [`crate::decode::AsyncDecoder::min_num_samples_to_enqueue_ahead`].
    fn min_num_samples_to_enqueue_ahead(&self) -> usize {
        0
    }

    /// See [`crate::decode::AsyncDecoder::end_of_video`]: flush what the decoder still holds.
    fn end_of_video(&mut self, _output_sender: &Sender<FrameResult>) {}
}
