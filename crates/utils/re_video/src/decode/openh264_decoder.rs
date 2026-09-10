//! In-process software H.264 decoding with Cisco's openh264.
//!
//! The fallback when no hardware decoder is usable: no child process, no pipe, one decoder
//! object per stream. Frames come out in presentation order; we match them back to the
//! submitted chunks by frame number.

use std::collections::BTreeMap;

use openh264::OpenH264API;
use openh264::decoder::{Decoder, DecoderConfig, Flush};
use openh264::formats::YUVSource as _;

use crate::decode::sync_decoder::SyncDecoder;
use crate::decode::{
    Chunk, DecodeError, Frame, FrameContent, FrameInfo, FrameResult, PixelFormat, Result, Time,
    YuvMatrixCoefficients, YuvPixelLayout, YuvRange,
};
use crate::h264::write_avc_chunk_to_nalu_stream;
use crate::nalu::AnnexBStreamState;
use crate::{Sender, VideoDataDescription};

struct PendingFrame {
    is_sync: bool,
    sample_idx: usize,
    presentation_timestamp: Time,
    decode_timestamp: Time,
    duration: Option<Time>,
}

pub struct OpenH264Decoder {
    debug_name: String,
    decoder: Decoder,
    /// `None` means the chunks are already an Annex-B bytestream.
    avcc: Option<re_mp4::Avc1Box>,
    annexb: AnnexBStreamState,
    /// Chunks handed to the decoder whose picture has not come out yet, by frame number.
    /// openh264 emits pictures in presentation order, so the lowest frame number is next.
    pending: BTreeMap<u32, PendingFrame>,
    yuv: (YuvRange, YuvMatrixCoefficients),
    annexb_buffer: Vec<u8>,
}

fn new_openh264() -> Result<Decoder> {
    // `NoFlush`: the crate's default flushes after every decode that produced no picture,
    // which on the single-threaded path leaks reordering slots (see moq-video's notes).
    let config = DecoderConfig::new().flush_after_decode(Flush::NoFlush);
    Decoder::with_api_config(OpenH264API::from_source(), config)
        .map_err(|err| DecodeError::OpenH264(format!("decoder init: {err}")))
}

fn avcc_from(video: &VideoDataDescription) -> Option<re_mp4::Avc1Box> {
    match video
        .encoding_details
        .as_ref()
        .and_then(|details| details.stsd.as_ref())
        .map(|stsd| &stsd.contents)
    {
        Some(re_mp4::StsdBoxContent::Avc1(avc1)) => {
            let mut avc1 = avc1.clone();
            // openh264 refuses SPS above level 5.2 (dsNoParamSets). x264 tags e.g. 120 fps clips
            // as level 6.2 regardless of size. level_idc only bounds buffer sizes, so clamp it.
            const MAX_OPENH264_LEVEL: u8 = 52;
            if avc1.avcc.avc_level_indication > MAX_OPENH264_LEVEL {
                avc1.avcc.avc_level_indication = MAX_OPENH264_LEVEL;
            }
            for sps in &mut avc1.avcc.sequence_parameter_sets {
                if let Some(level) = sps.bytes.get_mut(3) {
                    if *level > MAX_OPENH264_LEVEL {
                        *level = MAX_OPENH264_LEVEL;
                    }
                }
            }
            Some(avc1)
        }
        _ => None,
    }
}

impl OpenH264Decoder {
    pub fn new(
        debug_name: String,
        video: &VideoDataDescription,
        yuv: Option<(YuvRange, YuvMatrixCoefficients)>,
    ) -> Result<Self> {
        if video.codec != crate::VideoCodec::H264 {
            return Err(DecodeError::UnsupportedCodec(
                video.human_readable_codec_string(),
            ));
        }
        Ok(Self {
            debug_name,
            decoder: new_openh264()?,
            avcc: avcc_from(video),
            annexb: AnnexBStreamState::default(),
            pending: BTreeMap::new(),
            // Without a caller-supplied hint assume the most common SDR case.
            yuv: yuv.unwrap_or((YuvRange::Limited, YuvMatrixCoefficients::Bt709)),
            annexb_buffer: Vec::new(),
        })
    }

    fn frame_from_picture(
        picture: &openh264::decoder::DecodedYUV<'_>,
        frame_nr: u32,
        info: PendingFrame,
        yuv: (YuvRange, YuvMatrixCoefficients),
    ) -> Frame {
        let (width, height) = picture.dimensions();
        let (y_stride, u_stride, v_stride) = picture.strides();
        let chroma_width = width.div_ceil(2);
        let chroma_height = height.div_ceil(2);
        let mut data = Vec::with_capacity(width * height + 2 * chroma_width * chroma_height);
        for row in picture.y().chunks(y_stride).take(height) {
            data.extend_from_slice(&row[..width]);
        }
        for row in picture.u().chunks(u_stride).take(chroma_height) {
            data.extend_from_slice(&row[..chroma_width]);
        }
        for row in picture.v().chunks(v_stride).take(chroma_height) {
            data.extend_from_slice(&row[..chroma_width]);
        }
        Frame {
            content: FrameContent {
                data,
                width: width as u32,
                height: height as u32,
                format: PixelFormat::Yuv {
                    layout: YuvPixelLayout::Y_U_V420,
                    range: yuv.0,
                    coefficients: yuv.1,
                },
            },
            info: FrameInfo {
                is_sync: Some(info.is_sync),
                sample_idx: Some(info.sample_idx),
                frame_nr: Some(frame_nr),
                presentation_timestamp: info.presentation_timestamp,
                duration: info.duration,
                latest_decode_timestamp: Some(info.decode_timestamp),
            },
        }
    }
}

impl SyncDecoder for OpenH264Decoder {
    fn submit_chunk(
        &mut self,
        _should_stop: &std::sync::atomic::AtomicBool,
        chunk: Chunk,
        output_sender: &Sender<FrameResult>,
    ) {
        re_tracing::profile_function!();

        self.annexb_buffer.clear();
        let written = match &self.avcc {
            Some(avcc) => write_avc_chunk_to_nalu_stream(avcc, &mut self.annexb_buffer, &chunk, &mut self.annexb)
                .map_err(|err| DecodeError::OpenH264(format!("bad AVCC data: {err}"))),
            None => {
                self.annexb_buffer.extend_from_slice(&chunk.data);
                Ok(())
            }
        };
        if let Err(err) = written {
            output_sender.send(Err(err)).ok();
            return;
        }

        self.pending.insert(
            chunk.frame_nr,
            PendingFrame {
                is_sync: chunk.is_sync,
                sample_idx: chunk.sample_idx,
                presentation_timestamp: chunk.presentation_timestamp,
                decode_timestamp: chunk.decode_timestamp,
                duration: chunk.duration,
            },
        );

        match self.decoder.decode(&self.annexb_buffer) {
            Ok(Some(picture)) => {
                // The picture belongs to the oldest pending frame (presentation order).
                if let Some((frame_nr, info)) = self.pending.pop_first() {
                    let frame = Self::frame_from_picture(&picture, frame_nr, info, self.yuv);
                    output_sender.send(Ok(frame)).ok();
                }
            }
            // Parameter sets only, or a picture still held for reordering.
            Ok(None) => {}
            Err(err) => {
                // A refused access unit costs this picture, not the stream. Recover at the next keyframe.
                self.pending.remove(&chunk.frame_nr);
                re_log::trace!("{}: openh264 refused a chunk: {err}", self.debug_name);
            }
        }
    }

    fn reset(&mut self, video: &VideoDataDescription) {
        re_tracing::profile_function!();
        if let Ok(decoder) = new_openh264() {
            self.decoder = decoder;
        }
        self.avcc = avcc_from(video);
        self.annexb = AnnexBStreamState::default();
        self.pending.clear();
    }
}
