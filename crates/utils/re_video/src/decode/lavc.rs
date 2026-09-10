//! In-process decoding with libavcodec (linked, not the ffmpeg executable).
//!
//! One decoder object per stream. Hardware decoding goes through libavcodec's hwaccel
//! (VideoToolbox / D3D11VA / VAAPI), software decoding through its own multi-threaded
//! decoders. Frames are matched back to chunks by presentation timestamp.

use std::collections::BTreeMap;

use ffmpeg_next as ff;

use crate::decode::sync_decoder::SyncDecoder;
use crate::decode::{
    Chunk, DecodeError, Frame, FrameContent, FrameInfo, FrameResult, PixelFormat, Result, Time,
    YuvMatrixCoefficients, YuvPixelLayout, YuvRange,
};
use crate::h264::write_avc_chunk_to_nalu_stream;
use crate::h265::write_hevc_chunk_to_nalu_stream;
use crate::nalu::AnnexBStreamState;
use crate::{DecodeHardwareAcceleration, Sender, VideoDataDescription};

struct PendingFrame {
    is_sync: bool,
    sample_idx: usize,
    frame_nr: u32,
    presentation_timestamp: Time,
    decode_timestamp: Time,
    duration: Option<Time>,
}

/// How chunks reach libavcodec. MP4 carries AVCC/HVCC (length-prefixed, parameter sets in
/// the sample description); the decoders want Annex-B with in-band parameter sets.
enum Bitstream {
    Raw,
    Avc { avcc: re_mp4::Avc1Box, state: AnnexBStreamState },
    Hevc { hvcc: re_mp4::HevcBox, state: AnnexBStreamState },
}

impl Bitstream {
    fn for_video(video: &VideoDataDescription) -> Self {
        match video
            .encoding_details
            .as_ref()
            .and_then(|details| details.stsd.as_ref())
            .map(|stsd| &stsd.contents)
        {
            Some(re_mp4::StsdBoxContent::Avc1(avcc)) => Self::Avc {
                avcc: avcc.clone(),
                state: AnnexBStreamState::default(),
            },
            Some(re_mp4::StsdBoxContent::Hev1(hvcc) | re_mp4::StsdBoxContent::Hvc1(hvcc)) => {
                Self::Hevc {
                    hvcc: hvcc.clone(),
                    state: AnnexBStreamState::default(),
                }
            }
            _ => Self::Raw,
        }
    }

    fn write(&mut self, out: &mut Vec<u8>, chunk: &Chunk) -> Result<()> {
        out.clear();
        match self {
            Self::Raw => out.extend_from_slice(&chunk.data),
            Self::Avc { avcc, state } => write_avc_chunk_to_nalu_stream(avcc, out, chunk, state)
                .map_err(|err| DecodeError::Lavc(format!("bad AVCC data: {err}")))?,
            Self::Hevc { hvcc, state } => write_hevc_chunk_to_nalu_stream(hvcc, out, chunk, state)
                .map_err(|err| DecodeError::Lavc(format!("bad HVCC data: {err}")))?,
        }
        Ok(())
    }
}

fn codec_id(codec: &crate::VideoCodec) -> Option<ff::codec::Id> {
    Some(match codec {
        crate::VideoCodec::H264 => ff::codec::Id::H264,
        crate::VideoCodec::H265 => ff::codec::Id::HEVC,
        crate::VideoCodec::VP8 => ff::codec::Id::VP8,
        crate::VideoCodec::VP9 => ff::codec::Id::VP9,
        crate::VideoCodec::AV1 => ff::codec::Id::AV1,
        crate::VideoCodec::ImageSequence(_) => return None,
    })
}

fn init_once() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        ff::init().ok();
        ff::util::log::set_level(ff::util::log::Level::Error);
    });
}

/// The platform's hardware decoder device, if libavcodec can open one.
#[expect(unsafe_code)]
fn open_hw_device() -> Option<*mut ff::ffi::AVBufferRef> {
    let kind = if cfg!(target_os = "macos") {
        ff::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX
    } else if cfg!(target_os = "windows") {
        ff::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA
    } else if cfg!(target_os = "linux") {
        ff::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI
    } else {
        return None;
    };
    let mut device: *mut ff::ffi::AVBufferRef = std::ptr::null_mut();
    // SAFETY: plain FFI call with a valid out-pointer; a failure leaves `device` null.
    let rc = unsafe {
        ff::ffi::av_hwdevice_ctx_create(&mut device, kind, std::ptr::null(), std::ptr::null_mut(), 0)
    };
    (rc >= 0 && !device.is_null()).then_some(device)
}

#[expect(unsafe_code)]
fn open_decoder(
    codec: &crate::VideoCodec,
    hw: DecodeHardwareAcceleration,
) -> Result<(ff::decoder::Video, bool)> {
    init_once();
    let id = codec_id(codec).ok_or_else(|| DecodeError::UnsupportedCodec(format!("{codec:?}")))?;
    let decoder = ff::decoder::find(id)
        .ok_or_else(|| DecodeError::Lavc(format!("libavcodec has no decoder for {id:?}")))?;
    let mut context = ff::codec::context::Context::new_with_codec(decoder);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
    context.set_threading(ff::codec::threading::Config {
        kind: ff::codec::threading::Type::Frame,
        count: threads,
    });
    let mut hw_used = false;
    if hw != DecodeHardwareAcceleration::PreferSoftware {
        if let Some(device) = open_hw_device() {
            // SAFETY: `context` owns a valid AVCodecContext; we hand it a fresh reference to the device.
            unsafe {
                (*context.as_mut_ptr()).hw_device_ctx = ff::ffi::av_buffer_ref(device);
                ff::ffi::av_buffer_unref(&mut { device });
            }
            hw_used = true;
        } else if hw == DecodeHardwareAcceleration::PreferHardware {
            return Err(DecodeError::Lavc("no hardware decoder device available".to_owned()));
        }
    }
    let video = context
        .decoder()
        .video()
        .map_err(|err| DecodeError::Lavc(format!("open decoder: {err}")))?;
    Ok((video, hw_used))
}

pub struct LavcDecoder {
    debug_name: String,
    codec: crate::VideoCodec,
    hw: DecodeHardwareAcceleration,
    decoder: ff::decoder::Video,
    hw_used: bool,
    bitstream: Bitstream,
    /// Chunks handed to the decoder whose frame has not come out yet, by presentation timestamp.
    pending: BTreeMap<i64, PendingFrame>,
    yuv_hint: Option<(YuvRange, YuvMatrixCoefficients)>,
    packet_buffer: Vec<u8>,
    /// Scratch frames, reused across calls.
    frame: ff::util::frame::Video,
    sw_frame: ff::util::frame::Video,
    threads_ahead: usize,
}

impl LavcDecoder {
    pub fn new(
        debug_name: String,
        video: &VideoDataDescription,
        hw: DecodeHardwareAcceleration,
        yuv_hint: Option<(YuvRange, YuvMatrixCoefficients)>,
    ) -> Result<Self> {
        let (decoder, hw_used) = open_decoder(&video.codec, hw)?;
        re_log::debug!(
            "{debug_name}: libavcodec decoder for {:?}, hardware={hw_used}",
            video.codec
        );
        Ok(Self {
            debug_name,
            codec: video.codec.clone(),
            hw,
            decoder,
            hw_used,
            bitstream: Bitstream::for_video(video),
            pending: BTreeMap::new(),
            yuv_hint,
            packet_buffer: Vec::new(),
            frame: ff::util::frame::Video::empty(),
            sw_frame: ff::util::frame::Video::empty(),
            threads_ahead: std::thread::available_parallelism().map_or(4, |n| n.get()).min(8),
        })
    }

    pub fn min_num_samples_ahead(&self) -> usize {
        // Frame threading holds up to one frame per thread; reordering adds a couple more.
        if self.hw_used { 4 } else { self.threads_ahead + 2 }
    }

    fn drain(&mut self, output_sender: &Sender<FrameResult>) {
        while self.decoder.receive_frame(&mut self.frame).is_ok() {
            let result = self.frame_out();
            output_sender.send(result).ok();
        }
    }

    /// Turns the frame libavcodec just produced into ours, matching it to its chunk by timestamp.
    #[expect(unsafe_code)]
    fn frame_out(&mut self) -> FrameResult {
        let pts = self.frame.pts();
        let info = match pts.and_then(|pts| self.pending.remove(&pts)) {
            Some(info) => info,
            None => self
                .pending
                .pop_first()
                .map(|(_, info)| info)
                .ok_or_else(|| DecodeError::Lavc("frame without a pending chunk".to_owned()))?,
        };

        // Hardware frames live in device memory; bring them over. (Zero-copy import is a later step.)
        let source: &ff::util::frame::Video = if self.frame.format() == ff::format::Pixel::VIDEOTOOLBOX
            || self.frame.format() == ff::format::Pixel::D3D11
            || self.frame.format() == ff::format::Pixel::VAAPI
        {
            // SAFETY: both frames are valid; transfer allocates the software frame as needed.
            let rc = unsafe {
                ff::ffi::av_hwframe_transfer_data(self.sw_frame.as_mut_ptr(), self.frame.as_ptr(), 0)
            };
            if rc < 0 {
                return Err(DecodeError::Lavc(format!("hwframe transfer failed: {rc}")));
            }
            &self.sw_frame
        } else {
            &self.frame
        };

        let width = source.width() as usize;
        let height = source.height() as usize;
        let (layout, planes): (YuvPixelLayout, &[(usize, usize, usize)]) = match source.format() {
            ff::format::Pixel::YUV420P | ff::format::Pixel::YUVJ420P => {
                (YuvPixelLayout::Y_U_V420, &[(0, 1, 1), (1, 2, 2), (2, 2, 2)])
            }
            ff::format::Pixel::YUV422P | ff::format::Pixel::YUVJ422P => {
                (YuvPixelLayout::Y_U_V422, &[(0, 1, 1), (1, 2, 1), (2, 2, 1)])
            }
            ff::format::Pixel::YUV444P | ff::format::Pixel::YUVJ444P => {
                (YuvPixelLayout::Y_U_V444, &[(0, 1, 1), (1, 1, 1), (2, 1, 1)])
            }
            ff::format::Pixel::NV12 => (YuvPixelLayout::Y_UV420, &[(0, 1, 1), (1, 1, 2)]),
            ff::format::Pixel::GRAY8 => (YuvPixelLayout::Y400, &[(0, 1, 1)]),
            other => {
                return Err(DecodeError::Lavc(format!("unsupported pixel format {other:?}")));
            }
        };
        // (plane index, horizontal divisor, vertical divisor). NV12's UV plane is full width in bytes.
        let mut data = Vec::with_capacity(width * height * 3 / 2);
        for &(plane, dx, dy) in planes {
            let row_bytes = if layout == YuvPixelLayout::Y_UV420 && plane == 1 {
                width.div_ceil(dx) // two interleaved samples per chroma column = full width in bytes
            } else {
                width.div_ceil(dx)
            };
            let rows = height.div_ceil(dy);
            let stride = source.stride(plane);
            let bytes = source.data(plane);
            for row in bytes.chunks(stride).take(rows) {
                data.extend_from_slice(&row[..row_bytes]);
            }
        }

        let full = match source.color_range() {
            ff::util::color::Range::JPEG => true,
            ff::util::color::Range::MPEG => false,
            _ => matches!(
                source.format(),
                ff::format::Pixel::YUVJ420P | ff::format::Pixel::YUVJ422P | ff::format::Pixel::YUVJ444P
            ) || matches!(self.yuv_hint, Some((YuvRange::Full, _))),
        };
        let coefficients = match source.color_space() {
            ff::util::color::Space::BT709 => YuvMatrixCoefficients::Bt709,
            ff::util::color::Space::SMPTE170M | ff::util::color::Space::BT470BG => {
                YuvMatrixCoefficients::Bt601
            }
            ff::util::color::Space::RGB => YuvMatrixCoefficients::Identity,
            _ => self.yuv_hint.map_or(YuvMatrixCoefficients::Bt709, |(_, c)| c),
        };

        Ok(Frame {
            content: FrameContent {
                data,
                width: width as u32,
                height: height as u32,
                format: PixelFormat::Yuv {
                    layout,
                    range: if full { YuvRange::Full } else { YuvRange::Limited },
                    coefficients,
                },
            },
            info: FrameInfo {
                is_sync: Some(info.is_sync),
                sample_idx: Some(info.sample_idx),
                frame_nr: Some(info.frame_nr),
                presentation_timestamp: info.presentation_timestamp,
                duration: info.duration,
                latest_decode_timestamp: Some(info.decode_timestamp),
            },
        })
    }
}

impl SyncDecoder for LavcDecoder {
    fn submit_chunk(
        &mut self,
        _should_stop: &std::sync::atomic::AtomicBool,
        chunk: Chunk,
        output_sender: &Sender<FrameResult>,
    ) {
        re_tracing::profile_function!();

        if let Err(err) = self.bitstream.write(&mut self.packet_buffer, &chunk) {
            output_sender.send(Err(err)).ok();
            return;
        }
        let mut packet = ff::Packet::copy(&self.packet_buffer);
        packet.set_pts(Some(chunk.presentation_timestamp.0));
        packet.set_dts(Some(chunk.decode_timestamp.0));
        if chunk.is_sync {
            packet.set_flags(ff::codec::packet::Flags::KEY);
        }
        self.pending.insert(
            chunk.presentation_timestamp.0,
            PendingFrame {
                is_sync: chunk.is_sync,
                sample_idx: chunk.sample_idx,
                frame_nr: chunk.frame_nr,
                presentation_timestamp: chunk.presentation_timestamp,
                decode_timestamp: chunk.decode_timestamp,
                duration: chunk.duration,
            },
        );

        match self.decoder.send_packet(&packet) {
            Ok(()) => {}
            Err(ff::Error::Other { errno }) if errno == ff::util::error::EAGAIN => {
                // Output queue full: drain, then send again.
                self.drain(output_sender);
                if let Err(err) = self.decoder.send_packet(&packet) {
                    self.pending.remove(&chunk.presentation_timestamp.0);
                    re_log::trace!("{}: libavcodec refused a packet: {err}", self.debug_name);
                }
            }
            Err(err) => {
                // A refused packet costs this picture, not the stream.
                self.pending.remove(&chunk.presentation_timestamp.0);
                re_log::trace!("{}: libavcodec refused a packet: {err}", self.debug_name);
            }
        }
        self.drain(output_sender);
    }

    fn reset(&mut self, video: &VideoDataDescription) {
        re_tracing::profile_function!();
        match open_decoder(&self.codec, self.hw) {
            Ok((decoder, hw_used)) => {
                self.decoder = decoder;
                self.hw_used = hw_used;
            }
            Err(err) => re_log::warn_once!("{}: could not reopen libavcodec decoder: {err}", self.debug_name),
        }
        self.bitstream = Bitstream::for_video(video);
        self.pending.clear();
    }

    fn min_num_samples_to_enqueue_ahead(&self) -> usize {
        self.min_num_samples_ahead()
    }
}
