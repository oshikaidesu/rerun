use ahash::{HashMap, HashSet};
use re_mutex::Mutex;

use super::ImageDataToTextureError;
use super::image_data_to_texture::transfer_image_data_to_texture;
use crate::RenderContext;
use crate::resource_managers::ImageDataDesc;
use crate::wgpu_resources::{GpuTexture, GpuTexturePool, TextureDesc};

/// What is known about the alpha channel usage of a [`GpuTexture2D`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaChannelUsage {
    /// It is not known whether the alpha channel is in use.
    DontKnow,

    /// Either the texture format has no alpha channel,
    /// or the alpha channel is known to be set to 1.0 everywhere (fully opaque).
    Opaque,

    /// The alpha channel is known to contain values less than 1.0.
    AlphaChannelInUse,
}

/// Handle to a 2D resource.
///
/// Currently, this is solely a more strongly typed regular gpu texture handle.
#[derive(Clone)]
pub struct GpuTexture2D {
    texture: GpuTexture,
    alpha_channel_usage: AlphaChannelUsage,
}

impl std::fmt::Debug for GpuTexture2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            texture,
            alpha_channel_usage,
        } = self;
        f.debug_struct("GpuTexture2D")
            .field("handle", &texture.handle)
            .field("alpha_channel_usage", alpha_channel_usage)
            .finish()
    }
}

impl GpuTexture2D {
    /// Returns `None` if the `texture` is not 2D.
    pub fn new(texture: GpuTexture, alpha_channel_usage: AlphaChannelUsage) -> Option<Self> {
        if texture.texture.dimension() != wgpu::TextureDimension::D2 {
            return None;
        }

        let has_alpha_channel = texture_format_has_alpha_channel(texture.texture.format());

        let alpha_channel_usage = if has_alpha_channel {
            alpha_channel_usage
        } else {
            re_log::debug_assert!(
                alpha_channel_usage != AlphaChannelUsage::AlphaChannelInUse,
                "alpha_channel_usage is AlphaChannelInUse but texture format {:?} has no alpha channel",
                texture.texture.format()
            );

            AlphaChannelUsage::Opaque
        };

        Some(Self {
            texture,
            alpha_channel_usage,
        })
    }

    #[inline]
    pub fn handle(&self) -> crate::wgpu_resources::GpuTextureHandle {
        self.texture.handle
    }

    /// What is known about the alpha channel state of this texture.
    #[inline]
    pub fn alpha_channel_usage(&self) -> AlphaChannelUsage {
        self.alpha_channel_usage
    }

    /// Width of the texture.
    #[inline]
    pub fn width(&self) -> u32 {
        self.texture.texture.width()
    }

    /// Height of the texture.
    #[inline]
    pub fn height(&self) -> u32 {
        self.texture.texture.height()
    }

    /// Width and height of the texture.
    #[inline]
    pub fn width_height(&self) -> [u32; 2] {
        [self.width(), self.height()]
    }

    #[inline]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.texture.texture.format()
    }
}

impl AsRef<GpuTexture> for GpuTexture2D {
    #[inline(always)]
    fn as_ref(&self) -> &GpuTexture {
        &self.texture
    }
}

impl std::ops::Deref for GpuTexture2D {
    type Target = GpuTexture;

    #[inline(always)]
    fn deref(&self) -> &GpuTexture {
        &self.texture
    }
}

impl std::borrow::Borrow<GpuTexture> for GpuTexture2D {
    #[inline(always)]
    fn borrow(&self) -> &GpuTexture {
        &self.texture
    }
}

#[derive(thiserror::Error, Debug)]
pub enum TextureManager2DError<DataCreationError> {
    /// Something went wrong when creating the GPU texture & uploading/converting the image data.
    #[error(transparent)]
    ImageDataToTextureError(#[from] ImageDataToTextureError),

    /// Something went wrong in a user-callback.
    #[error(transparent)]
    DataCreation(DataCreationError),
}

impl From<TextureManager2DError<never::Never>> for ImageDataToTextureError {
    fn from(err: TextureManager2DError<never::Never>) -> Self {
        match err {
            TextureManager2DError::ImageDataToTextureError(texture_creation) => texture_creation,
            TextureManager2DError::DataCreation(never) => match never {},
        }
    }
}

/// Texture manager for 2D textures.
///
/// The scope is intentionally limited to particular kinds of textures that currently
/// require this kind of handle abstraction/management.
/// More complex textures types are typically handled within renderer which utilize the texture pool directly.
/// This manager in contrast, deals with user provided texture data!
/// We might revisit this later and make this texture manager more general purpose.
///
/// Has intertior mutability.
pub struct TextureManager2D {
    white_texture_unorm: GpuTexture2D,
    zeroed_texture_float: GpuTexture2D,
    zeroed_texture_sint: GpuTexture2D,
    zeroed_texture_uint: GpuTexture2D,

    /// The mutable part of the manager.
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    /// Caches textures using a unique id, which in practice is the hash of the
    /// row id of the tensor data (`tensor_data_row_id`).
    ///
    /// Any texture which wasn't accessed on the previous frame is ejected from the cache
    /// during [`Self::begin_frame`].
    texture_cache: HashMap<u64, GpuTexture2D>,

    /// Textures supplied by an embedder whose RGB channels are already multiplied by alpha.
    premultiplied_texture_keys: HashSet<u64>,

    /// Draws the mip chain for [`TextureManager2D::create_with_mipmaps`]. Built on first use.
    mipmaps: Option<super::MipmapGenerator>,

    /// Textures owned by an embedder and sampled in place, keyed like [`Self::texture_cache`].
    ///
    /// Held separately so that cache eviction never leaves the pool as sole owner: reclamation
    /// destroys the underlying `wgpu::Texture`, which here belongs to the embedder.
    imported_textures: HashMap<u64, GpuTexture2D>,

    accessed_textures: HashSet<u64>,
}

impl Inner {
    fn begin_frame(&mut self, _frame_index: u64) {
        // Drop any textures that weren't accessed in the last frame
        self.texture_cache
            .retain(|k, _| self.accessed_textures.contains(k));
        self.premultiplied_texture_keys
            .retain(|k| self.texture_cache.contains_key(k));
        // Let go of imports the embedder stopped handing over. Safe to do implicitly: this only
        // drops our reference, never the embedder's texture.
        self.imported_textures
            .retain(|k, _| self.texture_cache.contains_key(k));
        self.accessed_textures.clear();
    }
}

impl TextureManager2D {
    /// Samples an existing GPU-resident image in place, without allocating or copying.
    ///
    /// The zero-copy counterpart to [`Self::copy_from_gpu_premultiplied`]. The embedder keeps
    /// ownership of `source` and must not write to it while a frame that samples it is in flight,
    /// which in practice means alternating between two textures per layer.
    pub fn import_gpu_premultiplied(
        &self,
        key: u64,
        render_ctx: &RenderContext,
        source: &wgpu::Texture,
    ) -> Result<GpuTexture2D, ExternalGpuTextureError> {
        if source.dimension() != wgpu::TextureDimension::D2 {
            return Err(ExternalGpuTextureError::Not2D);
        }

        let size = source.size();
        if size.width == 0 || size.height == 0 || size.depth_or_array_layers != 1 {
            return Err(ExternalGpuTextureError::InvalidSize(size));
        }

        let mut inner = self.inner.lock();

        // Re-imported on every call rather than cached by `key`, because a double-buffered
        // embedder alternates between two textures that share one key and one descriptor: there is
        // nothing in a `wgpu::Texture` to tell them apart, so the only safe answer is to take
        // whichever one was handed over now. Dropping the previous import is free (see
        // `GpuTexturePool::import`); the cost of a repeat is one texture view.
        let imported = render_ctx.gpu_resources.textures.import(
            source.clone(),
            &TextureDesc {
                label: format!("imported premultiplied image {key:016x}").into(),
                size,
                mip_level_count: source.mip_level_count(),
                sample_count: source.sample_count(),
                dimension: wgpu::TextureDimension::D2,
                format: source.format(),
                usage: source.usage(),
            },
        );
        let texture = GpuTexture2D::new(imported, AlphaChannelUsage::AlphaChannelInUse)
            .expect("imported image texture is 2D");

        inner.imported_textures.insert(key, texture.clone());
        inner.texture_cache.insert(key, texture.clone());
        inner.premultiplied_texture_keys.insert(key);
        inner.accessed_textures.insert(key);

        Ok(texture)
    }

    /// Stops sampling a texture previously handed over by [`Self::import_gpu_premultiplied`].
    ///
    /// Call this before the embedder drops its own texture. Returns whether there was one.
    pub fn release_imported(&self, key: u64) -> bool {
        let mut inner = self.inner.lock();
        let released = inner.imported_textures.remove(&key).is_some();
        if released {
            inner.texture_cache.remove(&key);
            inner.premultiplied_texture_keys.remove(&key);
            inner.accessed_textures.remove(&key);
        }
        released
    }

    /// Copies an existing GPU-resident image into the texture cache used by image visualizers.
    ///
    /// This is intended for embedders which already rendered an image on the same device.
    /// No pixels are read back to the CPU.
    pub fn copy_from_gpu_premultiplied(
        &self,
        key: u64,
        render_ctx: &RenderContext,
        source: &wgpu::Texture,
    ) -> Result<GpuTexture2D, ExternalGpuTextureError> {
        if source.dimension() != wgpu::TextureDimension::D2 {
            return Err(ExternalGpuTextureError::Not2D);
        }

        let size = source.size();
        if size.width == 0 || size.height == 0 || size.depth_or_array_layers != 1 {
            return Err(ExternalGpuTextureError::InvalidSize(size));
        }

        let mut inner = self.inner.lock();
        let texture = match inner.texture_cache.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                let texture = entry.get();
                if texture.width_height() != [size.width, size.height]
                    || texture.format() != source.format()
                {
                    return Err(ExternalGpuTextureError::DescriptorMismatch);
                }
                texture.clone()
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let texture = render_ctx.gpu_resources.textures.alloc(
                    &render_ctx.device,
                    &TextureDesc {
                        label: "external premultiplied image".into(),
                        size,
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: source.format(),
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_DST
                            | wgpu::TextureUsages::COPY_SRC,
                    },
                );
                entry
                    .insert(
                        GpuTexture2D::new(texture, AlphaChannelUsage::AlphaChannelInUse)
                            .expect("external image texture is 2D"),
                    )
                    .clone()
            }
        };

        let mut encoder =
            render_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("copy external premultiplied image"),
                });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &texture.texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            size,
        );
        render_ctx.queue.submit([encoder.finish()]);

        inner.premultiplied_texture_keys.insert(key);
        inner.accessed_textures.insert(key);
        Ok(texture)
    }

    /// Whether the cached image identified by `key` already has premultiplied alpha.
    pub fn is_premultiplied(&self, key: u64) -> bool {
        self.inner.lock().premultiplied_texture_keys.contains(&key)
    }

    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_pool: &GpuTexturePool,
    ) -> Self {
        re_tracing::profile_function!();

        // Create the single pixel white texture ad hoc - at this point during initialization we don't have
        // the render context yet and thus can't use the higher level `transfer_image_data_to_texture` function.
        let white_texture_unorm = GpuTexture2D {
            alpha_channel_usage: AlphaChannelUsage::Opaque,
            texture: texture_pool.alloc(
                device,
                &TextureDesc {
                    label: "white pixel - unorm".into(),
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                },
            ),
        };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &white_texture_unorm.texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let zeroed_texture_float =
            create_zero_texture(texture_pool, device, wgpu::TextureFormat::Rgba8Unorm);
        let zeroed_texture_sint =
            create_zero_texture(texture_pool, device, wgpu::TextureFormat::Rgba8Sint);
        let zeroed_texture_uint =
            create_zero_texture(texture_pool, device, wgpu::TextureFormat::Rgba8Uint);

        Self {
            white_texture_unorm,
            zeroed_texture_float,
            zeroed_texture_sint,
            zeroed_texture_uint,
            inner: Default::default(),
        }
    }

    /// Creates a new 2D texture resource and schedules data upload to the GPU.
    /// TODO(jleibs): All usages of this should be replaced with `get_or_create`, which is strictly preferable
    #[expect(clippy::unused_self)]
    pub fn create(
        &self,
        render_ctx: &RenderContext,
        creation_desc: ImageDataDesc<'_>,
    ) -> Result<GpuTexture2D, ImageDataToTextureError> {
        // TODO(andreas): Disabled the warning as we're moving towards using this texture manager for user-logged images.
        // However, it's still very much a concern especially once we add mipmapping. Something we need to keep in mind.
        //
        // if !resource.width.is_power_of_two() || !resource.width.is_power_of_two() {
        //     re_log::warn!(
        //         "Texture {:?} has the non-power-of-two (NPOT) resolution of {}x{}. \
        //         NPOT textures are slower and on WebGL can't handle mipmapping, UV wrapping and UV tiling",
        //         resource.label,
        //         resource.width,
        //         resource.height
        //     );
        // }

        // Currently we don't store any data in the texture manager.
        // In the future we might handle (lazy?) mipmap generation in here or keep track of lazy upload processing.

        let alpha_channel_usage = creation_desc.alpha_channel_usage;
        let texture = creation_desc.create_target_texture(
            render_ctx,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        );
        transfer_image_data_to_texture(render_ctx, creation_desc, &texture)?;
        Ok(GpuTexture2D::new(texture, alpha_channel_usage).expect("Texture is known to be 2D"))
    }

    /// Like [`Self::create`], but allocates the full mip chain and draws it on the GPU right
    /// after the upload (same frame encoder, so ordering holds). Level 0 is the image data.
    pub fn create_with_mipmaps(
        &self,
        render_ctx: &RenderContext,
        creation_desc: ImageDataDesc<'_>,
    ) -> Result<GpuTexture2D, ImageDataToTextureError> {
        let alpha_channel_usage = creation_desc.alpha_channel_usage;
        let [width, height] = creation_desc.width_height;
        let texture = render_ctx.gpu_resources.textures.alloc(
            &render_ctx.device,
            &TextureDesc {
                label: creation_desc.label.clone(),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: super::MipmapGenerator::mip_level_count(width, height),
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: creation_desc.target_texture_format(),
                usage: creation_desc.target_texture_usage_requirements()
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
            },
        );
        transfer_image_data_to_texture(render_ctx, creation_desc, &texture)?;
        let mut inner = self.inner.lock();
        let generator = inner.mipmaps.get_or_insert_with(|| super::MipmapGenerator::new(&render_ctx.device));
        let mut encoder = render_ctx.active_frame.before_view_builder_encoder.lock();
        generator.generate(&render_ctx.device, encoder.get(), &texture.texture);
        Ok(GpuTexture2D::new(texture, alpha_channel_usage).expect("Texture is known to be 2D"))
    }

    /// Creates a new 2D texture resource and schedules data upload to the GPU if a texture
    /// wasn't already created using the same key.
    pub fn get_or_create(
        &self,
        key: u64,
        render_ctx: &RenderContext,
        texture_desc: ImageDataDesc<'_>,
    ) -> Result<GpuTexture2D, ImageDataToTextureError> {
        self.get_or_create_with(key, render_ctx, || texture_desc)
    }

    /// Creates a new 2D texture resource and schedules data upload to the GPU if a texture
    /// wasn't already created using the same key.
    pub fn get_or_create_with<'a>(
        &self,
        key: u64,
        render_ctx: &RenderContext,
        create_texture_desc: impl FnOnce() -> ImageDataDesc<'a>,
    ) -> Result<GpuTexture2D, ImageDataToTextureError> {
        self.get_or_try_create_with(key, render_ctx, || -> Result<_, never::Never> {
            Ok(create_texture_desc())
        })
        .map_err(|err| err.into())
    }

    /// Creates a new 2D texture resource and schedules data upload to the GPU if a texture
    /// wasn't already created using the same key.
    pub fn get_or_try_create_with<'a, Err: std::fmt::Display>(
        &self,
        key: u64,
        render_ctx: &RenderContext,
        try_create_texture_desc: impl FnOnce() -> Result<ImageDataDesc<'a>, Err>,
    ) -> Result<GpuTexture2D, TextureManager2DError<Err>> {
        let mut inner = self.inner.lock();
        let texture_handle = match inner.texture_cache.entry(key) {
            std::collections::hash_map::Entry::Occupied(texture_handle) => {
                texture_handle.get().clone() // already inserted
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                // Run potentially expensive texture creation code:
                let tex_creation_desc = try_create_texture_desc()
                    .map_err(|err| TextureManager2DError::DataCreation(err))?;

                let alpha_channel_usage = tex_creation_desc.alpha_channel_usage;
                let texture = tex_creation_desc.create_target_texture(
                    render_ctx,
                    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                );
                transfer_image_data_to_texture(render_ctx, tex_creation_desc, &texture)?;
                entry
                    .insert(GpuTexture2D {
                        texture,
                        alpha_channel_usage,
                    })
                    .clone()
            }
        };

        inner.accessed_textures.insert(key);
        Ok(texture_handle)
    }

    /// Returns a single pixel white pixel with an rgba8unorm format.
    pub fn white_texture_unorm_handle(&self) -> &GpuTexture2D {
        &self.white_texture_unorm
    }

    /// Returns a single pixel white pixel with an rgba8unorm format.
    pub fn white_texture_unorm(&self) -> &GpuTexture {
        &self.white_texture_unorm.texture
    }

    /// Returns a single zero pixel with format [`wgpu::TextureFormat::Rgba8Unorm`].
    pub fn zeroed_texture_float(&self) -> &GpuTexture {
        &self.zeroed_texture_float.texture
    }

    /// Returns a single zero pixel with format [`wgpu::TextureFormat::Rgba8Sint`].
    pub fn zeroed_texture_sint(&self) -> &GpuTexture {
        &self.zeroed_texture_sint.texture
    }

    /// Returns a single zero pixel with format [`wgpu::TextureFormat::Rgba8Uint`].
    pub fn zeroed_texture_uint(&self) -> &GpuTexture {
        &self.zeroed_texture_uint.texture
    }

    pub(crate) fn begin_frame(&self, _frame_index: u64) {
        self.inner.lock().begin_frame(_frame_index);
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ExternalGpuTextureError {
    #[error("external GPU image must be a 2D texture")]
    Not2D,
    #[error("external GPU image has invalid size {0:?}")]
    InvalidSize(wgpu::Extent3d),
    #[error("external GPU image descriptor changed without changing its cache key")]
    DescriptorMismatch,
}

/// Returns whether the given [`wgpu::TextureFormat`] has an alpha channel.
fn texture_format_has_alpha_channel(format: wgpu::TextureFormat) -> bool {
    // As of writing, the set of formats with four channels is identical to the set of formats with alpha.
    // But for all we know this may change in the future, so let's be on the safe side.
    #[expect(clippy::match_same_arms)]
    match format {
        // Uncompressed color formats with alpha
        wgpu::TextureFormat::Rgba8Unorm
        | wgpu::TextureFormat::Rgba8UnormSrgb
        | wgpu::TextureFormat::Rgba8Snorm
        | wgpu::TextureFormat::Rgba8Uint
        | wgpu::TextureFormat::Rgba8Sint
        | wgpu::TextureFormat::Bgra8Unorm
        | wgpu::TextureFormat::Bgra8UnormSrgb
        | wgpu::TextureFormat::Rgb10a2Uint
        | wgpu::TextureFormat::Rgb10a2Unorm
        | wgpu::TextureFormat::Rgba16Uint
        | wgpu::TextureFormat::Rgba16Sint
        | wgpu::TextureFormat::Rgba16Unorm
        | wgpu::TextureFormat::Rgba16Snorm
        | wgpu::TextureFormat::Rgba16Float
        | wgpu::TextureFormat::Rgba32Uint
        | wgpu::TextureFormat::Rgba32Sint
        | wgpu::TextureFormat::Rgba32Float => true,

        // Compressed formats with alpha
        wgpu::TextureFormat::Bc1RgbaUnorm
        | wgpu::TextureFormat::Bc1RgbaUnormSrgb
        | wgpu::TextureFormat::Bc2RgbaUnorm
        | wgpu::TextureFormat::Bc2RgbaUnormSrgb
        | wgpu::TextureFormat::Bc3RgbaUnorm
        | wgpu::TextureFormat::Bc3RgbaUnormSrgb
        | wgpu::TextureFormat::Bc7RgbaUnorm
        | wgpu::TextureFormat::Bc7RgbaUnormSrgb
        | wgpu::TextureFormat::Etc2Rgb8A1Unorm
        | wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb
        | wgpu::TextureFormat::Etc2Rgba8Unorm
        | wgpu::TextureFormat::Etc2Rgba8UnormSrgb
        | wgpu::TextureFormat::Astc { .. } => true,

        // 1- and 2-channel formats (no alpha)
        wgpu::TextureFormat::R8Unorm
        | wgpu::TextureFormat::R8Snorm
        | wgpu::TextureFormat::R8Uint
        | wgpu::TextureFormat::R8Sint
        | wgpu::TextureFormat::R16Uint
        | wgpu::TextureFormat::R16Sint
        | wgpu::TextureFormat::R16Unorm
        | wgpu::TextureFormat::R16Snorm
        | wgpu::TextureFormat::R16Float
        | wgpu::TextureFormat::Rg8Unorm
        | wgpu::TextureFormat::Rg8Snorm
        | wgpu::TextureFormat::Rg8Uint
        | wgpu::TextureFormat::Rg8Sint
        | wgpu::TextureFormat::R32Uint
        | wgpu::TextureFormat::R32Sint
        | wgpu::TextureFormat::R32Float
        | wgpu::TextureFormat::Rg16Uint
        | wgpu::TextureFormat::Rg16Sint
        | wgpu::TextureFormat::Rg16Unorm
        | wgpu::TextureFormat::Rg16Snorm
        | wgpu::TextureFormat::Rg16Float
        | wgpu::TextureFormat::R64Uint
        | wgpu::TextureFormat::Rg32Uint
        | wgpu::TextureFormat::Rg32Sint
        | wgpu::TextureFormat::Rg32Float => false,

        // Packed formats without alpha
        wgpu::TextureFormat::Rgb9e5Ufloat | wgpu::TextureFormat::Rg11b10Ufloat => false,

        // Depth/stencil formats
        wgpu::TextureFormat::Stencil8
        | wgpu::TextureFormat::Depth16Unorm
        | wgpu::TextureFormat::Depth24Plus
        | wgpu::TextureFormat::Depth24PlusStencil8
        | wgpu::TextureFormat::Depth32Float
        | wgpu::TextureFormat::Depth32FloatStencil8 => false,

        // Video formats
        wgpu::TextureFormat::NV12 | wgpu::TextureFormat::P010 => false,

        // Compressed formats without alpha
        wgpu::TextureFormat::Bc4RUnorm
        | wgpu::TextureFormat::Bc4RSnorm
        | wgpu::TextureFormat::Bc5RgUnorm
        | wgpu::TextureFormat::Bc5RgSnorm
        | wgpu::TextureFormat::Bc6hRgbUfloat
        | wgpu::TextureFormat::Bc6hRgbFloat
        | wgpu::TextureFormat::Etc2Rgb8Unorm
        | wgpu::TextureFormat::Etc2Rgb8UnormSrgb
        | wgpu::TextureFormat::EacR11Unorm
        | wgpu::TextureFormat::EacR11Snorm
        | wgpu::TextureFormat::EacRg11Unorm
        | wgpu::TextureFormat::EacRg11Snorm => false,
    }
}

fn create_zero_texture(
    texture_pool: &GpuTexturePool,
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> GpuTexture2D {
    // Wgpu zeros out new textures automatically
    GpuTexture2D {
        alpha_channel_usage: if texture_format_has_alpha_channel(format) {
            AlphaChannelUsage::AlphaChannelInUse
        } else {
            AlphaChannelUsage::Opaque
        },
        texture: texture_pool.alloc(
            device,
            &TextureDesc {
                label: format!("zeroed pixel {format:?}").into(),
                format,
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
            },
        ),
    }
}
