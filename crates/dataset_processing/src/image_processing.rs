use image::{GrayImage, ImageBuffer};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

// --- Structs ---

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    alpha: f32,
    _padding: [f32; 3],
}

pub struct WgpuProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    param_buffer: wgpu::Buffer,
    resources: Option<GpuResources>,
}

struct GpuResources {
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    input_texture: wgpu::Texture,
    output_texture: wgpu::Texture,
    output_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl WgpuProcessor {
    pub async fn new(alpha: f32) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

        // 1. Compile Shader & Pipeline Layout (Done once)
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("masking_step.wgsl"))),
        });

        // 2. Uniforms (Done once)
        let params = ShaderParams {
            alpha,
            _padding: [0.0; 3],
        };
        let param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 3. Bind Group Layout (Defines the interface, not the specific textures)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Binding 0: Input Texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 1: Output Storage (RGBA8)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Binding 2: Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: RGB Image Texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            param_buffer,
            resources: None,
        }
    }

    /// Process a batch of images sequentially, reusing GPU resources where possible.
    pub fn process_batch(&mut self, masks: &[GrayImage]) -> Vec<GrayImage> {
        let mut results = Vec::with_capacity(masks.len());

        for mask_img in masks {
            let (width, height) = mask_img.dimensions();

            let need_new_resources = match &self.resources {
                Some(res) => res.width != width || res.height != height,
                None => true,
            };

            if need_new_resources {
                self.allocate_resources(width, height);
            }

            let res = self.resources.as_ref().unwrap();

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &res.input_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                mask_img,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, &res.bind_group, &[]);
                cpass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
            }

            // --- 4. Copy to Buffer ---
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &res.output_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &res.output_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(res.padded_bytes_per_row),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );

            self.queue.submit(Some(encoder.finish()));

            // --- 5. Map and Read (Polling Loop) ---
            let buffer_slice = res.output_buffer.slice(..);
            let (sender, mut receiver) = futures::channel::oneshot::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            loop {
                self.device.poll(wgpu::PollType::Poll).unwrap();
                if let Ok(Some(_)) = receiver.try_recv() {
                    break;
                }
                // Tiny sleep to prevent 100% CPU usage
                std::thread::sleep(std::time::Duration::from_micros(100));
            }

            let data = buffer_slice.get_mapped_range();
            let mut final_bytes = Vec::with_capacity((width * height * 3) as usize);

            for row in 0..height {
                let start = (row * res.padded_bytes_per_row) as usize;
                // Read row (RGBA pixels)
                let row_data = &data[start..start + (width * 4) as usize];
                // Extract RGB channels (skip alpha)
                for pixel in row_data.chunks(4) {
                    final_bytes.push(pixel[0]); // R
                    final_bytes.push(pixel[1]); // G
                    final_bytes.push(pixel[2]); // B
                }
            }
            drop(data);
            res.output_buffer.unmap();

            results.push(ImageBuffer::from_raw(width, height, final_bytes).unwrap());
        }

        results
    }

    /// Helper to allocate textures and bind groups
    fn allocate_resources(&mut self, width: u32, height: u32) {
        println!("(Re)Allocating GPU resources for {}x{}", width, height);

        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let input_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Input"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let image_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RGB Image"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let output_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm, // Still using the RGBA hack
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &input_texture.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &output_texture.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.param_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &image_texture.create_view(&Default::default()),
                    ),
                },
            ],
        });

        // Calculate padding for readback
        let unpadded_bytes_per_row = width * 4;
        let align = 256;
        let padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padding;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback"),
            size: (padded_bytes_per_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        self.resources = Some(GpuResources {
            width,
            height,
            padded_bytes_per_row,
            input_texture,
            output_texture,
            output_buffer,
            bind_group,
        });
    }
}
