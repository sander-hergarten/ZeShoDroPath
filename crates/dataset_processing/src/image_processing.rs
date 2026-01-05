use image::{GrayImage, ImageBuffer};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

// --- Structs ---

// Updated to match WGSL layout alignment rules
// alpha (4 bytes) + padding (4 bytes) + direction (8 bytes) = 16 bytes
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    alpha: f32,
    _pad: u32,           // Padding to align 'direction' to 8 bytes
    direction: [i32; 2], // vec2<i32>
}

pub struct WgpuProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    // We keep two separate buffers so we don't have to update them mid-frame
    param_buffer_x: wgpu::Buffer,
    param_buffer_y: wgpu::Buffer,
    resources: Option<GpuResources>,
}

struct GpuResources {
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
    input_texture: wgpu::Texture,
    intermediate_texture: wgpu::Texture, // NEW: Temp storage for Pass 1 result
    output_texture: wgpu::Texture,
    output_buffer: wgpu::Buffer,
    bind_group_pass_1: wgpu::BindGroup, // Input -> Intermediate (Horizontal)
    bind_group_pass_2: wgpu::BindGroup, // Intermediate -> Output (Vertical)
}

impl WgpuProcessor {
    pub async fn new(alpha: f32) -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

        // 1. Compile Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("masking_step.wgsl"))),
        });

        // 2. Uniforms: Create two separate buffers for X and Y passes
        let params_x = ShaderParams {
            alpha,
            _pad: 0,
            direction: [1, 0], // Horizontal
        };
        let param_buffer_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params X"),
            contents: bytemuck::cast_slice(&[params_x]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let params_y = ShaderParams {
            alpha,
            _pad: 0,
            direction: [0, 1], // Vertical
        };
        let param_buffer_y = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Y"),
            contents: bytemuck::cast_slice(&[params_y]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // 3. Bind Group Layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // Binding 0: Source Texture (texture_2d<f32>)
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
                // Binding 1: Destination Storage (texture_storage_2d<rgba8unorm, write>)
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
            param_buffer_x,
            param_buffer_y,
            resources: None,
        }
    }

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

            // 1. Upload Input Image
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

            // 2. Execute Two-Pass Blur
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_pipeline(&self.pipeline);

                let dispatch_x = width.div_ceil(16);
                let dispatch_y = height.div_ceil(16);

                // --- PASS 1: Horizontal (Input -> Intermediate) ---
                cpass.set_bind_group(0, &res.bind_group_pass_1, &[]);
                cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

                // --- PASS 2: Vertical (Intermediate -> Output) ---
                // Note: We need a memory barrier here implicitly, but wgpu handles
                // texture usage hazards between dispatches automatically in most cases.
                cpass.set_bind_group(0, &res.bind_group_pass_2, &[]);
                cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            // 3. Copy Result to Readback Buffer
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

            // 4. Readback
            let buffer_slice = res.output_buffer.slice(..);
            let (sender, mut receiver) = futures::channel::oneshot::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            loop {
                self.device.poll(wgpu::PollType::Poll).unwrap();
                if let Ok(Some(_)) = receiver.try_recv() {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(100));
            }

            let data = buffer_slice.get_mapped_range();
            let mut final_bytes = Vec::with_capacity((width * height) as usize);

            for row in 0..height {
                let start = (row * res.padded_bytes_per_row) as usize;
                let row_data = &data[start..start + (width * 4) as usize];
                // Extract R channel from RGBA
                for pixel in row_data.chunks(4) {
                    final_bytes.push(pixel[0]);
                }
            }
            drop(data);
            res.output_buffer.unmap();

            results.push(ImageBuffer::from_raw(width, height, final_bytes).unwrap());
        }

        results
    }

    fn allocate_resources(&mut self, width: u32, height: u32) {
        println!("(Re)Allocating GPU resources for {}x{}", width, height);

        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        // 1. Input Texture (R8Unorm)
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

        // 2. Intermediate Texture (Rgba8Unorm - must support storage write)
        // This holds the result of the Horizontal pass
        let intermediate_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Intermediate"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            // Needs STORAGE for Pass 1 output, and TEXTURE_BINDING for Pass 2 input
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // 3. Final Output Texture
        let output_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // Create Views
        let input_view = input_texture.create_view(&Default::default());
        let intermediate_view = intermediate_texture.create_view(&Default::default());
        let output_view = output_texture.create_view(&Default::default());

        // --- Bind Group 1: Horizontal Pass ---
        // Input: input_texture | Output: intermediate_texture | Params: param_buffer_x
        let bind_group_pass_1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group Pass 1 (Horiz)"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&intermediate_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.param_buffer_x.as_entire_binding(),
                },
            ],
        });

        // --- Bind Group 2: Vertical Pass ---
        // Input: intermediate_texture | Output: output_texture | Params: param_buffer_y
        let bind_group_pass_2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group Pass 2 (Vert)"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    // The output of pass 1 is the input of pass 2
                    resource: wgpu::BindingResource::TextureView(&intermediate_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.param_buffer_y.as_entire_binding(),
                },
            ],
        });

        // Buffer for Readback
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
            intermediate_texture,
            output_texture,
            output_buffer,
            bind_group_pass_1,
            bind_group_pass_2,
        });
    }
}
