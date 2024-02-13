use std::ffi::CStr;

use anyhow::{anyhow, Result};
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    vk::{
        make_api_version, ApplicationInfo, BlendFactor, BlendOp, ColorComponentFlags,
        ColorSpaceKHR, ComponentMapping, ComponentSwizzle, CompositeAlphaFlagsKHR, CullModeFlags,
        DeviceCreateInfo, DeviceQueueCreateInfo, DynamicState, Extent2D, Format, FrontFace, Image,
        ImageAspectFlags, ImageSubresourceRange, ImageUsageFlags, ImageView, ImageViewCreateInfo,
        ImageViewType, InstanceCreateInfo, LogicOp, Offset2D, PhysicalDevice,
        PhysicalDeviceFeatures, PhysicalDeviceType, PipelineColorBlendAttachmentState,
        PipelineColorBlendStateCreateInfo, PipelineDynamicStateCreateInfo,
        PipelineInputAssemblyStateCreateInfo, PipelineLayout, PipelineLayoutCreateInfo,
        PipelineRasterizationStateCreateInfo, PipelineShaderStageCreateInfo,
        PipelineVertexInputStateCreateInfo, PipelineViewportStateCreateInfo, PolygonMode,
        PresentModeKHR, PrimitiveTopology, Queue, QueueFlags, Rect2D, ShaderModule,
        ShaderModuleCreateInfo, ShaderStageFlags, SharingMode, SurfaceCapabilitiesKHR,
        SurfaceFormatKHR, SurfaceKHR, SwapchainCreateInfoKHR, SwapchainKHR, Viewport,
    },
    Device, Entry, Instance,
};
use once_cell::sync::Lazy;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn include_u32(bytes: &[u8]) -> Vec<u32> {
    (0..bytes.len() / 4)
        .map(|i| i * 4)
        .map(|i| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]))
        .collect()
}

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

static VERTEX_SHADER: Lazy<Vec<u32>> =
    Lazy::new(|| include_u32(include_bytes!("shaders/shader.vert.spv")));
static FRAGMENT_SHADER: Lazy<Vec<u32>> =
    Lazy::new(|| include_u32(include_bytes!("shaders/shader.frag.spv")));

fn main() -> Result<()> {
    let (window, event_loop) = init_window();
    Vulkan::new(&window)?;
    main_loop(window, event_loop);

    Ok(())
}

fn init_window() -> (Window, EventLoop<()>) {
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
        .with_title("Vulkan")
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);

    (window, event_loop)
}

fn main_loop(window: Window, event_loop: EventLoop<()>) {
    event_loop
        .run(move |event, elwt| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    println!("The close button was pressed; stopping");
                    elwt.exit();
                }
                Event::AboutToWait => {
                    // Application update code.

                    // Queue a RedrawRequested event.
                    //
                    // You only need to call this if you've determined that you need to redraw in
                    // applications which do not always need to. Applications that redraw continuously
                    // can render here instead.
                    window.request_redraw();
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    // Redraw the application.
                    //
                    // It's preferable for applications that do not render continuously to render in
                    // this event rather than in AboutToWait, since rendering in here allows
                    // the program to gracefully handle redraws requested by the OS.
                }
                _ => (),
            }
        })
        .unwrap();
}

struct Vulkan {
    instance: Instance,
    device: Device,

    swapchain: Swapchain,
    swapchain_khr: SwapchainKHR,
    swapchain_images: Vec<Image>,
    swapchain_image_views: Vec<ImageView>,

    surface: Surface,
    surface_khr: SurfaceKHR,

    graphics_queue: Queue,
    present_queue: Queue,

    vertex_shader: ShaderModule,
    fragment_shader: ShaderModule,

    pipeline_layout: PipelineLayout,
}

impl Vulkan {
    fn new(window: &Window) -> Result<Self> {
        let entry = Entry::linked();

        println!("Extension properties:");
        entry
            .enumerate_instance_extension_properties(None)?
            .into_iter()
            .for_each(|ep| {
                println!("  - {ep:?}");
            });

        println!("\nLayer properties:");
        entry
            .enumerate_instance_layer_properties()?
            .into_iter()
            .for_each(|lp| {
                println!("  - {lp:?}");
            });

        let app_info = ApplicationInfo::builder()
            .application_name(unsafe { CStr::from_bytes_with_nul_unchecked(b"Vulkan Tutorial\0") })
            .application_version(make_api_version(0, 0, 1, 0))
            .engine_name(unsafe { CStr::from_bytes_with_nul_unchecked(b"No Engine\0") })
            .engine_version(make_api_version(0, 0, 1, 0))
            .api_version(make_api_version(0, 1, 0, 0))
            .build();

        let layer_names = [b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const i8];

        let mut extension_names =
            ash_window::enumerate_required_extensions(window.raw_display_handle())
                .unwrap()
                .to_vec();
        extension_names.push(DebugUtils::name().as_ptr());

        println!("Creating instance");

        let instance_create_info = InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&extension_names)
            .build();

        let instance = unsafe { entry.create_instance(&instance_create_info, None)? };

        let surface = Surface::new(&entry, &instance);

        let surface_khr = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle(),
                window.raw_window_handle(),
                None,
            )
        }?;

        println!("Creating physical device and finding queue family index");

        let (physical_device, queue_families, swapchain_support) =
            unsafe { instance.enumerate_physical_devices()? }
                .into_iter()
                .find_map(|device| is_device_suitable(&instance, device, &surface, surface_khr))
                .ok_or_else(|| anyhow!("Could not find suitable physical device"))?;

        println!("Creating logical device");

        let device_queue_create_info_graphics = DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_families.graphics_queue)
            .queue_priorities(&[1.0f32])
            .build();

        let device_queue_create_info_present = DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_families.present_queue)
            .queue_priorities(&[1.0f32])
            .build();

        let device_queue_create_infos = [
            device_queue_create_info_graphics,
            device_queue_create_info_present,
        ];

        let device_features = PhysicalDeviceFeatures::builder().build();

        let device_extension_names = [Swapchain::name().as_ptr()];

        let device_create_info = DeviceCreateInfo::builder()
            .queue_create_infos(&device_queue_create_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(&device_extension_names)
            .build();

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };

        println!("Retrieving device queues");

        let graphics_queue = unsafe { device.get_device_queue(queue_families.graphics_queue, 0) };
        let present_queue = unsafe { device.get_device_queue(queue_families.present_queue, 0) };

        println!("Creating swap chain");

        let swapchain_create_info = swapchain_support.create_info(&queue_families);

        let swapchain = Swapchain::new(&instance, &device);
        let swapchain_khr = unsafe { swapchain.create_swapchain(&swapchain_create_info, None)? };
        let swapchain_images = unsafe { swapchain.get_swapchain_images(swapchain_khr) }?;

        let image_view_create_info = |image| {
            ImageViewCreateInfo::builder()
                .view_type(ImageViewType::TYPE_2D)
                .format(swapchain_create_info.image_format)
                .components(
                    ComponentMapping::builder()
                        .r(ComponentSwizzle::IDENTITY)
                        .g(ComponentSwizzle::IDENTITY)
                        .b(ComponentSwizzle::IDENTITY)
                        .a(ComponentSwizzle::IDENTITY)
                        .build(),
                )
                .subresource_range(
                    ImageSubresourceRange::builder()
                        .aspect_mask(ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .image(image)
                .build()
        };

        println!("Creating swap chain image views");

        let swapchain_image_views = swapchain_images
            .iter()
            .copied()
            .map(|image| unsafe {
                let image_view_create_info = image_view_create_info(image);
                device
                    .create_image_view(&image_view_create_info, None)
                    .map_err(anyhow::Error::from)
            })
            .collect::<Result<Vec<_>>>()?;

        println!("Creating shader modules");

        let shader_module_create_info_vert = ShaderModuleCreateInfo::builder()
            .code(VERTEX_SHADER.as_ref())
            .build();

        let shader_module_create_info_frag = ShaderModuleCreateInfo::builder()
            .code(FRAGMENT_SHADER.as_ref())
            .build();

        let vertex_shader =
            unsafe { device.create_shader_module(&shader_module_create_info_vert, None) }?;

        let fragment_shader =
            unsafe { device.create_shader_module(&shader_module_create_info_frag, None) }?;

        println!("Creating shader stages");

        let pipeline_shader_stage_create_info_vert = PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::VERTEX)
            .module(vertex_shader)
            .name(unsafe { CStr::from_bytes_with_nul_unchecked(b"main") })
            .build();

        let pipeline_shader_stage_create_info_frag = PipelineShaderStageCreateInfo::builder()
            .stage(ShaderStageFlags::VERTEX)
            .module(vertex_shader)
            .name(unsafe { CStr::from_bytes_with_nul_unchecked(b"main") })
            .build();

        println!("Creating pipeline state");

        let pipeline_dynamic_state_create_info = PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&[DynamicState::VIEWPORT, DynamicState::SCISSOR])
            .build();

        let pipeline_vertex_input_state_create_info = PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&[])
            .vertex_attribute_descriptions(&[])
            .build();

        let pipeline_input_assembly_state_create_info =
            PipelineInputAssemblyStateCreateInfo::builder()
                .topology(PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false)
                .build();

        let viewport = Viewport::builder()
            .x(0.)
            .y(0.)
            .width(swapchain_create_info.image_extent.width as f32)
            .height(swapchain_create_info.image_extent.height as f32)
            .min_depth(0.)
            .max_depth(1.)
            .build();

        let scissor = Rect2D::builder()
            .offset(Offset2D { x: 0, y: 0 })
            .extent(swapchain_create_info.image_extent)
            .build();

        let pipeline_viewport_state_create_info = PipelineViewportStateCreateInfo::builder()
            .viewports(&[viewport])
            .scissors(&[scissor])
            .build();

        let pipeline_rasterization_state_create_info =
            PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .rasterizer_discard_enable(false)
                .polygon_mode(PolygonMode::FILL)
                .line_width(1.)
                .cull_mode(CullModeFlags::BACK)
                .front_face(FrontFace::CLOCKWISE)
                .depth_bias_enable(false)
                .build();

        let pipeline_color_blend_attachment_state = PipelineColorBlendAttachmentState::builder()
            .color_write_mask(ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(BlendOp::ADD)
            .src_alpha_blend_factor(BlendFactor::ONE)
            .dst_alpha_blend_factor(BlendFactor::ZERO)
            .alpha_blend_op(BlendOp::ADD)
            .build();

        let pipeline_color_blend_state_create_info = PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(LogicOp::COPY)
            .attachments(&[pipeline_color_blend_attachment_state])
            .blend_constants([0., 0., 0., 0.])
            .build();

        let pipeline_layout_create_info = PipelineLayoutCreateInfo::builder()
            .set_layouts(&[])
            .push_constant_ranges(&[])
            .build();

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }?;

        Ok(Self {
            instance,
            device,

            swapchain,
            swapchain_khr,
            swapchain_images,
            swapchain_image_views,

            surface,
            surface_khr,

            graphics_queue,
            present_queue,

            vertex_shader,
            fragment_shader,

            pipeline_layout,
        })
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_shader_module(self.vertex_shader, None);
            self.device
                .destroy_shader_module(self.fragment_shader, None);
            self.swapchain_image_views
                .drain(..)
                .for_each(|swapchain_image_view| {
                    self.device.destroy_image_view(swapchain_image_view, None);
                });
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
            self.surface.destroy_surface(self.surface_khr, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None)
        };
    }
}

struct QueueFamilies {
    graphics_queue: u32,
    present_queue: u32,
}

impl QueueFamilies {
    fn new(
        instance: &Instance,
        device: PhysicalDevice,
        surface: &Surface,
        surface_khr: SurfaceKHR,
    ) -> Option<Self> {
        let mut graphics_queue = None;
        let mut present_queue = None;

        for (queue_family_index, info) in
            unsafe { instance.get_physical_device_queue_family_properties(device) }
                .iter()
                .enumerate()
                .map(|(queue_family_index, info)| (queue_family_index as u32, info))
        {
            if info.queue_flags.contains(QueueFlags::GRAPHICS) {
                graphics_queue = Some(queue_family_index);
            }

            if unsafe {
                surface
                    .get_physical_device_surface_support(device, queue_family_index, surface_khr)
                    .unwrap_or(false)
            } {
                present_queue = Some(queue_family_index)
            }
        }

        Some(Self {
            present_queue: present_queue?,
            graphics_queue: graphics_queue?,
        })
    }
}

struct SwapchainSupport {
    capabilities: SurfaceCapabilitiesKHR,
    formats: Vec<SurfaceFormatKHR>,
    present_modes: Vec<PresentModeKHR>,
    surface_khr: SurfaceKHR,
}

impl SwapchainSupport {
    fn new(device: PhysicalDevice, surface: &Surface, surface_khr: SurfaceKHR) -> Option<Self> {
        let capabilities = unsafe {
            surface
                .get_physical_device_surface_capabilities(device, surface_khr)
                .ok()?
        };
        let formats = unsafe {
            surface
                .get_physical_device_surface_formats(device, surface_khr)
                .ok()?
        };
        let present_modes = unsafe {
            surface
                .get_physical_device_surface_present_modes(device, surface_khr)
                .ok()?
        };

        if !formats.is_empty() && !present_modes.is_empty() {
            Some(Self {
                capabilities,
                formats,
                present_modes,
                surface_khr,
            })
        } else {
            None
        }
    }

    fn choose_format(&self) -> SurfaceFormatKHR {
        self.formats
            .iter()
            .copied()
            .find(|format| {
                println!("Format: {format:?}");
                format.format == Format::B8G8R8A8_UNORM
                    && format.color_space == ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(self.formats.first().copied().unwrap())
    }

    fn choose_present_mode(&self) -> PresentModeKHR {
        self.present_modes
            .iter()
            .copied()
            .find(|&present_mode| present_mode == PresentModeKHR::MAILBOX)
            .unwrap_or(PresentModeKHR::FIFO)
    }

    fn choose_swap_extent(&self) -> Extent2D {
        if self.capabilities.current_extent.width == u32::MAX {
            Extent2D {
                width: u32::clamp(
                    WIDTH,
                    self.capabilities.min_image_extent.width,
                    self.capabilities.max_image_extent.width,
                ),
                height: u32::clamp(
                    HEIGHT,
                    self.capabilities.min_image_extent.height,
                    self.capabilities.max_image_extent.height,
                ),
            }
        } else {
            self.capabilities.current_extent
        }
    }

    fn choose_image_count(&self) -> u32 {
        u32::clamp(
            self.capabilities.min_image_count + 1,
            self.capabilities.min_image_count,
            self.capabilities.max_image_count,
        )
    }

    fn create_info(&self, queue_families: &QueueFamilies) -> SwapchainCreateInfoKHR {
        // Look into VK_IMAGE_USAGE_TRANSFER_DST_BIT for compositing

        let is_same_queue = queue_families.present_queue == queue_families.graphics_queue;

        let format = self.choose_format();
        SwapchainCreateInfoKHR::builder()
            .min_image_count(self.choose_image_count())
            .present_mode(self.choose_present_mode())
            .image_extent(self.choose_swap_extent())
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_array_layers(1)
            .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(if is_same_queue {
                SharingMode::CONCURRENT
            } else {
                SharingMode::EXCLUSIVE
            })
            .queue_family_indices(&[queue_families.present_queue, queue_families.graphics_queue])
            .pre_transform(self.capabilities.current_transform)
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .surface(self.surface_khr)
            .clipped(true)
            .build()
    }
}

fn is_device_suitable(
    instance: &Instance,
    device: PhysicalDevice,
    surface: &Surface,
    surface_khr: SurfaceKHR,
) -> Option<(PhysicalDevice, QueueFamilies, SwapchainSupport)> {
    let properties = unsafe { instance.get_physical_device_properties(device) };
    let features = unsafe { instance.get_physical_device_features(device) };

    if properties.device_type != PhysicalDeviceType::DISCRETE_GPU || features.geometry_shader == 0 {
        return None;
    }

    if !check_device_extension_support(instance, device).unwrap_or(false) {
        return None;
    }

    let queue_families = QueueFamilies::new(instance, device, surface, surface_khr)?;
    let swapchain_support = SwapchainSupport::new(device, surface, surface_khr)?;

    Some((device, queue_families, swapchain_support))
}

fn check_device_extension_support(instance: &Instance, device: PhysicalDevice) -> Result<bool> {
    Ok(
        unsafe { instance.enumerate_device_extension_properties(device)? }
            .into_iter()
            .any(|extension| unsafe {
                let ext_name = CStr::from_ptr(extension.extension_name.as_ptr());
                ext_name == Swapchain::name()
            }),
    )
}
