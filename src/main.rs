use std::ffi::{c_char, CStr};

use anyhow::{anyhow, Result};
use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    vk::{
        make_api_version, ApplicationInfo, DeviceCreateInfo, DeviceQueueCreateInfo,
        InstanceCreateInfo, PhysicalDevice, PhysicalDeviceFeatures, PhysicalDeviceType, Queue,
        QueueFlags, SurfaceKHR,
    },
    Device, Entry, Instance,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

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
    graphics_queue: Queue,
    present_queue: Queue,
    surface: Surface,
    surface_khr: SurfaceKHR,
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

        let layer_names =
            [unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") }];
        let layers_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        let mut extension_names =
            ash_window::enumerate_required_extensions(window.raw_display_handle())
                .unwrap()
                .to_vec();
        extension_names.push(DebugUtils::name().as_ptr());

        // Create instance
        println!("Creating instance");

        let instance_create_info = InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(&layers_names_raw)
            .enabled_extension_names(&extension_names)
            .build();

        let instance = unsafe { entry.create_instance(&instance_create_info, None)? };

        // Create surface
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

        // Find physical device and queue family
        println!("Creating physical device and finding queue family index");

        let (physical_device, queue_families) = unsafe { instance.enumerate_physical_devices()? }
            .into_iter()
            .find_map(|device| is_device_suitable(&instance, device, &surface, surface_khr))
            .ok_or_else(|| anyhow!("Could not find suitable physical device"))?;

        // Create logical device and queues
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

        let device_create_info = DeviceCreateInfo::builder()
            .queue_create_infos(&device_queue_create_infos)
            .enabled_features(&device_features)
            .build();

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };

        println!("Retrieving device queues");

        let graphics_queue = unsafe { device.get_device_queue(queue_families.graphics_queue, 0) };
        let present_queue = unsafe { device.get_device_queue(queue_families.present_queue, 0) };

        Ok(Self {
            instance,
            device,
            graphics_queue,
            present_queue,
            surface,
            surface_khr,
        })
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        unsafe {
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

fn is_device_suitable(
    instance: &Instance,
    device: PhysicalDevice,
    surface: &Surface,
    surface_khr: SurfaceKHR,
) -> Option<(PhysicalDevice, QueueFamilies)> {
    let properties = unsafe { instance.get_physical_device_properties(device) };
    let features = unsafe { instance.get_physical_device_features(device) };

    if properties.device_type != PhysicalDeviceType::DISCRETE_GPU || features.geometry_shader == 0 {
        return None;
    }

    if !check_device_extension_support(instance, device).unwrap_or(false) {
        return None;
    }

    let queue_families = QueueFamilies::new(instance, device, surface, surface_khr)?;

    Some((device, queue_families))
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
