use std::ffi::{c_char, CStr};

use anyhow::{anyhow, Result};
use ash::{
    extensions::{ext::DebugUtils, khr::Swapchain},
    vk::{
        make_api_version, ApplicationInfo, DeviceCreateInfo, DeviceQueueCreateInfo,
        InstanceCreateInfo, PhysicalDevice, PhysicalDeviceFeatures, PhysicalDeviceType, Queue,
        QueueFlags,
    },
    Device, Entry, Instance,
};
use raw_window_handle::HasRawDisplayHandle;
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
    queue: Queue,
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

        // Find physical device and queue family
        println!("Creating physical device and finding queue family index");

        let (physical_device, queue_family_index) =
            unsafe { instance.enumerate_physical_devices()? }
                .into_iter()
                .find_map(|device| is_device_suitable(&instance, device))
                .ok_or_else(|| anyhow!("Could not find suitable physical device"))?;

        // Create logical device and queues
        println!("Creating logical device");

        let device_queue_create_info = DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&[1.0f32])
            .build();

        let device_features = PhysicalDeviceFeatures::builder().build();

        let device_create_info = DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&device_queue_create_info))
            .enabled_features(&device_features)
            .build();

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };

        println!("Retrieving device queue");

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Ok(Self {
            instance,
            device,
            queue,
        })
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None)
        };
    }
}

fn is_device_suitable(
    instance: &Instance,
    device: PhysicalDevice,
) -> Option<(PhysicalDevice, u32)> {
    let properties = unsafe { instance.get_physical_device_properties(device) };
    let features = unsafe { instance.get_physical_device_features(device) };

    if properties.device_type != PhysicalDeviceType::DISCRETE_GPU || features.geometry_shader == 0 {
        return None;
    }

    unsafe {
        instance
            .get_physical_device_queue_family_properties(device)
            .iter()
            .enumerate()
            .find_map(|(idx, info)| {
                if info.queue_flags.contains(QueueFlags::GRAPHICS) {
                    Some((device, idx as u32))
                } else {
                    None
                }
            })
    }
}
