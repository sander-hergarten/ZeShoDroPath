use crate::{CameraRenderGraph, Cubemap, RenderLayers, RngResource, Skybox, Viewport};

use bevy::prelude::*;
use bevy_capture::{CameraTargetHeadless, Capture, CaptureBundle, encoder::frames};

#[derive(Message)]
pub struct CaptureImageMessage;

pub fn headless_setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    windows: Query<&Window>,
    mut cubemap: ResMut<Cubemap>,
    mut rng: ResMut<RngResource>,
    mut images: ResMut<Assets<Image>>,
    // mut images_2: ResMut<Assets<Image>>,
) {
    cubemap.fill(&asset_server);

    let skybox_handle = cubemap.pick_random(&mut rng);

    {
        let window = windows.single().unwrap();
        let width = window.resolution.physical_width();
        let height = 512;
        let half_width = 512;

        commands.spawn((
            Camera {
                order: 1, // Set the main camera to render first
                viewport: Some(Viewport {
                    physical_position: UVec2::new(0, 0),
                    physical_size: UVec2::new(half_width, height),
                    ..default()
                }),
                ..default()
            }
            .target_headless(half_width, height, &mut images),
            Camera3d::default(),
            CaptureBundle::default(),
            Skybox {
                image: skybox_handle.clone(),
                brightness: 1000.0,
                ..default()
            },
            EnvironmentMapLight {
                diffuse_map: skybox_handle.clone(),
                specular_map: skybox_handle.clone(),
                intensity: 1000.0,        // lux
                rotation: Quat::IDENTITY, // spin if you need sunrise
                affects_lightmapped_mesh_diffuse: true,
            },
        ));

        commands.spawn((
            Camera {
                viewport: Some(Viewport {
                    physical_position: UVec2::new(0, 0),
                    physical_size: UVec2::new(half_width, height),
                    ..default()
                }),
                order: 0,
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            }
            .target_headless(half_width, height, &mut images),
            Camera3d::default(),
            CameraRenderGraph::new(bevy::core_pipeline::core_3d::graph::Core3d),
            RenderLayers::layer(1),
            CaptureBundle::default(),
        ));
    }

    commands.insert_resource(AmbientLight {
        color: Color::srgb_u8(210, 220, 240),
        brightness: 0.0,
        ..Default::default()
    });
}

pub fn capture_update(
    mut capture_main_query: Query<&mut Capture, (With<Camera>, Without<RenderLayers>)>,
    mut capture_mask_query: Query<&mut Capture, (With<Camera>, With<RenderLayers>)>,
    mut frame: Local<u32>,
    mut waited: Local<bool>,
) {
    if !*waited {
        *waited = true;
        return;
    }

    let mut capture_main = capture_main_query.single_mut().unwrap();
    let mut capture_mask = capture_mask_query.single_mut().unwrap();
    if !capture_main.is_capturing() {
        capture_main.start(frames::FramesEncoder::new("captures/simple/frames"));
    }
    if !capture_mask.is_capturing() {
        capture_mask.start(frames::FramesEncoder::new("captures/simple/mask"));
    }
    *frame += 1;
    if (*frame).is_multiple_of(50) {
        println!("{}", *frame);
    }
}
