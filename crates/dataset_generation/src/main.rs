//! Load a cubemap texture onto a cube like a skybox and cycle through different compressed texture formats
mod components;
mod features;
mod image_capture;
mod scene_generation;

use bevy::asset::AssetPath;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::{
    app::{RunMode, ScheduleRunnerPlugin},
    prelude::*,
    render::RenderPlugin,
    winit::WinitPlugin,
};
use bevy::{
    camera::{Viewport, visibility::RenderLayers},
    core_pipeline::Skybox,
    render::{
        camera::CameraRenderGraph,
        render_resource::{TextureViewDescriptor, TextureViewDimension},
    },
};
use bevy_rapier3d::prelude::*;
use components::*;
use features::*;
use image_capture::CaptureImageMessage;
use rand::prelude::*;
use smooth_bevy_cameras::{
    LookTransformPlugin,
    controllers::unreal::{UnrealCameraBundle, UnrealCameraController, UnrealCameraPlugin},
};
use std::fs;

#[cfg(feature = "headless")]
use crate::image_capture::{capture_update, headless_setup};

#[derive(Resource)]
struct RngResource {
    rng: SmallRng,
}

impl RngResource {
    fn sample_normal(&mut self, mean: f32, std_dev: f32) -> f32 {
        // Box-Muller transform
        let u1 = self.rng.random::<f32>();
        let u2 = self.rng.random::<f32>();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        mean + std_dev * z0
    }
}

#[derive(Resource)]
struct Cubemap {
    currently_loaded: Handle<Image>,
    all_handles: Vec<Handle<Image>>,
}

impl Cubemap {
    fn new() -> Self {
        Self {
            currently_loaded: Handle::<Image>::default(),
            all_handles: Vec::new(),
        }
    }
    fn fill(&mut self, assets: &Res<AssetServer>) {
        for entry in std::fs::read_dir("assets/hdr").expect("hdrs missing") {
            let path = entry.unwrap().path();
            let name = path.strip_prefix("assets").unwrap();
            self.all_handles
                .push(assets.load(AssetPath::from_path(name)));
        }
    }
    fn pick_random(&mut self, rng: &mut RngResource) -> Handle<Image> {
        self.currently_loaded =
            self.all_handles[rng.rng.random_range(0..self.all_handles.len())].clone();
        self.currently_loaded.clone()
    }
}

#[derive(Resource)]
struct SpawnRange {
    min: Vec3,
    max: Vec3,
}

#[derive(Resource)]
struct FeaturesSpawned(u16);

#[derive(Resource)]
struct Textures(Vec<Handle<Image>>, Vec<String>);

#[derive(Message)]
struct RegenerateSceneMessage;

#[derive(Message)]
struct GenerateSceneMessage;

#[derive(Message)]
struct ChangeEnvironmentMessage;

impl Textures {
    fn new() -> Self {
        Self(Vec::new(), Vec::new())
    }
    fn fill(&mut self, assets: &Res<AssetServer>) {
        let mut names = Vec::new();
        for entry in std::fs::read_dir("assets/textures").expect("textures folder missing") {
            let path = entry.unwrap().path();
            let name = path.strip_prefix("assets").unwrap();
            names.push(name.to_str().unwrap().to_string());
            self.0.push(assets.load(AssetPath::from_path(name)));
        }
        self.1 = names;
    }

    fn pick_random(&self, rng: &mut RngResource) -> &str {
        &self.1[rng.rng.random_range(0..self.1.len())]
    }
}

#[derive(Resource)]
struct FeatureTextures(Vec<Handle<Image>>);

impl FeatureTextures {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn fill(&mut self, assets: &Res<AssetServer>) {
        for entry in std::fs::read_dir("assets/features").expect("features folder missing") {
            self.0.push(assets.load(AssetPath::from_path(
                entry.unwrap().path().strip_prefix("assets").unwrap(),
            )));
        }
    }
}

#[derive(Message)]
struct ClearSceneMessage;

fn main() {
    let mut app = App::new();

    #[cfg(not(feature = "headless"))]
    {
        app.add_plugins(DefaultPlugins)
            .add_plugins(LookTransformPlugin)
            .add_plugins(UnrealCameraPlugin::default())
            .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
            .add_plugins(FrameTimeDiagnosticsPlugin::default());
    }

    #[cfg(feature = "headless")]
    {
        fs::create_dir_all("captures/simple").unwrap();
        app.add_plugins((
            DefaultPlugins
                .build()
                // Disable the WinitPlugin to prevent the creation of a window
                .disable::<WinitPlugin>()
                // Make sure pipelines are ready before rendering
                .set(RenderPlugin {
                    synchronous_pipeline_compilation: true,
                    ..default()
                }),
            // Add the ScheduleRunnerPlugin to run the app in loop mode
            ScheduleRunnerPlugin {
                run_mode: RunMode::Loop { wait: None },
            },
            // Add the CapturePlugin
            bevy_capture::CapturePlugin,
        ))
        .add_plugins(LookTransformPlugin)
        .add_plugins(UnrealCameraPlugin::default())
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(FrameTimeDiagnosticsPlugin::default());
    }

    app.insert_resource(Textures::new())
        .insert_resource(FeatureTextures::new())
        .insert_resource(FeaturesSpawned(0))
        .insert_resource(RngResource {
            rng: SmallRng::from_rng(&mut rand::rng()), // <- simpler RNG init
        })
        .insert_resource(SpawnRange {
            min: Vec3::new(-50.0, -50.0, -50.0),
            max: Vec3::new(50.0, 50.0, 50.0),
        })
        .insert_resource(Cubemap::new())
        .add_message::<SpawnFeatures>()
        .add_message::<GenerateSceneMessage>()
        .add_message::<ClearSceneMessage>()
        .add_message::<ChangeEnvironmentMessage>()
        .add_message::<RegenerateSceneMessage>()
        .add_message::<CaptureImageMessage>();

    app.add_systems(PreStartup, setup_assets);
    #[cfg(not(feature = "headless"))]
    app.add_systems(Startup, setup);

    #[cfg(feature = "headless")]
    app.add_systems(Startup, headless_setup);

    app.add_systems(Update, sync_cameras)
        .add_systems(Update, environment_loader.after(setup))
        .add_systems(Update, emit_regenerate_scene_event_on_button_press)
        .add_systems(
            Update,
            clear_scene.after(emit_regenerate_scene_event_on_button_press),
        )
        .add_systems(
            Update,
            generate_scene
                .after(clear_scene)
                .after(emit_regenerate_scene_event_on_button_press),
        )
        .add_systems(Update, spawn_features)
        .add_systems(
            Update,
            change_environment.after(emit_regenerate_scene_event_on_button_press),
        )
        .add_systems(Update, sample_random_point_system)
        // .add_systems(Update, regenerate_on_space_key);
        .add_systems(Update, regenerate_on_frame.after(change_environment))
        .add_systems(Update, rotate_camera);

    #[cfg(feature = "headless")]
    app.add_systems(Update, capture_update);

    app.run();
}

fn setup_assets(
    mut textures: ResMut<Textures>,
    mut feature_textures: ResMut<FeatureTextures>,
    asset_server: Res<AssetServer>,
) {
    textures.fill(&asset_server);
    feature_textures.fill(&asset_server);
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    windows: Query<&Window>,
    mut cubemap: ResMut<Cubemap>,
    mut rng: ResMut<RngResource>,
) {
    cubemap.fill(&asset_server);

    let skybox_handle = cubemap.pick_random(&mut rng);

    {
        let window = windows.single().unwrap();
        let width = window.resolution.physical_width();
        let height = window.resolution.physical_height();
        let half_width = width / 2;
        commands.spawn((
            Camera {
                order: 1, // Set the main camera to render first
                viewport: Some(Viewport {
                    physical_position: UVec2::new(0, 0),
                    physical_size: UVec2::new(half_width, height),
                    ..default()
                }),
                ..default()
            },
            Camera3d::default(),
            UnrealCameraBundle::new(
                UnrealCameraController::default(),
                Vec3::new(-2.0, 5.0, 5.0),
                Vec3::new(0., 0., 0.),
                Vec3::Y,
            ),
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
            Camera3d::default(),
            Camera {
                viewport: Some(Viewport {
                    physical_position: UVec2::new(half_width, 0),
                    physical_size: UVec2::new(half_width, height),
                    ..default()
                }),
                order: 0,
                clear_color: ClearColorConfig::Custom(Color::BLACK),
                ..default()
            },
            CameraRenderGraph::new(bevy::core_pipeline::core_3d::graph::Core3d),
            RenderLayers::layer(1),
        ));
    }

    commands.insert_resource(AmbientLight {
        color: Color::srgb_u8(210, 220, 240),
        brightness: 0.0,
        ..Default::default()
    });
}
fn generate_scene(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    textures: ResMut<Textures>,
    rng: ResMut<RngResource>,
    meshes: ResMut<Assets<Mesh>>,
    assets: Res<AssetServer>,
    spawn_range: Res<SpawnRange>,
    mut generate_scene_event: MessageReader<GenerateSceneMessage>,
    mut features_spawned: ResMut<FeaturesSpawned>,
) {
    if generate_scene_event.is_empty() {
        return;
    } else {
        generate_scene_event.clear();
    }

    let black = materials.add(StandardMaterial {
        base_color: Color::BLACK,
        unlit: true,
        ..default()
    });

    let scene =
        scene_generation::generate_scene(meshes, materials, rng, spawn_range, textures, assets);

    let textured_objects =
        scene
            .clone()
            .into_iter()
            .map(|(mesh, material, transform, collider)| {
                (
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(material.clone()),
                    transform,
                    collider.clone(),
                    IsCuboid,
                )
            });

    let blacks = (0..scene.len()).map(|_| black.clone()).collect::<Vec<_>>();
    let non_textured_objects =
        scene
            .into_iter()
            .zip(blacks)
            .map(|((mesh, _, transform, _), black)| {
                (
                    Mesh3d(mesh),
                    MeshMaterial3d(black),
                    transform,
                    IsCuboid,
                    RenderLayers::layer(1),
                )
            });

    commands.spawn_batch(textured_objects);
    commands.spawn_batch(non_textured_objects);
    features_spawned.0 = 0;
}

fn environment_loader(
    asset_server: Res<AssetServer>,
    mut images: ResMut<Assets<Image>>,
    cubemap: ResMut<Cubemap>,
    mut skyboxes: Query<&mut Skybox>,
    mut environment_map_light: Query<&mut EnvironmentMapLight>,
) {
    if asset_server
        .load_state(&cubemap.currently_loaded)
        .is_loaded()
    {
        let image = images.get_mut(&cubemap.currently_loaded).unwrap();

        if image.texture_descriptor.array_layer_count() == 1 {
            image.reinterpret_stacked_2d_as_array(image.height() / image.width());
            image.texture_view_descriptor = Some(TextureViewDescriptor {
                dimension: Some(TextureViewDimension::Cube),
                ..default()
            });
        }

        for mut skybox in &mut skyboxes {
            skybox.image = cubemap.currently_loaded.clone();
        }

        for mut environment_map_light in &mut environment_map_light {
            environment_map_light.diffuse_map = cubemap.currently_loaded.clone();
            environment_map_light.specular_map = cubemap.currently_loaded.clone();
        }
    }
}

fn change_environment(
    mut cubemap: ResMut<Cubemap>,
    mut skyboxes: Query<&mut Skybox>,
    mut environment_map_light: Query<&mut EnvironmentMapLight>,
    mut rng: ResMut<RngResource>,
    mut change_environment_event: MessageReader<ChangeEnvironmentMessage>,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
) {
    if change_environment_event.is_empty() {
        return;
    } else {
        change_environment_event.clear();
    }

    let new_cubemap = cubemap.pick_random(&mut rng);

    // Check if the asset is loaded before trying to access it
    if asset_server.load_state(&new_cubemap).is_loaded()
        && let Some(image) = images.get_mut(&new_cubemap)
        && image.texture_descriptor.array_layer_count() == 1
    {
        image.reinterpret_stacked_2d_as_array(image.height() / image.width());
        image.texture_view_descriptor = Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..default()
        });
    }

    for mut skybox in &mut skyboxes {
        skybox.image = new_cubemap.clone();
    }

    for mut environment_map_light in &mut environment_map_light {
        environment_map_light.diffuse_map = new_cubemap.clone();
        environment_map_light.specular_map = new_cubemap.clone();
        // Sample from normal distribution and ensure non-negative value
        let intensity = rng.sample_normal(1000.0, 800.0).max(40.0);
        environment_map_light.intensity = intensity;
    }
}

fn sync_cameras(
    main_camera: Query<&Transform, (With<Camera>, Without<RenderLayers>)>,
    mut second_camera: Query<&mut Transform, (With<Camera>, With<RenderLayers>)>,
) {
    if let (Ok(main_transform), Ok(mut second_transform)) =
        (main_camera.single(), second_camera.single_mut())
    {
        *second_transform = *main_transform;
    } else {
        panic!("to many cameras")
    }
}

fn clear_scene(
    mut commands: Commands,
    entities: Query<Entity, With<Mesh3d>>,
    mut clear_scene_event: MessageReader<ClearSceneMessage>,
) {
    if !clear_scene_event.is_empty() {
        clear_scene_event.clear();
        for entity in entities.iter() {
            commands.entity(entity).despawn();
        }
    }
}

fn emit_regenerate_scene_event_on_button_press(
    mut clear_scene_event_writer: MessageWriter<ClearSceneMessage>,
    mut generate_scene_event_writer: MessageWriter<GenerateSceneMessage>,
    mut change_environment_event_writer: MessageWriter<ChangeEnvironmentMessage>,
    mut regenerate_scene_event_writer: MessageReader<RegenerateSceneMessage>,
    // keyboard_input: Res<ButtonInput<KeyCode>>,
) {
    if regenerate_scene_event_writer.is_empty() {
        return;
    } else {
        regenerate_scene_event_writer.clear();
    }

    clear_scene_event_writer.write(ClearSceneMessage);
    generate_scene_event_writer.write(GenerateSceneMessage);
    change_environment_event_writer.write(ChangeEnvironmentMessage);
}

fn regenerate_on_space_key(
    mut regenerate_scene_event_writer: MessageWriter<RegenerateSceneMessage>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
) {
    if keyboard_input.just_pressed(KeyCode::Space) {
        regenerate_scene_event_writer.write(RegenerateSceneMessage);
    }
}

fn regenerate_on_frame(
    mut regenerate_scene_event_writer: MessageWriter<RegenerateSceneMessage>,
    mut counter: Local<u32>,
) {
    if *counter < 3 {
        *counter += 1;
        return;
    } else {
        *counter += 1
    }

    regenerate_scene_event_writer.write(RegenerateSceneMessage);
}

fn rotate_camera(
    mut rng: ResMut<RngResource>,
    spawn_range: Res<SpawnRange>,
    mut camera_transforms: Query<&mut Transform, With<Camera>>,
) {
    // let position = Vec3::new(
    //     rng.rng.random_range(spawn_range.min.x..spawn_range.max.x),
    //     rng.rng.random_range(spawn_range.min.y..spawn_range.max.y),
    //     rng.rng.random_range(spawn_range.min.z..spawn_range.max.z),
    // );

    let position = Vec3::ZERO;
    let random_rotation = Quat::from_euler(
        EulerRot::XYZ,
        rng.rng.random_range(0.0..2.0 * std::f32::consts::PI),
        rng.rng.random_range(0.0..2.0 * std::f32::consts::PI),
        rng.rng.random_range(0.0..2.0 * std::f32::consts::PI),
    );
    let trans = Transform {
        translation: position,
        rotation: random_rotation,
        ..default()
    };
    for mut transform in camera_transforms.iter_mut() {
        *transform = trans;
    }
}
