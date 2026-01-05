mod image_processing;
mod mean;

use std::ops::Deref;
use std::{
    cell::RefCell, collections::HashSet, fs, fs::File, io::Cursor, path::PathBuf, sync::Arc,
};

use arrow::array::{Array, BinaryBuilder, RecordBatch, UInt16Array};
use arrow::datatypes::{DataType, Field, Schema};
use futures::future::join_all;
use image::{DynamicImage, EncodableLayout, ImageBuffer, ImageReader, Pixel, PixelWithColorType};
use itertools::Itertools;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::{image_processing::WgpuProcessor, mean::MeanCalculator};

#[allow(dead_code)]
const FINAL_IMAGE_SIZE: usize = 512;
#[allow(dead_code)]
const INTERPOLATION_STEPS: usize = 10;
const PATH_TO_FRAMES: &str = "/Users/sanderhergarten/Documents/programming/reply_drone/crates/dataset_generation/captures/simple/frames";
const PATH_TO_MASKS: &str = "/Users/sanderhergarten/Documents/programming/reply_drone/crates/dataset_generation/captures/simple/mask";
const CHUNK_SIZE: usize = 500;
const OUTPUT_PARQUET_FILE: &str = "dataset.parquet";

struct Progress {
    paths: Vec<PathBuf>,
    processed: RefCell<usize>,
}

impl Progress {
    fn new(directory_containing_images: PathBuf) -> Option<Self> {
        let paths = fs::read_dir(directory_containing_images)
            .unwrap()
            .map(|entry| entry.unwrap())
            .map(|entry| entry.path())
            .collect::<Vec<_>>();
        Some(Self {
            paths,
            processed: RefCell::new(0),
        })
    }
    fn from_paths(paths: Vec<PathBuf>) -> Self {
        Self {
            paths,
            processed: RefCell::new(0),
        }
    }

    fn get_next_paths(&self, batch_size: usize) -> Vec<PathBuf> {
        let slice = &self.paths[*self.processed.borrow()..*self.processed.borrow() + batch_size];
        (*self.processed.borrow_mut()) += batch_size;
        slice.to_vec()
    }

    fn get_chunked_paths(&self) -> Option<Vec<PathBuf>> {
        let chunk_size = self
            .paths
            .len()
            .min(*self.processed.borrow() + CHUNK_SIZE)
            .checked_sub(*self.processed.borrow())
            .expect("tried out of bounds access");

        println!("{}", *self.processed.borrow());
        if chunk_size == 0 {
            None
        } else {
            Some(self.get_next_paths(chunk_size))
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mask_paths = balance_dataset();
    println!("total processing image {:?}", mask_paths.len());

    let schema = Arc::new(Schema::new(vec![
        Field::new("image", DataType::Binary, false),
        Field::new("clean", DataType::Binary, false),
        Field::new("noisy", DataType::Binary, false),
        Field::new("timestep", DataType::UInt16, false),
    ]));

    // Initialize Parquet writer
    let file = File::create(OUTPUT_PARQUET_FILE)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();

    let mut processors =
        pollster::block_on(join_all((0..INTERPOLATION_STEPS).map(|step| {
            WgpuProcessor::new((step as f32) / ((INTERPOLATION_STEPS - 1) as f32))
        })));

    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    let frame_progress = Progress::from_paths(frame_path_from_mask_path(&mask_paths));
    let mask_progress = Progress::from_paths(mask_paths);
    while let Some(masks_dyn) = load_batch_into_memory(&mask_progress) {
        let (_, images): (Vec<_>, Vec<_>) = load_batch_into_memory(&frame_progress)
            .unwrap()
            .into_iter()
            .map(|(paths, im)| (paths, im.to_rgb8()))
            .unzip();

        let masks = masks_dyn
            .into_iter()
            .map(|(_, im)| im.to_luma8())
            .collect_vec();

        let image_builders = (0..INTERPOLATION_STEPS)
            .map(|_| BinaryBuilder::new())
            .collect_vec();

        let (masks, l_processors): (Vec<Arc<dyn Array>>, Vec<WgpuProcessor>) = image_builders
            .into_par_iter()
            .zip(processors)
            .map(|(mut image_builder, mut processor)| {
                let results = processor.process_batch(&masks);

                // Create BinaryBuilder to collect image binary data (one row per image)
                (
                    images_to_array(results, Some(&mut image_builder)),
                    processor,
                )
            })
            .unzip();

        processors = l_processors;
        let image_array = images_to_array(images, None);

        masks.into_iter().enumerate().tuple_windows().for_each(
            |((timestamp, clean), (_, noisy))| {
                let timestamp_array: UInt16Array =
                    (0..clean.len()).map(|_| timestamp as u16).collect();
                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        image_array.clone(),
                        clean,
                        noisy,
                        Arc::new(timestamp_array) as Arc<dyn Array>,
                    ],
                )
                .unwrap();

                writer.write(&batch).expect("schema_mismatch");
                writer.flush().expect("idk");
            },
        );
    }

    // Finalize the Parquet file
    writer.close()?;

    Ok(())
}

fn images_to_array<P, Container>(
    images: Vec<ImageBuffer<P, Container>>,
    builder: Option<&mut BinaryBuilder>,
) -> Arc<dyn Array>
where
    P: Pixel + PixelWithColorType,
    Container: Deref<Target = [P::Subpixel]>,
    [P::Subpixel]: EncodableLayout,
{
    let image_builder = match builder {
        Some(b) => b,
        None => &mut BinaryBuilder::new(),
    };

    for img in &images {
        // Encode image to PNG format
        let mut buffer = Vec::new();
        let mut cursor = Cursor::new(&mut buffer);
        img.write_to(&mut cursor, image::ImageFormat::Png).unwrap();
        // Append each image as a separate row
        image_builder.append_value(&buffer);
    }
    Arc::new(image_builder.finish()) as Arc<dyn Array>
}

fn load_batch_into_memory(progress: &Progress) -> Option<Vec<(PathBuf, DynamicImage)>> {
    let paths = progress.get_chunked_paths()?;

    Some(
        paths
            .clone()
            .into_par_iter()
            .map(|path| {
                (
                    path.clone(),
                    ImageReader::open(path).unwrap().decode().unwrap().crop(
                        0,
                        0,
                        FINAL_IMAGE_SIZE as u32,
                        FINAL_IMAGE_SIZE as u32,
                    ),
                )
            })
            .collect(),
    )
}

fn balance_dataset() -> Vec<PathBuf> {
    let mut calc = pollster::block_on(MeanCalculator::new());
    let mut means_collector: Vec<(f32, PathBuf)> = Vec::default();

    let mask_progress =
        Progress::new(PathBuf::from(PATH_TO_MASKS)).expect("directory wasnt able to open");

    while let Some(batch) = load_batch_into_memory(&mask_progress) {
        let (paths, images): (Vec<_>, Vec<_>) = batch
            .into_iter()
            .map(|(path, dynamic_image)| (path, dynamic_image.to_luma8()))
            .unzip();

        let means = calc.calculate_batch(&images);
        means_collector.extend(means.into_iter().zip(paths).collect_vec());
    }

    let target_samples = means_collector.len().div_ceil(10);
    println!("target samples{}", target_samples);
    let (_, even_paths): (Vec<f32>, Vec<PathBuf>) =
        distribute_evenly(means_collector, target_samples)
            .into_iter()
            .unzip();
    even_paths
}

fn distribute_evenly(mut items: Vec<(f32, PathBuf)>, target_count: usize) -> Vec<(f32, PathBuf)> {
    // 1. Edge Case: If we have fewer items than we want, keep them all
    if items.len() <= target_count {
        return items;
    }

    items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let min_val = items.first().unwrap().0;
    let max_val = items.last().unwrap().0;
    let range = max_val - min_val;

    if range.abs() < f32::EPSILON {
        return items.into_iter().take(target_count).collect();
    }

    let mut result = Vec::with_capacity(target_count);
    let mut used_indices = HashSet::new();

    for i in 0..target_count {
        let fraction = i as f32 / (target_count - 1) as f32;
        let target_val = min_val + (range * fraction);

        // Find the closest unused item
        let mut best_idx = None;
        let mut best_dist = f32::MAX;

        for (idx, item) in items.iter().enumerate() {
            if used_indices.contains(&idx) {
                continue;
            }

            let dist = (item.0 - target_val).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(idx);
            }
        }

        if let Some(idx) = best_idx {
            used_indices.insert(idx);
            result.push(items[idx].clone());
        } else {
            // Fallback: if no unused item found, break early
            break;
        }
    }

    result
}

fn frame_path_from_mask_path(mask_paths: &[PathBuf]) -> Vec<PathBuf> {
    mask_paths
        .iter()
        .map(|k| PathBuf::from(PATH_TO_FRAMES).join(k.file_name().unwrap()))
        .collect_vec()
}
