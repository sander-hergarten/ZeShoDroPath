@group(0) @binding(0) var input_texture: texture_2d<f32>;
// Stores the sum of each 16x16 block
@group(0) @binding(1) var<storage, read_write> partial_sums: array<f32>;

// Shared memory for the 256 threads in this workgroup
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>
) {
    let x = global_id.x;
    let y = global_id.y;
    let dims = textureDimensions(input_texture);

    if (x < dims.x && y < dims.y) {
        sdata[local_idx] = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0).r;
    } else {
        sdata[local_idx] = 0.0;
    }
    workgroupBarrier();

    if (local_idx < 128u) { sdata[local_idx] += sdata[local_idx + 128u]; } workgroupBarrier();
    if (local_idx < 64u)  { sdata[local_idx] += sdata[local_idx + 64u]; }  workgroupBarrier();
    if (local_idx < 32u)  { sdata[local_idx] += sdata[local_idx + 32u]; }  workgroupBarrier();
    if (local_idx < 16u)  { sdata[local_idx] += sdata[local_idx + 16u]; }  workgroupBarrier();
    if (local_idx < 8u)   { sdata[local_idx] += sdata[local_idx + 8u]; }   workgroupBarrier();
    if (local_idx < 4u)   { sdata[local_idx] += sdata[local_idx + 4u]; }   workgroupBarrier();
    if (local_idx < 2u)   { sdata[local_idx] += sdata[local_idx + 2u]; }   workgroupBarrier();
    if (local_idx < 1u)   { sdata[local_idx] += sdata[local_idx + 1u]; }   workgroupBarrier();

    if (local_idx == 0u) {
        let output_index = group_id.y * num_groups.x + group_id.x;
        partial_sums[output_index] = sdata[0];
    }
}
