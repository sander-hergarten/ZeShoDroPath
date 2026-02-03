@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;


@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);

    let dims = vec2<i32>(textureDimensions(input_texture));

    if (x >= dims.x || y >= dims.y) { return; }

    let val = textureLoad(input_texture, vec2<i32>(x, y), 0).r;

    if val > 0.5{
        textureStore(output_texture, vec2<i32>(x, y), vec4<f32>(f32(x), f32(y), 0.0, 1.0));
    }else{

        textureStore(output_texture, vec2<i32>(x, y), vec4<f32>(-99999.0, -99999.0, 0.0, 1.0));
    }
     
}
