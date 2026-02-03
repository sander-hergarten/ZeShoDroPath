@group(0) @binding(0) var input_texture: texture_2d<f32>;

// Output texture
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

struct Params { 
    alpha: f32,           // Controls Radius (0.0 to 1.0)
    // _pad: f32,         // (Implicit padding bytes may exist here depending on alignment)
    direction: vec2<i32>, // (1, 0) for Horizontal pass, (0, 1) for Vertical pass
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x);
    let y = i32(global_id.y);
    let dims = vec2<i32>(textureDimensions(input_texture));

    if (x >= dims.x || y >= dims.y) { return; }

    // Max radius 45 is now perfectly fast because the loop is linear (1D)
    let max_radius = 45.0;
    let radius = i32(params.alpha * max_radius);

    var count = 0.0;
    var accumulated_val = 0.0;

    // LINEAR LOOP: We only loop from -radius to +radius
    for (var i = -radius; i <= radius; i++) {
        
        // We move along the X or Y axis depending on params.direction
        // if direction is (1,0), we shift x by i.
        // if direction is (0,1), we shift y by i.
        let offset = params.direction * i;
        
        let sample_x = x + offset.x;
        let sample_y = y + offset.y;

        // Boundary check
        if (sample_x >= 0 && sample_x < dims.x && sample_y >= 0 && sample_y < dims.y) {
            
            // Sample the neighbor
            let val = textureLoad(input_texture, vec2<i32>(sample_x, sample_y), 0).r;
            
            accumulated_val += val;
            count += 1.0;
        }
    }

    let final_val = accumulated_val / max(count, 1.0);

    textureStore(output_texture, vec2<i32>(x, y), vec4<f32>(final_val, final_val, final_val, 1.0));
}
