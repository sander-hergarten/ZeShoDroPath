@group(0) @binding(0) var input_texture: texture_storage_2d<rgba8unorm, read>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

struct Params { 
    alpha: f32,           // Controls Radius (0.0 to 1.0)
    direction: vec2<i32>, // (1, 0) for Horizontal pass, (0, 1) for Vertical pass
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    let my_cord = vec2<i32>(global_id.xy);

    let dims = vec2<i32>(textureDimensions(input_texture));

    if (my_cord.x >= dims.x || my_cord.y >= dims.y) { return; }

    var val = textureLoad(input_texture, vec2<i32>(my_cord.x, my_cord.y));
    var best_seed = vec2<f32>(val.x, val.y);
    
    var min_dist = 99999.9;
    let jump_dist= dims.x/4;

    if val.w>= 0.5{
        min_dist = distance(vec2<f32>(my_cord.xy), best_seed);
    }

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            if (x==0&& y==0) {continue;}

            let offset = vec2<i32>(x, y) * jump_dist ;
            let neighbor_coord = my_cord.xy + offset;

            if (neighbor_coord.x < 0 || neighbor_coord.x >= dims.x ||
                neighbor_coord.y < 0 || neighbor_coord.y >= dims.y) {continue;}

            let neighbor_data = textureLoad(input_texture,  neighbor_coord.xy);
            let neighbor_seed_suggestion = neighbor_data.xy;
            let neighbor_validity = neighbor_data.w;

            if (neighbor_validity > 0.5) {
                let dist = distance(vec2<f32>(my_cord.xy), neighbor_seed_suggestion);

                if (dist < min_dist) {
                    min_dist = dist;
                    best_seed = neighbor_seed_suggestion;
                    val.w = 1.0; 
                }
            }
        }
    }


    let distance_final = distance(vec2<f32>(my_cord.xy), best_seed.xy);
     
    textureStore(output_texture, my_cord.xy, vec4<f32>(vec3<f32>(distance_final), 1.0));

} 

