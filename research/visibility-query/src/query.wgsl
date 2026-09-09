enable wgpu_ray_query;
struct Ray { origin: vec3f, tmin: f32, direction: vec3f, tmax: f32 }
struct Hit { t: f32, id: u32, primitive: u32, front: u32, bary: vec2f, padding: vec2f }
@group(0) @binding(0) var scene: acceleration_structure;
@group(0) @binding(1) var<storage, read> rays: array<Ray>;
@group(0) @binding(2) var<storage, read_write> hits: array<Hit>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if i >= arrayLength(&rays) { return; }
    let r = rays[i];
    var result: Hit;
    result.t = -1.0;
    result.id = 0xfffffffeu; // unresolved, never report a budget exhaustion as a miss
    var tmin = r.tmin;
    for (var step=0u; step<16u; step+=1u) {
        var q: ray_query;
        rayQueryInitialize(&q, scene, RayDesc(0u, 255u, tmin, r.tmax, r.origin, r.direction));
        rayQueryProceed(&q);
        let hit = rayQueryGetCommittedIntersection(&q);
        if hit.kind == RAY_QUERY_INTERSECTION_NONE { result.id=0xffffffffu; break; }
        if (hit.instance_custom_data & 0x800000u) != 0u {
            let b=hit.barycentrics;
            let uv=select(vec2f(b.x+b.y,b.y),vec2f(b.x,b.x+b.y),hit.primitive_index==1u);
            let delta=uv-vec2f(0.5);
            if dot(delta,delta)<0.09 {
                tmin=bitcast<f32>(bitcast<u32>(hit.t)+1u);
                if tmin>=r.tmax {result.id=0xffffffffu;break;}
                continue;
            }
        }
        result.t=hit.t;
        result.id=hit.instance_custom_data & 0x7fffffu;
        result.primitive=hit.primitive_index;
        result.front=select(0u,1u,hit.front_face);
        result.bary=hit.barycentrics;
        break;
    }
    hits[i]=result;
}
