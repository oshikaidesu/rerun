use serde::Deserialize;
use std::time::Instant;
use wgpu::util::DeviceExt;
#[derive(Deserialize)]
struct Mesh {
    vertices: Vec<[f32; 3]>,
    indices: Vec<u32>,
    #[serde(default)]
    alpha_hole: bool,
}
#[derive(Deserialize)]
struct Instance {
    mesh: usize,
    id: u32,
    transform: [f32; 12],
}
#[derive(Deserialize)]
struct Packet {
    meshes: Vec<Mesh>,
    instances: Vec<Instance>,
    rays: Vec<[f32; 8]>,
    expected_ids: Vec<i32>,
    expected_t: Vec<f32>,
    #[serde(default)]
    alternate_instances: Option<Vec<Instance>>,
    #[serde(default)]
    alternate_expected_ids: Vec<i32>,
    #[serde(default)]
    alternate_expected_t: Vec<f32>,
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Hit {
    t: f32,
    id: u32,
    primitive: u32,
    front: u32,
    bary: [f32; 2],
    padding: [f32; 2],
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .unwrap();
    let supported = adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
    if !supported {
        println!(
            "{}",
            serde_json::json!({"adapter":adapter.get_info().name,"ray_query":false})
        );
        return;
    }
    let desc = wgpu::DeviceDescriptor {
        required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
        required_limits: wgpu::Limits::default()
            .using_minimum_supported_acceleration_structure_values(),
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        ..Default::default()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&desc)).unwrap();
    if args.len() < 2 {
        println!(
            "{}",
            serde_json::json!({"adapter":adapter.get_info().name,"ray_query":true,"device":true})
        );
        return;
    }
    let packet: Packet = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let repeat: usize = args.get(2).map_or(1, |x| x.parse().unwrap());
    let ray_data = packet.rays.repeat(repeat);
    assert!(!ray_data.is_empty());
    let mut buffers = Vec::new();
    let mut sizes = Vec::new();
    let mut blases = Vec::new();
    for mesh in &packet.meshes {
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: mesh.vertices.len() as u32,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(mesh.indices.len() as u32),
            flags: if mesh.alpha_hole {
                wgpu::AccelerationStructureGeometryFlags::empty()
            } else {
                wgpu::AccelerationStructureGeometryFlags::OPAQUE
            },
        };
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("shared mesh"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size.clone()],
            },
        );
        buffers.push((vb, ib));
        sizes.push(size);
        blases.push(blas);
    }
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("instances"),
        max_instances: packet.instances.len() as u32,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE
            | wgpu::AccelerationStructureFlags::ALLOW_UPDATE,
        update_mode: wgpu::AccelerationStructureUpdateMode::PreferUpdate,
    });
    for (i, item) in packet.instances.iter().enumerate() {
        tlas[i] = Some(wgpu::TlasInstance::new(
            &blases[item.mesh],
            item.transform,
            item.id
                | if packet.meshes[item.mesh].alpha_hole {
                    1 << 23
                } else {
                    0
                },
            255,
        ));
    }
    let entries: Vec<_> = blases
        .iter()
        .enumerate()
        .map(|(i, blas)| wgpu::BlasBuildEntry {
            blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &sizes[i],
                vertex_buffer: &buffers[i].0,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: Some(&buffers[i].1),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        })
        .collect();
    let start = Instant::now();
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.build_acceleration_structures(entries.iter(), Some(&tlas));
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    let rays = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rays"),
        contents: bytemuck::cast_slice(&ray_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bytes = (ray_data.len() * std::mem::size_of::<Hit>()) as u64;
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hits"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let candidate_mode = std::env::var_os("QUERY_MASK_CANDIDATE").is_some();
    let source = if candidate_mode {
        include_str!("query_candidate.wgsl")
    } else {
        include_str!("query.wgsl")
    };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("visibility"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("visibility query"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::AccelerationStructure(&tlas),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rays.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut query_times = Vec::new();
    let mut update_times = Vec::new();
    let mut errors = 0usize;
    let mut max_error = 0.0_f32;
    let mut mismatch_details = Vec::new();
    for update in [false, true] {
        for frame in 0..35 {
            let start = Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            if update {
                let items = if frame % 2 == 0 {
                    packet
                        .alternate_instances
                        .as_ref()
                        .unwrap_or(&packet.instances)
                } else {
                    &packet.instances
                };
                for (i, item) in items.iter().enumerate() {
                    tlas[i] = Some(wgpu::TlasInstance::new(
                        &blases[item.mesh],
                        item.transform,
                        item.id
                            | if packet.meshes[item.mesh].alpha_hole {
                                1 << 23
                            } else {
                                0
                            },
                        255,
                    ));
                }
                encoder.build_acceleration_structures(
                    std::iter::empty::<&wgpu::BlasBuildEntry<'_>>(),
                    Some(&tlas),
                );
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups((ray_data.len() as u32).div_ceil(64), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            encoder.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |r| {
                tx.send(r).unwrap();
            });
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            if frame >= 5 {
                if update {
                    update_times.push(elapsed)
                } else {
                    query_times.push(elapsed)
                }
            }
            let data = readback.slice(..).get_mapped_range();
            if frame >= 33 {
                let alternate = update && frame % 2 == 0 && packet.alternate_instances.is_some();
                let expected_ids = if alternate {
                    &packet.alternate_expected_ids
                } else {
                    &packet.expected_ids
                };
                let expected_t = if alternate {
                    &packet.alternate_expected_t
                } else {
                    &packet.expected_t
                };
                let hits: &[Hit] = bytemuck::cast_slice(&data);
                for (i, h) in hits.iter().enumerate() {
                    let j = i % packet.rays.len();
                    let expected = expected_ids[j];
                    let dt = (h.t - expected_t[j]).abs();
                    if expected >= 0 {
                        max_error = max_error.max(dt);
                    }
                    if h.id as i32 != expected || (expected >= 0 && dt > 0.05) {
                        errors += 1;
                        if mismatch_details.len() < 200 {
                            mismatch_details.push(serde_json::json!({"index":j,"alternate":alternate,"expected_id":expected,"expected_t":expected_t[j],"gpu_id":h.id,"gpu_t":h.t,"primitive":h.primitive,"bary":h.bary,"front":h.front}));
                        }
                    }
                }
            }
            drop(data);
            readback.unmap();
        }
    }
    if let Ok(path) = std::env::var("QUERY_MISMATCH_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&mismatch_details).unwrap()).unwrap();
    }
    println!(
        "{}",
        serde_json::json!({"adapter":adapter.get_info().name,"ray_query":true,"mask_mode":if candidate_mode {"native_candidate"} else {"closest_restart"},"moving_instances":packet.alternate_instances.is_some(),"alpha_mask_case":packet.meshes.iter().any(|m|m.alpha_hole),"unique_rays":packet.rays.len(),"rays":ray_data.len(),"meshes":blases.len(),"instances":packet.instances.len(),"triangle_count":packet.meshes.iter().map(|m|m.indices.len()/3).sum::<usize>(),"mismatches":errors,"max_hit_distance_error":max_error,"initial_build_ms":build_ms,"query_readback_ms_median":median(&mut query_times),"tlas_refresh_query_readback_ms_median":median(&mut update_times),"query_ms_samples":query_times,"update_ms_samples":update_times})
    );
    assert_eq!(
        errors, 0,
        "GPU visibility must agree with independent CPU oracle"
    );
}
