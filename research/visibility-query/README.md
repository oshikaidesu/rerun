# Visibility query research

This standalone, headless experiment does not modify the Motolii renderer or its dependency pin.
It follows the wgpu 29.0.4 `ray_traced_triangle` example and compares results with independent Trimesh/Embree packets.
It is not a finished reflection renderer.

## Run

```sh
cargo clippy --manifest-path research/visibility-query/Cargo.toml
cargo run --manifest-path research/visibility-query/Cargo.toml -- packet.json 1
```

The second argument repeats the input rays for throughput experiments.
Use 1 for the full-resolution packet; repeated rays are not independent samples.
The program prints adapter support, first acceleration-structure build latency, 30 post-warmup query/completion/readback samples, and result mismatches.
The update phase alternates instance transforms if the packet supplies alternate instances and expected results.
BLAS data remains shared and unchanged in that phase.
The result timer includes CPU submission and GPU readback completion, but not shading, ray generation, or full-frame rendering.

`QUERY_MISMATCH_OUTPUT=path.json` stores discrepancy details.
`QUERY_MASK_CANDIDATE=1` selects the candidate-filter experiment, which fails the alpha-hole case on the tested Metal/Naga 29.0.4 path.
The default shader uses repeated committed nearest-hit queries instead.
It reports `0xfffffffe` (unresolved) when its 16-step dispatch budget is exhausted and `0xffffffff` for a miss.
This bounded experiment does not implement a continuation queue.
The alpha mask is an analytic circle on a two-triangle quad, not an arbitrary texture/material implementation.
Bit 23 of the custom instance data is reserved for that test mask.

The production integration must use a capability-selected backend and a shared coverage/material contract.
Do not copy this whole experiment into a product renderer.
The experimental wgpu API is explicitly enabled only in this executable.
GPU agreement is checked against packet data; coplanar ties and near-grazing floating-point differences must be assessed separately, not hidden by increasing the distance tolerance.

## Reproduction data

Motolii's `docs/reviews/assets/2026-09-09-query-readiness` contains the Python reference study, packet generators, source transforms, and recorded results.
The generators require the geometry in the sibling `2026-09-09-glass-gallery` directory.
Large generated packets and target artifacts are intentionally not versioned here.

## References

- https://github.com/gfx-rs/wgpu/tree/v29.0.4/examples/features/src/ray_traced_triangle
- https://trimesh.org/trimesh.ray.ray_pyembree.html
- https://mail.casual-effects.com/research/McGuire2017LightField/index.html
