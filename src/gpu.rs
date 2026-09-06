//! GPU-accelerated classic-ML compute via [`wgpu`](https://docs.rs/wgpu) — a
//! portable compute API over Vulkan / Metal / DX12, so this runs on any system
//! with a GPU (and in CI via a software adapter). Requires the `gpu-compute`
//! feature.
//!
//! Two primitives cover the heavy inner loops of several classic estimators:
//! [`gemm`] (dense matrix multiply, behind covariance / linear algebra) and
//! [`pairwise_sqdist`] (the squared-distance matrix behind k-NN, k-means, and
//! Mahalanobis). Both run in `f32` — GPU compute is single-precision — so
//! results match the `f64` CPU path within floating-point tolerance, never
//! bit-for-bit. CPU stays the default everywhere; these are opt-in accelerators
//! with a CPU fallback ([`is_available`] reports whether a GPU was found).

use std::sync::OnceLock;

use crate::error::{Error, Result};

/// A cached wgpu device + queue. Created once on first use.
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// The process-wide GPU context, or an error string if none could be created.
fn context() -> Result<&'static GpuContext> {
    static CTX: OnceLock<std::result::Result<GpuContext, String>> = OnceLock::new();
    CTX.get_or_init(|| GpuContext::new().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| Error::Backend(format!("GPU compute unavailable: {e}")))
}

impl GpuContext {
    fn new() -> Result<GpuContext> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|e| Error::Backend(format!("no GPU adapter: {e}")))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("millwright-gpu"),
            required_features: wgpu::Features::empty(),
            // Ask only for what the adapter offers, so a modest / software
            // adapter (e.g. CI) still yields a device.
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| Error::Backend(format!("no GPU device: {e}")))?;
        Ok(GpuContext { device, queue })
    }
}

/// Whether a usable GPU was found. A caller can branch on this to fall back to
/// CPU when there is no GPU (the primitives themselves also error in that case).
pub fn is_available() -> bool {
    context().is_ok()
}

/// Dense matrix multiply on the GPU: `C = A · B`, all row-major `f32`, with `A`
/// `m × k`, `B` `k × n`, and the returned `C` `m × n`.
///
/// Runs in `f32`; results match an `f64` CPU multiply within tolerance. Errors
/// if no GPU is available or the slice lengths do not match the dimensions.
pub fn gemm(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Result<Vec<f32>> {
    if a.len() != m * k {
        return Err(Error::Shape(format!(
            "gemm: A is {m}x{k} = {} elems, got {}",
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(Error::Shape(format!(
            "gemm: B is {k}x{n} = {} elems, got {}",
            k * n,
            b.len()
        )));
    }
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    if k == 0 {
        return Ok(vec![0.0; m * n]);
    }
    let ctx = context()?;
    run_kernel(
        ctx,
        GEMM_WGSL,
        a,
        b,
        m * n,
        [m as u32, k as u32, n as u32, 0],
        ((m as u32).div_ceil(8), (n as u32).div_ceil(8)),
    )
}

/// The squared-Euclidean distance matrix on the GPU: for `X` (`n × d`) and `Y`
/// (`m × d`), returns the row-major `n × m` matrix `out[i*m + j] = ||xᵢ − yⱼ||²`.
///
/// Runs in `f32`; take `sqrt` for Euclidean distance. Errors if no GPU is
/// available or the slice lengths do not match the dimensions.
pub fn pairwise_sqdist(x: &[f32], n: usize, y: &[f32], m: usize, d: usize) -> Result<Vec<f32>> {
    if x.len() != n * d {
        return Err(Error::Shape(format!(
            "pairwise_sqdist: X is {n}x{d} = {} elems, got {}",
            n * d,
            x.len()
        )));
    }
    if y.len() != m * d {
        return Err(Error::Shape(format!(
            "pairwise_sqdist: Y is {m}x{d} = {} elems, got {}",
            m * d,
            y.len()
        )));
    }
    if n == 0 || m == 0 {
        return Ok(Vec::new());
    }
    if d == 0 {
        return Ok(vec![0.0; n * m]);
    }
    let ctx = context()?;
    run_kernel(
        ctx,
        DIST_WGSL,
        x,
        y,
        n * m,
        [n as u32, m as u32, d as u32, 0],
        ((n as u32).div_ceil(8), (m as u32).div_ceil(8)),
    )
}

/// Upload two `f32` input buffers, dispatch a 2-D compute kernel over an
/// `(grid_x, grid_y)` workgroup grid, and read back `out_len` `f32` results.
fn run_kernel(
    ctx: &GpuContext,
    shader: &str,
    in0: &[f32],
    in1: &[f32],
    out_len: usize,
    dims: [u32; 4],
    grid: (u32, u32),
) -> Result<Vec<f32>> {
    let device = &ctx.device;
    let queue = &ctx.queue;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("millwright-kernel"),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });

    // Upload the inputs into storage/uniform buffers.
    let storage_in = |label: &'static str, data: &[f32]| {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of_val(data) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
        buf
    };
    let buf0 = storage_in("in0", in0);
    let buf1 = storage_in("in1", in1);
    let dim_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dims"),
        size: std::mem::size_of_val(&dims) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&dim_buf, 0, bytemuck::cast_slice(&dims));
    let out_bytes = (out_len * std::mem::size_of::<f32>()) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("millwright-pipeline"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("millwright-bind"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf0.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dim_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("millwright-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("millwright-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(grid.0, grid.1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
    queue.submit(Some(encoder.finish()));

    // Map the staging buffer and block until the GPU work + mapping complete.
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::Wait)
        .map_err(|e| Error::Backend(format!("GPU poll failed: {e}")))?;
    rx.recv()
        .map_err(|_| Error::Backend("GPU mapping channel closed".into()))?
        .map_err(|e| Error::Backend(format!("GPU buffer map failed: {e}")))?;

    let data = slice.get_mapped_range();
    let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(out)
}

/// `C = A·B`: one thread per output element, accumulating over the shared dim.
const GEMM_WGSL: &str = r#"
struct Dims { m: u32, k: u32, n: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> d: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= d.m || col >= d.n) { return; }
    var acc = 0.0;
    for (var i: u32 = 0u; i < d.k; i = i + 1u) {
        acc = acc + a[row * d.k + i] * b[i * d.n + col];
    }
    c[row * d.n + col] = acc;
}
"#;

/// `out[i,j] = ||x_i - y_j||^2`: one thread per (query, reference) pair.
const DIST_WGSL: &str = r#"
struct Dims { n: u32, m: u32, d: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dm: Dims;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if (i >= dm.n || j >= dm.m) { return; }
    var acc = 0.0;
    for (var l: u32 = 0u; l < dm.d; l = l + 1u) {
        let diff = x[i * dm.d + l] - y[j * dm.d + l];
        acc = acc + diff * diff;
    }
    out[i * dm.m + j] = acc;
}
"#;
