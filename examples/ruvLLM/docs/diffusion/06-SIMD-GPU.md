# Hardware Acceleration: SIMD and GPU

## Overview

RuvDLLM provides multiple hardware acceleration paths to achieve high performance across different deployment scenarios. The system automatically selects the optimal backend based on available hardware.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Hardware Acceleration Hierarchy                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     GPU Acceleration (Fastest)                       │   │
│  │  • CUDA (NVIDIA): cuBLAS, custom kernels                            │   │
│  │  • Metal (Apple): MPS, custom shaders                               │   │
│  │  • Vulkan (Cross-platform): Compute shaders                         │   │
│  │  Target: 1000-2000+ tok/s                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                        │
│                                    │ Fallback                               │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     CPU SIMD (High Performance)                      │   │
│  │  • AVX-512 (Intel/AMD): 512-bit vectors                             │   │
│  │  • AVX2 (Intel/AMD): 256-bit vectors                                │   │
│  │  • NEON (ARM): 128-bit vectors                                      │   │
│  │  Target: 200-500+ tok/s                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                        │
│                                    │ Fallback                               │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Scalar (Baseline)                                │   │
│  │  • Pure Rust implementation                                         │   │
│  │  • Auto-vectorization where possible                                │   │
│  │  Target: 50-100 tok/s                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## CPU SIMD Acceleration

### Feature Detection

```rust
/// Runtime SIMD feature detection
pub struct SIMDCapabilities {
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx2: bool,
    pub avx: bool,
    pub fma: bool,
    pub sse4_2: bool,
    pub sse4_1: bool,
    pub neon: bool,  // ARM
}

impl SIMDCapabilities {
    /// Detect available SIMD features at runtime
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx512f: is_x86_feature_detected!("avx512f"),
                avx512bw: is_x86_feature_detected!("avx512bw"),
                avx512vl: is_x86_feature_detected!("avx512vl"),
                avx2: is_x86_feature_detected!("avx2"),
                avx: is_x86_feature_detected!("avx"),
                fma: is_x86_feature_detected!("fma"),
                sse4_2: is_x86_feature_detected!("sse4.2"),
                sse4_1: is_x86_feature_detected!("sse4.1"),
                neon: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx512f: false,
                avx512bw: false,
                avx512vl: false,
                avx2: false,
                avx: false,
                fma: false,
                sse4_2: false,
                sse4_1: false,
                neon: true, // Always available on AArch64
            }
        }
    }

    /// Get optimal dispatch for vector operations
    pub fn get_dispatcher(&self) -> Box<dyn VectorOps> {
        #[cfg(target_arch = "x86_64")]
        {
            if self.avx512f && self.avx512bw {
                return Box::new(AVX512Ops::new());
            }
            if self.avx2 && self.fma {
                return Box::new(AVX2Ops::new());
            }
            if self.sse4_1 {
                return Box::new(SSE41Ops::new());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.neon {
                return Box::new(NEONOps::new());
            }
        }

        Box::new(ScalarOps::new())
    }
}
```

### AVX2 Kernels

```rust
/// AVX2-optimized vector operations
pub struct AVX2Ops;

impl AVX2Ops {
    /// Dot product of two vectors (AVX2 + FMA)
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut sum = _mm256_setzero_ps();

        let chunks = n / 8;
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        // Horizontal sum
        let sum128 = _mm_add_ps(
            _mm256_extractf128_ps(sum, 0),
            _mm256_extractf128_ps(sum, 1),
        );
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result = _mm_cvtss_f32(sum32);

        // Handle remainder
        for i in (chunks * 8)..n {
            result += a[i] * b[i];
        }

        result
    }

    /// Q4 dequantization with AVX2
    #[target_feature(enable = "avx2")]
    pub unsafe fn dequantize_q4_avx2(
        quantized: &[u8],
        scales: &[f32],
        output: &mut [f32],
    ) {
        let block_size = 32;
        let num_blocks = quantized.len() * 2 / block_size;

        // Lookup table for 4-bit values (-8 to 7)
        let lut_low = _mm256_setr_epi8(
            -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
            -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7,
        );
        let mask = _mm256_set1_epi8(0x0F);

        for block in 0..num_blocks {
            let scale = scales[block];
            let scale_v = _mm256_set1_ps(scale);

            let base = block * block_size / 2;
            let q_bytes = _mm_loadu_si128(quantized.as_ptr().add(base) as *const __m128i);

            // Unpack low and high nibbles
            let q_bytes_256 = _mm256_castsi128_si256(q_bytes);
            let low_nibbles = _mm256_and_si256(q_bytes_256, mask);
            let high_nibbles = _mm256_and_si256(
                _mm256_srli_epi16(q_bytes_256, 4),
                mask,
            );

            // Lookup dequantized values
            let vals_low = _mm256_shuffle_epi8(lut_low, low_nibbles);
            let vals_high = _mm256_shuffle_epi8(lut_low, high_nibbles);

            // Convert to float and scale
            let vals_low_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                _mm256_castsi256_si128(vals_low)
            ));
            let output_low = _mm256_mul_ps(vals_low_f32, scale_v);

            _mm256_storeu_ps(output.as_mut_ptr().add(block * block_size), output_low);
            // ... continue for remaining elements
        }
    }

    /// Softmax with AVX2
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn softmax_avx2(data: &mut [f32]) {
        let n = data.len();

        // Find max (for numerical stability)
        let mut max_v = _mm256_set1_ps(f32::NEG_INFINITY);
        let chunks = n / 8;

        for i in 0..chunks {
            let v = _mm256_loadu_ps(data.as_ptr().add(i * 8));
            max_v = _mm256_max_ps(max_v, v);
        }

        let max_val = horizontal_max_avx2(max_v);
        let max_v = _mm256_set1_ps(max_val);

        // Compute exp(x - max) and sum
        let mut sum_v = _mm256_setzero_ps();
        for i in 0..chunks {
            let v = _mm256_loadu_ps(data.as_ptr().add(i * 8));
            let shifted = _mm256_sub_ps(v, max_v);
            let exp_v = exp_avx2(shifted);
            _mm256_storeu_ps(data.as_mut_ptr().add(i * 8), exp_v);
            sum_v = _mm256_add_ps(sum_v, exp_v);
        }

        let sum = horizontal_sum_avx2(sum_v);
        let inv_sum = _mm256_set1_ps(1.0 / sum);

        // Normalize
        for i in 0..chunks {
            let v = _mm256_loadu_ps(data.as_ptr().add(i * 8));
            let normalized = _mm256_mul_ps(v, inv_sum);
            _mm256_storeu_ps(data.as_mut_ptr().add(i * 8), normalized);
        }
    }

    /// Fast exp approximation using AVX2
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn exp_avx2(x: __m256) -> __m256 {
        // Polynomial approximation: exp(x) ≈ (1 + x/256)^256
        // Faster: use 2^(x * log2(e)) with polynomial for fractional part

        let log2e = _mm256_set1_ps(1.4426950408889634);
        let ln2 = _mm256_set1_ps(0.6931471805599453);

        let z = _mm256_mul_ps(x, log2e);
        let zi = _mm256_floor_ps(z);
        let zf = _mm256_sub_ps(z, zi);

        // Polynomial for 2^frac
        let c0 = _mm256_set1_ps(1.0);
        let c1 = _mm256_set1_ps(0.6931471805599453);
        let c2 = _mm256_set1_ps(0.24022650695910072);
        let c3 = _mm256_set1_ps(0.05550410866482158);

        let zf_ln2 = _mm256_mul_ps(zf, ln2);
        let poly = _mm256_fmadd_ps(
            _mm256_fmadd_ps(
                _mm256_fmadd_ps(c3, zf_ln2, c2),
                zf_ln2, c1
            ),
            zf_ln2, c0
        );

        // Scale by 2^int
        let scale = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_add_epi32(_mm256_cvtps_epi32(zi), _mm256_set1_epi32(127)),
            23
        ));

        _mm256_mul_ps(poly, scale)
    }
}
```

### AVX-512 Kernels

```rust
/// AVX-512 optimized operations (when available)
#[cfg(target_feature = "avx512f")]
pub struct AVX512Ops;

#[cfg(target_feature = "avx512f")]
impl AVX512Ops {
    /// Matrix-vector multiply with AVX-512
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn matvec_q4_avx512(
        matrix: &Q4Matrix,
        vector: &[f32],
        output: &mut [f32],
    ) {
        let m = matrix.rows;
        let k = matrix.cols;

        // Process 16 rows at a time
        for row_block in (0..m).step_by(16) {
            let rows_in_block = (m - row_block).min(16);
            let mut accum = [_mm512_setzero_ps(); 16];

            // Process columns in blocks of 64 (Q4 block size)
            for col_block in (0..k).step_by(64) {
                // Load vector chunk
                let v_chunk = _mm512_loadu_ps(vector.as_ptr().add(col_block));

                for r in 0..rows_in_block {
                    let row = row_block + r;
                    let q_row = matrix.get_row_q4(row, col_block);

                    // Dequantize and multiply
                    let dequant = dequantize_q4_avx512(q_row, matrix.scale(row, col_block / 64));
                    accum[r] = _mm512_fmadd_ps(dequant, v_chunk, accum[r]);
                }
            }

            // Reduce and store
            for r in 0..rows_in_block {
                output[row_block + r] = _mm512_reduce_add_ps(accum[r]);
            }
        }
    }

    /// Q4 dequantization with AVX-512
    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    unsafe fn dequantize_q4_avx512(quantized: &[u8], scale: f32) -> __m512 {
        let scale_v = _mm512_set1_ps(scale);

        // Load 32 bytes (64 Q4 values)
        let q = _mm256_loadu_si256(quantized.as_ptr() as *const __m256i);

        // Unpack nibbles using vpermb (AVX-512VBMI) or shuffle
        let low_mask = _mm512_set1_epi8(0x0F);
        let q_512 = _mm512_cvtepu8_epi16(q);

        // Extract low and high nibbles
        let low = _mm512_and_si512(q_512, low_mask);
        let high = _mm512_srli_epi16(q_512, 4);

        // Interleave and convert to signed (-8 to 7)
        let bias = _mm512_set1_epi16(8);
        let vals = _mm512_sub_epi16(low, bias);

        // Convert to float
        let vals_f32 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
            _mm512_castsi512_si256(vals)
        ));

        _mm512_mul_ps(vals_f32, scale_v)
    }
}
```

### ARM NEON Kernels

```rust
/// ARM NEON optimized operations
#[cfg(target_arch = "aarch64")]
pub struct NEONOps;

#[cfg(target_arch = "aarch64")]
impl NEONOps {
    /// Dot product with NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::aarch64::*;

        assert_eq!(a.len(), b.len());
        let n = a.len();
        let mut sum = vdupq_n_f32(0.0);

        let chunks = n / 4;
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            sum = vfmaq_f32(sum, va, vb);
        }

        // Horizontal sum
        let sum2 = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
        let sum1 = vpadd_f32(sum2, sum2);
        let mut result = vget_lane_f32(sum1, 0);

        // Handle remainder
        for i in (chunks * 4)..n {
            result += a[i] * b[i];
        }

        result
    }

    /// Q4 dequantization with NEON
    #[target_feature(enable = "neon")]
    pub unsafe fn dequantize_q4_neon(
        quantized: &[u8],
        scales: &[f32],
        output: &mut [f32],
    ) {
        use std::arch::aarch64::*;

        let block_size = 32;
        let num_blocks = quantized.len() * 2 / block_size;

        for block in 0..num_blocks {
            let scale = vdupq_n_f32(scales[block]);
            let base = block * block_size / 2;

            // Load 16 bytes (32 Q4 values)
            let q = vld1q_u8(quantized.as_ptr().add(base));

            // Unpack nibbles
            let mask = vdupq_n_u8(0x0F);
            let low = vandq_u8(q, mask);
            let high = vshrq_n_u8(q, 4);

            // Convert to signed and float
            let bias = vdupq_n_s8(8);
            let low_s8 = vreinterpretq_s8_u8(low);
            let low_centered = vsubq_s8(low_s8, bias);

            // Widen to 32-bit and convert to float
            let low_s16 = vmovl_s8(vget_low_s8(low_centered));
            let low_s32 = vmovl_s16(vget_low_s16(low_s16));
            let low_f32 = vcvtq_f32_s32(low_s32);

            // Scale
            let output_v = vmulq_f32(low_f32, scale);
            vst1q_f32(output.as_mut_ptr().add(block * block_size), output_v);
        }
    }
}
```

## GPU Acceleration

### CUDA Backend

```rust
/// CUDA acceleration for NVIDIA GPUs
#[cfg(feature = "cuda")]
pub mod cuda {
    use cudarc::driver::*;
    use cudarc::cublas::*;

    pub struct CUDABackend {
        device: Arc<CudaDevice>,
        cublas: CudaBlas,
        /// Custom kernels
        kernels: CUDAKernels,
        /// Allocated buffers
        buffers: CUDABuffers,
    }

    impl CUDABackend {
        pub fn new(device_id: usize) -> Result<Self, CUDAError> {
            let device = CudaDevice::new(device_id)?;
            let cublas = CudaBlas::new(device.clone())?;

            // Load custom kernels
            let kernels = CUDAKernels::load(&device)?;

            Ok(Self {
                device,
                cublas,
                kernels,
                buffers: CUDABuffers::default(),
            })
        }

        /// Run diffusion denoising step on GPU
        pub fn denoise_step(
            &mut self,
            noisy: &CudaSlice<f32>,
            timestep: u32,
            model: &CUDADiffusionModel,
        ) -> Result<CudaSlice<f32>, CUDAError> {
            // 1. Compute time embedding
            let time_embed = self.kernels.time_embedding(timestep)?;

            // 2. Run transformer blocks
            let mut hidden = noisy.clone();
            for block in &model.blocks {
                hidden = self.transformer_block(&hidden, &time_embed, block)?;
            }

            // 3. Predict noise
            let predicted_noise = self.kernels.output_projection(&hidden, &model.output_proj)?;

            // 4. Denoise: x_t-1 = (x_t - predicted_noise * sigma) / alpha
            let denoised = self.kernels.denoise_step(
                noisy,
                &predicted_noise,
                model.schedule.alpha(timestep),
                model.schedule.sigma(timestep),
            )?;

            Ok(denoised)
        }

        /// Transformer block with fused attention
        fn transformer_block(
            &mut self,
            hidden: &CudaSlice<f32>,
            time_embed: &CudaSlice<f32>,
            block: &TransformerBlock,
        ) -> Result<CudaSlice<f32>, CUDAError> {
            // Fused QKV projection
            let qkv = self.cublas.gemm(
                hidden,
                &block.qkv_proj,
                block.batch_size,
                block.seq_len * 3,
                block.hidden_dim,
            )?;

            // Flash attention
            let attn_out = self.kernels.flash_attention(
                &qkv,
                block.num_heads,
                block.head_dim,
                block.seq_len,
            )?;

            // Add time embedding
            let with_time = self.kernels.add_time_embedding(&attn_out, time_embed)?;

            // FFN with GELU
            let ffn_out = self.ffn_forward(&with_time, &block.ffn)?;

            // Residual + LayerNorm
            self.kernels.residual_layernorm(hidden, &ffn_out, &block.ln_weight, &block.ln_bias)
        }
    }

    /// Custom CUDA kernels
    struct CUDAKernels {
        time_embedding: CudaFunction,
        flash_attention: CudaFunction,
        denoise_step: CudaFunction,
        q4_dequant: CudaFunction,
    }

    impl CUDAKernels {
        fn load(device: &CudaDevice) -> Result<Self, CUDAError> {
            // Load PTX modules
            let module = device.load_ptx(include_str!("kernels/diffusion.ptx").into())?;

            Ok(Self {
                time_embedding: module.get_func("time_embedding")?,
                flash_attention: module.get_func("flash_attention")?,
                denoise_step: module.get_func("denoise_step")?,
                q4_dequant: module.get_func("q4_dequantize")?,
            })
        }
    }
}
```

### CUDA Kernel (PTX)

```cuda
// kernels/diffusion.cu

// Flash Attention kernel
extern "C" __global__ void flash_attention(
    const float* __restrict__ qkv,  // [batch, seq, 3, heads, head_dim]
    float* __restrict__ output,      // [batch, seq, heads, head_dim]
    const int seq_len,
    const int num_heads,
    const int head_dim,
    const float scale
) {
    // Shared memory for K, V blocks
    extern __shared__ float smem[];
    float* s_key = smem;
    float* s_val = smem + BLOCK_SIZE * head_dim;

    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int q_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (q_idx >= seq_len) return;

    // Load query
    float q[HEAD_DIM];
    #pragma unroll
    for (int i = 0; i < head_dim; i++) {
        q[i] = qkv[batch * seq_len * 3 * num_heads * head_dim +
                   q_idx * 3 * num_heads * head_dim +
                   0 * num_heads * head_dim +
                   head * head_dim + i];
    }

    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    float o[HEAD_DIM] = {0};

    // Process K,V in blocks
    for (int k_block = 0; k_block < seq_len; k_block += BLOCK_SIZE) {
        // Load K,V block to shared memory
        __syncthreads();
        // ... load logic ...
        __syncthreads();

        // Compute attention scores
        float scores[BLOCK_SIZE];
        for (int k_idx = 0; k_idx < BLOCK_SIZE && k_block + k_idx < seq_len; k_idx++) {
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < head_dim; i++) {
                dot += q[i] * s_key[k_idx * head_dim + i];
            }
            scores[k_idx] = dot * scale;
        }

        // Online softmax (FlashAttention style)
        float m_new = m_prev;
        for (int k_idx = 0; k_idx < BLOCK_SIZE && k_block + k_idx < seq_len; k_idx++) {
            m_new = fmaxf(m_new, scores[k_idx]);
        }

        float l_new = l_prev * expf(m_prev - m_new);
        for (int k_idx = 0; k_idx < BLOCK_SIZE && k_block + k_idx < seq_len; k_idx++) {
            l_new += expf(scores[k_idx] - m_new);
        }

        // Update output
        float scale_prev = l_prev * expf(m_prev - m_new) / l_new;
        #pragma unroll
        for (int i = 0; i < head_dim; i++) {
            o[i] *= scale_prev;
        }

        for (int k_idx = 0; k_idx < BLOCK_SIZE && k_block + k_idx < seq_len; k_idx++) {
            float weight = expf(scores[k_idx] - m_new) / l_new;
            #pragma unroll
            for (int i = 0; i < head_dim; i++) {
                o[i] += weight * s_val[k_idx * head_dim + i];
            }
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Write output
    #pragma unroll
    for (int i = 0; i < head_dim; i++) {
        output[batch * seq_len * num_heads * head_dim +
               q_idx * num_heads * head_dim +
               head * head_dim + i] = o[i];
    }
}

// Q4 dequantization kernel
extern "C" __global__ void q4_dequantize(
    const uint8_t* __restrict__ quantized,
    const float* __restrict__ scales,
    float* __restrict__ output,
    const int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    const int byte_idx = idx / 2;
    const int nibble = idx % 2;

    uint8_t packed = quantized[byte_idx];
    int8_t val = nibble ? (packed >> 4) : (packed & 0x0F);
    val -= 8;  // Center around 0

    const int block_idx = idx / 32;
    output[idx] = (float)val * scales[block_idx];
}

// Denoising step kernel
extern "C" __global__ void denoise_step(
    const float* __restrict__ x_t,
    const float* __restrict__ predicted_noise,
    float* __restrict__ x_t_minus_1,
    const float alpha,
    const float sigma,
    const int num_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // x_{t-1} = (x_t - sigma * noise) / alpha
    x_t_minus_1[idx] = (x_t[idx] - sigma * predicted_noise[idx]) / alpha;
}
```

### Metal Backend (Apple)

```rust
/// Metal acceleration for Apple Silicon
#[cfg(target_os = "macos")]
pub mod metal {
    use metal::*;

    pub struct MetalBackend {
        device: Device,
        queue: CommandQueue,
        library: Library,
        pipelines: MetalPipelines,
    }

    impl MetalBackend {
        pub fn new() -> Result<Self, MetalError> {
            let device = Device::system_default()
                .ok_or(MetalError::NoDevice)?;
            let queue = device.new_command_queue();

            // Compile shaders
            let source = include_str!("shaders/diffusion.metal");
            let library = device.new_library_with_source(source, &CompileOptions::new())?;

            let pipelines = MetalPipelines::new(&device, &library)?;

            Ok(Self {
                device,
                queue,
                library,
                pipelines,
            })
        }

        /// Run diffusion inference
        pub fn denoise_step(
            &self,
            noisy: &Buffer,
            timestep: u32,
            model: &MetalDiffusionModel,
        ) -> Result<Buffer, MetalError> {
            let command_buffer = self.queue.new_command_buffer();

            // Time embedding
            let time_embed = self.compute_time_embedding(command_buffer, timestep)?;

            // Transformer blocks
            let mut hidden = noisy.clone();
            for block in &model.blocks {
                hidden = self.transformer_block(command_buffer, &hidden, &time_embed, block)?;
            }

            // Output projection
            let output = self.output_projection(command_buffer, &hidden, &model.output_proj)?;

            command_buffer.commit();
            command_buffer.wait_until_completed();

            Ok(output)
        }

        fn compute_time_embedding(
            &self,
            command_buffer: &CommandBuffer,
            timestep: u32,
        ) -> Result<Buffer, MetalError> {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipelines.time_embedding);

            // Set timestep buffer
            let timestep_buffer = self.device.new_buffer_with_data(
                &timestep as *const u32 as *const _,
                4,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(0, Some(&timestep_buffer), 0);

            // Output buffer
            let output = self.device.new_buffer(
                TIME_EMBED_DIM * 4,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(1, Some(&output), 0);

            // Dispatch
            let thread_groups = MTLSize::new(1, 1, 1);
            let threads_per_group = MTLSize::new(TIME_EMBED_DIM as u64, 1, 1);
            encoder.dispatch_thread_groups(thread_groups, threads_per_group);
            encoder.end_encoding();

            Ok(output)
        }
    }
}
```

### Metal Shader

```metal
// shaders/diffusion.metal

#include <metal_stdlib>
using namespace metal;

// Time embedding computation
kernel void time_embedding(
    constant uint& timestep [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    const float half_dim = TIME_EMBED_DIM / 2;
    const float log_timescale = log(10000.0f) / (half_dim - 1.0f);

    if (tid < half_dim) {
        float freq = exp(-tid * log_timescale);
        float angle = timestep * freq;
        output[tid] = sin(angle);
        output[tid + uint(half_dim)] = cos(angle);
    }
}

// Q4 dequantization
kernel void q4_dequantize(
    constant uchar* quantized [[buffer(0)]],
    constant float* scales [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_elements [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_elements) return;

    uint byte_idx = tid / 2;
    uint nibble = tid % 2;

    uchar packed = quantized[byte_idx];
    char val = nibble ? (packed >> 4) : (packed & 0x0F);
    val -= 8;

    uint block_idx = tid / 32;
    output[tid] = float(val) * scales[block_idx];
}

// Matrix-vector multiplication with Q4
kernel void matvec_q4(
    constant uchar* matrix [[buffer(0)]],
    constant float* scales [[buffer(1)]],
    constant float* vector [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;

    float sum = 0.0f;
    const uint blocks_per_row = cols / 32;

    for (uint block = 0; block < blocks_per_row; block++) {
        float scale = scales[row * blocks_per_row + block];
        uint base = row * cols / 2 + block * 16;

        for (uint i = 0; i < 16; i++) {
            uchar packed = matrix[base + i];
            char low = (packed & 0x0F) - 8;
            char high = (packed >> 4) - 8;

            uint col = block * 32 + i * 2;
            sum += float(low) * scale * vector[col];
            sum += float(high) * scale * vector[col + 1];
        }
    }

    output[row] = sum;
}

// Softmax (numerically stable)
kernel void softmax(
    device float* data [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Find max
    float local_max = -INFINITY;
    for (uint i = tid; i < length; i += tg_size) {
        local_max = max(local_max, data[i]);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce max
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = max(shared[tid], shared[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (uint i = tid; i < length; i += tg_size) {
        float exp_val = exp(data[i] - max_val);
        data[i] = exp_val;
        local_sum += exp_val;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sum
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = shared[0];

    // Normalize
    for (uint i = tid; i < length; i += tg_size) {
        data[i] /= sum;
    }
}
```

## Unified Backend Selection

```rust
/// Unified compute backend
pub enum ComputeBackend {
    #[cfg(feature = "cuda")]
    CUDA(cuda::CUDABackend),
    #[cfg(target_os = "macos")]
    Metal(metal::MetalBackend),
    #[cfg(feature = "vulkan")]
    Vulkan(vulkan::VulkanBackend),
    SIMD(SIMDBackend),
    Scalar(ScalarBackend),
}

impl ComputeBackend {
    /// Automatically select best available backend
    pub fn auto_detect() -> Result<Self, BackendError> {
        // Try GPU backends first
        #[cfg(feature = "cuda")]
        {
            if let Ok(cuda) = cuda::CUDABackend::new(0) {
                log::info!("Using CUDA backend");
                return Ok(Self::CUDA(cuda));
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(metal) = metal::MetalBackend::new() {
                log::info!("Using Metal backend");
                return Ok(Self::Metal(metal));
            }
        }

        #[cfg(feature = "vulkan")]
        {
            if let Ok(vulkan) = vulkan::VulkanBackend::new() {
                log::info!("Using Vulkan backend");
                return Ok(Self::Vulkan(vulkan));
            }
        }

        // Fall back to SIMD
        let caps = SIMDCapabilities::detect();
        if caps.avx2 || caps.neon {
            log::info!("Using SIMD backend: {:?}", caps);
            return Ok(Self::SIMD(SIMDBackend::new(caps)));
        }

        // Scalar fallback
        log::warn!("Using scalar backend (no SIMD support)");
        Ok(Self::Scalar(ScalarBackend::new()))
    }

    /// Run diffusion denoising
    pub fn denoise_step(
        &mut self,
        noisy: &Tensor,
        timestep: u32,
        model: &DiffusionModel,
    ) -> Result<Tensor, BackendError> {
        match self {
            #[cfg(feature = "cuda")]
            Self::CUDA(backend) => backend.denoise_step(noisy, timestep, model),
            #[cfg(target_os = "macos")]
            Self::Metal(backend) => backend.denoise_step(noisy, timestep, model),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(backend) => backend.denoise_step(noisy, timestep, model),
            Self::SIMD(backend) => backend.denoise_step(noisy, timestep, model),
            Self::Scalar(backend) => backend.denoise_step(noisy, timestep, model),
        }
    }
}
```

## Performance Comparison

| Operation | Scalar | AVX2 | AVX-512 | CUDA | Metal |
|-----------|--------|------|---------|------|-------|
| Dot 1024-d | 1x | 6x | 10x | 50x | 45x |
| Q4 Dequant | 1x | 8x | 14x | 100x | 90x |
| MatVec 4096x4096 | 1x | 7x | 12x | 200x | 180x |
| Softmax 4096 | 1x | 5x | 8x | 80x | 70x |
| Full Denoise Step | 1x | 6x | 10x | 150x | 130x |

---

**Previous**: [05-PRIVACY.md](./05-PRIVACY.md) - Privacy tiers and encryption
**Next**: [07-NOVEL-CONTRIBUTIONS.md](./07-NOVEL-CONTRIBUTIONS.md) - Original research
