# Performance Benchmarks

## Overview

This document defines performance targets, benchmark methodology, and expected results for RuvDLLM. All benchmarks are designed to be reproducible and comparable to existing solutions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Performance Targets Summary                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INFERENCE (7B model, 8 denoising steps)                                    │
│  ├─ GPU (CUDA/Metal): <50ms latency, 1000+ tok/s                           │
│  ├─ CPU (AVX2): <100ms latency, 200+ tok/s                                 │
│  └─ CPU (Scalar): <500ms latency, 50+ tok/s                                │
│                                                                              │
│  ADAPTATION                                                                  │
│  ├─ TALoRA retrieval: <0.1ms per query                                      │
│  ├─ DGR overhead: <10% per step                                             │
│  └─ MicroLoRA application: <1ms total                                       │
│                                                                              │
│  MEMORY (7B Q4 model)                                                        │
│  ├─ Base model: ~4GB                                                         │
│  ├─ With TALoRA (100K patterns): +200MB                                     │
│  └─ Total inference: <6GB                                                    │
│                                                                              │
│  FEDERATION                                                                  │
│  ├─ Round latency: <2 minutes                                               │
│  ├─ Bandwidth per round: <10MB                                              │
│  └─ Minimum nodes: 3                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Inference Benchmarks

### Latency Targets

| Configuration | Target | Stretch Goal | Baseline (Python) |
|--------------|--------|--------------|-------------------|
| GPU (7B, 8 steps) | <50ms | <30ms | ~200ms |
| GPU (13B, 8 steps) | <100ms | <60ms | ~400ms |
| CPU AVX2 (7B, 8 steps) | <200ms | <100ms | ~2000ms |
| CPU AVX512 (7B, 8 steps) | <150ms | <80ms | N/A |
| CPU NEON (7B, 8 steps) | <250ms | <150ms | ~2500ms |

### Throughput Targets

| Configuration | Target | Stretch Goal |
|--------------|--------|--------------|
| GPU (batch=1) | 1000+ tok/s | 2000+ tok/s |
| GPU (batch=8) | 4000+ tok/s | 8000+ tok/s |
| CPU AVX2 (batch=1) | 200+ tok/s | 500+ tok/s |
| CPU NEON (batch=1) | 150+ tok/s | 300+ tok/s |

### Benchmark Code

```rust
/// Inference latency benchmark
pub fn benchmark_inference_latency(
    model: &DiffusionModel,
    backend: &mut ComputeBackend,
    config: &BenchmarkConfig,
) -> LatencyResults {
    let mut results = LatencyResults::default();

    // Warmup
    for _ in 0..config.warmup_iterations {
        let _ = generate_sample(model, backend, config.prompt_length, config.output_length);
    }

    // Benchmark
    let mut latencies = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        let start = Instant::now();
        let _ = generate_sample(model, backend, config.prompt_length, config.output_length);
        latencies.push(start.elapsed());
    }

    // Statistics
    latencies.sort();
    results.p50 = latencies[latencies.len() / 2];
    results.p90 = latencies[latencies.len() * 9 / 10];
    results.p99 = latencies[latencies.len() * 99 / 100];
    results.mean = latencies.iter().sum::<Duration>() / latencies.len() as u32;

    results
}

/// Throughput benchmark
pub fn benchmark_throughput(
    model: &DiffusionModel,
    backend: &mut ComputeBackend,
    config: &BenchmarkConfig,
) -> ThroughputResults {
    let mut results = ThroughputResults::default();

    let duration = Duration::from_secs(config.duration_secs);
    let start = Instant::now();
    let mut total_tokens = 0usize;

    while start.elapsed() < duration {
        let output = generate_sample(model, backend, config.prompt_length, config.output_length);
        total_tokens += output.len();
    }

    results.tokens_per_second = total_tokens as f64 / start.elapsed().as_secs_f64();
    results.total_tokens = total_tokens;

    results
}
```

## Adaptation Benchmarks

### TALoRA Retrieval

| Metric | Target | Notes |
|--------|--------|-------|
| Single query (10K patterns) | <0.1ms | HNSW with ef=50 |
| Single query (100K patterns) | <0.2ms | HNSW with ef=50 |
| Batch query (10 queries) | <0.5ms | Parallelized |
| Index build (100K patterns) | <30s | One-time cost |

### DGR Overhead

| Metric | Target | Notes |
|--------|--------|-------|
| Uncertainty computation | <1ms | SIMD-optimized |
| Position selection | <0.1ms | Top-k selection |
| Per-position retrieval | <0.5ms | With TALoRA |
| Total DGR overhead | <10% | Per denoising step |

### Benchmark Code

```rust
/// TALoRA retrieval benchmark
pub fn benchmark_talora_retrieval(
    talora: &TALoRAManager,
    config: &TALoRABenchmarkConfig,
) -> TALoRAResults {
    let queries: Vec<Vec<f32>> = (0..config.num_queries)
        .map(|_| random_embedding(config.hidden_dim))
        .collect();

    let mut results = TALoRAResults::default();

    // Warmup
    for query in queries.iter().take(10) {
        let _ = talora.retrieve(query, 500, config.top_k);
    }

    // Benchmark
    let start = Instant::now();
    for query in &queries {
        let _ = talora.retrieve(query, 500, config.top_k);
    }
    let total_time = start.elapsed();

    results.avg_latency = total_time / config.num_queries as u32;
    results.queries_per_second = config.num_queries as f64 / total_time.as_secs_f64();

    results
}

/// DGR overhead benchmark
pub fn benchmark_dgr_overhead(
    model: &DiffusionModel,
    dgr: &DGRSystem,
    config: &DGRBenchmarkConfig,
) -> DGRResults {
    let input = random_input(config.seq_length);

    // Without DGR
    let start = Instant::now();
    for _ in 0..config.iterations {
        let _ = model.forward(&input, 500, None);
    }
    let baseline_time = start.elapsed();

    // With DGR
    let start = Instant::now();
    for _ in 0..config.iterations {
        let (hidden, logits) = model.forward_with_hidden(&input, 500);
        let dgr_result = dgr.process(&logits, &hidden, 500);
        let mut output = logits.clone();
        dgr.apply(&mut output, &dgr_result);
    }
    let dgr_time = start.elapsed();

    DGRResults {
        baseline_latency: baseline_time / config.iterations as u32,
        dgr_latency: dgr_time / config.iterations as u32,
        overhead_percent: (dgr_time.as_secs_f64() / baseline_time.as_secs_f64() - 1.0) * 100.0,
    }
}
```

## Memory Benchmarks

### Memory Targets

| Component | Target | Notes |
|-----------|--------|-------|
| 7B Q4 model weights | ~3.5GB | 4-bit quantization |
| KV cache (2048 ctx) | ~500MB | Q8 cache |
| TALoRA banks (100K) | ~200MB | 8-bit compressed |
| Working memory | ~500MB | Intermediate tensors |
| **Total (inference)** | **<6GB** | Fits in 8GB GPU |

### Memory Benchmark Code

```rust
/// Memory usage benchmark
pub fn benchmark_memory(config: &MemoryBenchmarkConfig) -> MemoryResults {
    let mut results = MemoryResults::default();

    // Baseline
    let baseline = get_memory_usage();

    // Load model
    let model = DiffusionModel::load_q4(&config.model_path).unwrap();
    results.model_size = get_memory_usage() - baseline;

    // Create TALoRA
    let talora = TALoRAManager::new(config.talora_config.clone());
    let after_talora = get_memory_usage();
    results.talora_empty = after_talora - baseline - results.model_size;

    // Populate TALoRA
    for _ in 0..config.num_patterns {
        let adapter = random_adapter(config.talora_config.fine_rank, config.hidden_dim);
        let embedding = random_embedding(config.hidden_dim);
        talora.store(adapter, &embedding, 100);
    }
    results.talora_populated = get_memory_usage() - after_talora;

    // Inference working memory
    let start_mem = get_memory_usage();
    let _ = generate_sample(&model, &config.prompt, config.output_length);
    results.working_memory = get_peak_memory() - start_mem;

    results.total = get_memory_usage();
    results
}
```

## Federation Benchmarks

### Federation Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Round latency (10 nodes) | <2 min | Including aggregation |
| Bandwidth per node | <10MB | Compressed deltas |
| Convergence (100 rounds) | 90%+ | Of centralized baseline |
| Byzantine tolerance | 30%+ | Malicious nodes |

### Benchmark Code

```rust
/// Federation round benchmark
pub fn benchmark_federation_round(
    coordinator: &FederationCoordinator,
    nodes: &[MockNode],
    config: &FederationBenchmarkConfig,
) -> FederationResults {
    let mut results = FederationResults::default();

    // Generate mock updates
    let updates: Vec<_> = nodes
        .iter()
        .map(|n| n.generate_update())
        .collect();

    // Benchmark aggregation
    let start = Instant::now();
    let aggregated = coordinator.aggregate(updates).unwrap();
    results.aggregation_time = start.elapsed();

    // Benchmark distribution
    let start = Instant::now();
    for node in nodes {
        node.receive_update(&aggregated);
    }
    results.distribution_time = start.elapsed();

    // Measure bandwidth
    results.update_size_bytes = updates.iter().map(|u| u.size_bytes()).sum::<usize>();
    results.aggregated_size_bytes = aggregated.size_bytes();

    results
}
```

## Quality Benchmarks

### Perplexity Targets

| Configuration | Target PPL | vs Baseline |
|--------------|------------|-------------|
| Base diffusion | <20 | Baseline |
| + TALoRA | <18 | -10% |
| + DGR | <17 | -15% |
| + Federation | <16 | -20% |

### Task-Specific Metrics

| Task | Metric | Target |
|------|--------|--------|
| Text completion | BLEU | >0.4 |
| Summarization | ROUGE-L | >0.35 |
| Translation | BLEU | >0.3 |
| Code generation | Pass@1 | >0.2 |

### Benchmark Code

```rust
/// Quality benchmark on standard datasets
pub fn benchmark_quality(
    model: &DiffusionModel,
    talora: Option<&TALoRAManager>,
    dgr: Option<&DGRSystem>,
    dataset: &QualityBenchmarkDataset,
) -> QualityResults {
    let mut results = QualityResults::default();

    // Perplexity
    let mut total_log_prob = 0.0;
    let mut total_tokens = 0;

    for sample in &dataset.samples {
        let log_probs = model.compute_log_probs(&sample.text);
        total_log_prob += log_probs.sum();
        total_tokens += sample.text.len();
    }

    results.perplexity = (-total_log_prob / total_tokens as f64).exp();

    // Task metrics
    for (task_name, task_data) in &dataset.tasks {
        let predictions: Vec<String> = task_data.inputs
            .iter()
            .map(|input| generate_with_adapters(model, input, talora, dgr))
            .collect();

        let metrics = compute_task_metrics(task_name, &predictions, &task_data.targets);
        results.task_metrics.insert(task_name.clone(), metrics);
    }

    results
}
```

## Comparison with Baselines

### vs dLLM (Python)

| Metric | dLLM (Python) | RuvDLLM | Improvement |
|--------|---------------|---------|-------------|
| Inference latency (GPU) | ~200ms | <50ms | 4x |
| Memory overhead | ~4GB | ~500MB | 8x |
| Tokens/sec (GPU) | ~250 | 1000+ | 4x |
| Real-time adaptation | No | Yes | ∞ |

### vs DiffuLLaMA

| Metric | DiffuLLaMA | RuvDLLM | Notes |
|--------|------------|---------|-------|
| Timestep awareness | None | TALoRA | Novel |
| Dynamic retrieval | None | DGR | Novel |
| CPU inference | Slow | Fast | SIMD |
| Federation | None | DAF | Novel |

### vs Standard LLM (AR)

| Metric | AR LLM | RuvDLLM | Trade-off |
|--------|--------|---------|-----------|
| Steps to generate | N | 8 | Lower |
| Parallelism | None | Full | Higher |
| Edit capability | Limited | Native | Higher |
| Model size | Same | Same | Neutral |

## Benchmark Suite

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench --features "cuda benchmark"

# Run specific benchmark
cargo bench --features "benchmark" -- inference_latency

# Generate report
cargo bench --features "benchmark" -- --save-baseline main

# Compare with baseline
cargo bench --features "benchmark" -- --baseline main
```

### Benchmark Configuration

```toml
# bench.toml
[inference]
warmup_iterations = 10
iterations = 100
prompt_length = 128
output_length = 256
denoising_steps = 8

[talora]
num_patterns = 100000
num_queries = 1000
top_k = 5

[dgr]
iterations = 100
seq_length = 512

[memory]
num_patterns = 100000

[federation]
num_nodes = 10
rounds = 100

[quality]
dataset = "benchmarks/quality_eval"
tasks = ["completion", "summarization", "translation"]
```

## Continuous Benchmarking

### CI Integration

```yaml
# .github/workflows/benchmark.yml
name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run benchmarks
        run: cargo bench --features benchmark -- --save-baseline pr

      - name: Compare with main
        run: |
          git fetch origin main
          git checkout origin/main -- target/criterion
          cargo bench --features benchmark -- --baseline main

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: target/criterion
```

### Performance Regression Detection

```rust
/// Check for performance regressions
pub fn check_regressions(
    current: &BenchmarkResults,
    baseline: &BenchmarkResults,
    thresholds: &RegressionThresholds,
) -> Vec<Regression> {
    let mut regressions = Vec::new();

    // Latency regression
    let latency_ratio = current.inference_latency.p50.as_secs_f64()
        / baseline.inference_latency.p50.as_secs_f64();
    if latency_ratio > thresholds.latency_increase_percent / 100.0 + 1.0 {
        regressions.push(Regression {
            metric: "inference_latency_p50",
            baseline: baseline.inference_latency.p50,
            current: current.inference_latency.p50,
            threshold: thresholds.latency_increase_percent,
        });
    }

    // Throughput regression
    let throughput_ratio = current.throughput.tokens_per_second
        / baseline.throughput.tokens_per_second;
    if throughput_ratio < 1.0 - thresholds.throughput_decrease_percent / 100.0 {
        regressions.push(Regression {
            metric: "throughput",
            baseline: baseline.throughput.tokens_per_second,
            current: current.throughput.tokens_per_second,
            threshold: thresholds.throughput_decrease_percent,
        });
    }

    // Memory regression
    let memory_ratio = current.memory.total as f64 / baseline.memory.total as f64;
    if memory_ratio > thresholds.memory_increase_percent / 100.0 + 1.0 {
        regressions.push(Regression {
            metric: "memory_usage",
            baseline: baseline.memory.total,
            current: current.memory.total,
            threshold: thresholds.memory_increase_percent,
        });
    }

    regressions
}
```

## Reporting

### Benchmark Report Template

```markdown
# RuvDLLM Benchmark Report

**Date**: {{date}}
**Commit**: {{commit}}
**Hardware**: {{hardware}}

## Summary

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Inference Latency (GPU) | <50ms | {{gpu_latency}}ms | {{status}} |
| Throughput (GPU) | >1000 tok/s | {{throughput}} tok/s | {{status}} |
| Memory Usage | <6GB | {{memory}}GB | {{status}} |
| TALoRA Retrieval | <0.1ms | {{talora_latency}}ms | {{status}} |
| DGR Overhead | <10% | {{dgr_overhead}}% | {{status}} |

## Detailed Results

### Inference
{{inference_details}}

### Adaptation
{{adaptation_details}}

### Memory
{{memory_details}}

### Quality
{{quality_details}}
```

---

**Previous**: [09-API-REFERENCE.md](./09-API-REFERENCE.md) - Module API design

---

## Appendix: Hardware Test Configurations

### GPU Configurations
- NVIDIA RTX 4090 (24GB)
- NVIDIA A100 (40GB)
- Apple M2 Max (32GB unified)

### CPU Configurations
- AMD EPYC 7763 (64 cores, AVX2)
- Intel Xeon W-3375 (32 cores, AVX-512)
- Apple M2 Max (12 cores, NEON)

### Memory Configurations
- 32GB DDR5-5600
- 64GB DDR5-4800
- 128GB DDR4-3200
