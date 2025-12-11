# Benchmark Specifications

## TRM + RuvLLM Performance Validation

---

## 1. Benchmark Overview

### 1.1 Benchmark Categories

| Category | Purpose | Metrics |
|----------|---------|---------|
| **Latency** | Response time validation | ms, p50/p95/p99 |
| **Throughput** | Queries per second | QPS |
| **Quality** | Reasoning accuracy | % correct |
| **Learning** | K prediction accuracy | % within 2 of optimal |
| **Memory** | Resource consumption | MB |
| **Scaling** | Performance vs K | ms/iteration |

### 1.2 Target Metrics Summary

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| K=1 latency | <10ms | <5ms |
| K=10 latency | <100ms | <50ms |
| K=20 latency | <200ms | <100ms |
| Throughput (K=5) | >100 QPS | >200 QPS |
| Memory footprint | <100MB | <50MB |
| WASM bundle | <50MB | <30MB |
| K prediction accuracy | >75% | >85% |

---

## 2. Latency Benchmarks

### 2.1 Single Iteration Latency

```rust
// benches/trm_latency.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use ruvllm::trm::{TrmEngine, TrmConfig, MlpLatentUpdater, AttentionLatentUpdater};

fn bench_single_latent_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("Single Latent Update");

    for dim in [64, 128, 256, 512] {
        let mut mlp = MlpLatentUpdater::new(dim, dim);
        let question = vec![0.1; dim];
        let answer = vec![0.2; dim];
        let mut latent = vec![0.0; dim];

        group.bench_with_input(BenchmarkId::new("MLP", dim), &dim, |b, _| {
            b.iter(|| {
                mlp.update(&question, &answer, &mut latent);
            });
        });
    }

    group.finish();
}

fn bench_single_answer_refinement(c: &mut Criterion) {
    let config = TrmConfig::builder()
        .hidden_dim(256)
        .embedding_dim(256)
        .build()
        .unwrap();

    let mut refiner = AnswerRefiner::new(&config);
    let question = vec![0.1; 256];
    let latent = vec![0.5; 256];
    let mut answer = vec![0.2; 256 * 10];

    c.bench_function("Answer Refinement (256d, 10 tokens)", |b| {
        b.iter(|| {
            refiner.refine(&question, &latent, &mut answer);
        });
    });
}

fn bench_confidence_scoring(c: &mut Criterion) {
    let config = TrmConfig::default();
    let scorer = ConfidenceScorer::new(&config);
    let answer = vec![0.5; 256 * 10];

    c.bench_function("Confidence Scoring", |b| {
        b.iter(|| {
            scorer.score(&answer)
        });
    });
}
```

**Target Results**:
```
Single Latent Update/MLP/64      time: [45.2 µs 46.1 µs 47.0 µs]
Single Latent Update/MLP/128     time: [89.3 µs 91.2 µs 93.1 µs]
Single Latent Update/MLP/256     time: [178 µs 182 µs 186 µs]
Single Latent Update/MLP/512     time: [356 µs 364 µs 372 µs]

Answer Refinement (256d, 10tok)  time: [234 µs 241 µs 248 µs]
Confidence Scoring               time: [12.3 µs 12.8 µs 13.2 µs]
```

### 2.2 Full K-Iteration Latency

```rust
fn bench_full_k_iterations(c: &mut Criterion) {
    let config = TrmConfig::builder()
        .hidden_dim(256)
        .embedding_dim(256)
        .use_attention(false)  // MLP variant
        .build()
        .unwrap();

    let mut engine = TrmEngine::new(config);
    let question = vec![0.1; 256];
    let answer = vec![0.2; 256 * 10];

    let mut group = c.benchmark_group("Full TRM K-Iterations");

    for k in [1, 2, 5, 10, 15, 20] {
        group.bench_with_input(BenchmarkId::new("K", k), &k, |b, &k| {
            b.iter(|| {
                engine.reason(&question, &answer, Some(k))
            });
        });
    }

    group.finish();
}
```

**Target Results**:
```
Full TRM K-Iterations/K/1        time: [2.1 ms 2.2 ms 2.3 ms]
Full TRM K-Iterations/K/2        time: [4.2 ms 4.4 ms 4.6 ms]
Full TRM K-Iterations/K/5        time: [10.3 ms 10.8 ms 11.2 ms]
Full TRM K-Iterations/K/10       time: [20.5 ms 21.4 ms 22.3 ms]
Full TRM K-Iterations/K/15       time: [30.8 ms 32.1 ms 33.4 ms]
Full TRM K-Iterations/K/20       time: [41.0 ms 42.8 ms 44.6 ms]
```

### 2.3 MLP vs Attention Comparison

```rust
fn bench_mlp_vs_attention(c: &mut Criterion) {
    let question = vec![0.1; 256];
    let answer = vec![0.2; 256 * 10];

    let mut group = c.benchmark_group("MLP vs Attention");

    // MLP variant
    let mlp_config = TrmConfig::builder()
        .hidden_dim(256)
        .use_attention(false)
        .build()
        .unwrap();
    let mut mlp_engine = TrmEngine::new(mlp_config);

    group.bench_function("MLP K=10", |b| {
        b.iter(|| mlp_engine.reason(&question, &answer, Some(10)));
    });

    // Attention variant
    let attn_config = TrmConfig::builder()
        .hidden_dim(256)
        .use_attention(true)
        .num_heads(8)
        .build()
        .unwrap();
    let mut attn_engine = TrmEngine::new(attn_config);

    group.bench_function("Attention K=10", |b| {
        b.iter(|| attn_engine.reason(&question, &answer, Some(10)));
    });

    group.finish();
}
```

**Target Results**:
```
MLP vs Attention/MLP K=10        time: [21.4 ms 22.1 ms 22.8 ms]
MLP vs Attention/Attention K=10  time: [45.2 ms 46.8 ms 48.4 ms]
```

---

## 3. Throughput Benchmarks

### 3.1 Queries Per Second

```rust
fn bench_throughput(c: &mut Criterion) {
    let config = TrmConfig::builder()
        .hidden_dim(256)
        .build()
        .unwrap();

    let engine = Arc::new(Mutex::new(TrmEngine::new(config)));
    let question = vec![0.1; 256];
    let answer = vec![0.2; 256 * 10];

    let mut group = c.benchmark_group("Throughput");
    group.throughput(Throughput::Elements(1));

    // Single-threaded
    group.bench_function("Single Thread K=5", |b| {
        let mut engine = engine.lock().unwrap();
        b.iter(|| {
            engine.reason(&question, &answer, Some(5))
        });
    });

    group.finish();
}

fn bench_concurrent_throughput(c: &mut Criterion) {
    use std::sync::Arc;
    use rayon::prelude::*;

    let config = TrmConfig::default();

    c.bench_function("Concurrent 8 Threads K=5", |b| {
        b.iter(|| {
            let results: Vec<_> = (0..8).into_par_iter()
                .map(|_| {
                    let mut engine = TrmEngine::new(config.clone());
                    let question = vec![0.1; 256];
                    let answer = vec![0.2; 256 * 10];
                    engine.reason(&question, &answer, Some(5))
                })
                .collect();
            results
        });
    });
}
```

**Target Results**:
```
Throughput/Single Thread K=5     QPS: ~90
Throughput/8 Threads K=5         QPS: ~600 (total across threads)
```

---

## 4. Quality Benchmarks

### 4.1 Sudoku-Extreme Benchmark

```rust
// benches/trm_quality.rs

/// Sudoku-Extreme benchmark following TRM paper methodology
fn bench_sudoku_extreme() {
    let test_cases = load_sudoku_extreme_dataset();  // 1000 puzzles

    let config = TrmConfig::builder()
        .hidden_dim(256)
        .max_k(30)
        .latent_iterations(6)  // n=6 per paper
        .build()
        .unwrap();

    let mut engine = TrmEngine::new(config);
    let encoder = SudokuEncoder::new();
    let decoder = SudokuDecoder::new();

    let mut correct = 0;
    let mut total_k = 0;

    for (i, puzzle) in test_cases.iter().enumerate() {
        let question = encoder.encode_puzzle(puzzle);
        let initial = encoder.encode_empty_solution();

        let result = engine.reason(&question, &initial, None);
        let solution = decoder.decode(&result.answer);

        if verify_sudoku(&solution) && matches_puzzle(puzzle, &solution) {
            correct += 1;
        }

        total_k += result.iterations_used;

        if i % 100 == 0 {
            println!("Progress: {}/1000, Accuracy: {:.1}%",
                     i, correct as f32 / (i + 1) as f32 * 100.0);
        }
    }

    let accuracy = correct as f32 / test_cases.len() as f32 * 100.0;
    let avg_k = total_k as f32 / test_cases.len() as f32;

    println!("Sudoku-Extreme Results:");
    println!("  Accuracy: {:.1}%", accuracy);
    println!("  Avg K: {:.1}", avg_k);
    println!("  Target: >80%");

    assert!(accuracy > 80.0, "Sudoku accuracy below target");
}
```

**Target**: >80% accuracy (TRM paper: ~87% with MLP)

### 4.2 Maze-Hard Benchmark

```rust
fn bench_maze_hard() {
    let test_cases = load_maze_hard_dataset();

    let config = TrmConfig::builder()
        .hidden_dim(256)
        .max_k(20)
        .build()
        .unwrap();

    let mut engine = TrmEngine::new(config);
    let encoder = MazeEncoder::new();
    let decoder = MazeDecoder::new();

    let mut correct = 0;
    let mut path_valid = 0;

    for maze in &test_cases {
        let question = encoder.encode_maze(maze);
        let initial = encoder.encode_start_position(maze);

        let result = engine.reason(&question, &initial, None);
        let path = decoder.decode(&result.answer);

        if verify_maze_solution(maze, &path) {
            correct += 1;
        }

        if is_valid_path(maze, &path) {
            path_valid += 1;
        }
    }

    let accuracy = correct as f32 / test_cases.len() as f32 * 100.0;
    let validity = path_valid as f32 / test_cases.len() as f32 * 100.0;

    println!("Maze-Hard Results:");
    println!("  Solve Rate: {:.1}%", accuracy);
    println!("  Valid Paths: {:.1}%", validity);
    println!("  Target: >70%");

    assert!(accuracy > 70.0, "Maze solve rate below target");
}
```

**Target**: >70% solve rate

---

## 5. Learning Benchmarks

### 5.1 K Prediction Accuracy

```rust
fn bench_k_prediction_accuracy() {
    let sona = SonaEngine::new(256);
    let bridge = TrmSonaBridge::new(sona);
    let mut trm = TrmEngine::new(TrmConfig::default());

    // Training phase: 500 queries with known optimal K
    let training_set: Vec<(Vec<f32>, usize)> = generate_training_queries(500);

    for (query, optimal_k) in &training_set {
        // Run with generous max_k
        let result = trm.reason(query, &vec![0.0; 256], Some(30));

        // Tell SONA the true optimal K
        bridge.learn_from_trm_with_optimal_k(query, &result, optimal_k);
    }

    bridge.sona.force_learn();

    // Test phase: 200 queries
    let test_set: Vec<(Vec<f32>, usize)> = generate_test_queries(200);

    let mut within_1 = 0;
    let mut within_2 = 0;
    let mut total_error = 0.0;

    for (query, optimal_k) in &test_set {
        let predicted_k = bridge.predict_optimal_k(query, 30);
        let error = (predicted_k as i32 - *optimal_k as i32).abs();

        if error <= 1 { within_1 += 1; }
        if error <= 2 { within_2 += 1; }
        total_error += error as f32;
    }

    let accuracy_1 = within_1 as f32 / test_set.len() as f32 * 100.0;
    let accuracy_2 = within_2 as f32 / test_set.len() as f32 * 100.0;
    let mae = total_error / test_set.len() as f32;

    println!("K Prediction Results:");
    println!("  Within 1 of optimal: {:.1}%", accuracy_1);
    println!("  Within 2 of optimal: {:.1}%", accuracy_2);
    println!("  Mean Absolute Error: {:.2}", mae);
    println!("  Target: >75% within 2");

    assert!(accuracy_2 > 75.0, "K prediction accuracy below target");
}
```

**Target**: >75% within 2 of optimal K

### 5.2 Learning Speed

```rust
fn bench_learning_convergence() {
    let sona = SonaEngine::new(256);
    let bridge = TrmSonaBridge::new(sona);

    // Track accuracy over training
    let mut accuracies = vec![];

    for batch in 0..10 {
        // Train with 50 examples per batch
        for _ in 0..50 {
            let query = generate_random_query(256);
            let optimal_k = determine_optimal_k(&query);

            let result = TrmResult {
                iterations_used: optimal_k,
                // ... other fields
            };

            bridge.learn_from_trm(&query, &result, 0.9);
        }

        bridge.sona.force_learn();

        // Evaluate
        let accuracy = evaluate_k_prediction(&bridge, 100);
        accuracies.push(accuracy);

        println!("Batch {}: Accuracy {:.1}%", batch, accuracy);
    }

    // Should improve over time
    assert!(accuracies.last().unwrap() > accuracies.first().unwrap(),
            "Learning should improve accuracy");

    // Should reach >70% by batch 5
    assert!(accuracies[4] > 70.0, "Should reach 70% by 250 examples");
}
```

---

## 6. Memory Benchmarks

### 6.1 Memory Footprint

```rust
fn bench_memory_footprint() {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

    struct TrackingAllocator;

    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOCATED.fetch_add(layout.size(), Ordering::SeqCst);
            System.alloc(layout)
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            ALLOCATED.fetch_sub(layout.size(), Ordering::SeqCst);
            System.dealloc(ptr, layout)
        }
    }

    // Measure TRM engine memory
    let before = ALLOCATED.load(Ordering::SeqCst);

    let config = TrmConfig::builder()
        .hidden_dim(256)
        .embedding_dim(256)
        .build()
        .unwrap();

    let engine = TrmEngine::new(config);

    let after = ALLOCATED.load(Ordering::SeqCst);
    let trm_memory = (after - before) / 1024 / 1024;

    println!("TRM Engine Memory: {}MB", trm_memory);
    assert!(trm_memory < 50, "TRM engine should use <50MB");

    // Measure with SONA
    let bridge = TrmSonaBridge::new(SonaEngine::new(256));
    let after_sona = ALLOCATED.load(Ordering::SeqCst);
    let total_memory = (after_sona - before) / 1024 / 1024;

    println!("TRM + SONA Memory: {}MB", total_memory);
    assert!(total_memory < 100, "TRM + SONA should use <100MB");
}
```

**Target**: <100MB total

### 6.2 WASM Bundle Size

```bash
#!/bin/bash
# scripts/measure_wasm_size.sh

# Build WASM
wasm-pack build --target web --release

# Measure sizes
echo "WASM Bundle Sizes:"
ls -lh pkg/*.wasm | awk '{print $5, $9}'

# With compression
gzip -k pkg/*.wasm
echo "Compressed:"
ls -lh pkg/*.wasm.gz | awk '{print $5, $9}'

# Target: <50MB uncompressed, <15MB compressed
```

**Target**: <50MB uncompressed, <15MB gzipped

---

## 7. Benchmark Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TRM + RuvLLM Benchmark Dashboard                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LATENCY                          │  THROUGHPUT                             │
│  ─────────                        │  ──────────                             │
│  K=1:   ████████░░ 2.2ms (<10)    │  Single:  █████████░ 90 QPS (>100)     │
│  K=5:   ████████░░ 10.8ms (<50)   │  8-Thread: ██████████ 600 QPS          │
│  K=10:  ███████░░░ 21.4ms (<100)  │                                         │
│  K=20:  ████████░░ 42.8ms (<200)  │  MEMORY                                 │
│                                    │  ──────                                 │
│  QUALITY                           │  TRM Engine:  ████░░░ 35MB (<50)       │
│  ───────                           │  + SONA:      ██████░ 65MB (<100)      │
│  Sudoku: ████████░░ 82% (>80)      │  WASM Bundle: ████░░░ 42MB (<50)       │
│  Maze:   ███████░░░ 74% (>70)      │                                         │
│                                    │                                         │
│  LEARNING                          │  SCALING                                │
│  ────────                          │  ───────                                │
│  K Pred: ████████░░ 78% (>75)      │  Linear scaling with K ✓               │
│  MAE:    ██░░░░░░░░ 1.8 (<2)       │  MLP 2x faster than Attn ✓             │
│                                    │                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. CI Integration

```yaml
# .github/workflows/benchmark.yml

name: TRM Benchmarks

on:
  push:
    branches: [main, feature/trm-*]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Run Benchmarks
        run: |
          cd examples/ruvLLM
          cargo bench --features trm -- --save-baseline new

      - name: Compare to Baseline
        run: |
          cargo bench --features trm -- --baseline main --save-baseline new

      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: target/criterion/

      - name: Check Regression
        run: |
          # Fail if >10% regression
          python scripts/check_benchmark_regression.py --threshold 0.1
```

---

**Next**: [07_OPTIMIZATION.md](./07_OPTIMIZATION.md) - Performance Optimization Guide
