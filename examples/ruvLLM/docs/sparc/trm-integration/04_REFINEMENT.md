# SPARC Phase 4: Refinement

## TDD Implementation Plan

---

## 1. Test-Driven Development Strategy

### 1.1 TDD Principles for TRM Integration

```
RED → GREEN → REFACTOR

1. Write a failing test for desired behavior
2. Implement minimum code to pass
3. Refactor for clarity and performance
4. Repeat
```

### 1.2 Test Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        Test Pyramid                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                         ┌───────┐                               │
│                         │  E2E  │  5%                           │
│                         │ Tests │                               │
│                       ┌─┴───────┴─┐                             │
│                       │Integration│  20%                        │
│                       │   Tests   │                             │
│                     ┌─┴───────────┴─┐                           │
│                     │  Unit Tests   │  75%                      │
│                     │               │                           │
│                     └───────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Implementation Milestones

### Milestone 1: Core TRM Engine (Week 1-2)

#### Sprint 1.1: Configuration & Types

**Tests First**:
```rust
// tests/trm/config_test.rs

#[test]
fn test_default_config() {
    let config = TrmConfig::default();
    assert_eq!(config.hidden_dim, 256);
    assert_eq!(config.max_k, 20);
    assert_eq!(config.default_k, 5);
}

#[test]
fn test_config_builder() {
    let config = TrmConfig::builder()
        .hidden_dim(512)
        .max_k(30)
        .use_attention(true)
        .build();

    assert_eq!(config.hidden_dim, 512);
    assert!(config.use_attention);
}

#[test]
fn test_config_validation() {
    let result = TrmConfig::builder()
        .hidden_dim(0)  // Invalid
        .build();

    assert!(result.is_err());
}
```

**Implementation**:
- [ ] `src/trm/config.rs` - Configuration struct
- [ ] `src/trm/types.rs` - Core types (TrmResult, TrmTrajectory)
- [ ] `src/trm/error.rs` - Error types

#### Sprint 1.2: MLP Latent Updater

**Tests First**:
```rust
// tests/trm/mlp_test.rs

#[test]
fn test_mlp_updater_dimensions() {
    let updater = MlpLatentUpdater::new(256, 256);

    let question = vec![0.1; 256];
    let answer = vec![0.2; 256];
    let mut latent = vec![0.0; 256];

    updater.update(&question, &answer, &mut latent);

    assert_eq!(latent.len(), 256);
    // Latent should be modified
    assert!(latent.iter().any(|&x| x != 0.0));
}

#[test]
fn test_mlp_updater_convergence() {
    let mut updater = MlpLatentUpdater::new(64, 64);

    let question = vec![1.0; 64];
    let answer = vec![1.0; 64];
    let mut latent = vec![0.0; 64];

    // Run multiple iterations
    for _ in 0..10 {
        updater.update(&question, &answer, &mut latent);
    }

    // Should converge to stable state
    let prev_latent = latent.clone();
    updater.update(&question, &answer, &mut latent);

    let diff: f32 = latent.iter()
        .zip(prev_latent.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(diff < 0.1, "Latent should converge");
}

#[test]
fn test_mlp_layer_norm() {
    let mut updater = MlpLatentUpdater::new(64, 64);

    let question = vec![100.0; 64];  // Large values
    let answer = vec![100.0; 64];
    let mut latent = vec![0.0; 64];

    updater.update(&question, &answer, &mut latent);

    // Layer norm should keep values bounded
    let max_val = latent.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(max_val < 10.0, "Layer norm should bound values");
}
```

**Implementation**:
- [ ] `src/trm/mlp.rs` - MLP latent updater
- [ ] Weight initialization (Xavier)
- [ ] GELU activation
- [ ] Gated residual connection
- [ ] Layer normalization

#### Sprint 1.3: Attention Latent Updater

**Tests First**:
```rust
// tests/trm/attention_test.rs

#[test]
fn test_attention_updater_dimensions() {
    let updater = AttentionLatentUpdater::new(256, 256, 8);

    let question = vec![0.1; 256];
    let answer = vec![0.2; 256];
    let mut latent = vec![0.0; 256];

    updater.update(&question, &answer, &mut latent);

    assert_eq!(latent.len(), 256);
}

#[test]
fn test_attention_head_count() {
    // Must be divisible by num_heads
    let result = std::panic::catch_unwind(|| {
        AttentionLatentUpdater::new(256, 256, 7)  // 256 not divisible by 7
    });

    assert!(result.is_err());
}

#[test]
fn test_attention_scores_normalized() {
    let updater = AttentionLatentUpdater::new(64, 64, 4);

    // Access internal attention scores
    let scores = updater.get_last_attention_scores();

    // Each row should sum to 1 (softmax)
    for row in scores.chunks(2) {  // 2 context positions
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
```

**Implementation**:
- [ ] `src/trm/attention.rs` - Multi-head cross-attention
- [ ] Query/Key/Value projections
- [ ] Scaled dot-product attention
- [ ] Output projection with residual

#### Sprint 1.4: Answer Refiner

**Tests First**:
```rust
// tests/trm/refiner_test.rs

#[test]
fn test_answer_refiner_preserves_structure() {
    let config = TrmConfig::default();
    let refiner = AnswerRefiner::new(&config);

    let question = vec![0.1; 256];
    let latent = vec![0.5; 256];
    let mut answer = vec![1.0; 256 * 10];  // 10 tokens

    let original = answer.clone();
    refiner.refine(&question, &latent, &mut answer);

    // Answer should be modified but not drastically
    let diff: f32 = answer.iter()
        .zip(original.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(diff > 0.0, "Answer should be refined");
    assert!(diff < answer.len() as f32, "Refinement should be gradual");
}

#[test]
fn test_answer_refiner_residual_scale() {
    let config = TrmConfig::default();
    let refiner = AnswerRefiner::new(&config);

    let question = vec![0.0; 256];  // Zero input
    let latent = vec![0.0; 256];
    let mut answer = vec![1.0; 256];

    refiner.refine(&question, &latent, &mut answer);

    // With zero inputs, answer should change minimally
    let max_diff = answer.iter().map(|x| (x - 1.0).abs()).fold(0.0f32, f32::max);
    assert!(max_diff < 0.2, "Zero input should cause minimal change");
}
```

**Implementation**:
- [ ] `src/trm/refiner.rs` - Answer refinement network
- [ ] Latent expansion
- [ ] Two-layer transformation
- [ ] Residual connection with 0.1 scale

#### Sprint 1.5: Confidence Scorer

**Tests First**:
```rust
// tests/trm/confidence_test.rs

#[test]
fn test_confidence_range() {
    let config = TrmConfig::default();
    let scorer = ConfidenceScorer::new(&config);

    let answer = vec![0.5; 256];
    let confidence = scorer.score(&answer);

    assert!(confidence >= 0.0 && confidence <= 1.0);
}

#[test]
fn test_confidence_improves_with_structure() {
    let config = TrmConfig::default();
    let scorer = ConfidenceScorer::new(&config);

    // Random noise
    let random_answer: Vec<f32> = (0..256).map(|i| (i as f32 * 0.618) % 1.0).collect();
    let random_conf = scorer.score(&random_answer);

    // Structured answer (more coherent)
    let structured_answer = vec![0.8; 256];
    let structured_conf = scorer.score(&structured_answer);

    assert!(structured_conf > random_conf, "Structure should increase confidence");
}
```

**Implementation**:
- [ ] `src/trm/confidence.rs` - Confidence scoring
- [ ] Answer embedding pooling
- [ ] Learned confidence head

### Milestone 2: TRM Engine Integration (Week 2-3)

#### Sprint 2.1: Main TRM Engine

**Tests First**:
```rust
// tests/trm/engine_test.rs

#[test]
fn test_trm_engine_creation() {
    let config = TrmConfig::default();
    let engine = TrmEngine::new(config);

    assert_eq!(engine.config().hidden_dim, 256);
}

#[test]
fn test_trm_single_iteration() {
    let config = TrmConfig::builder()
        .hidden_dim(64)
        .embedding_dim(64)
        .build()
        .unwrap();

    let mut engine = TrmEngine::new(config);

    let question = vec![0.1; 64];
    let answer = vec![0.2; 64];

    let result = engine.reason(&question, &answer, Some(1));

    assert_eq!(result.iterations_used, 1);
    assert!(result.confidence >= 0.0);
    assert_eq!(result.trajectory.states.len(), 1);
}

#[test]
fn test_trm_early_stopping() {
    let config = TrmConfig::builder()
        .hidden_dim(64)
        .confidence_threshold(0.5)  // Low threshold
        .build()
        .unwrap();

    let mut engine = TrmEngine::new(config);

    let question = vec![1.0; 64];
    let answer = vec![1.0; 64];

    let result = engine.reason(&question, &answer, Some(20));

    assert!(result.early_stopped || result.iterations_used <= 20);
}

#[test]
fn test_trm_trajectory_recording() {
    let config = TrmConfig::builder()
        .hidden_dim(64)
        .build()
        .unwrap();

    let mut engine = TrmEngine::new(config);

    let result = engine.reason(&vec![0.1; 64], &vec![0.2; 64], Some(5));

    assert_eq!(result.trajectory.states.len(), result.iterations_used);

    for state in &result.trajectory.states {
        assert_eq!(state.latent_state.len(), 64);
        assert!(state.confidence >= 0.0 && state.confidence <= 1.0);
        assert!(state.latency_us > 0);
    }
}

#[test]
fn test_trm_latency_target() {
    let config = TrmConfig::builder()
        .hidden_dim(256)
        .build()
        .unwrap();

    let mut engine = TrmEngine::new(config);

    let start = std::time::Instant::now();
    let result = engine.reason(&vec![0.1; 256], &vec![0.2; 256], Some(10));
    let elapsed = start.elapsed();

    // Target: <100ms for K=10
    assert!(elapsed.as_millis() < 100, "K=10 should complete in <100ms");
}
```

**Implementation**:
- [ ] `src/trm/engine.rs` - Main TRM engine
- [ ] LatentUpdater enum (MLP/Attention)
- [ ] Main reasoning loop
- [ ] Early stopping logic
- [ ] Trajectory recording
- [ ] Pre-allocated buffers

### Milestone 3: SONA Integration (Week 3-4)

#### Sprint 3.1: TRM-SONA Bridge

**Tests First**:
```rust
// tests/trm/sona_bridge_test.rs

#[test]
fn test_trajectory_conversion() {
    let sona = SonaEngine::new(64);
    let bridge = TrmSonaBridge::new(sona);

    let trm_result = TrmResult {
        answer: vec![0.5; 64],
        confidence: 0.8,
        iterations_used: 5,
        early_stopped: false,
        trajectory: TrmTrajectory {
            states: vec![
                TrmIterationState {
                    iteration: 0,
                    latent_state: vec![0.1; 64],
                    answer_state: vec![0.2; 64],
                    confidence: 0.3,
                    latency_us: 1000,
                },
            ],
            optimal_k: 5,
            total_latency_us: 5000,
        },
        latency_us: 5000,
    };

    let query = vec![0.1; 64];
    bridge.learn_from_trm(&query, &trm_result, 0.9);

    // Verify learning was recorded
    let stats = bridge.sona.stats();
    assert!(stats.trajectories_buffered > 0);
}

#[test]
fn test_optimal_k_prediction_default() {
    let sona = SonaEngine::new(64);
    let bridge = TrmSonaBridge::new(sona);

    // No patterns yet, should return default
    let k = bridge.predict_optimal_k(&vec![0.1; 64], 20);
    assert_eq!(k, 5);
}

#[test]
fn test_optimal_k_prediction_learned() {
    let sona = SonaEngine::new(64);
    let bridge = TrmSonaBridge::new(sona);

    // Train with several patterns
    for i in 0..50 {
        let query = vec![(i as f32) * 0.01; 64];
        let result = TrmResult {
            iterations_used: 3,  // Always K=3
            // ... other fields
        };
        bridge.learn_from_trm(&query, &result, 0.9);
    }

    bridge.sona.force_learn();

    // Similar query should predict K~3
    let k = bridge.predict_optimal_k(&vec![0.25; 64], 20);
    assert!(k >= 2 && k <= 5, "Should predict K close to trained value");
}
```

**Implementation**:
- [ ] `src/trm/sona_bridge.rs` - Bridge implementation
- [ ] Trajectory conversion
- [ ] K prediction with similarity lookup
- [ ] Learning submission

#### Sprint 3.2: Router Update for K Prediction

**Tests First**:
```rust
// tests/trm/router_test.rs

#[test]
fn test_router_trm_decision() {
    let router = FastGRNNRouter::new(128, 64);

    let embedding = vec![0.5; 128];
    let decision = router.route_with_trm(&embedding);

    assert!(decision.k_value >= 1 && decision.k_value <= 20);
    assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
}

#[test]
fn test_router_complexity_scaling() {
    let router = FastGRNNRouter::new(128, 64);

    // Simple query (low entropy embedding)
    let simple = vec![1.0; 128];
    let simple_decision = router.route_with_trm(&simple);

    // Complex query (high entropy embedding)
    let complex: Vec<f32> = (0..128).map(|i| (i as f32 * 0.618) % 1.0).collect();
    let complex_decision = router.route_with_trm(&complex);

    assert!(complex_decision.k_value >= simple_decision.k_value,
            "Complex queries should get more iterations");
}
```

**Implementation**:
- [ ] Update `src/router.rs` - Add K prediction output head
- [ ] TrmRoutingDecision struct
- [ ] Complexity estimation

### Milestone 4: SIMD Optimization (Week 4-5)

#### Sprint 4.1: SIMD Matrix Operations

**Tests First**:
```rust
// tests/simd_test.rs

#[test]
fn test_simd_matmul_correctness() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 0.0, 0.0, 1.0];
    let mut c = vec![0.0; 4];

    SimdOps::matmul(&a, &b, &mut c, 2, 2, 2);

    // Expected: [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
    assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_simd_layer_norm() {
    let mut x = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0; 4];
    let beta = vec![0.0; 4];

    SimdOps::layer_norm_inplace(&mut x, &gamma, &beta, 1e-5);

    // Check mean is ~0
    let mean: f32 = x.iter().sum::<f32>() / 4.0;
    assert!(mean.abs() < 1e-5);

    // Check variance is ~1
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 4.0;
    assert!((var - 1.0).abs() < 1e-4);
}

#[test]
fn test_simd_gelu() {
    let mut x = vec![-1.0, 0.0, 1.0, 2.0];

    SimdOps::gelu_inplace(&mut x);

    assert!(x[0] < 0.0);    // GELU(-1) < 0
    assert!(x[1] == 0.0);   // GELU(0) = 0
    assert!(x[2] > 0.0);    // GELU(1) > 0
    assert!(x[3] > x[2]);   // GELU(2) > GELU(1)
}

#[test]
fn test_simd_performance() {
    let size = 256 * 256;
    let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
    let mut c = vec![0.0; size];

    let start = std::time::Instant::now();
    for _ in 0..100 {
        SimdOps::matmul(&a, &b, &mut c, 256, 256, 256);
    }
    let elapsed = start.elapsed();

    // 100 256x256 matmuls should complete in <1s
    assert!(elapsed.as_secs() < 1);
}
```

**Implementation**:
- [ ] AVX2 matrix multiplication
- [ ] SSE4.1 fallback
- [ ] SIMD layer normalization
- [ ] SIMD GELU/sigmoid

### Milestone 5: Benchmarking (Week 5)

#### Sprint 5.1: Benchmark Suite

**Tests/Benchmarks**:
```rust
// benches/trm_recursive.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_trm_k_values(c: &mut Criterion) {
    let config = TrmConfig::builder()
        .hidden_dim(256)
        .build()
        .unwrap();

    let mut engine = TrmEngine::new(config);
    let question = vec![0.1; 256];
    let answer = vec![0.2; 256];

    let mut group = c.benchmark_group("TRM K-values");

    for k in [1, 5, 10, 15, 20] {
        group.bench_with_input(BenchmarkId::new("K", k), &k, |b, &k| {
            b.iter(|| engine.reason(&question, &answer, Some(k)));
        });
    }

    group.finish();
}

fn bench_trm_variants(c: &mut Criterion) {
    let question = vec![0.1; 256];
    let answer = vec![0.2; 256];

    let mut group = c.benchmark_group("TRM Variants");

    // MLP variant
    let mlp_config = TrmConfig::builder()
        .use_attention(false)
        .build()
        .unwrap();
    let mut mlp_engine = TrmEngine::new(mlp_config);

    group.bench_function("MLP", |b| {
        b.iter(|| mlp_engine.reason(&question, &answer, Some(10)));
    });

    // Attention variant
    let attn_config = TrmConfig::builder()
        .use_attention(true)
        .num_heads(8)
        .build()
        .unwrap();
    let mut attn_engine = TrmEngine::new(attn_config);

    group.bench_function("Attention", |b| {
        b.iter(|| attn_engine.reason(&question, &answer, Some(10)));
    });

    group.finish();
}

fn bench_sona_integration(c: &mut Criterion) {
    let sona = SonaEngine::new(256);
    let bridge = TrmSonaBridge::new(sona);

    let mut trm = TrmEngine::new(TrmConfig::default());

    c.bench_function("TRM+SONA full pipeline", |b| {
        let query = vec![0.1; 256];
        let answer = vec![0.2; 256];

        b.iter(|| {
            let k = bridge.predict_optimal_k(&query, 20);
            let result = trm.reason(&query, &answer, Some(k));
            bridge.learn_from_trm(&query, &result, 0.8);
        });
    });
}

criterion_group!(benches, bench_trm_k_values, bench_trm_variants, bench_sona_integration);
criterion_main!(benches);
```

**Implementation**:
- [ ] `benches/trm_recursive.rs` - TRM benchmarks
- [ ] `benches/trm_sona.rs` - Integration benchmarks

### Milestone 6: WASM Compilation (Week 5-6)

#### Sprint 6.1: WASM Build

**Tests**:
```rust
// tests/wasm_test.rs

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test]
fn test_wasm_trm_engine() {
    let engine = WasmTrmEngine::new(256);
    let result = engine.reason(&[0.1; 256], &[0.2; 256], 5);

    assert!(result.confidence() >= 0.0);
    assert!(result.iterations_used() <= 5);
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test]
fn test_wasm_memory_usage() {
    // Should not exceed 50MB
    let engine = WasmTrmEngine::new(256);

    // Run several iterations
    for _ in 0..100 {
        let _ = engine.reason(&[0.1; 256], &[0.2; 256], 10);
    }

    // Memory should be stable (pool reuse)
    let memory = wasm_bindgen::memory();
    let pages = memory.buffer().byte_length() / 65536;
    assert!(pages < 800, "Should use <50MB"); // 800 pages * 64KB = 50MB
}
```

**Implementation**:
- [ ] WASM feature flag
- [ ] Memory pool for WASM
- [ ] SIMD128 detection
- [ ] JavaScript bindings

---

## 3. Iteration Plan

### Week 1: Foundation
- Day 1-2: Config, types, error handling
- Day 3-4: MLP latent updater
- Day 5: Testing and debugging

### Week 2: Core Engine
- Day 1-2: Attention latent updater
- Day 3: Answer refiner
- Day 4: Confidence scorer
- Day 5: Main TRM engine

### Week 3: SONA Integration
- Day 1-2: TRM-SONA bridge
- Day 3: K prediction
- Day 4: Router updates
- Day 5: Integration testing

### Week 4: Optimization
- Day 1-2: SIMD matrix operations
- Day 3: SIMD layer norm, activations
- Day 4: Memory optimization
- Day 5: Performance profiling

### Week 5: Benchmarking & WASM
- Day 1-2: Benchmark suite
- Day 3: WASM compilation
- Day 4: JS bindings
- Day 5: Browser testing

### Week 6: Release Preparation
- Day 1-2: Documentation
- Day 3: Package configuration
- Day 4: CI/CD setup
- Day 5: Release

---

## 4. Quality Gates

### Gate 1: Unit Tests Pass
- All 75+ unit tests passing
- Code coverage >80%

### Gate 2: Performance Targets
- K=10 latency <100ms
- Memory <100MB
- WASM bundle <50MB

### Gate 3: Integration Tests
- TRM+SONA pipeline working
- Memory caching working
- Router K prediction working

### Gate 4: Benchmarks
- Sudoku benchmark >80%
- Maze benchmark >70%
- K prediction accuracy >75%

### Gate 5: Release Ready
- Documentation complete
- Package builds on all platforms
- Examples working

---

**Next**: [05_COMPLETION.md](./05_COMPLETION.md) - Integration & Release
