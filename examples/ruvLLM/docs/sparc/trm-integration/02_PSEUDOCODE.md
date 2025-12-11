# SPARC Phase 2: Pseudocode

## TinyRecursiveModels Algorithm Design

---

## 1. Core TRM Algorithm

### 1.1 Main Recursive Reasoning Loop

```
ALGORITHM TRM_Recursive_Reasoning
INPUT:
    question: Embedded question tensor [batch, seq_len, dim]
    initial_answer: Initial answer embedding [batch, ans_len, dim]
    K: Maximum improvement iterations
    n: Latent update iterations per K step

OUTPUT:
    final_answer: Refined answer [batch, ans_len, dim]
    confidence: Confidence score [0, 1]
    trajectory: List of intermediate states

BEGIN
    // Initialize
    latent := ZEROS([batch, hidden_dim])
    answer := initial_answer
    trajectory := EMPTY_LIST

    // Main recursive loop (K iterations)
    FOR k := 1 TO K DO

        // Phase 1: Recursive latent updates (n iterations)
        FOR i := 1 TO n DO
            // Combine question context, current answer, and latent state
            combined := CONCAT(question_pooled, answer_pooled, latent)

            // Update latent through MLP or Attention
            IF use_attention THEN
                latent := Attention_Latent_Update(combined, latent)
            ELSE
                latent := MLP_Latent_Update(combined, latent)
            END IF

            // Apply layer normalization
            latent := LayerNorm(latent)
        END FOR

        // Phase 2: Answer refinement using updated latent
        answer := Answer_Refine(question, answer, latent)

        // Record trajectory for SONA learning
        trajectory.APPEND({
            iteration: k,
            latent_state: latent.CLONE(),
            answer_state: answer.CLONE(),
            confidence: Compute_Confidence(answer)
        })

        // Early stopping if confident
        IF Compute_Confidence(answer) > CONFIDENCE_THRESHOLD THEN
            BREAK
        END IF

    END FOR

    final_answer := answer
    confidence := Compute_Confidence(final_answer)

    RETURN (final_answer, confidence, trajectory)
END
```

### 1.2 MLP Latent Update

```
ALGORITHM MLP_Latent_Update
INPUT:
    combined: Combined input [batch, combined_dim]
    latent: Current latent state [batch, hidden_dim]

OUTPUT:
    new_latent: Updated latent state [batch, hidden_dim]

BEGIN
    // Two-layer MLP with residual connection
    hidden := Linear_1(combined)  // combined_dim -> hidden_dim * 4
    hidden := GELU(hidden)
    hidden := Dropout(hidden, p=0.1)

    delta := Linear_2(hidden)     // hidden_dim * 4 -> hidden_dim

    // Residual connection with gating
    gate := Sigmoid(Linear_gate(latent))  // hidden_dim -> hidden_dim
    new_latent := gate * latent + (1 - gate) * delta

    RETURN new_latent
END
```

### 1.3 Attention Latent Update

```
ALGORITHM Attention_Latent_Update
INPUT:
    combined: Combined context [batch, seq_len, dim]
    latent: Current latent state [batch, hidden_dim]

OUTPUT:
    new_latent: Updated latent state [batch, hidden_dim]

BEGIN
    // Multi-head cross-attention
    num_heads := 8
    head_dim := hidden_dim / num_heads

    // Latent as query
    Q := Linear_Q(latent.UNSQUEEZE(1))  // [batch, 1, hidden_dim]

    // Combined context as key/value
    K := Linear_K(combined)              // [batch, seq_len, hidden_dim]
    V := Linear_V(combined)              // [batch, seq_len, hidden_dim]

    // Reshape for multi-head
    Q := Q.RESHAPE([batch, 1, num_heads, head_dim]).TRANSPOSE(1, 2)
    K := K.RESHAPE([batch, seq_len, num_heads, head_dim]).TRANSPOSE(1, 2)
    V := V.RESHAPE([batch, seq_len, num_heads, head_dim]).TRANSPOSE(1, 2)

    // Scaled dot-product attention
    scores := MATMUL(Q, K.TRANSPOSE(-2, -1)) / SQRT(head_dim)
    attention := Softmax(scores, dim=-1)
    attended := MATMUL(attention, V)

    // Reshape back
    attended := attended.TRANSPOSE(1, 2).RESHAPE([batch, 1, hidden_dim])
    attended := attended.SQUEEZE(1)  // [batch, hidden_dim]

    // Output projection with residual
    output := Linear_O(attended)
    new_latent := LayerNorm(latent + output)

    RETURN new_latent
END
```

### 1.4 Answer Refinement

```
ALGORITHM Answer_Refine
INPUT:
    question: Question embedding [batch, q_len, dim]
    answer: Current answer [batch, a_len, dim]
    latent: Updated latent state [batch, hidden_dim]

OUTPUT:
    refined_answer: Improved answer [batch, a_len, dim]

BEGIN
    // Expand latent to match answer sequence length
    latent_expanded := latent.UNSQUEEZE(1).EXPAND([batch, a_len, hidden_dim])

    // Combine answer with latent information
    combined := CONCAT(answer, latent_expanded, dim=-1)

    // Transform through answer refinement network
    hidden := Linear_1(combined)  // (dim + hidden_dim) -> dim * 2
    hidden := GELU(hidden)
    hidden := Linear_2(hidden)    // dim * 2 -> dim

    // Residual refinement
    refined_answer := answer + 0.1 * hidden  // Small refinement step

    RETURN refined_answer
END
```

---

## 2. SONA Integration Algorithms

### 2.1 Trajectory Recording for TRM

```
ALGORITHM Record_TRM_Trajectory
INPUT:
    query_embedding: Query embedding [dim]
    trajectory: List of TRM iteration states
    final_quality: Task success score [0, 1]
    optimal_k: Actual K used (for learning)

OUTPUT:
    sona_trajectory: SONA-compatible trajectory

BEGIN
    sona_trajectory := NEW QueryTrajectory()
    sona_trajectory.embedding := query_embedding
    sona_trajectory.quality := final_quality
    sona_trajectory.metadata := {
        "optimal_k": optimal_k,
        "task_type": Infer_Task_Type(query_embedding),
        "early_stopped": optimal_k < MAX_K
    }

    FOR EACH step IN trajectory DO
        sona_step := NEW TrajectoryStep()
        sona_step.activations := step.latent_state
        sona_step.token_id := step.iteration  // Use iteration as "token"
        sona_step.confidence := step.confidence
        sona_step.latency_us := Measure_Step_Latency()

        sona_trajectory.steps.APPEND(sona_step)
    END FOR

    RETURN sona_trajectory
END
```

### 2.2 Optimal K Prediction

```
ALGORITHM Predict_Optimal_K
INPUT:
    query_embedding: Query embedding [dim]
    sona_engine: SONA engine reference
    max_k: Maximum allowed K

OUTPUT:
    predicted_k: Predicted optimal recursion depth

BEGIN
    // Search for similar queries in ReasoningBank
    similar_patterns := sona_engine.find_patterns(query_embedding, k=5)

    IF similar_patterns.IS_EMPTY() THEN
        // No similar patterns, use default
        RETURN DEFAULT_K  // e.g., 5
    END IF

    // Weighted average of similar queries' optimal K
    total_weight := 0.0
    weighted_k := 0.0

    FOR EACH pattern IN similar_patterns DO
        similarity := Cosine_Similarity(query_embedding, pattern.embedding)
        k_value := pattern.metadata["optimal_k"]

        weighted_k := weighted_k + similarity * k_value
        total_weight := total_weight + similarity
    END FOR

    predicted_k := ROUND(weighted_k / total_weight)
    predicted_k := CLAMP(predicted_k, 1, max_k)

    RETURN predicted_k
END
```

### 2.3 Learn Optimal K via MicroLoRA

```
ALGORITHM Learn_K_From_Trajectory
INPUT:
    trajectory: Completed TRM trajectory with quality score
    sona_engine: SONA engine reference

OUTPUT:
    learning_applied: Boolean

BEGIN
    // Only learn from successful trajectories
    IF trajectory.quality < QUALITY_THRESHOLD THEN
        RETURN FALSE
    END IF

    // Extract features for K prediction
    features := {
        "query_embedding": trajectory.embedding,
        "optimal_k": trajectory.metadata["optimal_k"],
        "task_type": trajectory.metadata["task_type"],
        "convergence_pattern": Extract_Convergence_Pattern(trajectory)
    }

    // Apply MicroLoRA update for K prediction
    gradient := Compute_K_Prediction_Gradient(features)
    sona_engine.micro_lora.apply_gradient(gradient)

    // Store in ReasoningBank for future similarity lookup
    pattern := NEW LearnedPattern()
    pattern.embedding := trajectory.embedding
    pattern.metadata := features
    pattern.strategy := "trm_recursive"

    sona_engine.reasoning_bank.store(pattern)

    RETURN TRUE
END
```

---

## 3. Router Integration

### 3.1 Adaptive Recursion Routing

```
ALGORITHM Route_With_Adaptive_K
INPUT:
    query: User query string
    ruvllm: RuvLLM orchestrator
    trm_engine: TRM recursive engine

OUTPUT:
    response: Final response with metadata

BEGIN
    // Step 1: Embed query
    embedding := ruvllm.embed(query)

    // Step 2: Check memory for exact/similar match
    cached_result := ruvllm.memory_search(embedding, threshold=0.95)
    IF cached_result.IS_FOUND() THEN
        RETURN cached_result.response  // Skip recursion entirely
    END IF

    // Step 3: Predict optimal K
    predicted_k := Predict_Optimal_K(embedding, ruvllm.sona)

    // Step 4: Router decides execution strategy
    routing_decision := ruvllm.router.route({
        "embedding": embedding,
        "predicted_k": predicted_k,
        "complexity_score": Estimate_Complexity(embedding)
    })

    // Step 5: Execute TRM recursion
    IF routing_decision.use_trm THEN
        // Use TRM recursive reasoning
        (answer, confidence, trajectory) := trm_engine.reason(
            query=embedding,
            initial_answer=Generate_Initial_Answer(embedding),
            K=routing_decision.k_value,
            n=routing_decision.n_value
        )

        // Record for learning
        sona_trajectory := Record_TRM_Trajectory(
            embedding, trajectory, confidence, routing_decision.k_value
        )
        ruvllm.sona.submit_trajectory(sona_trajectory)

        // Cache result
        ruvllm.memory_store(embedding, answer, confidence)

        response := {
            "text": Decode_Answer(answer),
            "confidence": confidence,
            "k_used": routing_decision.k_value,
            "source": "trm_recursive"
        }
    ELSE
        // Use standard LLM path
        response := ruvllm.standard_inference(embedding)
    END IF

    RETURN response
END
```

---

## 4. SIMD-Optimized Operations

### 4.1 SIMD Matrix Multiply

```
ALGORITHM SIMD_MatMul
INPUT:
    A: Matrix [M, K]
    B: Matrix [K, N]

OUTPUT:
    C: Matrix [M, N]

BEGIN
    C := ZEROS([M, N])

    // Tile sizes optimized for cache
    TILE_M := 64
    TILE_N := 64
    TILE_K := 256

    // Parallel over M tiles
    PARALLEL FOR m_tile := 0 TO M STEP TILE_M DO
        FOR n_tile := 0 TO N STEP TILE_N DO
            // Local accumulator (stays in registers)
            acc := ZEROS([TILE_M, TILE_N])

            FOR k_tile := 0 TO K STEP TILE_K DO
                // Load tiles
                A_tile := A[m_tile:m_tile+TILE_M, k_tile:k_tile+TILE_K]
                B_tile := B[k_tile:k_tile+TILE_K, n_tile:n_tile+TILE_N]

                // SIMD inner loop (8 floats at a time with AVX2)
                FOR i := 0 TO TILE_M DO
                    FOR j := 0 TO TILE_N STEP 8 DO
                        vec_acc := LOAD_AVX(acc[i, j:j+8])

                        FOR kk := 0 TO TILE_K DO
                            a_broadcast := BROADCAST_AVX(A_tile[i, kk])
                            b_vec := LOAD_AVX(B_tile[kk, j:j+8])
                            vec_acc := FMA_AVX(a_broadcast, b_vec, vec_acc)
                        END FOR

                        STORE_AVX(acc[i, j:j+8], vec_acc)
                    END FOR
                END FOR
            END FOR

            // Write back
            C[m_tile:m_tile+TILE_M, n_tile:n_tile+TILE_N] := acc
        END FOR
    END FOR

    RETURN C
END
```

### 4.2 SIMD Layer Normalization

```
ALGORITHM SIMD_LayerNorm
INPUT:
    x: Input tensor [batch, dim]
    gamma: Scale parameter [dim]
    beta: Shift parameter [dim]
    eps: Epsilon for numerical stability

OUTPUT:
    y: Normalized tensor [batch, dim]

BEGIN
    y := ZEROS_LIKE(x)

    PARALLEL FOR b := 0 TO batch DO
        // Compute mean using SIMD horizontal sum
        sum_vec := ZEROS_AVX()
        FOR d := 0 TO dim STEP 8 DO
            x_vec := LOAD_AVX(x[b, d:d+8])
            sum_vec := ADD_AVX(sum_vec, x_vec)
        END FOR
        mean := HORIZONTAL_SUM_AVX(sum_vec) / dim

        // Compute variance using SIMD
        var_vec := ZEROS_AVX()
        mean_broadcast := BROADCAST_AVX(mean)
        FOR d := 0 TO dim STEP 8 DO
            x_vec := LOAD_AVX(x[b, d:d+8])
            diff := SUB_AVX(x_vec, mean_broadcast)
            var_vec := FMA_AVX(diff, diff, var_vec)
        END FOR
        variance := HORIZONTAL_SUM_AVX(var_vec) / dim

        // Normalize
        inv_std := 1.0 / SQRT(variance + eps)
        inv_std_broadcast := BROADCAST_AVX(inv_std)

        FOR d := 0 TO dim STEP 8 DO
            x_vec := LOAD_AVX(x[b, d:d+8])
            gamma_vec := LOAD_AVX(gamma[d:d+8])
            beta_vec := LOAD_AVX(beta[d:d+8])

            // (x - mean) * inv_std * gamma + beta
            normalized := SUB_AVX(x_vec, mean_broadcast)
            normalized := MUL_AVX(normalized, inv_std_broadcast)
            normalized := FMA_AVX(normalized, gamma_vec, beta_vec)

            STORE_AVX(y[b, d:d+8], normalized)
        END FOR
    END FOR

    RETURN y
END
```

---

## 5. WASM Compilation Considerations

### 5.1 Memory Management

```
ALGORITHM WASM_Memory_Pool
// Pre-allocate buffers to avoid repeated allocations

GLOBAL:
    latent_buffer: [MAX_BATCH, HIDDEN_DIM]
    answer_buffer: [MAX_BATCH, MAX_ANS_LEN, DIM]
    scratch_buffer: [SCRATCH_SIZE]

ALGORITHM Allocate_From_Pool
INPUT:
    size: Required size in floats

OUTPUT:
    ptr: Pointer to allocated memory

BEGIN
    IF pool_offset + size > SCRATCH_SIZE THEN
        // Reset pool (assumes single-threaded WASM)
        pool_offset := 0
    END IF

    ptr := scratch_buffer + pool_offset
    pool_offset := pool_offset + size

    RETURN ptr
END
```

### 5.2 SIMD.js Fallback

```
ALGORITHM WASM_SIMD_Dispatch
// Detect SIMD support and dispatch appropriately

BEGIN
    IF WASM_SIMD_SUPPORTED THEN
        // Use wasm32 SIMD128 intrinsics
        USE simd128_implementations
    ELSE
        // Fallback to scalar
        USE scalar_implementations
    END IF
END
```

---

## 6. Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRM + SONA Data Flow                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Query ──┬──► Embedding ──► Memory Search ──┬──► Cache Hit ──► Out │
│           │                                   │                     │
│           │                                   │                     │
│           │              ┌────────────────────┘                     │
│           │              │ Cache Miss                               │
│           │              ▼                                          │
│           │       ┌─────────────┐                                   │
│           │       │ K Predictor │◄─── SONA ReasoningBank            │
│           │       └──────┬──────┘                                   │
│           │              │ predicted_k                              │
│           │              ▼                                          │
│           │       ┌─────────────┐                                   │
│           │       │   Router    │──► routing_decision               │
│           │       └──────┬──────┘                                   │
│           │              │                                          │
│           │              ▼                                          │
│           │    ┌──────────────────────┐                             │
│           │    │  TRM Recursive Loop  │                             │
│           │    │  ────────────────────│                             │
│           │    │  FOR k=1 TO K:       │                             │
│           │    │    Latent Update (n) │                             │
│           │    │    Answer Refine     │                             │
│           │    │    Record Trajectory │                             │
│           │    └──────────┬───────────┘                             │
│           │               │                                         │
│           │               ▼                                         │
│           │        ┌─────────────┐                                  │
│           │        │   Answer    │                                  │
│           │        └──────┬──────┘                                  │
│           │               │                                         │
│           └───────────────┼─────────────────────────────────────┐   │
│                           │                                     │   │
│                           ▼                                     │   │
│                    ┌─────────────┐                              │   │
│                    │ SONA Learn  │◄─── Trajectory + Quality     │   │
│                    │  MicroLoRA  │                              │   │
│                    │  EWC++      │                              │   │
│                    │  Pattern    │                              │   │
│                    └─────────────┘                              │   │
│                                                                 │   │
└─────────────────────────────────────────────────────────────────────┘
```

---

**Next**: [03_ARCHITECTURE.md](./03_ARCHITECTURE.md) - System Design
