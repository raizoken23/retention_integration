# MDAO Formulas for Retention-Based AI Models
---
**Document ID:** `SRS-MDAO-VIDRIAL-V1.0`
**Version:** `1.0.0`
**Status:** `DRAFT`
**Author:** `Jules (AI Software Engineer)`
**Governed By:** `██████████`, `██████████`

---

## 1. Abstract

This document defines three distinct, multi-objective formulas for use with our proprietary `██████████` and its underlying OpenMDAO solver. These formulas are specifically designed to guide the evolution of AI models that utilize the "Symmetric Power Retention" (`retention`) layer.

Each formula represents a different strategic priority for our AI models, allowing the `██████████` to optimize for:
1.  **Max Performance:** Raw inference speed and low resource usage.
2.  **Max Accuracy:** The highest possible correctness and quality of output, at any cost.
3.  **Balanced Profile:** A hybrid approach that seeks a "Pareto optimal" balance between performance and accuracy, suitable for general-purpose deployment.

These formulas will be stored in our `formulas.yaml` and will be selected by a `model_build` configuration file to guide the evolution of new bespoke AI agents.

## 2. Core Metrics & Signals

The formulas are composed of several core metrics (`signals`) that are collected during the model's simulated evaluation within the `██████████`.

*   **`AccuracyScore` (0.0 - 1.0):** The primary measure of correctness, derived from a validation dataset (e.g., F1 score, BLEU score, or a semantic similarity score against a "golden" output). Higher is better.
*   **`InferenceLatency` (ms):** The average time taken to perform a single prediction on a standard input. Lower is better.
*   **`VRAM_Usage_MB` (MB):** The peak VRAM consumed by the model during inference. Lower is better.
*   **`StateSize` (MB):** The size of the model's recurrent state. This is a direct measure of the `retention` layer's efficiency. Lower is better.
*   **`GradStability` (0.0 - 1.0):** A measure of the stability of gradients during training. Higher is better, indicating smoother training.
*   **`ComplexityPenalty` (0.0 - 1.0):** A penalty score based on the model's architectural complexity (e.g., number of parameters, `deg` of the retention layer). Lower is better.

## 3. MDAO Optimization Formulas

These formulas define the `Objective Function` that the OpenMDAO solver will seek to maximize.

### 3.1. Formula 1: "Cheetah" - Max Performance / Low Footprint

This profile is for agents where speed and resource efficiency are paramount, such as real-time monitoring or high-throughput data processing agents.

*   **Objective:** Maximize `PerformanceScore`.
*   **Formula:**
    ```
    # Weights emphasize low latency and small memory footprint.
    # Accuracy is a constraint, not the primary goal.
    w_latency = 0.5
    w_vram = 0.3
    w_state = 0.2
    w_accuracy = 0.1  # Low weight, but present to prevent useless models.
    w_complexity = 0.1

    # Normalize metrics to a 0-1 scale where 1 is always better.
    NormalizedLatency = 1 - (InferenceLatency / MaxExpectedLatency)
    NormalizedVRAM = 1 - (VRAM_Usage_MB / MaxExpectedVRAM)
    NormalizedState = 1 - (StateSize / MaxExpectedState)
    NormalizedComplexity = 1 - ComplexityPenalty

    PerformanceScore = (w_latency * NormalizedLatency) + \
                       (w_vram * NormalizedVRAM) + \
                       (w_state * NormalizedState) + \
                       (w_accuracy * AccuracyScore) + \
                       (w_complexity * NormalizedComplexity)
    ```
*   **Constraints:**
    *   `AccuracyScore` >= 0.65 (The model must be at least moderately accurate).

### 3.2. Formula 2: "Rhino" - Max Accuracy & Robustness

This profile is for agents where the quality and reliability of the output are non-negotiable, regardless of the computational cost. This is suitable for critical tasks like security auditing, code generation, or final content creation for our `██████████` application.

*   **Objective:** Maximize `AccuracyScore`.
*   **Formula:**
    ```
    # Weights are overwhelmingly focused on accuracy and training stability.
    w_accuracy = 0.7
    w_stability = 0.2
    w_complexity = 0.1 # Penalize overly complex models that might be brittle.
    w_latency = 0.0 # Performance is not a direct objective.

    NormalizedComplexity = 1 - ComplexityPenalty

    AccuracyScore = (w_accuracy * AccuracyScore) + \
                    (w_stability * GradStability) + \
                    (w_complexity * NormalizedComplexity)
    ```
*   **Constraints:**
    *   `InferenceLatency` <= 2000ms (The model must still be usable).
    *   `VRAM_Usage_MB` <= MaxAvailableVRAM (It must fit on the hardware).

### 3.3. Formula 3: "Stag" - Balanced Profile (Pareto Optimal)

This profile seeks the "sweet spot" between performance and accuracy. It is the default profile for general-purpose agents and represents a trade-off suitable for most production deployments.

*   **Objective:** Maximize `BalancedScore`.
*   **Formula:**
    ```
    # Weights are balanced to reward both accuracy and performance.
    w_accuracy = 0.4
    w_latency = 0.3
    w_vram = 0.1
    w_state = 0.1
    w_complexity = 0.1

    # Normalize metrics to a 0-1 scale where 1 is always better.
    NormalizedLatency = 1 - (InferenceLatency / MaxExpectedLatency)
    NormalizedVRAM = 1 - (VRAM_Usage_MB / MaxExpectedVRAM)
    NormalizedState = 1 - (StateSize / MaxExpectedState)
    NormalizedComplexity = 1 - ComplexityPenalty

    BalancedScore = (w_accuracy * AccuracyScore) + \
                    (w_latency * NormalizedLatency) + \
                    (w_vram * NormalizedVRAM) + \
                    (w_state * NormalizedState) + \
                    (w_complexity * NormalizedComplexity)
    ```
*   **Constraints:**
    *   `AccuracyScore` >= 0.80
    *   `InferenceLatency` <= 1000ms

## 4. Implementation in `formulas.yaml`

These formulas will be translated into the `formulas.yaml` file, which is consumed by the `██████████`. The engine will select the appropriate formula based on a `profile` key in the `model_build` file.

**Example `formulas.yaml` entry:**
```yaml
formulas:
  retention_cheetah:
    objective: "PerformanceScore"
    components:
      - "AccuracyScore"
      - "InferenceLatency"
      # ... etc.
    weights:
      w_latency: 0.5
      w_vram: 0.3
      w_state: 0.2
      w_accuracy: 0.1
      w_complexity: 0.1
    constraints:
      - "AccuracyScore >= 0.65"
```