# Citadel Integration Blueprint: High-Performance AI Engine (Project "Vidrial")
---
**Document ID:** `SRS-BLUEPRINT-AIE-V1.0`
**Version:** `1.0.0`
**Status:** `DRAFT`
**Author:** `Jules (AI Software Engineer)`
**Governed By:** `CGRF-v2.0`

---

## 1. Executive Abstract

This document provides the complete architectural blueprint and Software Requirements Specification (SRS) for the integration of the high-performance "Symmetric Power Retention" library (internally codenamed "Vidrial") into the Citadel ecosystem.

The integration of this library is a strategic imperative. Its core value proposition—enabling larger, more powerful AI models to run efficiently on our existing infrastructure—directly enhances our capabilities across all four architectural pillars. This blueprint details the creation of a new **AIEngine Service** within the Control Layer, modifications to the **Build Orchestrator**, the definition of a new **SAKE protocol for model generation**, and the specification of new **MCP API endpoints** to expose these powerful new capabilities to the rest of the ecosystem.

This document serves as the canonical reference for this integration, ensuring it is executed in a manner consistent with Citadel's principles of security, automation, and governance.

## 2. Architectural Changes

To integrate the Vidrial library, we will introduce a new service and modify our existing build pipeline.

### 2.1. New Service: The `AIEngine` (Control Layer)

A new microservice, the `AIEngine`, will be introduced into the Control Layer. This service will be responsible for:

*   **Model Lifecycle Management:** Loading, unloading, and managing the state of trained AI models that utilize the Vidrial `retention` layer.
*   **Inference Execution:** Providing a secure, high-performance runtime for executing inference requests against the loaded models.
*   **Resource Management:** Monitoring and managing the VRAM and CPU resources consumed by the AI models to ensure system stability.

The `AIEngine` will be a containerized Python service, managed by Docker within our VPS cluster, and will communicate with other services via the Redis Message Bus.

### 2.2. Modification: `build_orchestrator.py`

The `build_orchestrator.py` script will be extended with a new, distinct stage:

*   **`build_ai_model` Stage:** This stage will be triggered when a `.sake` file of type `model_build` is detected.
    *   It will parse the SAKE file to extract model hyperparameters and training data locations.
    *   It will execute a new training script (`tools/train_model.py`) that uses the `retention` library to construct and train the specified model.
    *   Upon successful training, it will serialize the trained model weights and its configuration into a versioned artifact bundle (e.g., `model_v1.0.0.tar.gz`).
    *   It will then deploy this artifact to a designated location accessible by the `AIEngine` service.

This modification aligns perfectly with our **Automation Centric** pillar, treating AI model creation as a fully automated, reproducible build step.

## 3. New Protocol: SAKE for Model Building

To declaratively define and govern the creation of new AI models, we will introduce a new SAKE protocol.

### 3.1. `sake_model_build.json` Schema

This new SAKE file type will govern the `build_ai_model` stage.

**Example `sake_model_build.json`:**
```json
{
  "sake_version": "1.0",
  "task_type": "model_build",
  "metadata": {
    "task_id": "build-sec-analyst-v1",
    "author": "SentinelAI",
    "description": "Build a new security analysis model optimized for log anomaly detection."
  },
  "governance": {
    "max_training_time_seconds": 3600,
    "required_dataset_access": ["/secure/logs/processed/2025-q3.jsonl"]
  },
  "taskir": {
    "model_name": "LogAnomalyDetector",
    "model_version": "1.0.0",
    "architecture": "PowerRetention_Classifier",
    "hyperparameters": {
      "retention_layer": {
        "deg": 4,
        "chunk_size": 256,
        "gating": true
      },
      "classifier_head": {
        "hidden_layers": 2,
        "dropout": 0.3
      }
    },
    "training_data_uri": "vds://security_logs/processed/2025-q3.jsonl",
    "validation_data_uri": "vds://security_logs/processed/2025-q4-validation.jsonl"
  }
}
```

This protocol embodies our **Data-Driven Metaprogramming** pillar by treating the AI models themselves as artifacts defined entirely by data.

## 4. New MCP API Endpoints

To expose the `AIEngine`'s capabilities, we will add the following endpoints to the `/mcp` gateway:

*   **`model.predict`**
    *   **Description:** Runs inference on a deployed model.
    *   **Params:**
        *   `model_name` (string): The name of the model to use (e.g., "LogAnomalyDetector").
        *   `model_version` (string, optional): The version of the model. Defaults to latest.
        *   `input_data` (object): The data to run inference on.
    *   **Returns:** (object) The model's prediction.

*   **`model.list`**
    *   **Description:** Lists all available and loaded models.
    *   **Params:** None
    *   **Returns:** (array) A list of objects, each describing a model.

*   **`model.get_status`**
    *   **Description:** Checks the health and status of a specific model.
    *   **Params:**
        *   `model_name` (string): The name of the model.
        *   `model_version` (string, optional): The version of the model.
    *   **Returns:** (object) The model's status, including resource usage and uptime.

These endpoints will be protected by our standard OAuth2 and RBAC security, aligning with the **Security First** and **Decoupled Control** pillars.

## 5. Software Requirements Specification (SRS) for `AIEngine`

### 5.1. Functional Requirements (FR)

*   **FR-AIE-001:** The service MUST be able to load a trained model artifact (specified by path) into memory.
*   **FR-AIE-002:** The service MUST expose an internal API (via Redis Pub/Sub) to handle `predict` requests.
*   **FR-AIE-003:** The `predict` function MUST accept input data, run it through the loaded model, and return the model's output.
*   **FR-AIE-004:** The service MUST handle multiple models concurrently, routing requests to the correct model based on `model_name` and `model_version`.
*   **FR-AIE-005:** The service MUST provide status information, including loaded models and their resource consumption.

### 5.2. Non-Functional Requirements (NFR)

*   **NFR-AIE-001 (Performance):** The `predict` endpoint must have a p95 latency of less than 500ms for a standard-sized input.
*   **NFR-AIE-002 (Scalability):** The architecture must support horizontal scaling by adding more `AIEngine` instances to the VPS cluster.
*   **NFR-AIE-003 (Security):** The service must run in an isolated container. All model artifacts loaded must be scanned for vulnerabilities.
*   **NFR-AIE-004 (Reliability):** The service must have a 99.9% uptime and include automated restart-on-failure mechanisms.
*   **NFR-AIE-005 (Observability):** The service MUST emit structured logs (JSON format) for all major events (model load, prediction, error) and expose metrics for Prometheus scraping.