# SwiftLLM: The Core Engine of NEO

`swiftllm` is the core Python package of the NEO project. It implements a high-performance LLM inference engine supporting PagedAttention, CPU-GPU offloading, and asymmetric pipelining.

## Package Structure

The package is organized into two primary planes:

### 1. Control Plane (`swiftllm/server/`)
The control plane manages the high-level life cycle of inference requests and makes scheduling decisions.

*   **`engine.py`**: Contains `Engine` and `AsyncEngine`. It coordinates all other server components and provides the external API.
*   **`scheduler.py`**: The "brain" of the system. It implements a First-Come-First-Serve (FCFS) policy and decides whether to use Sequential or Pipelined mode based on performance predictions.
*   **`block_manager.py`**: Manages KV cache blocks on both GPU and CPU. It handles the mapping between virtual sequence IDs and physical block IDs.
*   **`executor.py`**: An abstraction layer for running the model. It supports `SingleProcExecutor` (local) and `RayExecutor` (distributed).
*   **`profiler.py`**: Measures hardware performance to build tables for the performance predictor.
*   **`tokenization_engine.py`**: Handles asynchronous tokenization of requests.

### 2. Data Plane (`swiftllm/worker/`)
The data plane resides on the accelerator (GPU) and handles the actual tensor computations.

*   **`model.py` (`LlamaModel`)**: The main model class that executes the forward pass, manages weights, and coordinates with the swapper.
*   **`layers/`**: Implements the transformer layers, specifically optimized for NEO's pipelining logic.
*   **`block_swapper.py`**: Manages the moving of KV cache blocks between GPU memory and CPU main memory using optimized CUDA/C++ kernels.
*   **`buffer.py`**: Manages pre-allocated activation buffers to avoid runtime allocation overhead.

## Key Package-wide Components

*   **`structs.py`**: Defines the fundamental data structures:
    *   `Request`: State and data for a single user request.
    *   `SubBatch`: A collection of requests prepared for a single forward iteration.
    *   `BatchPerfData`: Metrics used by the scheduler to balance load.
*   **`perfpredictor.py`**: Uses profiling data to estimate GPU and CPU execution times for various batch sizes and sequence lengths.
*   **`engine_config.py` & `model_config.py`**: Centralized configuration management for the engine and the Llama model architecture.

## Interaction Flow

1.  **Request Arrival**: `AsyncEngine` receives a prompt, sends it to `TokenizationEngine`, and adds it to the `Scheduler`'s queue.
2.  **Scheduling**: The `Scheduler` picks requests from the queue, allocates blocks via `BlockManager`, and forms one or two `SubBatch`es.
3.  **Execution**: `Executor` dispatches the batch(es) to `LlamaModel`.
4.  **Computation**: `LlamaModel` executes the forward pass, using `Swapper` for any required KV cache offloading, and calls into `pacpu` (CPU) or CUDA (GPU) kernels for attention.
5.  **Token Generation**: Output tokens are sent back to the `Request` objects, and finished requests are freed by the `BlockManager`.
