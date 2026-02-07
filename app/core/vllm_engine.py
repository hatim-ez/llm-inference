"""vLLM Engine wrapper for model inference.

This module provides a high-performance wrapper around vLLM's AsyncLLMEngine
for continuous batching and concurrent request processing.
"""

import asyncio
import time
import uuid
from typing import AsyncIterator, Optional

import structlog

logger = structlog.get_logger(__name__)


class VLLMEngine:
    """Wrapper around vLLM AsyncLLMEngine for inference with continuous batching.

    Uses AsyncLLMEngine instead of LLM for:
    - Continuous batching of concurrent requests
    - Non-blocking async generation
    - Better throughput under load
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_length: int = 4096,
        max_num_seqs: int = 8,  # Reduced default for better batching behavior
        enforce_eager: bool = False,
        swap_space: int = 4,
    ):
        """Initialize vLLM engine configuration.

        Args:
            model_path: Path to the model weights.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use.
            max_model_length: Maximum sequence length.
            max_num_seqs: Maximum number of concurrent sequences (default 8 for batching).
            enforce_eager: Whether to disable CUDA graphs.
            swap_space: CPU swap space in GB.
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_length = max_model_length
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.swap_space = swap_space

        self._engine = None  # AsyncLLMEngine instead of LLM
        self._tokenizer = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the vLLM AsyncLLMEngine for continuous batching."""
        async with self._lock:
            if self._initialized:
                return

            logger.info(
                "initializing_vllm_async_engine",
                model_path=self.model_path,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_num_seqs=self.max_num_seqs,
            )

            try:
                from vllm import AsyncLLMEngine
                from vllm.engine.arg_utils import AsyncEngineArgs

                # Configure AsyncLLMEngine for continuous batching
                engine_args = AsyncEngineArgs(
                    model=self.model_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_length,
                    max_num_seqs=self.max_num_seqs,
                    enforce_eager=self.enforce_eager,
                    swap_space=self.swap_space,
                    trust_remote_code=True,
                )

                # Initialize AsyncLLMEngine
                self._engine = AsyncLLMEngine.from_engine_args(engine_args)

                self._initialized = True
                logger.info(
                    "vllm_async_engine_initialized_successfully",
                    max_num_seqs=self.max_num_seqs,
                    continuous_batching="enabled",
                )

            except Exception as e:
                logger.error("failed_to_initialize_vllm", error=str(e))
                raise

    async def shutdown(self) -> None:
        """Shutdown the vLLM engine."""
        async with self._lock:
            if self._engine is not None:
                logger.info("shutting_down_vllm_engine")
                # Abort any pending requests and cleanup
                self._engine = None
                self._initialized = False

    @property
    def is_ready(self) -> bool:
        """Check if the engine is ready for inference."""
        return self._initialized and self._engine is not None

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        n: int = 1,
    ) -> dict:
        """Generate completion for a prompt using async engine.

        Uses AsyncLLMEngine for continuous batching - multiple concurrent
        requests are automatically batched together for GPU efficiency.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            stop: Stop sequences.
            presence_penalty: Presence penalty.
            frequency_penalty: Frequency penalty.
            n: Number of completions.

        Returns:
            Dictionary with generated text and metadata.
        """
        if not self.is_ready:
            raise RuntimeError("Engine is not initialized")

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            n=n,
        )

        start_time = time.time()
        request_id = str(uuid.uuid4())

        logger.debug(
            "starting_generation",
            request_id=request_id,
            prompt_length=len(prompt),
            max_tokens=max_tokens,
        )

        # Use AsyncLLMEngine.generate() for continuous batching
        # This allows multiple requests to be batched together automatically
        final_output = None
        async for output in self._engine.generate(prompt, sampling_params, request_id):
            final_output = output

        generation_time = time.time() - start_time

        if final_output is None:
            raise RuntimeError("No output generated")

        # Extract results
        results = []
        for i, completion in enumerate(final_output.outputs):
            results.append(
                {
                    "index": i,
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "tokens": len(completion.token_ids),
                }
            )

        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = sum(r["tokens"] for r in results)

        logger.info(
            "generation_complete",
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            generation_time_seconds=generation_time,
            tokens_per_second=completion_tokens / generation_time if generation_time > 0 else 0,
        )

        return {
            "id": request_id,
            "results": results,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "generation_time": generation_time,
        }

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ) -> AsyncIterator[dict]:
        """Generate completion with true streaming using AsyncLLMEngine.

        Uses native vLLM streaming for real token-by-token output.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            stop: Stop sequences.
            presence_penalty: Presence penalty.
            frequency_penalty: Frequency penalty.

        Yields:
            Dictionary with token chunks as they are generated.
        """
        if not self.is_ready:
            raise RuntimeError("Engine is not initialized")

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            n=1,
        )

        request_id = str(uuid.uuid4())
        previous_text = ""

        # Stream tokens as they are generated using AsyncLLMEngine
        async for output in self._engine.generate(prompt, sampling_params, request_id):
            if output.outputs:
                completion = output.outputs[0]
                # Get only the new text since last iteration
                new_text = completion.text[len(previous_text):]
                previous_text = completion.text

                if new_text:
                    yield {
                        "id": request_id,
                        "content": new_text,
                        "finish_reason": completion.finish_reason,
                    }

    async def generate_multimodal(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> dict:
        """Generate completion for multimodal input (text + images).

        Args:
            messages: List of message dictionaries with text and image content.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            stop: Stop sequences.

        Returns:
            Dictionary with generated text and metadata.
        """
        if not self.is_ready:
            raise RuntimeError("Engine is not initialized")

        # Process messages to create prompt with image placeholders
        # Note: Actual multimodal processing depends on the model architecture
        # This is a simplified implementation

        from app.core.image_processor import ImageProcessor

        processor = ImageProcessor()
        processed_prompt, images = await processor.process_messages(messages)

        # For now, use text-only generation
        # Full multimodal support requires model-specific handling
        return await self.generate(
            prompt=processed_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
