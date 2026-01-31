"""vLLM Engine wrapper for model inference."""

import asyncio
import time
import uuid
from typing import AsyncIterator, Optional

import structlog

logger = structlog.get_logger(__name__)


class VLLMEngine:
    """Wrapper around vLLM for inference."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_length: int = 4096,
        max_num_seqs: int = 128,
        enforce_eager: bool = False,
        swap_space: int = 4,
    ):
        """Initialize vLLM engine configuration.

        Args:
            model_path: Path to the model weights.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use.
            max_model_length: Maximum sequence length.
            max_num_seqs: Maximum number of concurrent sequences.
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

        self._llm = None
        self._tokenizer = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the vLLM engine."""
        async with self._lock:
            if self._initialized:
                return

            logger.info(
                "initializing_vllm_engine",
                model_path=self.model_path,
                gpu_memory_utilization=self.gpu_memory_utilization,
            )

            try:
                from vllm import LLM

                # Initialize vLLM in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self._llm = await loop.run_in_executor(
                    None,
                    lambda: LLM(
                        model=self.model_path,
                        tensor_parallel_size=self.tensor_parallel_size,
                        gpu_memory_utilization=self.gpu_memory_utilization,
                        max_model_len=self.max_model_length,
                        max_num_seqs=self.max_num_seqs,
                        enforce_eager=self.enforce_eager,
                        swap_space=self.swap_space,
                        trust_remote_code=True,
                    ),
                )

                self._initialized = True
                logger.info("vllm_engine_initialized_successfully")

            except Exception as e:
                logger.error("failed_to_initialize_vllm", error=str(e))
                raise

    async def shutdown(self) -> None:
        """Shutdown the vLLM engine."""
        async with self._lock:
            if self._llm is not None:
                logger.info("shutting_down_vllm_engine")
                # vLLM doesn't have explicit cleanup, but we can clear references
                self._llm = None
                self._initialized = False

    @property
    def is_ready(self) -> bool:
        """Check if the engine is ready for inference."""
        return self._initialized and self._llm is not None

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
        """Generate completion for a prompt.

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

        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self._llm.generate([prompt], sampling_params),
        )

        generation_time = time.time() - start_time
        output = outputs[0]

        # Extract results
        results = []
        for i, completion in enumerate(output.outputs):
            results.append(
                {
                    "index": i,
                    "text": completion.text,
                    "finish_reason": completion.finish_reason,
                    "tokens": len(completion.token_ids),
                }
            )

        prompt_tokens = len(output.prompt_token_ids)
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
        """Generate completion with streaming.

        Args:
            prompt: Input prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            stop: Stop sequences.
            presence_penalty: Presence penalty.
            frequency_penalty: Frequency penalty.

        Yields:
            Dictionary with token chunks.
        """
        if not self.is_ready:
            raise RuntimeError("Engine is not initialized")

        # Note: vLLM's streaming API requires AsyncLLMEngine
        # For this implementation, we simulate streaming from non-streaming output
        # In production, use vLLM's native streaming with AsyncLLMEngine

        result = await self.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            n=1,
        )

        text = result["results"][0]["text"]
        request_id = result["id"]

        # Simulate streaming by yielding character chunks
        chunk_size = 4
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            yield {
                "id": request_id,
                "content": chunk,
                "finish_reason": None if i + chunk_size < len(text) else result["results"][0]["finish_reason"],
            }
            # Small delay to simulate streaming
            await asyncio.sleep(0.01)

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
