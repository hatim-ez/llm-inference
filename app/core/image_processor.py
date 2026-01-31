"""Image processing utilities for multimodal inference."""

import base64
import io
from typing import Optional
from urllib.parse import urlparse

import httpx
import structlog
from PIL import Image

logger = structlog.get_logger(__name__)


class ImageProcessor:
    """Process images for multimodal LLM inference."""

    # Supported image formats
    SUPPORTED_FORMATS = {"JPEG", "PNG", "GIF", "WEBP", "BMP"}

    # Maximum image dimensions
    MAX_WIDTH = 2048
    MAX_HEIGHT = 2048

    # Maximum file size (10 MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024

    def __init__(self, max_width: int = MAX_WIDTH, max_height: int = MAX_HEIGHT):
        """Initialize image processor.

        Args:
            max_width: Maximum image width.
            max_height: Maximum image height.
        """
        self.max_width = max_width
        self.max_height = max_height
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def load_image_from_url(self, url: str) -> Image.Image:
        """Load image from URL.

        Args:
            url: Image URL.

        Returns:
            PIL Image object.

        Raises:
            ValueError: If URL is invalid or image cannot be loaded.
        """
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

        logger.debug("loading_image_from_url", url=url[:100])

        client = await self._get_http_client()
        response = await client.get(url)
        response.raise_for_status()

        # Check content length
        content_length = response.headers.get("content-length")
        if content_length and int(content_length) > self.MAX_FILE_SIZE:
            raise ValueError(f"Image too large: {content_length} bytes")

        # Load image
        image_data = response.content
        if len(image_data) > self.MAX_FILE_SIZE:
            raise ValueError(f"Image too large: {len(image_data)} bytes")

        image = Image.open(io.BytesIO(image_data))
        return self._validate_and_process(image)

    def load_image_from_base64(self, data: str) -> Image.Image:
        """Load image from base64 string.

        Args:
            data: Base64-encoded image data (with or without data URL prefix).

        Returns:
            PIL Image object.

        Raises:
            ValueError: If image cannot be decoded.
        """
        # Handle data URL format
        if data.startswith("data:"):
            # Extract base64 part
            try:
                _, data = data.split(",", 1)
            except ValueError:
                raise ValueError("Invalid data URL format")

        logger.debug("loading_image_from_base64", data_length=len(data))

        try:
            image_bytes = base64.b64decode(data)
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {e}")

        if len(image_bytes) > self.MAX_FILE_SIZE:
            raise ValueError(f"Image too large: {len(image_bytes)} bytes")

        image = Image.open(io.BytesIO(image_bytes))
        return self._validate_and_process(image)

    def _validate_and_process(self, image: Image.Image) -> Image.Image:
        """Validate and process image.

        Args:
            image: PIL Image object.

        Returns:
            Processed PIL Image object.

        Raises:
            ValueError: If image format is not supported.
        """
        # Check format
        if image.format and image.format.upper() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {image.format}")

        # Convert to RGB if necessary
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        # Resize if too large
        if image.width > self.max_width or image.height > self.max_height:
            logger.debug(
                "resizing_image",
                original_size=(image.width, image.height),
                max_size=(self.max_width, self.max_height),
            )
            image.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)

        return image

    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string.

        Args:
            image: PIL Image object.
            format: Output format (PNG, JPEG, etc.).

        Returns:
            Base64-encoded string.
        """
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def process_messages(
        self, messages: list[dict]
    ) -> tuple[str, list[Image.Image]]:
        """Process chat messages and extract images.

        Args:
            messages: List of chat messages with potential image content.

        Returns:
            Tuple of (text prompt, list of images).
        """
        prompt_parts = []
        images = []
        image_count = 0

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content")

            if isinstance(content, str):
                prompt_parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                # Multimodal content
                message_text = []
                for part in content:
                    if part.get("type") == "text":
                        message_text.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url", {})
                        url = image_url.get("url", "")

                        try:
                            if url.startswith("data:"):
                                image = self.load_image_from_base64(url)
                            else:
                                image = await self.load_image_from_url(url)

                            images.append(image)
                            image_count += 1
                            message_text.append(f"[Image {image_count}]")

                        except Exception as e:
                            logger.warning(
                                "failed_to_load_image",
                                url=url[:50],
                                error=str(e),
                            )
                            message_text.append("[Failed to load image]")

                prompt_parts.append(f"{role}: {' '.join(message_text)}")

        # Combine into single prompt
        prompt = "\n".join(prompt_parts)

        if images:
            logger.info(
                "processed_multimodal_messages",
                num_images=len(images),
                prompt_length=len(prompt),
            )

        return prompt, images

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
