"""
Persistent context manager module.
Extends SimpleContextManager with file loading capabilities for cross-session memory.
"""

import logging
from pathlib import Path
from typing import Any

from amplifier_core import ModuleCoordinator
from amplifier_module_context_simple import SimpleContextManager

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the persistent context manager.

    Args:
        coordinator: Module coordinator
        config: Optional configuration

    Returns:
        Optional cleanup function
    """
    config = config or {}

    # Extract persistent-specific config
    memory_files = config.get("memory_files", [])

    # Create context manager with base config
    context = PersistentContextManager(
        memory_files=memory_files,
        max_tokens=config.get("max_tokens", 200_000),
        compact_threshold=config.get("compact_threshold", 0.92),
    )

    # Load memory files during initialization
    await context.initialize()

    await coordinator.mount("context", context)
    logger.info(f"Mounted PersistentContextManager with {len(memory_files)} memory files")
    return


class PersistentContextManager(SimpleContextManager):
    """
    Context manager with persistent file loading.
    Loads context files at session start for cross-session memory.
    """

    def __init__(
        self, memory_files: list[str] | None = None, max_tokens: int = 200_000, compact_threshold: float = 0.92
    ):
        """
        Initialize the persistent context manager.

        Args:
            memory_files: List of file paths to load at startup
            max_tokens: Maximum context size in tokens
            compact_threshold: Threshold for triggering compaction (0.0-1.0)
        """
        super().__init__(max_tokens, compact_threshold)
        self.memory_files = memory_files or []

    async def initialize(self) -> None:
        """
        Initialize and load memory files at session start.
        """
        await self._load_memory_files()

    async def _load_memory_files(self) -> None:
        """
        Load all configured memory files as system messages.
        Files are loaded in order and added to context.
        Missing files are skipped with a warning.
        """
        if not self.memory_files:
            logger.debug("No memory files configured")
            return

        logger.info(f"Loading {len(self.memory_files)} memory files")

        for file_path in self.memory_files:
            path = Path(file_path)

            # Expand home directory if present
            if str(file_path).startswith("~/"):
                path = path.expanduser()

            # Skip missing files with warning
            if not path.exists():
                logger.warning(f"Memory file not found, skipping: {file_path}")
                continue

            try:
                # Read file content
                content = path.read_text(encoding="utf-8")

                # Skip empty files
                if not content.strip():
                    logger.debug(f"Skipping empty file: {file_path}")
                    continue

                # Add as system message with clear label
                await self.add_message({"role": "system", "content": f"[Context from {path.name}]\n\n{content}"})

                logger.info(f"Loaded context file: {path.name} ({len(content)} chars)")

            except Exception as e:
                logger.error(f"Failed to load memory file {file_path}: {e}")
                continue

        # Log final state
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        logger.info(f"Context initialized with {len(system_msgs)} system messages")
