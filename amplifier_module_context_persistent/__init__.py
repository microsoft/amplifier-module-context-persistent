"""
Persistent context manager module.

Implements a standalone context manager that provides:
  • Message storage with token accounting and compaction (matches SimpleContextManager behaviour)
  • Optional loading of persistent memory files on initialization
"""

from __future__ import annotations

# Amplifier module metadata
__amplifier_module_type__ = "context"

import logging
from pathlib import Path
from typing import Any

from amplifier_core import ModuleCoordinator

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

    context = PersistentContextManager(
        memory_files=config.get("memory_files", []),
        max_tokens=config.get("max_tokens", 200_000),
        compact_threshold=config.get("compact_threshold", 0.92),
    )

    await context.initialize()
    await coordinator.mount("context", context)
    logger.info(
        "Mounted PersistentContextManager "
        f"(memory_files={len(context.memory_files)}, max_tokens={context.max_tokens}, "
        f"compact_threshold={context.compact_threshold:.2f})"
    )
    return


class PersistentContextManager:
    """
    Context manager with message storage, token counting, compaction, and
    optional persistent memory file loading.
    """

    def __init__(
        self,
        memory_files: list[str] | None = None,
        max_tokens: int = 200_000,
        compact_threshold: float = 0.92,
    ):
        """
        Initialize the persistent context manager.

        Args:
            memory_files: List of file paths to load at startup
            max_tokens: Maximum context size in tokens
            compact_threshold: Threshold for triggering compaction (0.0-1.0)
        """
        self.memory_files = memory_files or []
        self.max_tokens = max_tokens
        self.compact_threshold = compact_threshold
        self.messages: list[dict[str, Any]] = []
        self._token_count = 0

    async def initialize(self) -> None:
        """Load configured memory files and validate initial context."""
        await self._load_memory_files()
        await self._validate_startup_context()

    async def add_message(self, message: dict[str, Any]) -> None:
        """Add a message to the context, compacting if necessary."""
        # Estimate tokens for this message
        message_tokens = len(str(message)) // 4

        # Check projected usage
        projected_total = self._token_count + message_tokens
        usage = projected_total / self.max_tokens

        if usage >= self.compact_threshold:
            logger.info(
                "Projected usage %.1f%% >= threshold %.1f%%, compacting before adding message",
                usage * 100,
                self.compact_threshold * 100,
            )
            await self.compact()

            # Re-check after compaction
            projected_total = self._token_count + message_tokens
            if projected_total > self.max_tokens:
                error_msg = (
                    f"Cannot add message: would exceed context limit "
                    f"({projected_total:,} tokens > {self.max_tokens:,} max). "
                    f"Current context: {self._token_count:,} tokens, "
                    f"message: {message_tokens:,} tokens. "
                    "Suggestions: reduce memory files, use shorter messages, "
                    "or increase max_tokens in context config."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        self.messages.append(message)
        self._token_count += message_tokens
        logger.debug(
            "Added message: %s - %d total messages, %d tokens (%.1f%%)",
            message.get("role", "unknown"),
            len(self.messages),
            self._token_count,
            (self._token_count / self.max_tokens) * 100,
        )

    async def get_messages(self) -> list[dict[str, Any]]:
        """Return a shallow copy of stored messages."""
        return self.messages.copy()

    async def set_messages(self, messages: list[dict[str, Any]]) -> None:
        """Replace stored messages (used for session resume)."""
        self.messages = messages.copy()
        self._recalculate_tokens()
        logger.info("Restored %d messages to context", len(messages))

    async def clear(self) -> None:
        """Remove all stored messages."""
        self.messages = []
        self._token_count = 0
        logger.info("Context cleared")

    async def should_compact(self) -> bool:
        """Return True if compaction is recommended."""
        usage = self._token_count / self.max_tokens
        should = usage >= self.compact_threshold
        if should:
            logger.info("Context at %.1f%% capacity, compaction recommended", usage * 100)
        return should

    async def compact(self) -> None:
        """
        Compact the context while preserving tool_use/tool_result pairs.

        Tool usage continuity is required by providers that expect tool result
        messages immediately following tool_calls. We treat these as atomic units.
        """
        logger.info("Compacting context with %d messages", len(self.messages))

        keep_indices = set()

        # Always keep system messages
        for idx, msg in enumerate(self.messages):
            if msg.get("role") == "system":
                keep_indices.add(idx)

        # Keep last 10 messages
        for idx in range(max(0, len(self.messages) - 10), len(self.messages)):
            keep_indices.add(idx)

        # Ensure tool-use/tool-result integrity
        expanded = keep_indices.copy()
        for idx in keep_indices:
            msg = self.messages[idx]

            # Preserve assistant tool_calls with subsequent tool results
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                num_calls = len(msg["tool_calls"])
                kept_results = 0
                offset = 1
                while kept_results < num_calls and idx + offset < len(self.messages):
                    next_msg = self.messages[idx + offset]
                    if next_msg.get("role") == "tool":
                        expanded.add(idx + offset)
                        kept_results += 1
                    else:
                        logger.warning(
                            "Assistant tool message at %d expected %d tool results, "
                            "encountered non-tool message at %d after %d results",
                            idx,
                            num_calls,
                            idx + offset,
                            kept_results,
                        )
                        break
                    offset += 1

            # Ensure tool message keeps the assistant with tool_calls
            # Walk backwards to find it (may be multiple tool results after one assistant)
            elif msg.get("role") == "tool":
                # Find the assistant message with tool_calls that this result belongs to
                for j in range(idx - 1, -1, -1):
                    check_msg = self.messages[j]
                    if check_msg.get("role") == "assistant" and check_msg.get("tool_calls"):
                        expanded.add(j)
                        logger.debug(
                            "Preserving tool group: message %d (assistant with tool_calls) includes tool result at %d",
                            j,
                            idx,
                        )
                        break
                    if check_msg.get("role") != "tool":
                        # Hit a non-tool, non-assistant-with-tool_calls message before finding the assistant
                        # This shouldn't happen in well-formed conversation but log if it does
                        logger.warning(
                            "Tool result at %d has no matching assistant with tool_calls (found %s at %d instead)",
                            idx,
                            check_msg.get("role"),
                            j,
                        )
                        break

        compacted = [self.messages[i] for i in sorted(expanded)]

        # Deduplicate non-tool messages while preserving tool interactions
        seen = set()
        final: list[dict[str, Any]] = []
        for msg in compacted:
            if msg.get("role") == "tool" or (msg.get("role") == "assistant" and msg.get("tool_calls")):
                final.append(msg)
            else:
                key = (msg.get("role"), msg.get("content", "")[:100])
                if key not in seen:
                    seen.add(key)
                    final.append(msg)

        old_count = len(self.messages)
        self.messages = final
        self._recalculate_tokens()

        tool_use_count = sum(1 for m in final if m.get("tool_calls"))
        tool_result_count = sum(1 for m in final if m.get("role") == "tool")
        logger.info(
            "Compacted %d → %d messages (%d tool_use, %d tool_result preserved)",
            old_count,
            len(final),
            tool_use_count,
            tool_result_count,
        )

    # ------------------------------------------------------------------
    # Persistent helpers
    # ------------------------------------------------------------------
    async def _load_memory_files(self) -> None:
        """Load configured memory files as system messages."""
        if not self.memory_files:
            logger.debug("Persistent context has no memory files configured")
            return

        logger.info("Loading %d memory files", len(self.memory_files))

        for file_path in self.memory_files:
            path = Path(file_path).expanduser()

            if not path.exists():
                logger.warning("Memory file not found, skipping: %s", file_path)
                continue

            try:
                content = path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.error("Failed to read memory file %s: %s", file_path, exc)
                continue

            if not content.strip():
                logger.debug("Skipping empty memory file: %s", file_path)
                continue

            await self.add_message(
                {
                    "role": "system",
                    "content": f"[Context from {path.name}]\n\n{content}",
                }
            )

            logger.info("Loaded context file: %s (%d chars)", path.name, len(content))

        system_messages = [msg for msg in self.messages if msg.get("role") == "system"]
        logger.info("Persistent context initialized with %d system messages", len(system_messages))

    async def _validate_startup_context(self) -> None:
        """Ensure context limits are not exceeded after loading memory files."""
        if self._token_count <= self.max_tokens:
            return

        system_messages = [m for m in self.messages if m.get("role") == "system"]
        error_lines = [
            "Memory files exceed context limit!",
            "",
            f"Total tokens: {self._token_count:,} > {self.max_tokens:,} max",
            f"Loaded {len(system_messages)} system messages from {len(self.memory_files)} files",
            "",
            "File breakdown:",
        ]

        for msg in system_messages:
            content = msg.get("content", "")
            first_line = content.split("\n", 1)[0]
            filename = first_line.replace("[Context from ", "").replace("]", "")
            msg_tokens = len(content) // 4
            error_lines.append(f"  - {filename}: ~{msg_tokens:,} tokens")

        error_lines.extend(
            [
                "",
                "Suggestions:",
                "  1. Remove or reduce memory files in your profile",
                "  2. Increase max_tokens in context config",
                "  3. Split large files into smaller focused files",
            ]
        )

        error_msg = "\n".join(error_lines)
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _recalculate_tokens(self) -> None:
        """Recalculate token usage after compaction or state restore."""
        self._token_count = sum(len(str(msg)) // 4 for msg in self.messages)
