"""
Persistent context manager module.

Implements a standalone context manager that provides:
  • Message storage with token accounting and internal compaction
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
    Context manager with message storage, token counting, internal compaction,
    and optional persistent memory file loading.

    Owns memory policy: orchestrators ask for messages via get_messages_for_request(),
    and this context manager decides how to fit them within limits. Compaction is
    handled internally - orchestrators don't know or care about compaction.
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
        """Add a message to the context.

        Messages are always accepted. Compaction happens internally when
        get_messages_for_request() is called before LLM requests.

        Tool results MUST be added even if over threshold, otherwise
        tool_use/tool_result pairing breaks.
        """
        # Estimate tokens for this message
        message_tokens = len(str(message)) // 4

        # Add message (no rejection - compaction happens internally)
        self.messages.append(message)
        self._token_count += message_tokens

        # Warn if significantly over threshold (but don't reject)
        usage = self._token_count / self.max_tokens
        if usage > 1.0:
            logger.warning(
                "Context at %.1f%% of max (%d/%d tokens). Compaction will run on next get_messages_for_request() call.",
                usage * 100,
                self._token_count,
                self.max_tokens,
            )

        logger.debug(
            "Added message: %s - %d total messages, %d tokens (%.1f%%)",
            message.get("role", "unknown"),
            len(self.messages),
            self._token_count,
            usage * 100,
        )

    async def get_messages_for_request(self, token_budget: int | None = None) -> list[dict[str, Any]]:
        """
        Get messages ready for an LLM request.

        Handles compaction internally if needed. Orchestrators call this before
        every LLM request and trust the context manager to return messages that
        fit within limits.

        Args:
            token_budget: Optional token limit. If None, uses configured max.

        Returns:
            Messages ready for LLM request, compacted if necessary.
        """
        budget = token_budget or self.max_tokens

        # Check if compaction needed
        if self._should_compact(budget):
            await self._compact_internal()

        return self.messages.copy()

    async def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages (raw, uncompacted) for transcripts/debugging."""
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

    def _should_compact(self, budget: int | None = None) -> bool:
        """Internal: Check if compaction is recommended."""
        effective_budget = budget or self.max_tokens
        usage = self._token_count / effective_budget
        should = usage >= self.compact_threshold
        if should:
            logger.info("Context at %.1f%% capacity, compaction needed", usage * 100)
        return should

    async def _compact_internal(self) -> None:
        """
        Internal: Compact the context while preserving tool_use/tool_result pairs.

        Anthropic API requires that tool_use blocks in message N have matching tool_result
        blocks in message N+1. These pairs are treated as atomic units during compaction:
        - If keeping a message with tool_calls, keep the next message (tool_result)
        - If keeping a tool message, keep the previous message (tool_use)

        This preserves conversation state integrity per IMPLEMENTATION_PHILOSOPHY.md:
        "Data integrity: Ensure data consistency and reliability"
        """
        logger.info("Compacting context with %d messages", len(self.messages))

        # Step 1: Determine base keep set
        keep_indices = set()

        # Keep all system messages
        for i, msg in enumerate(self.messages):
            if msg.get("role") == "system":
                keep_indices.add(i)

        # Keep last 10 messages
        for i in range(max(0, len(self.messages) - 10), len(self.messages)):
            keep_indices.add(i)

        # Step 2: Expand to preserve tool pairs (atomic units)
        # IMPORTANT: Must iterate until no new messages added, because:
        # - If tool_result in keep_indices → adds tool_use to expanded
        # - That tool_use → must add ALL its tool_results (not just those in keep_indices)
        expanded = keep_indices.copy()
        processed_indices = set()

        changed = True
        while changed:
            changed = False
            # Process indices we haven't processed yet
            to_process = expanded - processed_indices
            for i in to_process:
                processed_indices.add(i)
                msg = self.messages[i]

                # If keeping assistant with tool_calls, MUST keep ALL matching tool results
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    # Collect the tool_call IDs we need to find results for
                    expected_tool_ids = set()
                    for tc in msg["tool_calls"]:
                        if tc:
                            tool_id = tc.get("id") or tc.get("tool_call_id")
                            if tool_id:
                                expected_tool_ids.add(tool_id)

                    # Scan forward to find matching tool results
                    tool_results_kept = 0
                    for j in range(i + 1, len(self.messages)):
                        candidate = self.messages[j]
                        if candidate.get("role") == "tool":
                            tool_id = candidate.get("tool_call_id")
                            if tool_id in expected_tool_ids:
                                if j not in expanded:
                                    expanded.add(j)
                                    changed = True
                                expected_tool_ids.discard(tool_id)
                                tool_results_kept += 1
                                if not expected_tool_ids:
                                    break  # Found all tool results

                    if expected_tool_ids:
                        logger.warning(
                            "Message %d has %d tool_calls but only %d matching tool results found (missing IDs: %s)",
                            i,
                            len(msg["tool_calls"]),
                            tool_results_kept,
                            expected_tool_ids,
                        )

                    logger.debug(
                        "Preserving tool group: message %d (assistant with %d tool_calls) + %d tool result messages",
                        i,
                        len(msg["tool_calls"]),
                        tool_results_kept,
                    )

                # If keeping tool message, MUST keep the assistant with tool_calls
                elif msg.get("role") == "tool":
                    # Walk backwards to find assistant with tool_calls
                    for j in range(i - 1, -1, -1):
                        check_msg = self.messages[j]
                        if check_msg.get("role") == "assistant" and check_msg.get("tool_calls"):
                            if j not in expanded:
                                expanded.add(j)
                                changed = True
                            logger.debug(
                                "Preserving tool group: message %d (assistant with tool_calls) includes tool result at %d",
                                j,
                                i,
                            )
                            break
                        if check_msg.get("role") != "tool":
                            logger.warning(
                                "Tool result at %d has no matching assistant with tool_calls (found %s at %d instead)",
                                i,
                                check_msg.get("role"),
                                j,
                            )
                            break

        # Step 3: Build ordered compacted list
        compacted = [self.messages[i] for i in sorted(expanded)]

        # Deduplicate non-tool messages while preserving tool interactions
        seen = set()
        final: list[dict[str, Any]] = []
        for msg in compacted:
            if msg.get("role") == "tool" or (msg.get("role") == "assistant" and msg.get("tool_calls")):
                final.append(msg)
            else:
                # Content may be a string or list of blocks - convert to string for hashing
                content = msg.get("content", "")
                content_str = str(content) if not isinstance(content, str) else content
                key = (msg.get("role"), content_str[:100])
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
