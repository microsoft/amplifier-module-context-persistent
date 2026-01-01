"""
Persistent context manager module.

Implements a file-based context manager that provides:
  • Append-only message storage to context-messages.jsonl (source of truth)
  • Ephemeral compaction (never modifies stored history)
  • Optional loading of persistent memory files on initialization
  • Self-contained session resume (loads from own file, ignores CLI's set_messages)

Architecture:
  - context-messages.jsonl is the source of truth (append-only, never overwritten)
  - add_message() appends to file immediately
  - get_messages_for_request() reads file, compacts EPHEMERALLY for LLM request
  - get_messages() reads file, returns FULL history (for transcripts/debugging)

File Separation:
  - context-persistent owns: context-messages.jsonl (full history with system messages)
  - CLI SessionStore owns: transcript.jsonl (filtered user/assistant only)
  - These are separate files to prevent conflicts during session resume

This design ensures conversation history is never lost, even during compaction.
"""

from __future__ import annotations

# Amplifier module metadata
__amplifier_module_type__ = "context"

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from amplifier_core import ModuleCoordinator

logger = logging.getLogger(__name__)

# Use distinct filename to avoid collision with CLI's transcript.jsonl
CONTEXT_MESSAGES_FILENAME = "context-messages.jsonl"


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the persistent context manager.

    Args:
        coordinator: Module coordinator
        config: Optional configuration
            - transcript_path: Path to messages file (auto-generated if not provided)
            - memory_files: List of file paths to load at startup
            - max_tokens: Maximum context size (default: 200,000)
            - compact_threshold: Trigger compaction at this usage (default: 0.92)
            - target_usage: Compact down to this usage (default: 0.50)

    Returns:
        Optional cleanup function
    """
    config = config or {}

    # Auto-generate transcript_path from session_id if not provided
    transcript_path = config.get("transcript_path")
    if not transcript_path:
        session_id = getattr(coordinator, "session_id", None)
        if session_id:
            # Use standard Amplifier session directory structure
            project_slug = _get_project_slug()
            session_dir = Path.home() / ".amplifier" / "projects" / project_slug / "sessions" / session_id
            transcript_path = session_dir / CONTEXT_MESSAGES_FILENAME
            logger.debug("Auto-generated transcript_path: %s", transcript_path)

    context = PersistentContextManager(
        transcript_path=transcript_path,
        memory_files=config.get("memory_files", []),
        max_tokens=config.get("max_tokens", 200_000),
        compact_threshold=config.get("compact_threshold", 0.92),
        target_usage=config.get("target_usage", 0.50),
    )

    await context.initialize()
    await coordinator.mount("context", context)
    logger.info(
        "Mounted PersistentContextManager "
        f"(transcript_path={context.transcript_path}, "
        f"loaded_from_file={context._loaded_from_file}, "
        f"memory_files={len(context.memory_files)}, max_tokens={context.max_tokens}, "
        f"compact_threshold={context.compact_threshold:.2f})"
    )
    return


def _get_project_slug() -> str:
    """Get project slug from current directory (matches CLI's project_utils)."""
    import hashlib
    cwd = Path.cwd()
    # Create slug from path: last component + hash prefix
    name = cwd.name or "root"
    path_hash = hashlib.sha256(str(cwd).encode()).hexdigest()[:8]
    return f"{name}-{path_hash}"


class PersistentContextManager:
    """
    File-based context manager with append-only storage and ephemeral compaction.

    Key Principle: The transcript file is the source of truth. Compaction NEVER
    modifies the file - it only returns a compacted VIEW for the current LLM request.

    Session Resume: When the transcript file already exists (session resume),
    we load from it and ignore any set_messages() calls from the CLI. This ensures
    our complete history (including system messages) is preserved, rather than
    being overwritten by the CLI's filtered transcript.

    Owns memory policy: orchestrators ask for messages via get_messages_for_request(),
    and this context manager decides how to fit them within limits. Compaction is
    handled internally and ephemerally - the original history is always preserved.
    """

    def __init__(
        self,
        transcript_path: str | Path | None = None,
        memory_files: list[str] | None = None,
        max_tokens: int = 200_000,
        compact_threshold: float = 0.92,
        target_usage: float = 0.50,
    ):
        """
        Initialize the persistent context manager.

        Args:
            transcript_path: Path to context-messages.jsonl file (source of truth)
            memory_files: List of file paths to load at startup
            max_tokens: Maximum context size in tokens
            compact_threshold: Threshold for triggering compaction (0.0-1.0)
            target_usage: Target usage after compaction (0.0-1.0)
        """
        self.transcript_path = Path(transcript_path) if transcript_path else None
        self.memory_files = memory_files or []
        self.max_tokens = max_tokens
        self.compact_threshold = compact_threshold
        self.target_usage = target_usage

        # In-memory cache for performance (mirrors file, never authoritative)
        self._messages_cache: list[dict[str, Any]] | None = None
        self._cache_valid = False

        # Track if we loaded from an existing file (for session resume)
        # When True, set_messages() is ignored to prevent CLI overwriting our state
        self._loaded_from_file = False

    async def initialize(self) -> None:
        """
        Initialize the context manager.

        If transcript file exists (session resume), load from it.
        Otherwise, load configured memory files for fresh session.
        """
        # Check if we're resuming an existing session
        if self.transcript_path and self.transcript_path.exists():
            messages = await self._read_all_messages()
            if messages:
                self._loaded_from_file = True
                logger.info(
                    "Resumed from existing transcript: %d messages from %s",
                    len(messages),
                    self.transcript_path,
                )
                # Skip memory file loading - we already have our state
                return

        # Fresh session: load memory files
        await self._load_memory_files()
        await self._validate_startup_context()

    async def add_message(self, message: dict[str, Any]) -> None:
        """
        Add a message to the context by appending to transcript file.

        This is an APPEND-ONLY operation. The file is never overwritten.

        Messages are always accepted. Compaction happens ephemerally when
        get_messages_for_request() is called before LLM requests.

        Tool results MUST be added even if over threshold, otherwise
        tool_use/tool_result pairing breaks.
        """
        # Add timestamp if not present
        if "timestamp" not in message:
            message = {**message, "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds")}

        if self.transcript_path:
            # Append to file (atomic append, never overwrites)
            self._append_to_file(message)
            # Invalidate cache
            self._cache_valid = False
        else:
            # Fallback: in-memory only (for testing or when no path configured)
            if self._messages_cache is None:
                self._messages_cache = []
            self._messages_cache.append(message)

        logger.debug(
            "Added message: %s (appended to %s)",
            message.get("role", "unknown"),
            self.transcript_path or "memory",
        )

    async def get_messages_for_request(
        self,
        token_budget: int | None = None,
        provider: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get messages ready for an LLM request.

        Reads from file, applies EPHEMERAL compaction if needed.
        The compacted result is a LOCAL variable - the file is NEVER modified.

        Args:
            token_budget: Optional explicit token limit (deprecated, prefer provider).
            provider: Optional provider instance for dynamic budget calculation.
                If provided, budget = context_window - max_output_tokens - safety_margin.

        Returns:
            Messages ready for LLM request, compacted if necessary.
        """
        # Read full history from file
        all_messages = await self._read_all_messages()

        budget = self._calculate_budget(token_budget, provider)
        token_count = self._estimate_tokens(all_messages)

        # Check if compaction needed
        if self._should_compact(token_count, budget):
            # Compact EPHEMERALLY - returns new list, doesn't modify file
            compacted = self._compact_ephemeral(all_messages, budget)
            logger.info(
                "Ephemeral compaction: %d -> %d messages for this request",
                len(all_messages),
                len(compacted),
            )
            return compacted

        return all_messages

    async def get_messages(self) -> list[dict[str, Any]]:
        """
        Get ALL messages (full history, never compacted) for transcripts/debugging.

        This reads directly from the file - the complete, unmodified history.
        """
        return await self._read_all_messages()

    async def set_messages(self, messages: list[dict[str, Any]]) -> None:
        """
        Set messages directly (for initial session setup or migration).

        IMPORTANT: If we already loaded from our own file (session resume),
        this call is IGNORED. This prevents the CLI's filtered transcript
        from overwriting our complete history.

        Use cases where this IS applied:
        - Fresh sessions (no existing transcript file)
        - Migration from context-simple to context-persistent
        - Explicit state restoration (rare)

        Use cases where this is IGNORED:
        - Session resume when our transcript file already exists
        """
        # If we already loaded from file, ignore CLI's set_messages
        # This preserves our complete history (including system messages)
        if self._loaded_from_file:
            logger.info(
                "Ignoring set_messages(%d messages) - already loaded %d messages from file",
                len(messages),
                len(self._messages_cache or []),
            )
            return

        if self.transcript_path:
            # Write all messages to file (overwrite for initial load)
            self._write_all_to_file(messages)
            self._cache_valid = False
        else:
            self._messages_cache = list(messages)

        logger.info("Set %d messages to context", len(messages))

    async def clear(self) -> None:
        """
        Clear all messages by truncating the transcript file.

        Use with caution - this permanently removes history.
        """
        if self.transcript_path and self.transcript_path.exists():
            self.transcript_path.write_text("")
        self._messages_cache = []
        self._cache_valid = False
        logger.info("Context cleared")

    # ------------------------------------------------------------------
    # File Operations (Append-Only)
    # ------------------------------------------------------------------

    def _append_to_file(self, message: dict[str, Any]) -> None:
        """Append a single message to the transcript file."""
        if not self.transcript_path:
            return

        # Ensure parent directory exists
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)

        # Append line (atomic for single writes)
        with self.transcript_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

    def _write_all_to_file(self, messages: list[dict[str, Any]]) -> None:
        """Write all messages to file (used only for set_messages)."""
        if not self.transcript_path:
            return

        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)

        with self.transcript_path.open("w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    async def _read_all_messages(self) -> list[dict[str, Any]]:
        """Read all messages from the transcript file."""
        # Use cache if valid
        if self._cache_valid and self._messages_cache is not None:
            return list(self._messages_cache)

        if not self.transcript_path or not self.transcript_path.exists():
            return list(self._messages_cache or [])

        messages = []
        try:
            with self.transcript_path.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            messages.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning("Skipping malformed line %d: %s", line_num, e)
        except OSError as e:
            logger.error("Failed to read transcript file: %s", e)
            return list(self._messages_cache or [])

        # Update cache
        self._messages_cache = messages
        self._cache_valid = True

        return list(messages)

    # ------------------------------------------------------------------
    # Ephemeral Compaction (NEVER modifies file)
    # ------------------------------------------------------------------

    def _should_compact(self, token_count: int, budget: int) -> bool:
        """Check if compaction is needed."""
        usage = token_count / budget if budget > 0 else 0
        should = usage >= self.compact_threshold
        if should:
            logger.info("Context at %.1f%% capacity, compaction needed", usage * 100)
        return should

    def _compact_ephemeral(
        self, messages: list[dict[str, Any]], budget: int
    ) -> list[dict[str, Any]]:
        """
        Compact messages EPHEMERALLY for a single LLM request.

        This returns a NEW list - the original messages list is NOT modified.
        The transcript file is NEVER touched by this method.

        Strategy:
        1. Always keep system messages
        2. Always keep recent messages (last 10% or minimum 10)
        3. Truncate old tool results
        4. Remove oldest messages if still over budget
        5. Preserve tool_use/tool_result pairs as atomic units
        """
        target_tokens = int(budget * self.target_usage)

        # Step 1: Identify what to keep
        keep_indices = set()

        # Always keep system messages
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                keep_indices.add(i)

        # Always keep recent messages (last 10% or minimum 10)
        recent_count = max(10, int(len(messages) * 0.10))
        for i in range(max(0, len(messages) - recent_count), len(messages)):
            keep_indices.add(i)

        # Step 2: Expand to preserve tool pairs
        expanded = self._expand_for_tool_pairs(messages, keep_indices)

        # Step 3: Build initial compacted list with truncated tool results
        compacted = []
        for i in sorted(expanded):
            msg = messages[i]
            # Truncate old tool results (outside recent window)
            if msg.get("role") == "tool" and i < len(messages) - recent_count:
                msg = self._truncate_tool_result(msg)
            compacted.append(msg)

        # Step 4: If still over budget, remove more old messages
        current_tokens = self._estimate_tokens(compacted)
        if current_tokens > target_tokens:
            compacted = self._remove_oldest_until_target(
                compacted, target_tokens, recent_count
            )

        return compacted

    def _expand_for_tool_pairs(
        self, messages: list[dict[str, Any]], keep_indices: set[int]
    ) -> set[int]:
        """
        Expand keep_indices to include complete tool_use/tool_result pairs.

        CRITICAL: Tool pairs must be kept together to avoid LLM API errors.
        """
        expanded = keep_indices.copy()
        changed = True

        while changed:
            changed = False
            to_check = list(expanded)

            for i in to_check:
                msg = messages[i]

                # If keeping assistant with tool_calls, keep all matching tool results
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg.get("tool_calls", []):
                        tc_id = tc.get("id") or tc.get("tool_call_id")
                        if tc_id:
                            for j, m in enumerate(messages):
                                if m.get("tool_call_id") == tc_id and j not in expanded:
                                    expanded.add(j)
                                    changed = True

                # If keeping tool result, keep matching assistant with tool_calls
                elif msg.get("role") == "tool":
                    tool_call_id = msg.get("tool_call_id")
                    if tool_call_id:
                        for j in range(i - 1, -1, -1):
                            check = messages[j]
                            if check.get("role") == "assistant" and check.get("tool_calls"):
                                for tc in check.get("tool_calls", []):
                                    if (tc.get("id") or tc.get("tool_call_id")) == tool_call_id:
                                        if j not in expanded:
                                            expanded.add(j)
                                            changed = True
                                        break
                            if check.get("role") != "tool":
                                break

        return expanded

    def _truncate_tool_result(self, msg: dict[str, Any], max_chars: int = 250) -> dict[str, Any]:
        """
        Truncate a tool result message content.

        Returns a NEW dict - does not modify the original.
        """
        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) <= max_chars:
            return msg

        original_tokens = len(content) // 4
        truncated = {
            **msg,
            "content": f"[truncated: ~{original_tokens:,} tokens] {content[:max_chars]}...",
            "_truncated": True,
        }
        return truncated

    def _remove_oldest_until_target(
        self, messages: list[dict[str, Any]], target_tokens: int, protected_recent: int
    ) -> list[dict[str, Any]]:
        """
        Remove oldest non-protected messages until under target.

        Protected: system messages, recent messages, complete tool pairs.
        """
        # Build protection set
        protected = set()

        for i, msg in enumerate(messages):
            # Protect system messages
            if msg.get("role") == "system":
                protected.add(i)

        # Protect recent messages
        for i in range(max(0, len(messages) - protected_recent), len(messages)):
            protected.add(i)

        # Protect tool pairs (if one is protected, both must be)
        for i, msg in enumerate(messages):
            if i in protected:
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    for tc in msg.get("tool_calls", []):
                        tc_id = tc.get("id") or tc.get("tool_call_id")
                        for j, m in enumerate(messages):
                            if m.get("tool_call_id") == tc_id:
                                protected.add(j)
                elif msg.get("role") == "tool":
                    tc_id = msg.get("tool_call_id")
                    for j, m in enumerate(messages):
                        if m.get("role") == "assistant" and m.get("tool_calls"):
                            for tc in m.get("tool_calls", []):
                                if (tc.get("id") or tc.get("tool_call_id")) == tc_id:
                                    protected.add(j)

        # Remove oldest unprotected until under target
        result = list(messages)
        current_tokens = self._estimate_tokens(result)

        for i in range(len(messages)):
            if current_tokens <= target_tokens:
                break
            if i not in protected:
                msg = messages[i]
                # Skip if this would orphan a tool pair
                if self._would_orphan_tool_pair(messages, i, protected):
                    continue
                result[i] = None  # Mark for removal
                current_tokens -= len(str(msg)) // 4

        return [m for m in result if m is not None]

    def _would_orphan_tool_pair(
        self, messages: list[dict[str, Any]], index: int, protected: set[int]
    ) -> bool:
        """Check if removing this message would orphan a tool pair."""
        msg = messages[index]

        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Check if any tool result for this is protected
            for tc in msg.get("tool_calls", []):
                tc_id = tc.get("id") or tc.get("tool_call_id")
                for j, m in enumerate(messages):
                    if m.get("tool_call_id") == tc_id and j in protected:
                        return True

        elif msg.get("role") == "tool":
            # Check if the assistant with tool_calls is protected
            tc_id = msg.get("tool_call_id")
            for j, m in enumerate(messages):
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    for tc in m.get("tool_calls", []):
                        if (tc.get("id") or tc.get("tool_call_id")) == tc_id:
                            if j in protected:
                                return True

        return False

    # ------------------------------------------------------------------
    # Persistent Memory Files
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

        # Read back to check
        messages = await self._read_all_messages()
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        logger.info("Persistent context initialized with %d system messages", len(system_messages))

    async def _validate_startup_context(self) -> None:
        """Ensure context limits are not exceeded after loading memory files."""
        messages = await self._read_all_messages()
        token_count = self._estimate_tokens(messages)

        if token_count <= self.max_tokens:
            return

        system_messages = [m for m in messages if m.get("role") == "system"]
        error_lines = [
            "Memory files exceed context limit!",
            "",
            f"Total tokens: {token_count:,} > {self.max_tokens:,} max",
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
    # Helpers
    # ------------------------------------------------------------------

    def _calculate_budget(self, token_budget: int | None, provider: Any | None) -> int:
        """Calculate effective token budget from provider or fallback to config."""
        if token_budget is not None:
            return token_budget

        if provider is not None:
            try:
                info = provider.get_info()
                defaults = info.defaults or {}
                context_window = defaults.get("context_window")
                max_output_tokens = defaults.get("max_output_tokens")

                if context_window and max_output_tokens:
                    safety_margin = 1000
                    budget = context_window - max_output_tokens - safety_margin
                    logger.debug(
                        "Calculated budget from provider: %d (context=%d, output=%d, safety=%d)",
                        budget,
                        context_window,
                        max_output_tokens,
                        safety_margin,
                    )
                    return budget
            except Exception as e:
                logger.debug("Could not get budget from provider: %s", e)

        return self.max_tokens

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Rough token estimation (chars / 4)."""
        return sum(len(str(msg)) // 4 for msg in messages)
