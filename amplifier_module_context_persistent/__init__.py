"""
Persistent context manager module.

Extends SimpleContextManager with file-based persistence:
  • Append-only message storage to context-messages.jsonl (source of truth)
  • Ephemeral compaction inherited from context-simple (never modifies stored history)
  • Optional loading of persistent memory files on initialization
  • Self-contained session resume (loads from own file, ignores CLI's set_messages)

Architecture:
  - Inherits compaction logic from context-simple (single implementation)
  - context-messages.jsonl is the source of truth (append-only, never overwritten)
  - add_message() appends to file immediately
  - get_messages_for_request() reads file, uses inherited ephemeral compaction
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

# Import the base class - compaction logic lives here (single implementation)
from amplifier_module_context_simple import SimpleContextManager

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
            - truncate_boundary: Truncate tool results in first N% of history (default: 0.50)
            - protected_recent: Always protect last N% of messages (default: 0.10)
            - truncate_chars: Characters to keep when truncating tool results (default: 250)

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
        truncate_boundary=config.get("truncate_boundary", 0.50),
        protected_recent=config.get("protected_recent", 0.10),
        truncate_chars=config.get("truncate_chars", 250),
        hooks=getattr(coordinator, "hooks", None),
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


class PersistentContextManager(SimpleContextManager):
    """
    File-based context manager extending SimpleContextManager with persistence.

    Inherits all compaction logic from SimpleContextManager (single implementation).
    Adds file-based storage as the source of truth.

    Key Principle: The transcript file is the source of truth. Compaction NEVER
    modifies the file - it only returns a compacted VIEW for the current LLM request.
    Compaction logic is inherited from SimpleContextManager.

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
        truncate_boundary: float = 0.50,
        protected_recent: float = 0.10,
        truncate_chars: int = 250,
        hooks: Any = None,
    ):
        """
        Initialize the persistent context manager.

        Args:
            transcript_path: Path to context-messages.jsonl file (source of truth)
            memory_files: List of file paths to load at startup
            max_tokens: Maximum context size in tokens
            compact_threshold: Threshold for triggering compaction (0.0-1.0)
            target_usage: Target usage after compaction (0.0-1.0)
            truncate_boundary: Truncate tool results in first N% of history (0.0-1.0)
            protected_recent: Always protect last N% of messages (0.0-1.0)
            truncate_chars: Characters to keep when truncating tool results
        """
        # Initialize base class with compaction config
        super().__init__(
            max_tokens=max_tokens,
            compact_threshold=compact_threshold,
            target_usage=target_usage,
            protected_recent=protected_recent,
            truncate_chars=truncate_chars,
            hooks=hooks,
        )

        # File-based persistence attributes
        self.transcript_path = Path(transcript_path) if transcript_path else None
        self.memory_files = memory_files or []

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
            messages = self._read_all_from_file()
            if messages:
                self._loaded_from_file = True
                # Sync to in-memory state (used by parent's compaction)
                self.messages = messages
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

        # Add to in-memory list (used by parent's compaction)
        self.messages.append(message)

        # Persist to file
        if self.transcript_path:
            self._append_to_file(message)

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

        Syncs from file (if present), then uses inherited ephemeral compaction.
        The compacted result is a LOCAL variable - the file is NEVER modified.

        Args:
            token_budget: Optional explicit token limit (deprecated, prefer provider).
            provider: Optional provider instance for dynamic budget calculation.
                If provided, budget = context_window - max_output_tokens - safety_margin.

        Returns:
            Messages ready for LLM request, compacted if necessary.
        """
        # Sync from file to ensure we have latest state
        if self.transcript_path and self.transcript_path.exists():
            self.messages = self._read_all_from_file()

        # Use parent's implementation (ephemeral compaction)
        return await super().get_messages_for_request(token_budget, provider)

    async def get_messages(self) -> list[dict[str, Any]]:
        """
        Get ALL messages (full history, never compacted) for transcripts/debugging.

        This reads directly from the file - the complete, unmodified history.
        """
        if self.transcript_path and self.transcript_path.exists():
            self.messages = self._read_all_from_file()

        # Use parent's implementation (returns copy of self.messages)
        return await super().get_messages()

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
                len(self.messages),
            )
            return

        # Set in-memory state
        self.messages = list(messages)

        # Persist to file
        if self.transcript_path:
            self._write_all_to_file(messages)

        logger.info("Set %d messages to context", len(messages))

    async def clear(self) -> None:
        """
        Clear all messages by truncating the transcript file.

        Use with caution - this permanently removes history.
        """
        if self.transcript_path and self.transcript_path.exists():
            self.transcript_path.write_text("")

        # Use parent's clear
        await super().clear()

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

    def _read_all_from_file(self) -> list[dict[str, Any]]:
        """Read all messages from the transcript file."""
        if not self.transcript_path or not self.transcript_path.exists():
            return list(self.messages)

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
            return list(self.messages)

        return messages

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

        # Log final state
        system_messages = [msg for msg in self.messages if msg.get("role") == "system"]
        logger.info("Persistent context initialized with %d system messages", len(system_messages))

    async def _validate_startup_context(self) -> None:
        """Ensure context limits are not exceeded after loading memory files."""
        token_count = self._estimate_tokens(self.messages)

        if token_count <= self.max_tokens:
            return

        system_messages = [m for m in self.messages if m.get("role") == "system"]
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
