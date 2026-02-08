"""Tests for timestamp injection feature in persistent context."""

import pytest
from datetime import datetime, UTC
from pathlib import Path
import tempfile
import json

from amplifier_module_context_persistent import PersistentContextManager


@pytest.mark.asyncio
async def test_timestamps_added_automatically():
    """Verify timestamps are always added to message metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript_path = Path(tmpdir) / "test-transcript.jsonl"
        context = PersistentContextManager(transcript_path=transcript_path)
        await context.initialize()

        # Add message without timestamp
        message = {"role": "user", "content": "Hello"}
        await context.add_message(message)

        # Verify timestamp was added in metadata
        messages = await context.get_messages()
        assert len(messages) == 1
        assert "metadata" in messages[0]
        assert "timestamp" in messages[0]["metadata"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"


@pytest.mark.asyncio
async def test_timestamps_do_not_mutate_caller_dict():
    """Verify that add_message does not mutate the caller's dictionary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript_path = Path(tmpdir) / "test-transcript.jsonl"
        context = PersistentContextManager(transcript_path=transcript_path)
        await context.initialize()

        # Add message without timestamp
        original_message = {"role": "user", "content": "Hello"}
        await context.add_message(original_message)

        # Verify caller's dict was NOT modified
        assert "metadata" not in original_message
        assert "timestamp" not in original_message

        # But the stored message has metadata with timestamp
        messages = await context.get_messages()
        assert len(messages) == 1
        assert "metadata" in messages[0]
        assert "timestamp" in messages[0]["metadata"]


@pytest.mark.asyncio
async def test_timestamp_format():
    """Verify timestamp format is ISO 8601 with milliseconds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript_path = Path(tmpdir) / "test-transcript.jsonl"
        context = PersistentContextManager(transcript_path=transcript_path)
        await context.initialize()

        await context.add_message({"role": "user", "content": "Test"})

        messages = await context.get_messages()
        timestamp = messages[0]["metadata"]["timestamp"]

        # Should match format: 2026-02-06T10:00:00.123+00:00
        # Year-Month-DayTHour:Minute:Second.Milliseconds+00:00
        assert "T" in timestamp
        assert timestamp.endswith("+00:00")
        assert "." in timestamp  # Has milliseconds

        # Should be parseable
        parsed = datetime.fromisoformat(timestamp)
        assert isinstance(parsed, datetime)

        # Should be recent (within last second)
        now = datetime.now(UTC)
        delta = (now - parsed).total_seconds()
        assert delta < 1.0, f"Timestamp {timestamp} is not recent (delta: {delta}s)"


@pytest.mark.asyncio
async def test_timestamps_preserved_when_present():
    """Verify existing timestamps in metadata are preserved, not overwritten."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript_path = Path(tmpdir) / "test-transcript.jsonl"
        context = PersistentContextManager(transcript_path=transcript_path)
        await context.initialize()

        # Add message WITH existing timestamp in metadata
        original_timestamp = "2026-02-06T10:00:00.123+00:00"
        message = {
            "role": "user",
            "content": "Hello",
            "metadata": {"timestamp": original_timestamp},
        }
        await context.add_message(message)

        # Verify original timestamp was preserved
        messages = await context.get_messages()
        assert len(messages) == 1
        assert messages[0]["metadata"]["timestamp"] == original_timestamp


@pytest.mark.asyncio
async def test_timestamps_preserve_existing_metadata():
    """Verify existing metadata fields are preserved when adding timestamp."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript_path = Path(tmpdir) / "test-transcript.jsonl"
        context = PersistentContextManager(transcript_path=transcript_path)
        await context.initialize()

        # Add message with existing metadata (like hook-injected messages)
        message = {
            "role": "system",
            "content": "Hook message",
            "metadata": {"source": "hook", "custom_field": "value"},
        }
        await context.add_message(message)

        # Verify existing metadata was preserved AND timestamp added
        messages = await context.get_messages()
        assert len(messages) == 1
        assert messages[0]["metadata"]["source"] == "hook"
        assert messages[0]["metadata"]["custom_field"] == "value"
        assert "timestamp" in messages[0]["metadata"]


@pytest.mark.asyncio
async def test_timestamps_all_message_types():
    """Verify timestamps work for all message types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript_path = Path(tmpdir) / "test-transcript.jsonl"
        context = PersistentContextManager(transcript_path=transcript_path)
        await context.initialize()

        # User message
        await context.add_message({"role": "user", "content": "Question"})

        # Assistant message
        await context.add_message({"role": "assistant", "content": "Answer"})

        # Tool message
        await context.add_message(
            {"role": "tool", "tool_call_id": "1", "content": "Result"}
        )

        # System message
        await context.add_message({"role": "system", "content": "Instructions"})

        messages = await context.get_messages()
        assert len(messages) == 4

        # All should have timestamps in metadata
        for msg in messages:
            assert "metadata" in msg
            assert "timestamp" in msg["metadata"]
            # Verify format
            timestamp = msg["metadata"]["timestamp"]
            parsed = datetime.fromisoformat(timestamp)
            assert parsed is not None


@pytest.mark.asyncio
async def test_timestamps_preserved_on_resume():
    """Verify timestamps from set_messages() are preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript_path = Path(tmpdir) / "test-transcript.jsonl"
        context = PersistentContextManager(transcript_path=transcript_path)
        await context.initialize()

        # Restore messages from a saved session (with existing timestamps in metadata)
        saved_messages = [
            {
                "role": "user",
                "content": "Message 1",
                "metadata": {"timestamp": "2026-02-06T10:00:00.123+00:00"},
            },
            {
                "role": "assistant",
                "content": "Response 1",
                "metadata": {"timestamp": "2026-02-06T10:00:05.456+00:00"},
            },
        ]

        await context.set_messages(saved_messages)

        # Verify timestamps were preserved
        messages = await context.get_messages()
        assert len(messages) == 2
        assert messages[0]["metadata"]["timestamp"] == "2026-02-06T10:00:00.123+00:00"
        assert messages[1]["metadata"]["timestamp"] == "2026-02-06T10:00:05.456+00:00"

        # Add new message - should get new timestamp
        await context.add_message({"role": "user", "content": "Message 2"})

        messages = await context.get_messages()
        assert len(messages) == 3
        # First two preserved
        assert messages[0]["metadata"]["timestamp"] == "2026-02-06T10:00:00.123+00:00"
        assert messages[1]["metadata"]["timestamp"] == "2026-02-06T10:00:05.456+00:00"
        # Third got new timestamp
        assert "timestamp" in messages[2]["metadata"]
        assert messages[2]["metadata"]["timestamp"] != "2026-02-06T10:00:00.123+00:00"


@pytest.mark.asyncio
async def test_timestamps_with_hook_messages():
    """Verify timestamps work correctly with hook-injected messages that have metadata.source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript_path = Path(tmpdir) / "test-transcript.jsonl"
        context = PersistentContextManager(transcript_path=transcript_path)
        await context.initialize()

        # Simulate hook-injected message (kernel adds metadata.source = "hook")
        hook_message = {
            "role": "system",
            "content": "Hook-injected content",
            "metadata": {"source": "hook"},
        }
        await context.add_message(hook_message)

        messages = await context.get_messages()
        assert len(messages) == 1
        # Should preserve source and add timestamp
        assert messages[0]["metadata"]["source"] == "hook"
        assert "timestamp" in messages[0]["metadata"]


@pytest.mark.asyncio
async def test_timestamps_persisted_to_file():
    """Verify timestamps are written to the transcript file in metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript_path = Path(tmpdir) / "test-transcript.jsonl"
        context = PersistentContextManager(transcript_path=transcript_path)
        await context.initialize()

        # Add messages
        await context.add_message({"role": "user", "content": "Test 1"})
        await context.add_message({"role": "assistant", "content": "Response 1"})

        # Read file directly
        with open(transcript_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Parse each line and verify metadata.timestamp exists
        for line in lines:
            msg = json.loads(line.strip())
            assert "metadata" in msg
            assert "timestamp" in msg["metadata"]
            # Verify it's in the correct format
            timestamp = msg["metadata"]["timestamp"]
            parsed = datetime.fromisoformat(timestamp)
            assert parsed is not None


@pytest.mark.asyncio
async def test_timestamps_loaded_from_file_on_resume():
    """Verify timestamps are preserved when loading from file on session resume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        transcript_path = Path(tmpdir) / "test-transcript.jsonl"
        
        # First session - write messages with timestamps
        context1 = PersistentContextManager(transcript_path=transcript_path)
        await context1.initialize()
        await context1.add_message({"role": "user", "content": "Message 1"})
        await context1.add_message({"role": "assistant", "content": "Response 1"})
        
        messages1 = await context1.get_messages()
        original_timestamps = [msg["metadata"]["timestamp"] for msg in messages1]

        # Second session - resume from file
        context2 = PersistentContextManager(transcript_path=transcript_path)
        await context2.initialize()

        # Verify messages loaded with original timestamps
        messages2 = await context2.get_messages()
        assert len(messages2) == 2
        assert messages2[0]["metadata"]["timestamp"] == original_timestamps[0]
        assert messages2[1]["metadata"]["timestamp"] == original_timestamps[1]
