# Persistent Context Manager Module

A context manager that extends SimpleContextManager with persistent file loading capabilities for cross-session memory.

## Purpose

Enable AI assistants to maintain context across sessions by loading project-specific context files (AGENTS.md, PROJECT.md, etc.) at session start. This allows teams to maintain consistent AI behavior and knowledge without manually injecting context each time.

## Features

- **Inherits from SimpleContextManager**: All base functionality (message storage, token counting, compaction)
- **File Loading at Session Start**: Automatically loads configured context files
- **Graceful Error Handling**: Skips missing files with warnings
- **Clear Context Labeling**: Each loaded file is labeled in the system message
- **Home Directory Expansion**: Supports `~/` paths

## Prerequisites

- **Python 3.11+**
- **[UV](https://github.com/astral-sh/uv)** - Fast Python package manager

### Installing UV

```bash
# macOS/Linux/WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Installation

```bash
cd amplifier-module-context-persistent
uv pip install -e .
```

## Configuration

In your Amplifier config.toml:

```toml
[modules]
context = "context-persistent"

[context]
# List of context files to load at session start
memory_files = [
    "./AGENTS.md",
    "./PROJECT.md",
    "~/.amplifier/AGENTS.md"
]

# Standard context manager settings
max_tokens = 200000
compact_threshold = 0.92
```

## How It Works

1. **Session Start**: When mounted, the module loads all configured files
2. **File Processing**: Each file is read and added as a system message
3. **Context Labeling**: Files are clearly labeled (e.g., `[Context from AGENTS.md]`)
4. **Error Handling**: Missing or unreadable files are skipped with warnings
5. **Normal Operation**: After loading, behaves like SimpleContextManager

## File Format Support

Currently supports plain text files (Markdown, text, etc.). Files are read as UTF-8 and inserted as-is into system messages.

### Example Loaded Message

```python
{
    'role': 'system',
    'content': '[Context from AGENTS.md]\n\n# AGENTS.md\n\nThis file provides guidance...'
}
```

## API

The module exposes the same API as SimpleContextManager with one additional method:

### PersistentContextManager

```python
class PersistentContextManager(SimpleContextManager):
    def __init__(self, memory_files: list[str] = None, max_tokens: int = 200_000, compact_threshold: float = 0.92)
    async def initialize(self) -> None  # Loads memory files

    # Inherited from SimpleContextManager:
    async def add_message(self, message: dict[str, Any]) -> None
    async def get_messages() -> list[dict[str, Any]]  # Raw access for debugging
    async def get_messages_for_request(
        token_budget: int | None = None,
        provider: Any | None = None,  # Dynamic budget from provider.get_info().defaults
    ) -> list[dict[str, Any]]  # For LLM requests
    async def clear(self) -> None
```

## Compaction Strategy

Inherits compaction strategy from SimpleContextManager with full tool pair preservation:

- **Keeps**: All system messages + last 10 conversation messages
- **Deduplicates**: Non-tool messages (based on role and first 100 chars of content)
- **Preserves tool pairs**: Tool_use and tool_result messages are treated as atomic units
  - If keeping assistant message with tool_calls, keeps all subsequent tool result messages
  - If keeping tool message, walks backwards to find and keep the assistant with tool_calls
  - Never deduplicates tool-related messages (each has unique ID)

### Tool Pair Preservation

Anthropic API requires that every tool_use in message N has matching tool_result messages immediately following. The context manager preserves these as atomic units during compaction to maintain conversation state integrity and prevent API errors.

**Critical implementation detail**: When an assistant message has multiple tool_calls, there are multiple consecutive tool_result messages after it. The compaction logic walks backwards through these tool results to find the originating assistant message, ensuring the entire tool group is preserved as an atomic unit. This prevents orphaned tool results that would cause API validation errors.

## Philosophy

This module follows the ruthless simplicity principle:

- **Simple file loading**: Just read files and add as system messages
- **No complex watching**: Files are loaded once at session start
- **Fail gracefully**: Missing files don't crash the system
- **Clear boundaries**: Memory files are read-only

## Common Use Cases

### Team Context Files

Maintain consistent AI behavior across a development team:

```toml
memory_files = [
    "./TEAM_STYLE_GUIDE.md",
    "./PROJECT_ARCHITECTURE.md",
    "./API_CONVENTIONS.md"
]
```

### Project-Specific Knowledge

Load project-specific documentation:

```toml
memory_files = [
    "./docs/DESIGN_DECISIONS.md",
    "./docs/KNOWN_ISSUES.md",
    "./docs/DEPLOYMENT_GUIDE.md"
]
```

### Personal Preferences

Individual developer preferences:

```toml
memory_files = [
    "~/.amplifier/AGENTS.md",
    "./amplifier/local/AGENTS.md"
]
```

## Logging

The module logs at various levels:

- **INFO**: Successful file loads, mount status
- **WARNING**: Missing files
- **ERROR**: File read failures
- **DEBUG**: Empty files, detailed operations

## Limitations

- **Read-only**: Files are loaded at start, not monitored for changes
- **Text only**: Currently only supports text-based files
- **No auto-save**: Does not persist context back to files
- **Sequential loading**: Files are loaded in order specified

## Future Enhancements (Not Implemented)

Potential future features that maintain simplicity:

- File watching for reload on changes
- TOML/JSON parsing for structured context
- Selective context persistence
- File change detection

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
