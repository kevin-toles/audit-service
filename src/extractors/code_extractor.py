"""
EEP-5.1: Code Block Extraction from Chapters

Extracts code blocks from markdown chapter content with metadata.

WBS Mapping:
- 5.1.1: Python code block detection (```python markers)
- 5.1.2: Language identification from fence markers
- 5.1.3: Code metadata extraction (line numbers, context)
- 5.1.4: Multi-block extraction from single chapter

Patterns Applied:
- Pydantic models for structured output
- Regex-based parsing with precompiled patterns (performance)
- Constants for repeated values (Anti-Pattern S1192)

Anti-Patterns Avoided:
- S1192: Extracted constants for repeated strings
- S3776: Small focused methods, low cognitive complexity
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Constants (Anti-Pattern S1192 Prevention)
# =============================================================================

_UNKNOWN_LANGUAGE = "unknown"
_CONTEXT_LINES = 3  # Number of lines to capture before/after code block

# Language aliases to normalize
_LANGUAGE_ALIASES: dict[str, str] = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "ts": "typescript",
    "sh": "shell",
    "bash": "shell",
    "zsh": "shell",
    "yml": "yaml",
    "rb": "ruby",
}

# Precompiled regex for code fence detection
# Matches ```language (optional) at start of line
_CODE_FENCE_PATTERN = re.compile(
    r"^```(\w*)\s*$",  # Opening fence with optional language
    re.MULTILINE,
)


# =============================================================================
# Pydantic Models
# =============================================================================


class CodeBlock(BaseModel):
    """Extracted code block with metadata.

    WBS 5.1.3: Contains code content and extraction metadata.
    """

    code: str = Field(description="Raw code content without fence markers")
    language: str = Field(default=_UNKNOWN_LANGUAGE, description="Programming language")
    start_line: int = Field(ge=1, description="Starting line number (1-indexed)")
    end_line: int = Field(ge=1, description="Ending line number (1-indexed)")
    index: int = Field(ge=0, description="Block index within document")
    context_before: str = Field(default="", description="Text before code block")
    context_after: str = Field(default="", description="Text after code block")


# =============================================================================
# CodeExtractor Class
# =============================================================================


class CodeExtractor:
    """Extracts code blocks from markdown content.

    WBS 5.1: Parses markdown to find fenced code blocks with metadata.

    Usage:
        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(markdown_content)

        # Filter by language
        python_blocks = extractor.extract_code_blocks(content, language_filter="python")

        # Batch processing
        results = extractor.extract_from_chapters([
            {"chapter_id": "ch1", "content": "..."},
            {"chapter_id": "ch2", "content": "..."},
        ])
    """

    def __init__(self) -> None:
        """Initialize the code extractor."""
        pass  # Stateless extractor

    def extract_code_blocks(
        self,
        content: str,
        language_filter: str | None = None,
    ) -> list[CodeBlock]:
        """Extract all code blocks from markdown content.

        WBS 5.1.1-5.1.4: Full extraction with language detection and metadata.

        Args:
            content: Markdown content to parse
            language_filter: Optional language to filter by (e.g., "python")

        Returns:
            List of CodeBlock objects in order of appearance
        """
        if not content or not content.strip():
            return []

        blocks: list[CodeBlock] = []
        lines = content.split("\n")
        total_lines = len(lines)

        i = 0
        block_index = 0

        while i < total_lines:
            line = lines[i]

            # Check for opening fence
            match = _CODE_FENCE_PATTERN.match(line)
            if match:
                language_raw = match.group(1) or ""
                language = self._normalize_language(language_raw)

                # Find closing fence
                start_line = i + 1  # 1-indexed, points to first line of code
                code_start_idx = i + 1
                closing_idx = self._find_closing_fence(lines, code_start_idx)

                if closing_idx is not None:
                    # Extract code content
                    code_lines = lines[code_start_idx:closing_idx]
                    code = "\n".join(code_lines)

                    # Extract context
                    context_before = self._get_context_before(lines, i)
                    context_after = self._get_context_after(lines, closing_idx)

                    block = CodeBlock(
                        code=code,
                        language=language,
                        start_line=start_line + 1,  # Convert to 1-indexed
                        end_line=closing_idx + 1,  # 1-indexed
                        index=block_index,
                        context_before=context_before,
                        context_after=context_after,
                    )

                    # Apply language filter if specified
                    if language_filter is None or language == language_filter:
                        blocks.append(block)
                        block_index += 1

                    # Move past closing fence
                    i = closing_idx + 1
                else:
                    # Unclosed code block - skip the opening fence line
                    i += 1
            else:
                i += 1

        return blocks

    def extract_from_chapters(
        self,
        chapters: list[dict[str, Any]],
        language_filter: str | None = None,
    ) -> dict[str, list[CodeBlock]]:
        """Extract code blocks from multiple chapters.

        WBS 5.1.4: Batch processing for multiple chapters.

        Args:
            chapters: List of dicts with 'chapter_id' and 'content' keys
            language_filter: Optional language to filter by

        Returns:
            Dict mapping chapter_id to list of CodeBlock objects
        """
        results: dict[str, list[CodeBlock]] = {}

        for chapter in chapters:
            chapter_id = chapter.get("chapter_id", "unknown")
            content = chapter.get("content", "")
            blocks = self.extract_code_blocks(content, language_filter)
            results[chapter_id] = blocks

        return results

    def _normalize_language(self, language: str) -> str:
        """Normalize language identifier.

        WBS 5.1.2: Convert aliases to standard names.

        Args:
            language: Raw language string from fence marker

        Returns:
            Normalized language name
        """
        if not language:
            return _UNKNOWN_LANGUAGE

        lang_lower = language.lower().strip()

        # Check aliases first
        if lang_lower in _LANGUAGE_ALIASES:
            return _LANGUAGE_ALIASES[lang_lower]

        return lang_lower

    def _find_closing_fence(self, lines: list[str], start_idx: int) -> int | None:
        """Find the closing fence index.

        Args:
            lines: All lines in document
            start_idx: Index to start searching from (after opening fence)

        Returns:
            Index of closing fence line, or None if not found
        """
        for i in range(start_idx, len(lines)):
            if lines[i].strip() == "```":
                return i
        return None

    def _get_context_before(self, lines: list[str], fence_idx: int) -> str:
        """Get text context before a code block.

        WBS 5.1.3: Extract surrounding context.

        Args:
            lines: All lines in document
            fence_idx: Index of opening fence

        Returns:
            Text from lines before the fence
        """
        start = max(0, fence_idx - _CONTEXT_LINES)
        context_lines = lines[start:fence_idx]
        return "\n".join(context_lines).strip()

    def _get_context_after(self, lines: list[str], closing_fence_idx: int) -> str:
        """Get text context after a code block.

        WBS 5.1.3: Extract surrounding context.

        Args:
            lines: All lines in document
            closing_fence_idx: Index of closing fence

        Returns:
            Text from lines after the fence
        """
        total = len(lines)
        end = min(total, closing_fence_idx + 1 + _CONTEXT_LINES)
        context_lines = lines[closing_fence_idx + 1 : end]
        return "\n".join(context_lines).strip()
