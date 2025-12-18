"""
EEP-5.1 Tests: Code Block Extraction from Chapters

TDD RED Phase: Write failing tests first.

Tests for extracting code blocks from chapter markdown content.

WBS Mapping:
- 5.1.1: Python code block detection (```python markers)
- 5.1.2: Language identification from fence markers
- 5.1.3: Code metadata extraction (line numbers, context)
- 5.1.4: Multi-block extraction from single chapter

Anti-Patterns Avoided:
- #12: FakeExtractor for testing (no real I/O)
- S1192: Constants for repeated strings
- S3776: Small, focused test methods
"""

from __future__ import annotations

import pytest

# =============================================================================
# Constants (Anti-Pattern S1192 Prevention)
# =============================================================================

_SAMPLE_PYTHON_BLOCK = '''```python
def hello():
    return "Hello, World!"
```'''

_SAMPLE_JAVASCRIPT_BLOCK = '''```javascript
function hello() {
    return "Hello, World!";
}
```'''

_SAMPLE_MULTI_BLOCK_MARKDOWN = '''# Chapter 1: Code Examples

Here is some Python code:

```python
def chunk_text(text: str, chunk_size: int = 100) -> list[str]:
    """Split text into chunks."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
```

And here is some JavaScript:

```javascript
function chunkText(text, chunkSize = 100) {
    const chunks = [];
    for (let i = 0; i < text.length; i += chunkSize) {
        chunks.push(text.slice(i, i + chunkSize));
    }
    return chunks;
}
```

More text after the code blocks.
'''

_SAMPLE_NO_CODE_MARKDOWN = '''# Chapter Without Code

This chapter contains only prose and no code examples.

## Section 1

Some explanatory text here.
'''

_SAMPLE_NESTED_BACKTICKS = '''# Edge Case

Here is a code block with nested triple backticks:

```python
markdown = """
Some markdown with \\```python
code fence
\\```
"""
```
'''

_SAMPLE_CODE_WITH_CONTEXT = '''# Error Handling Patterns

The following shows proper exception handling:

```python
class CustomError(Exception):
    """Custom exception for domain errors."""
    pass

def validate(data: dict) -> bool:
    if not data:
        raise CustomError("Empty data")
    return True
```

Always use custom exceptions for domain-specific errors.
'''


# =============================================================================
# EEP-5.1.1: Python Code Block Detection Tests
# =============================================================================


class TestPythonCodeBlockDetection:
    """Tests for detecting Python code blocks in markdown."""

    def test_code_extractor_class_exists(self) -> None:
        """CodeExtractor class exists and is importable."""
        from src.extractors.code_extractor import CodeExtractor

        assert CodeExtractor is not None

    def test_extract_code_blocks_method_exists(self) -> None:
        """extract_code_blocks() method exists."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        assert hasattr(extractor, "extract_code_blocks")
        assert callable(extractor.extract_code_blocks)

    def test_extract_single_python_block(self) -> None:
        """Extracts a single Python code block.

        AC-5.1.1: Detect Python code blocks using ```python markers.
        """
        from src.extractors.code_extractor import CodeExtractor, CodeBlock

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(_SAMPLE_PYTHON_BLOCK)

        assert len(blocks) == 1
        assert isinstance(blocks[0], CodeBlock)
        assert blocks[0].language == "python"
        assert "def hello():" in blocks[0].code

    def test_extract_returns_empty_list_for_no_code(self) -> None:
        """Returns empty list when no code blocks found."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(_SAMPLE_NO_CODE_MARKDOWN)

        assert blocks == []

    def test_extracts_only_python_when_filtered(self) -> None:
        """Filters to Python-only blocks when language specified."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(
            _SAMPLE_MULTI_BLOCK_MARKDOWN,
            language_filter="python"
        )

        assert len(blocks) == 1
        assert blocks[0].language == "python"


# =============================================================================
# EEP-5.1.2: Language Identification Tests
# =============================================================================


class TestLanguageIdentification:
    """Tests for identifying code language from fence markers."""

    def test_identifies_python_language(self) -> None:
        """Correctly identifies Python from ```python marker.

        AC-5.1.2: Identify language from fence markers.
        """
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(_SAMPLE_PYTHON_BLOCK)

        assert blocks[0].language == "python"

    def test_identifies_javascript_language(self) -> None:
        """Correctly identifies JavaScript from ```javascript marker."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(_SAMPLE_JAVASCRIPT_BLOCK)

        assert blocks[0].language == "javascript"

    def test_identifies_multiple_languages(self) -> None:
        """Identifies multiple different languages in same document."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(_SAMPLE_MULTI_BLOCK_MARKDOWN)

        languages = {b.language for b in blocks}
        assert "python" in languages
        assert "javascript" in languages

    def test_handles_unknown_language(self) -> None:
        """Handles code blocks without language specification."""
        from src.extractors.code_extractor import CodeExtractor

        markdown = '''```
some code without language
```'''
        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(markdown)

        assert len(blocks) == 1
        assert blocks[0].language == "unknown"

    def test_normalizes_language_aliases(self) -> None:
        """Normalizes language aliases (py → python, js → javascript)."""
        from src.extractors.code_extractor import CodeExtractor

        markdown = '''```py
x = 1
```'''
        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(markdown)

        assert blocks[0].language == "python"


# =============================================================================
# EEP-5.1.3: Code Metadata Extraction Tests
# =============================================================================


class TestCodeMetadataExtraction:
    """Tests for extracting metadata from code blocks."""

    def test_code_block_has_line_number(self) -> None:
        """CodeBlock includes starting line number.

        AC-5.1.3: Extract metadata including line numbers.
        """
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(_SAMPLE_CODE_WITH_CONTEXT)

        assert hasattr(blocks[0], "start_line")
        assert blocks[0].start_line > 0

    def test_code_block_has_end_line(self) -> None:
        """CodeBlock includes ending line number."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(_SAMPLE_CODE_WITH_CONTEXT)

        assert hasattr(blocks[0], "end_line")
        assert blocks[0].end_line > blocks[0].start_line

    def test_code_block_has_context_before(self) -> None:
        """CodeBlock captures context before the code block."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(_SAMPLE_CODE_WITH_CONTEXT)

        assert hasattr(blocks[0], "context_before")
        assert "exception handling" in blocks[0].context_before.lower()

    def test_code_block_has_context_after(self) -> None:
        """CodeBlock captures context after the code block."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(_SAMPLE_CODE_WITH_CONTEXT)

        assert hasattr(blocks[0], "context_after")
        # Should capture the text that comes after

    def test_code_block_has_raw_code(self) -> None:
        """CodeBlock stores the raw code content without fence markers."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(_SAMPLE_PYTHON_BLOCK)

        assert "```" not in blocks[0].code
        assert "def hello():" in blocks[0].code


# =============================================================================
# EEP-5.1.4: Multi-Block Extraction Tests
# =============================================================================


class TestMultiBlockExtraction:
    """Tests for extracting multiple code blocks from a chapter."""

    def test_extracts_multiple_blocks(self) -> None:
        """Extracts all code blocks from multi-block document.

        AC-5.1.4: Multi-block extraction from single chapter.
        """
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(_SAMPLE_MULTI_BLOCK_MARKDOWN)

        assert len(blocks) == 2

    def test_blocks_ordered_by_appearance(self) -> None:
        """Extracted blocks are ordered by their appearance."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(_SAMPLE_MULTI_BLOCK_MARKDOWN)

        # Python block appears before JavaScript in the sample
        assert blocks[0].language == "python"
        assert blocks[1].language == "javascript"

    def test_each_block_has_unique_index(self) -> None:
        """Each block has a unique index for identification."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(_SAMPLE_MULTI_BLOCK_MARKDOWN)

        indices = [b.index for b in blocks]
        assert len(indices) == len(set(indices))  # All unique

    def test_handles_nested_backticks(self) -> None:
        """Correctly handles nested or escaped backticks."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(_SAMPLE_NESTED_BACKTICKS)

        # Should extract exactly one block
        assert len(blocks) == 1


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestCodeExtractionEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input_returns_empty_list(self) -> None:
        """Returns empty list for empty input."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks("")

        assert blocks == []

    def test_handles_unclosed_code_block(self) -> None:
        """Handles unclosed code blocks gracefully."""
        from src.extractors.code_extractor import CodeExtractor

        markdown = '''```python
def broken():
    pass
'''  # No closing ```
        extractor = CodeExtractor()
        # Should not raise, may return partial or skip
        blocks = extractor.extract_code_blocks(markdown)
        assert isinstance(blocks, list)

    def test_handles_whitespace_only_input(self) -> None:
        """Handles whitespace-only input."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks("   \n\n   ")

        assert blocks == []

    def test_extract_code_preserves_indentation(self) -> None:
        """Preserves original code indentation."""
        from src.extractors.code_extractor import CodeExtractor

        markdown = '''```python
def example():
    if True:
        return 42
```'''
        extractor = CodeExtractor()
        blocks = extractor.extract_code_blocks(markdown)

        # Check indentation preserved
        assert "    if True:" in blocks[0].code
        assert "        return 42" in blocks[0].code


# =============================================================================
# Pydantic Models Tests
# =============================================================================


class TestCodeBlockModel:
    """Tests for CodeBlock Pydantic model."""

    def test_code_block_model_exists(self) -> None:
        """CodeBlock Pydantic model exists."""
        from src.extractors.code_extractor import CodeBlock

        assert CodeBlock is not None

    def test_code_block_is_pydantic_model(self) -> None:
        """CodeBlock is a Pydantic BaseModel."""
        from pydantic import BaseModel
        from src.extractors.code_extractor import CodeBlock

        assert issubclass(CodeBlock, BaseModel)

    def test_code_block_serializes_to_dict(self) -> None:
        """CodeBlock can serialize to dictionary."""
        from src.extractors.code_extractor import CodeBlock

        block = CodeBlock(
            code="def test(): pass",
            language="python",
            start_line=1,
            end_line=1,
            index=0,
            context_before="",
            context_after="",
        )

        data = block.model_dump()
        assert data["code"] == "def test(): pass"
        assert data["language"] == "python"


# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestBatchCodeExtraction:
    """Tests for extracting code from multiple chapters."""

    def test_extract_from_chapters_method_exists(self) -> None:
        """extract_from_chapters() method exists for batch processing."""
        from src.extractors.code_extractor import CodeExtractor

        extractor = CodeExtractor()
        assert hasattr(extractor, "extract_from_chapters")
        assert callable(extractor.extract_from_chapters)

    def test_batch_extraction_multiple_chapters(self) -> None:
        """Extracts code blocks from multiple chapters at once."""
        from src.extractors.code_extractor import CodeExtractor

        chapters = [
            {"chapter_id": "ch1", "content": _SAMPLE_PYTHON_BLOCK},
            {"chapter_id": "ch2", "content": _SAMPLE_JAVASCRIPT_BLOCK},
        ]

        extractor = CodeExtractor()
        results = extractor.extract_from_chapters(chapters)

        assert "ch1" in results
        assert "ch2" in results
        assert len(results["ch1"]) == 1
        assert len(results["ch2"]) == 1

    def test_batch_extraction_with_empty_chapter(self) -> None:
        """Handles chapters with no code blocks."""
        from src.extractors.code_extractor import CodeExtractor

        chapters = [
            {"chapter_id": "ch1", "content": _SAMPLE_NO_CODE_MARKDOWN},
            {"chapter_id": "ch2", "content": _SAMPLE_PYTHON_BLOCK},
        ]

        extractor = CodeExtractor()
        results = extractor.extract_from_chapters(chapters)

        assert results["ch1"] == []
        assert len(results["ch2"]) == 1

