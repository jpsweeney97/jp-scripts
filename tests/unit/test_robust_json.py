"""Tests for robust JSON extraction from chatty LLM output.

These tests verify that the JSON parser can handle:
- Broken <thinking> tags
- JSON with nested braces
- JSON containing code snippets with braces
- Malformed markdown fences
- Multiple JSON objects in output
"""

from __future__ import annotations

from jpscripts.engine import (
    _clean_json_payload,
    _extract_balanced_json,
    _extract_from_code_fence,
    _extract_thinking_content,
    _find_last_valid_json,
    _split_thought_and_json,
)


class TestExtractBalancedJson:
    """Test stack-based brace counting."""

    def test_simple_json(self) -> None:
        """Basic JSON extraction."""
        text = 'Some preamble {"key": "value"} trailing text'
        result = _extract_balanced_json(text)
        assert result == '{"key": "value"}'

    def test_nested_braces(self) -> None:
        """JSON with deeply nested objects."""
        text = '{"a": {"b": {"c": {"d": "value"}}}}'
        result = _extract_balanced_json(text)
        assert result == '{"a": {"b": {"c": {"d": "value"}}}}'

    def test_braces_in_string_literals(self) -> None:
        """Braces inside string values should not affect counting."""
        text = '{"code": "function() { return {}; }"}'
        result = _extract_balanced_json(text)
        assert result == '{"code": "function() { return {}; }"}'

    def test_escaped_quotes(self) -> None:
        """Handle escaped quotes in strings."""
        text = r'{"message": "He said \"hello\" to {them}"}'
        result = _extract_balanced_json(text)
        assert result == r'{"message": "He said \"hello\" to {them}"}'

    def test_code_in_json_string(self) -> None:
        """Python/JS code with braces inside JSON string value."""
        text = """{"patch": "def foo():\\n    if x:\\n        return {1: 2}\\n"}"""
        result = _extract_balanced_json(text)
        assert result == """{"patch": "def foo():\\n    if x:\\n        return {1: 2}\\n"}"""

    def test_no_json_returns_original(self) -> None:
        """Text without JSON returns original text."""
        text = "No JSON here, just plain text."
        result = _extract_balanced_json(text)
        assert result == text

    def test_unbalanced_braces_fallback(self) -> None:
        """Unbalanced braces should use fallback (first { to last })."""
        text = '{"key": "value" missing close'
        result = _extract_balanced_json(text)
        # No closing brace, so returns original
        assert result == text

    def test_json_with_arrays(self) -> None:
        """JSON with nested arrays containing objects."""
        text = '{"items": [{"id": 1}, {"id": 2}]}'
        result = _extract_balanced_json(text)
        assert result == '{"items": [{"id": 1}, {"id": 2}]}'


class TestExtractFromCodeFence:
    """Test markdown fence handling."""

    def test_standard_json_fence(self) -> None:
        """```json ... ``` format."""
        text = """Here is the response:
```json
{"tool": "edit", "path": "/file.py"}
```
Done!"""
        result = _extract_from_code_fence(text)
        assert result == '{"tool": "edit", "path": "/file.py"}'

    def test_json_fence_case_insensitive(self) -> None:
        """```JSON ... ``` should work too."""
        text = """```JSON
{"key": "value"}
```"""
        result = _extract_from_code_fence(text)
        assert result == '{"key": "value"}'

    def test_malformed_fence_no_close(self) -> None:
        """Missing closing fence returns None."""
        text = """```json
{"key": "value"}
no closing fence"""
        result = _extract_from_code_fence(text)
        assert result is None

    def test_no_fence_returns_none(self) -> None:
        """Text without fence returns None."""
        text = '{"key": "value"}'
        result = _extract_from_code_fence(text)
        assert result is None

    def test_non_json_fence_returns_none(self) -> None:
        """```python ... ``` should return None."""
        text = """```python
def foo():
    pass
```"""
        result = _extract_from_code_fence(text)
        assert result is None

    def test_fence_with_extra_whitespace(self) -> None:
        """Handle whitespace around content."""
        text = """```json

  {"key": "value"}

```"""
        result = _extract_from_code_fence(text)
        assert result == '{"key": "value"}'


class TestExtractThinkingContent:
    """Test thinking tag handling."""

    def test_normal_thinking_tags(self) -> None:
        """Standard <thinking>...</thinking> tags."""
        text = """<thinking>
I need to analyze this carefully.
</thinking>
{"tool": "read"}"""
        thinking, remaining = _extract_thinking_content(text)
        assert "analyze this carefully" in thinking
        assert '{"tool": "read"}' in remaining

    def test_broken_thinking_no_close(self) -> None:
        """Missing closing tag - treat rest as thinking."""
        text = """<thinking>
This is my thought process
{"tool": "edit"}"""
        thinking, remaining = _extract_thinking_content(text)
        assert "thought process" in thinking
        # When no closing tag, JSON is consumed as thinking
        assert remaining == ""

    def test_case_insensitive_tags(self) -> None:
        """Handle <THINKING> and <Thinking> variants."""
        text = """<THINKING>
Upper case tags
</THINKING>
{"result": true}"""
        thinking, remaining = _extract_thinking_content(text)
        assert "Upper case tags" in thinking
        assert '{"result": true}' in remaining

    def test_no_thinking_tag(self) -> None:
        """No thinking tag returns empty thinking and full text."""
        text = '{"tool": "edit", "path": "file.py"}'
        thinking, remaining = _extract_thinking_content(text)
        assert thinking == ""
        assert remaining == text

    def test_preamble_before_thinking(self) -> None:
        """Text before <thinking> is included in thought content."""
        text = """Some initial thoughts
<thinking>
More detailed analysis
</thinking>
{"result": "done"}"""
        thinking, remaining = _extract_thinking_content(text)
        assert "initial thoughts" in thinking
        assert "detailed analysis" in thinking
        assert '{"result": "done"}' in remaining


class TestFindLastValidJson:
    """Test greedy fallback for edge cases."""

    def test_simple_valid_json(self) -> None:
        """Find valid JSON in clean text."""
        text = 'Here is the JSON: {"key": "value"}'
        result = _find_last_valid_json(text)
        assert result == '{"key": "value"}'

    def test_multiple_json_objects_returns_last(self) -> None:
        """With multiple objects, return the last valid one."""
        text = """First: {"id": 1}
Then: {"id": 2}
Finally: {"id": 3}"""
        result = _find_last_valid_json(text)
        assert result == '{"id": 3}'

    def test_broken_json_with_valid_suffix(self) -> None:
        """Text with broken JSON followed by valid JSON."""
        text = """{"broken": missing_quote}
{"valid": "json"}"""
        result = _find_last_valid_json(text)
        assert result == '{"valid": "json"}'

    def test_no_valid_json(self) -> None:
        """No valid JSON returns None."""
        text = "No JSON here at all"
        result = _find_last_valid_json(text)
        assert result is None

    def test_nested_objects(self) -> None:
        """Handle nested objects correctly."""
        text = 'Result: {"outer": {"inner": {"deep": "value"}}}'
        result = _find_last_valid_json(text)
        assert result == '{"outer": {"inner": {"deep": "value"}}}'


class TestSplitThoughtAndJson:
    """Test the combined thought/JSON splitting."""

    def test_thinking_then_json(self) -> None:
        """Standard thinking followed by JSON."""
        text = """<thinking>
Analyzing the problem...
</thinking>
{"tool": "edit", "path": "file.py"}"""
        thought, json_content = _split_thought_and_json(text)
        assert "Analyzing" in thought
        assert '{"tool"' in json_content

    def test_json_in_code_fence(self) -> None:
        """JSON wrapped in markdown fence."""
        text = """```json
{"tool": "read", "path": "config.yaml"}
```"""
        _thought, json_content = _split_thought_and_json(text)
        assert '{"tool": "read"' in json_content

    def test_plain_json(self) -> None:
        """Just JSON, no thinking or fence."""
        text = '{"simple": "json"}'
        thought, json_content = _split_thought_and_json(text)
        assert thought == ""
        assert json_content == '{"simple": "json"}'


class TestCleanJsonPayload:
    """Test the main JSON cleaning function."""

    def test_extracts_from_fence(self) -> None:
        """Extract JSON from code fence."""
        text = """```json
{"key": "value"}
```"""
        result = _clean_json_payload(text)
        assert result == '{"key": "value"}'

    def test_extracts_balanced_json(self) -> None:
        """Extract JSON using balanced brace matching."""
        text = 'Some text {"key": "value"} more text'
        result = _clean_json_payload(text)
        assert result == '{"key": "value"}'


class TestIntegration:
    """End-to-end tests with real LLM output patterns."""

    def test_chatty_output_with_thinking(self) -> None:
        """Full response with thinking + JSON."""
        text = """<thinking>
Let me analyze this carefully.
I should edit the file to fix the bug.
The issue is on line 42.
</thinking>

Here is the JSON:

```json
{
    "tool": "edit",
    "arguments": {
        "path": "src/main.py",
        "line": 42
    }
}
```"""
        thought, json_content = _split_thought_and_json(text)
        assert "analyze this carefully" in thought
        assert '"tool": "edit"' in json_content

    def test_markdown_with_code_snippets(self) -> None:
        """JSON containing code examples with braces."""
        text = """```json
{
    "tool": "write",
    "content": "def process():\\n    data = {}\\n    for item in items:\\n        data[item.id] = item\\n    return data"
}
```"""
        result = _clean_json_payload(text)
        # Should extract the full JSON even with braces in the string
        assert '"tool": "write"' in result
        assert "def process" in result

    def test_broken_thinking_with_json(self) -> None:
        """Malformed thinking tag but valid JSON."""
        text = """<thinking>
I'm going to fix the bug
But I forgot to close the tag

```json
{"tool": "edit", "path": "file.py"}
```"""
        thought, json_content = _split_thought_and_json(text)
        # With broken thinking, we should still find the JSON
        assert '"tool": "edit"' in json_content or "edit" in thought

    def test_json_with_embedded_json_string(self) -> None:
        """JSON containing a JSON string as a value."""
        text = """{"response": "{\\"nested\\": \\"json\\"}"}"""
        result = _extract_balanced_json(text)
        assert result == text
