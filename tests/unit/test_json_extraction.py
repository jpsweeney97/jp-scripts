"""Tests for JSON extraction with balanced brace matching."""

from __future__ import annotations

import json

from jpscripts.agent.parsing import _clean_json_payload, _extract_balanced_json


class TestExtractBalancedJson:
    """Test balanced brace JSON extraction."""

    def test_simple_json_object(self) -> None:
        """Extracts a simple JSON object."""
        text = '{"key": "value"}'
        assert _extract_balanced_json(text) == '{"key": "value"}'

    def test_nested_braces_in_string(self) -> None:
        """Handles braces inside string values correctly."""
        text = '{"message": "Use { and } in code"}'
        result = _extract_balanced_json(text)
        # Should extract the complete JSON, not stop at the } inside the string
        assert result == '{"message": "Use { and } in code"}'
        # Verify it's valid JSON
        parsed = json.loads(result)
        assert parsed["message"] == "Use { and } in code"

    def test_escaped_quotes_in_string(self) -> None:
        """Handles escaped quotes correctly."""
        text = r'{"message": "He said \"hello\""}'
        result = _extract_balanced_json(text)
        assert result == r'{"message": "He said \"hello\""}'

    def test_nested_objects(self) -> None:
        """Handles nested JSON objects."""
        text = '{"outer": {"inner": "value"}}'
        result = _extract_balanced_json(text)
        assert result == '{"outer": {"inner": "value"}}'

    def test_json_with_prose_before(self) -> None:
        """Extracts JSON from text with leading prose."""
        text = 'Here is the data: {"key": "value"}'
        result = _extract_balanced_json(text)
        assert result == '{"key": "value"}'

    def test_json_with_prose_after(self) -> None:
        """Extracts JSON from text with trailing prose."""
        text = '{"key": "value"} That was the data.'
        result = _extract_balanced_json(text)
        assert result == '{"key": "value"}'

    def test_multiple_json_objects_extracts_first(self) -> None:
        """Extracts only the first complete JSON object."""
        text = '{"first": 1} {"second": 2}'
        result = _extract_balanced_json(text)
        assert result == '{"first": 1}'

    def test_no_braces_returns_original(self) -> None:
        """Returns original text when no braces found."""
        text = "No JSON here"
        assert _extract_balanced_json(text) == "No JSON here"

    def test_unbalanced_braces_fallback(self) -> None:
        """Falls back to first-to-last for unbalanced braces."""
        text = '{"key": "missing close quote}'
        result = _extract_balanced_json(text)
        # Fallback: first { to last }
        assert result == '{"key": "missing close quote}'

    def test_empty_json_object(self) -> None:
        """Handles empty JSON object."""
        text = "{}"
        assert _extract_balanced_json(text) == "{}"

    def test_backslash_not_escape_sequence(self) -> None:
        """Handles backslash that is not an escape sequence."""
        text = r'{"path": "C:\\Users\\name"}'
        result = _extract_balanced_json(text)
        assert result == r'{"path": "C:\\Users\\name"}'


class TestCleanJsonPayload:
    """Test the full JSON payload cleaner."""

    def test_raw_json(self) -> None:
        """Handles raw JSON string."""
        text = '{"key": "value"}'
        assert _clean_json_payload(text) == '{"key": "value"}'

    def test_markdown_fence(self) -> None:
        """Extracts JSON from markdown code fence."""
        text = '```json\n{"key": "value"}\n```'
        assert _clean_json_payload(text) == '{"key": "value"}'

    def test_markdown_fence_case_insensitive(self) -> None:
        """Handles case-insensitive fence markers."""
        text = '```JSON\n{"key": "value"}\n```'
        assert _clean_json_payload(text) == '{"key": "value"}'

    def test_prose_with_embedded_json(self) -> None:
        """Extracts JSON from prose."""
        text = 'Here is the result: {"key": "value"} Thanks!'
        result = _clean_json_payload(text)
        assert result == '{"key": "value"}'

    def test_empty_input(self) -> None:
        """Handles empty input."""
        assert _clean_json_payload("") == ""
        assert _clean_json_payload("   ") == ""

    def test_whitespace_stripped(self) -> None:
        """Strips leading/trailing whitespace."""
        text = '  {"key": "value"}  '
        result = _clean_json_payload(text)
        assert result == '{"key": "value"}'

    def test_complex_nested_with_prose(self) -> None:
        """Handles complex nested JSON with surrounding prose."""
        json_obj = {
            "thought_process": "I need to analyze {this} carefully",
            "nested": {"inner": "value with } brace"},
            "final_message": "Done",
        }
        json_str = json.dumps(json_obj)
        text = f"Let me think about this...\n\n{json_str}\n\nThat's my answer."
        result = _clean_json_payload(text)
        # Should be valid JSON that parses correctly
        parsed = json.loads(result)
        assert parsed["thought_process"] == "I need to analyze {this} carefully"
        assert parsed["nested"]["inner"] == "value with } brace"

    def test_fence_preferred_over_brace_extraction(self) -> None:
        """Markdown fence is preferred over brace extraction."""
        text = 'Before { brace ```json\n{"inside": "fence"}\n``` after } brace'
        result = _clean_json_payload(text)
        assert result == '{"inside": "fence"}'
