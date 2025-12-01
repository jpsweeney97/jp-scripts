"""Agent response parsing and JSON extraction.

This module provides functions for parsing agent responses:
- JSON extraction from various formats (balanced braces, code fences)
- Thinking content extraction
- Response validation and parsing
"""

from __future__ import annotations

from .models import AgentResponse


def _extract_balanced_json(text: str) -> str:
    """Extract first complete JSON object using balanced brace matching.

    Properly handles:
    - Nested braces in string values
    - Escape sequences
    - Unmatched braces (falls back to first { to last })
    """
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Fallback: unbalanced braces, return from start to last }
    end = text.rfind("}")
    if end > start:
        return text[start : end + 1]
    return text


def _extract_from_code_fence(text: str) -> str | None:
    """Extract JSON from markdown code fence without regex.

    Handles ```json ... ``` and ``` json ... ``` formats.
    Returns None if no valid JSON fence is found.
    """
    text_lower = text.lower()

    # Find opening fence: ```json or ``` followed by json
    fence_start = text_lower.find("```json")
    if fence_start == -1:
        fence_start = text_lower.find("```")
        if fence_start == -1:
            return None
        # Check if "json" follows on same line
        line_end = text.find("\n", fence_start)
        prefix = text_lower[fence_start:line_end] if line_end != -1 else text_lower[fence_start:]
        if "json" not in prefix:
            return None

    # Find content start (after the opening fence line)
    content_start = text.find("\n", fence_start)
    if content_start == -1:
        return None
    content_start += 1

    # Find closing fence
    fence_end = text.find("```", content_start)
    if fence_end == -1:
        return None

    return text[content_start:fence_end].strip()


def _extract_thinking_content(text: str) -> tuple[str, str]:
    """Extract thinking content and remaining text without regex.

    Returns:
        Tuple of (thinking_content, remaining_text)
        If no thinking tag found, returns ("", original_text)
    """
    text_lower = text.lower()

    # Find opening tag (case-insensitive)
    open_tag = "<thinking>"
    close_tag = "</thinking>"

    open_idx = text_lower.find(open_tag)
    if open_idx == -1:
        return "", text

    # Find closing tag
    close_idx = text_lower.find(close_tag, open_idx + len(open_tag))
    if close_idx == -1:
        # Malformed: opening tag without closing - treat rest as thinking
        thinking = text[open_idx + len(open_tag) :].strip()
        preamble = text[:open_idx].strip()
        return f"{preamble}\n{thinking}".strip() if preamble else thinking, ""

    # Extract parts
    preamble = text[:open_idx].strip()
    thinking = text[open_idx + len(open_tag) : close_idx].strip()
    remaining = text[close_idx + len(close_tag) :].strip()

    thought_parts = [p for p in (preamble, thinking) if p]
    return "\n\n".join(thought_parts), remaining


def _find_last_valid_json(text: str) -> str | None:
    """Find the last valid JSON object by searching backwards.

    This is a fallback for when balanced brace extraction fails.
    Attempts to parse candidate substrings to verify they are valid JSON.
    """
    import json as json_module

    # Find all potential JSON end positions (closing braces)
    end_positions = [i for i, c in enumerate(text) if c == "}"]

    for end_pos in reversed(end_positions):
        # Try to find matching opening brace using forward scan
        depth = 0
        in_string = False
        escape_next = False
        start_pos = -1

        # Scan forward from beginning to find the opening brace that matches this end
        for i, char in enumerate(text[: end_pos + 1]):
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                if depth == 0:
                    start_pos = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and i == end_pos:
                    # Found matching braces
                    candidate = text[start_pos : end_pos + 1]
                    try:
                        json_module.loads(candidate)
                        return candidate
                    except json_module.JSONDecodeError:
                        break  # Try earlier end position

    return None


def _clean_json_payload(text: str) -> str:
    """Extract JSON content from raw agent output, tolerating code fences and stray prose."""
    stripped = text.strip()
    if not stripped:
        return stripped

    # Try markdown fence first (without regex)
    fence_content = _extract_from_code_fence(stripped)
    if fence_content:
        return fence_content

    # Use balanced brace extraction for proper handling
    extracted = _extract_balanced_json(stripped)

    # If balanced extraction returns the same text (no JSON found),
    # try the fallback to find last valid JSON
    if extracted == stripped:
        fallback = _find_last_valid_json(stripped)
        if fallback:
            return fallback

    return extracted


def _split_thought_and_json(payload: str) -> tuple[str, str]:
    """Separate thinking content from JSON payload for strict validation."""
    stripped = payload.strip()
    if not stripped:
        return "", ""

    # Use state-machine based thinking extraction (no regex)
    thinking_content, remaining = _extract_thinking_content(stripped)
    if thinking_content:
        json_candidate = _clean_json_payload(remaining or stripped)
        return thinking_content, json_candidate

    json_content = _clean_json_payload(stripped)
    if not json_content:
        return stripped, ""

    json_start = stripped.find(json_content)
    preamble = stripped[:json_start].strip() if json_start != -1 else ""
    return preamble, json_content


def parse_agent_response(payload: str) -> AgentResponse:
    """Parse and validate a JSON agent response."""
    thought_content, json_content = _split_thought_and_json(payload)
    response = AgentResponse.model_validate_json(json_content)
    if thought_content:
        response.thought_process = thought_content
    return response


__all__ = [
    "_clean_json_payload",
    "_extract_balanced_json",
    "_extract_from_code_fence",
    "_extract_thinking_content",
    "_find_last_valid_json",
    "_split_thought_and_json",
    "parse_agent_response",
]
