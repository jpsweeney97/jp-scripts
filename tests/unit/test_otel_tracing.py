"""
Unit tests for OpenTelemetry integration in agent tracing.

Tests verify span emission without requiring actual OTel infrastructure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from jpscripts.agent.middleware import StepContext, TracingMiddleware
from jpscripts.agent.tracing import TraceRecorder
from jpscripts.core.cost_tracker import TokenUsage


class MockSpan:
    """Mock span that records attributes and events."""

    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any]) -> None:
        self.events.append((name, attributes))

    def __enter__(self) -> MockSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class MockTracer:
    """Mock tracer that creates MockSpans."""

    def __init__(self) -> None:
        self.spans: list[MockSpan] = []

    def start_as_current_span(self, name: str) -> MockSpan:
        span = MockSpan()
        self.spans.append(span)
        return span


class TestTracingMiddlewareOTel:
    """Test TracingMiddleware OpenTelemetry span emission."""

    @pytest.fixture
    def middleware(self, tmp_path: Path) -> TracingMiddleware[MagicMock]:
        recorder = TraceRecorder(trace_dir=tmp_path, trace_id="test-trace")
        return TracingMiddleware(
            trace_recorder=recorder,
            persona="test-agent",
        )

    @pytest.fixture
    def mock_tracer(self) -> MockTracer:
        return MockTracer()

    def test_emit_span_sets_persona_attribute(
        self,
        middleware: TracingMiddleware[MagicMock],
        mock_tracer: MockTracer,
    ) -> None:
        """Span should have agent.persona attribute."""
        ctx: StepContext[MagicMock] = StepContext(history=[])

        with patch(
            "jpscripts.agent.middleware._get_tracer",
            return_value=mock_tracer,
        ):
            middleware._emit_otel_span(ctx)

        assert len(mock_tracer.spans) == 1
        span = mock_tracer.spans[0]
        assert span.attributes.get("agent.persona") == "test-agent"

    def test_emit_span_sets_files_touched(
        self,
        middleware: TracingMiddleware[MagicMock],
        mock_tracer: MockTracer,
    ) -> None:
        """Span should record files_touched as attribute."""
        ctx: StepContext[MagicMock] = StepContext(
            history=[],
            files_touched={Path("file1.py"), Path("file2.py")},
        )

        with patch(
            "jpscripts.agent.middleware._get_tracer",
            return_value=mock_tracer,
        ):
            middleware._emit_otel_span(ctx)

        span = mock_tracer.spans[0]
        files = span.attributes.get("code.files_touched")
        assert files is not None
        assert "file1.py" in files
        assert "file2.py" in files

    def test_emit_span_sets_usage_attributes(
        self,
        middleware: TracingMiddleware[MagicMock],
        mock_tracer: MockTracer,
    ) -> None:
        """Span should record token usage attributes."""
        ctx: StepContext[MagicMock] = StepContext(
            history=[],
            usage_snapshot=TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
            ),
        )

        with patch(
            "jpscripts.agent.middleware._get_tracer",
            return_value=mock_tracer,
        ):
            middleware._emit_otel_span(ctx)

        span = mock_tracer.spans[0]
        assert span.attributes.get("usage.prompt_tokens") == 100
        assert span.attributes.get("usage.completion_tokens") == 50
        assert span.attributes.get("usage.total_tokens") == 150

    def test_emit_span_adds_tool_output_event(
        self,
        middleware: TracingMiddleware[MagicMock],
        mock_tracer: MockTracer,
    ) -> None:
        """Span should add tool_output event when present."""
        ctx: StepContext[MagicMock] = StepContext(
            history=[],
            metadata={"tool_output": "Command executed successfully"},
        )

        with patch(
            "jpscripts.agent.middleware._get_tracer",
            return_value=mock_tracer,
        ):
            middleware._emit_otel_span(ctx)

        span = mock_tracer.spans[0]
        assert len(span.events) == 1
        event_name, event_attrs = span.events[0]
        assert event_name == "tool_output"
        assert "Command executed successfully" in event_attrs.get("output", "")

    def test_emit_span_noop_when_no_tracer(
        self,
        middleware: TracingMiddleware[MagicMock],
    ) -> None:
        """No-op when tracer is not available."""
        ctx: StepContext[MagicMock] = StepContext(history=[])

        with patch(
            "jpscripts.agent.middleware._get_tracer",
            return_value=None,
        ):
            # Should not raise
            middleware._emit_otel_span(ctx)

    def test_emit_span_with_tool_call_event(
        self,
        middleware: TracingMiddleware[MagicMock],
        mock_tracer: MockTracer,
    ) -> None:
        """Span should add tool_call event when response has tool_call."""
        # Create a mock response with tool_call
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.model_dump.return_value = {"name": "write_file", "args": {}}
        mock_response.tool_call = mock_tool_call

        ctx: StepContext[MagicMock] = StepContext(
            history=[],
            response=mock_response,
        )

        with patch(
            "jpscripts.agent.middleware._get_tracer",
            return_value=mock_tracer,
        ):
            middleware._emit_otel_span(ctx)

        span = mock_tracer.spans[0]
        tool_call_events = [e for e in span.events if e[0] == "tool_call"]
        assert len(tool_call_events) == 1


class TestGetTracer:
    """Test _get_tracer function."""

    def test_returns_none_when_otel_not_installed(self) -> None:
        """_get_tracer returns None when OpenTelemetry is not installed."""
        # Reset global state
        import jpscripts.agent.tracing as tracing
        from jpscripts.agent.tracing import _get_tracer

        tracing._otel_tracer = None
        tracing._otel_provider_configured = False

        with patch.object(tracing, "_load_otel_deps", return_value=None):
            result = _get_tracer()

        assert result is None

    def test_returns_cached_tracer(self) -> None:
        """_get_tracer returns cached tracer if already initialized."""
        import jpscripts.agent.tracing as tracing
        from jpscripts.agent.tracing import _get_tracer

        mock_tracer = MagicMock()
        tracing._otel_tracer = mock_tracer

        try:
            result = _get_tracer()
            assert result is mock_tracer
        finally:
            tracing._otel_tracer = None
