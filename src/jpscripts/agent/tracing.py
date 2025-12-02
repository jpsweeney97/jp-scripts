"""Trace recording and OpenTelemetry integration.

This module provides:
- TraceRecorder: Persists agent execution traces to JSONL files
- OpenTelemetry integration for distributed tracing
"""

from __future__ import annotations

import asyncio
import gzip
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from jpscripts.core.console import get_logger

from .models import (
    AgentTraceStep,
    BatchSpanProcessorProtocol,
    OTLPSpanExporterProtocol,
    ResourceProtocol,
    TraceModuleProtocol,
    TracerProtocol,
    TracerProviderProtocol,
)

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# OpenTelemetry Module State
# -----------------------------------------------------------------------------

_otel_trace_module: TraceModuleProtocol | None = None
_otel_resource_cls: type[ResourceProtocol] | None = None
_otel_tracer_provider_cls: type[TracerProviderProtocol] | None = None
_otel_span_processor_cls: type[BatchSpanProcessorProtocol] | None = None
_otel_exporter_cls: type[OTLPSpanExporterProtocol] | None = None
_otel_tracer: TracerProtocol | None = None
_otel_provider_configured = False


# -----------------------------------------------------------------------------
# TraceRecorder Class
# -----------------------------------------------------------------------------


class TraceRecorder:
    MAX_TRACE_SIZE = 10 * 1024 * 1024  # 10MB
    ARCHIVE_MAX_AGE_DAYS = 30

    def __init__(self, trace_dir: Path, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self.trace_dir = self._ensure_trace_dir(trace_dir)
        self._path = self.trace_dir / f"{self.trace_id}.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_trace_dir(self, trace_dir: Path) -> Path:
        primary = trace_dir.expanduser()
        try:
            primary.mkdir(parents=True, exist_ok=True)
            return primary
        except PermissionError:
            fallback = (Path.cwd() / ".jpscripts" / "traces").resolve()
            try:
                fallback.mkdir(parents=True, exist_ok=True)
                logger.warning(
                    "TraceRecorder falling back to %s due to permission issues", fallback
                )
                return fallback
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Failed to create fallback trace dir %s: %s", fallback, exc)
                raise

    async def append(self, step: AgentTraceStep) -> None:
        payload = step.model_dump_json()
        await asyncio.to_thread(self._write_line, payload)

    def _write_line(self, line: str) -> None:
        self._rotate_if_needed()
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _rotate_if_needed(self) -> None:
        """Compress trace file if it exceeds size limit."""
        if not self._path.exists():
            return
        try:
            if self._path.stat().st_size < self.MAX_TRACE_SIZE:
                return
        except OSError:
            return

        # Compress to .jsonl.gz with timestamp
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        archive_path = self.trace_dir / f"{self.trace_id}.{timestamp}.jsonl.gz"

        try:
            with self._path.open("rb") as f_in, gzip.open(archive_path, "wb") as f_out:
                f_out.writelines(f_in)

            # Truncate original file
            self._path.write_text("")

            # Clean up old archives
            self._cleanup_old_archives()
        except OSError as exc:
            logger.debug("Failed to rotate trace file: %s", exc)

    def _cleanup_old_archives(self) -> None:
        """Delete .jsonl.gz archives older than 30 days."""
        cutoff = datetime.now(UTC) - timedelta(days=self.ARCHIVE_MAX_AGE_DAYS)
        for archive in self.trace_dir.glob("*.jsonl.gz"):
            try:
                mtime = datetime.fromtimestamp(archive.stat().st_mtime, tz=UTC)
                if mtime < cutoff:
                    archive.unlink()
                    logger.debug("Deleted old trace archive: %s", archive)
            except OSError:
                pass  # Ignore cleanup failures


# -----------------------------------------------------------------------------
# OpenTelemetry Integration
# -----------------------------------------------------------------------------


def _load_otel_deps() -> (
    tuple[
        TraceModuleProtocol,
        type[ResourceProtocol],
        type[TracerProviderProtocol],
        type[BatchSpanProcessorProtocol],
        type[OTLPSpanExporterProtocol],
    ]
    | None
):
    """Load OpenTelemetry dependencies if available.

    Returns None if opentelemetry packages are not installed.
    """
    global _otel_trace_module, _otel_resource_cls, _otel_tracer_provider_cls
    global _otel_span_processor_cls, _otel_exporter_cls

    if _otel_trace_module is not None:
        return (
            _otel_trace_module,
            cast(type[ResourceProtocol], _otel_resource_cls),
            cast(type[TracerProviderProtocol], _otel_tracer_provider_cls),
            cast(type[BatchSpanProcessorProtocol], _otel_span_processor_cls),
            cast(type[OTLPSpanExporterProtocol], _otel_exporter_cls),
        )

    try:
        from opentelemetry import trace as trace_module  # pyright: ignore[reportMissingImports]
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # pyright: ignore[reportMissingImports]
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # pyright: ignore[reportMissingImports]
        from opentelemetry.sdk.trace import TracerProvider  # pyright: ignore[reportMissingImports]
        from opentelemetry.sdk.trace.export import (  # pyright: ignore[reportMissingImports]
            BatchSpanProcessor,
        )

        _otel_trace_module = cast(TraceModuleProtocol, trace_module)
        _otel_resource_cls = cast(type[ResourceProtocol], Resource)
        _otel_tracer_provider_cls = cast(type[TracerProviderProtocol], TracerProvider)
        _otel_span_processor_cls = cast(type[BatchSpanProcessorProtocol], BatchSpanProcessor)
        _otel_exporter_cls = cast(type[OTLPSpanExporterProtocol], OTLPSpanExporter)

        return (
            _otel_trace_module,
            _otel_resource_cls,
            _otel_tracer_provider_cls,
            _otel_span_processor_cls,
            _otel_exporter_cls,
        )
    except ImportError:
        return None


def _get_tracer() -> TracerProtocol | None:
    """Get or create an OpenTelemetry tracer.

    Returns None if OpenTelemetry is not available or not configured.
    """
    global _otel_tracer, _otel_provider_configured

    if _otel_tracer is not None:
        return _otel_tracer

    deps = _load_otel_deps()
    if deps is None:
        return None

    trace_module, resource_cls, provider_cls, processor_cls, exporter_cls = deps

    if not _otel_provider_configured:
        import os

        otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otel_endpoint:
            try:
                resource = resource_cls.create({"service.name": "jpscripts"})
                provider = provider_cls(resource=resource)
                exporter = exporter_cls(endpoint=f"{otel_endpoint}/v1/traces")
                processor = processor_cls(exporter)
                provider.add_span_processor(processor)
                trace_module.set_tracer_provider(provider)
                _otel_provider_configured = True
            except Exception as exc:  # pragma: no cover
                logger.debug("Failed to configure OpenTelemetry: %s", exc)
                return None
        else:
            return None

    _otel_tracer = trace_module.get_tracer("jpscripts.agent")
    return _otel_tracer


__all__ = [
    "TraceRecorder",
    "_get_tracer",
    "_load_otel_deps",
]
