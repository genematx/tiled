"""The `RequestSpec` dataclass shared by the sans-I/O protocol modules.

Protocol modules (`_array_protocol`, `_container_protocol`) build these pure,
declarative request descriptions; the sync and async shells turn them into
actual I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class RequestSpec:
    "A declarative description of one HTTP request. Contains no I/O state."

    method: str
    url_path: str
    params: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    content: Optional[bytes] = None
