"""Low-overhead HTTP server logging defaults for Polar services."""

from __future__ import annotations

import os


_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})


def uvicorn_access_log_enabled() -> bool:
    """Return whether per-request Uvicorn access logging is explicitly enabled.

    Gateway and rollout endpoints are high-frequency internal control-plane
    paths.  Formatting and synchronously writing one line for every poll and
    model request is expensive on shared job-output filesystems, so access
    logging is opt-in.  Uvicorn's error logger remains enabled either way.
    """

    return os.environ.get("POLAR_UVICORN_ACCESS_LOG", "").strip().lower() in (
        _TRUTHY_VALUES
    )


__all__ = ["uvicorn_access_log_enabled"]
