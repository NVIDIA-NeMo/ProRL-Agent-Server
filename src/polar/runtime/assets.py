"""Host-side validation for runtime assets used by rollout tasks."""

from __future__ import annotations

from pathlib import Path

from polar.runtime.models import RuntimeSpec


class RuntimeAssetUnavailableError(RuntimeError):
    """A local runtime image or bind source disappeared before dispatch."""


_REMOTE_APPTAINER_IMAGE_PREFIXES = (
    "docker://",
    "http://",
    "https://",
    "library://",
    "oras://",
    "shub://",
)


def validate_runtime_assets(spec: RuntimeSpec | None) -> None:
    """Fail before session fan-out when required Apptainer assets are missing.

    Docker image names and remote Apptainer transports are intentionally not
    interpreted as host paths.  Apptainer bind sources are local by definition,
    so they are checked as well.  An empty read-only directory is treated as a
    broken asset: code/runtime bundles are commonly mounted read-only, and an
    interrupted cleanup can otherwise leave the mount point behind while
    deleting everything the task needs (for example a shared agent runtime).
    """

    if spec is None or spec.backend != "apptainer":
        return

    problems: list[str] = []
    image = spec.image.strip()
    if not image.startswith(_REMOTE_APPTAINER_IMAGE_PREFIXES):
        image_path = Path(image).expanduser()
        if not image_path.is_file():
            problems.append(f"apptainer image is not a file: {image_path}")
        else:
            try:
                image_size = image_path.stat().st_size
            except OSError as exc:
                problems.append(f"apptainer image is not readable: {image_path}: {exc}")
            else:
                if image_size <= 0:
                    problems.append(f"apptainer image is empty: {image_path}")

    volumes = spec.kwargs.get("volumes", [])
    if volumes is None:
        volumes = []
    if not isinstance(volumes, (list, tuple)):
        problems.append("runtime kwargs.volumes must be a list or tuple")
        volumes = []

    active_volume_groups: list[tuple[str, list[object] | tuple[object, ...]]] = [
        ("bind volume", volumes),
    ]
    if spec.allow_internet:
        active_volume_groups.append(
            ("internet bind volume", tuple(spec.internet_volumes))
        )

    for label, volume_group in active_volume_groups:
        for index, raw_volume in enumerate(volume_group):
            source, read_only, parse_error = _parse_bind_source(raw_volume)
            if parse_error is not None:
                problems.append(f"{label} {index}: {parse_error}")
                continue

            source_path = Path(source).expanduser()
            if not source_path.exists():
                problems.append(f"bind source does not exist: {source_path}")
                continue
            if read_only and source_path.is_dir():
                try:
                    empty = next(source_path.iterdir(), None) is None
                except OSError as exc:
                    problems.append(f"bind source is not readable: {source_path}: {exc}")
                else:
                    if empty:
                        problems.append(f"read-only bind source is empty: {source_path}")

    if problems:
        raise RuntimeAssetUnavailableError(
            "runtime asset validation failed before session dispatch: "
            + "; ".join(problems)
        )


def _parse_bind_source(raw_volume: object) -> tuple[str, bool, str | None]:
    if not isinstance(raw_volume, str):
        return "", False, f"expected string, got {type(raw_volume).__name__}"
    value = raw_volume.strip()
    if not value:
        return "", False, "bind specification is empty"

    parts = value.split(":", 2)
    source = parts[0].strip()
    if not source:
        return "", False, "bind source is empty"
    options = parts[2].split(",") if len(parts) == 3 else []
    return source, "ro" in options, None
