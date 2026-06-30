from __future__ import annotations

from pathlib import Path

import pytest

from polar.runtime.assets import RuntimeAssetUnavailableError, validate_runtime_assets
from polar.runtime.models import RuntimeSpec


def _apptainer_spec(
    image: Path,
    volumes: list[object] | None = None,
    *,
    allow_internet: bool = True,
    internet_volumes: list[str] | None = None,
) -> RuntimeSpec:
    return RuntimeSpec(
        backend="apptainer",
        image=str(image),
        allow_internet=allow_internet,
        internet_volumes=internet_volumes or [],
        kwargs={"volumes": volumes or []},
    )


def test_validates_local_apptainer_image_and_read_only_bundle(tmp_path: Path) -> None:
    image = tmp_path / "task.sif"
    image.write_bytes(b"sif")
    bundle = tmp_path / "agent-runtime"
    (bundle / "bin").mkdir(parents=True)
    (bundle / "bin" / "agent").write_text("ready")

    validate_runtime_assets(
        _apptainer_spec(image, [f"{bundle}:/opt/agent:ro"])
    )


def test_rejects_missing_image_before_session_fanout(tmp_path: Path) -> None:
    missing_image = tmp_path / "missing.sif"

    with pytest.raises(RuntimeAssetUnavailableError, match="image is not a file"):
        validate_runtime_assets(_apptainer_spec(missing_image))


def test_rejects_empty_image_before_session_fanout(tmp_path: Path) -> None:
    empty_image = tmp_path / "empty.sif"
    empty_image.touch()

    with pytest.raises(RuntimeAssetUnavailableError, match="image is empty"):
        validate_runtime_assets(_apptainer_spec(empty_image))


def test_rejects_missing_bind_source(tmp_path: Path) -> None:
    image = tmp_path / "task.sif"
    image.write_bytes(b"sif")
    missing_bundle = tmp_path / "missing-runtime"

    with pytest.raises(RuntimeAssetUnavailableError, match="bind source does not exist"):
        validate_runtime_assets(
            _apptainer_spec(image, [f"{missing_bundle}:/opt/agent:ro"])
        )


def test_rejects_read_only_bundle_left_empty_by_cleanup(tmp_path: Path) -> None:
    image = tmp_path / "task.sif"
    image.write_bytes(b"sif")
    empty_bundle = tmp_path / "empty-runtime"
    empty_bundle.mkdir()

    with pytest.raises(RuntimeAssetUnavailableError, match="read-only bind source is empty"):
        validate_runtime_assets(
            _apptainer_spec(image, [f"{empty_bundle}:/opt/agent:ro"])
        )


def test_internet_bind_assets_are_validated_only_for_online_runtime(tmp_path: Path) -> None:
    image = tmp_path / "task.sif"
    image.write_bytes(b"sif")
    missing_proxy = tmp_path / "missing-proxy"
    internet_volume = f"{missing_proxy}:/polar/proxy:ro"

    with pytest.raises(RuntimeAssetUnavailableError, match="bind source does not exist"):
        validate_runtime_assets(
            _apptainer_spec(
                image,
                allow_internet=True,
                internet_volumes=[internet_volume],
            )
        )

    validate_runtime_assets(
        _apptainer_spec(
            image,
            allow_internet=False,
            internet_volumes=[internet_volume],
        )
    )


def test_does_not_treat_docker_or_remote_apptainer_images_as_local_files() -> None:
    validate_runtime_assets(RuntimeSpec(backend="docker", image="repo/image:latest"))
    validate_runtime_assets(
        RuntimeSpec(backend="apptainer", image="docker://repo/image:latest")
    )
