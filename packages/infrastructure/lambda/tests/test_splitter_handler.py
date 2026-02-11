"""Tests for the splitter Lambda handler."""

# pylint: disable=protected-access

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── _survey_from_key ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("cubes/GRS.fits", "grs"),
        ("a/b/NGC1234.fits", "ngc1234"),
        ("FILE.fits", "file"),
    ],
    ids=["nested-dir", "deep-nested", "root-level"],
)
def test_survey_from_key(splitter: Any, key: str, expected: str) -> None:
    """``_survey_from_key`` extracts lowercase stem from various key formats."""
    assert splitter._survey_from_key(key) == expected


# ── handler routing ─────────────────────────────────────────────────────────


def test_routes_manifest(splitter: Any, eventbridge_event: Any, lambda_context: MagicMock) -> None:
    """Manifest keys delegate to ``_handle_manifest``."""
    event = eventbridge_event("manifests/sweep.json")
    with patch.object(splitter, "_handle_manifest", return_value={"statusCode": 200}) as mock:
        splitter.handler(event, lambda_context)
    mock.assert_called_once_with("manifests/sweep.json")


def test_routes_fits(splitter: Any, eventbridge_event: Any, lambda_context: MagicMock) -> None:
    """FITS keys delegate to ``_handle_fits`` with ``DEFAULT_BETA``."""
    event = eventbridge_event("cubes/grs.fits")
    with patch.object(splitter, "_handle_fits", return_value={"statusCode": 200}) as mock:
        splitter.handler(event, lambda_context)
    mock.assert_called_once_with(
        "cubes/grs.fits", survey="grs", beta_values=[splitter.DEFAULT_BETA]
    )


def test_routes_unsupported_key(
    splitter: Any, eventbridge_event: Any, lambda_context: MagicMock
) -> None:
    """Non-FITS, non-manifest keys return 400."""
    event = eventbridge_event("data/image.png")
    result = splitter.handler(event, lambda_context)
    assert result["statusCode"] == 400


def test_json_outside_manifests_prefix(
    splitter: Any, eventbridge_event: Any, lambda_context: MagicMock
) -> None:
    """A ``.json`` key *not* under ``manifests/`` hits the else branch."""
    event = eventbridge_event("data/foo.json")
    result = splitter.handler(event, lambda_context)
    assert result["statusCode"] == 400


# ── _handle_manifest ────────────────────────────────────────────────────────


def test_handle_manifest_parses_and_delegates(splitter: Any) -> None:
    """``_handle_manifest`` reads S3 JSON and calls ``_handle_fits``."""
    manifest = {"cube_key": "cubes/test.fits", "survey": "grs", "beta_values": [1.0, 2.0]}
    body_stream = io.BytesIO(json.dumps(manifest).encode())
    mock_resp = {"Body": body_stream}

    with (
        patch.object(splitter.s3, "get_object", return_value=mock_resp) as mock_s3,
        patch.object(splitter, "_handle_fits", return_value={"statusCode": 200}) as mock_fits,
    ):
        splitter._handle_manifest("manifests/test.json")

    mock_s3.assert_called_once_with(Bucket="test-bucket", Key="manifests/test.json")
    mock_fits.assert_called_once_with("cubes/test.fits", survey="grs", beta_values=[1.0, 2.0])


# ── _handle_fits ────────────────────────────────────────────────────────────


def _make_mock_hdul(shape: tuple[int, ...]) -> MagicMock:
    """Build a mock HDUList whose primary HDU has ``data`` of given *shape*."""
    data = np.ones(shape, dtype=np.float64)
    hdu = MagicMock()
    hdu.data = data
    hdul = MagicMock()
    hdul.__enter__ = MagicMock(return_value=[hdu])
    hdul.__exit__ = MagicMock(return_value=False)
    return hdul


def test_handle_fits_single_chunk(splitter: Any) -> None:
    """A 3-D cube with <=500 spectra produces exactly 1 chunk and 1 message."""
    # shape (4, 10, 10) → 100 spectra → 1 chunk
    hdul = _make_mock_hdul((4, 10, 10))
    run_id = "fixed-uuid"

    with (
        patch.object(splitter.s3, "download_file"),
        patch.object(splitter, "fits") as mock_fits_mod,
        patch.object(splitter, "os") as mock_os,
        patch.object(splitter, "np") as mock_np,
        patch.object(splitter.s3, "upload_file"),
        patch.object(splitter.sqs, "send_message"),
        patch.object(
            splitter.uuid, "uuid4", return_value=MagicMock(hex=run_id, __str__=lambda _: run_id)
        ),
    ):
        mock_fits_mod.open.return_value = hdul
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        # Let np operations pass through to real numpy
        mock_np.nan_to_num.side_effect = np.nan_to_num
        mock_np.mgrid = np.mgrid
        mock_np.savez_compressed = MagicMock()
        mock_np.float64 = np.float64

        result = splitter._handle_fits("cubes/test.fits", survey="grs", beta_values=[5.0])

    assert result["statusCode"] == 200
    assert result["body"]["n_chunks"] == 1
    assert result["body"]["n_messages"] == 1
    assert result["body"]["n_spectra"] == 100


def test_handle_fits_multiple_chunks(splitter: Any) -> None:
    """A cube with >500 spectra produces the correct number of chunks."""
    # shape (4, 30, 30) → 900 spectra → ceil(900/500) = 2 chunks
    hdul = _make_mock_hdul((4, 30, 30))
    run_id = "fixed-uuid"

    with (
        patch.object(splitter.s3, "download_file"),
        patch.object(splitter, "fits") as mock_fits_mod,
        patch.object(splitter, "os") as mock_os,
        patch.object(splitter, "np") as mock_np,
        patch.object(splitter.s3, "upload_file"),
        patch.object(splitter.sqs, "send_message"),
        patch.object(
            splitter.uuid, "uuid4", return_value=MagicMock(hex=run_id, __str__=lambda _: run_id)
        ),
    ):
        mock_fits_mod.open.return_value = hdul
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_np.nan_to_num.side_effect = np.nan_to_num
        mock_np.mgrid = np.mgrid
        mock_np.savez_compressed = MagicMock()
        mock_np.float64 = np.float64

        result = splitter._handle_fits("cubes/test.fits", survey="grs", beta_values=[5.0])

    assert result["statusCode"] == 200
    assert result["body"]["n_chunks"] == 2
    assert result["body"]["n_spectra"] == 900


def test_handle_fits_multiple_betas(splitter: Any) -> None:
    """Number of messages equals ``n_chunks * len(beta_values)``."""
    # 100 spectra → 1 chunk, 3 betas → 3 messages
    hdul = _make_mock_hdul((4, 10, 10))
    run_id = "fixed-uuid"

    with (
        patch.object(splitter.s3, "download_file"),
        patch.object(splitter, "fits") as mock_fits_mod,
        patch.object(splitter, "os") as mock_os,
        patch.object(splitter, "np") as mock_np,
        patch.object(splitter.s3, "upload_file"),
        patch.object(splitter.sqs, "send_message"),
        patch.object(
            splitter.uuid, "uuid4", return_value=MagicMock(hex=run_id, __str__=lambda _: run_id)
        ),
    ):
        mock_fits_mod.open.return_value = hdul
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_np.nan_to_num.side_effect = np.nan_to_num
        mock_np.mgrid = np.mgrid
        mock_np.savez_compressed = MagicMock()
        mock_np.float64 = np.float64

        result = splitter._handle_fits(
            "cubes/test.fits", survey="grs", beta_values=[1.0, 5.0, 10.0]
        )

    assert result["body"]["n_messages"] == 3
    assert result["body"]["beta_values"] == [1.0, 5.0, 10.0]


def test_handle_fits_non_3d_raises(splitter: Any) -> None:
    """A 2-D array raises ``ValueError``."""
    hdul = _make_mock_hdul((100, 50))
    run_id = "fixed-uuid"

    with (
        patch.object(splitter.s3, "download_file"),
        patch.object(splitter, "fits") as mock_fits_mod,
        patch.object(splitter, "os") as mock_os,
        patch.object(
            splitter.uuid, "uuid4", return_value=MagicMock(hex=run_id, __str__=lambda _: run_id)
        ),
    ):
        mock_fits_mod.open.return_value = hdul
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()

        with pytest.raises(ValueError, match="Expected 3D FITS cube"):
            splitter._handle_fits("cubes/test.fits", survey="grs", beta_values=[5.0])


def test_handle_fits_nan_replacement(splitter: Any) -> None:
    """NaN values in the cube are replaced with zeros in uploaded chunks."""
    data = np.ones((4, 5, 5), dtype=np.float64)
    data[0, 0, 0] = np.nan
    hdu = MagicMock()
    hdu.data = data
    hdul = MagicMock()
    hdul.__enter__ = MagicMock(return_value=[hdu])
    hdul.__exit__ = MagicMock(return_value=False)
    run_id = "fixed-uuid"

    saved_arrays: list[np.ndarray] = []

    def capture_savez(_path: str, **kwargs: Any) -> None:
        saved_arrays.append(kwargs["spectra"].copy())

    with (
        patch.object(splitter.s3, "download_file"),
        patch.object(splitter, "fits") as mock_fits_mod,
        patch.object(splitter, "os") as mock_os,
        patch.object(splitter, "np") as mock_np,
        patch.object(splitter.s3, "upload_file"),
        patch.object(splitter.sqs, "send_message"),
        patch.object(
            splitter.uuid, "uuid4", return_value=MagicMock(hex=run_id, __str__=lambda _: run_id)
        ),
    ):
        mock_fits_mod.open.return_value = hdul
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_np.nan_to_num.side_effect = np.nan_to_num
        mock_np.mgrid = np.mgrid
        mock_np.savez_compressed.side_effect = capture_savez
        mock_np.float64 = np.float64

        splitter._handle_fits("cubes/test.fits", survey="grs", beta_values=[5.0])

    assert len(saved_arrays) == 1
    assert not np.isnan(saved_arrays[0]).any()


def test_handle_fits_sqs_message_body(splitter: Any) -> None:
    """SQS message body contains chunk_key, survey, beta, and run_id."""
    hdul = _make_mock_hdul((4, 5, 5))
    run_id = "fixed-uuid"

    sent_messages: list[dict[str, Any]] = []

    def capture_send(**kwargs: Any) -> dict[str, str]:
        sent_messages.append(kwargs)
        return {"MessageId": "msg-001"}

    with (
        patch.object(splitter.s3, "download_file"),
        patch.object(splitter, "fits") as mock_fits_mod,
        patch.object(splitter, "os") as mock_os,
        patch.object(splitter, "np") as mock_np,
        patch.object(splitter.s3, "upload_file"),
        patch.object(splitter.sqs, "send_message", side_effect=capture_send),
        patch.object(
            splitter.uuid, "uuid4", return_value=MagicMock(hex=run_id, __str__=lambda _: run_id)
        ),
    ):
        mock_fits_mod.open.return_value = hdul
        mock_os.path.basename.side_effect = lambda p: p.rsplit("/", 1)[-1]
        mock_os.remove = MagicMock()
        mock_np.nan_to_num.side_effect = np.nan_to_num
        mock_np.mgrid = np.mgrid
        mock_np.savez_compressed = MagicMock()
        mock_np.float64 = np.float64

        splitter._handle_fits("cubes/test.fits", survey="grs", beta_values=[5.0])

    assert len(sent_messages) == 1
    body = json.loads(sent_messages[0]["MessageBody"])
    assert body["survey"] == "grs"
    assert body["beta"] == 5.0
    assert body["run_id"] == run_id
    assert body["chunk_key"].startswith("chunks/")
