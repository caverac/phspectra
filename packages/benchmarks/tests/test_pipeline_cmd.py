"""Tests for the ``benchmarks pipeline`` command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from benchmarks.cli import main
from benchmarks.commands.pipeline import (
    _bucket_name,
    _discover_run_id,
    _finish_pipeline,
    _get_run_item,
    _parse_params,
    _poll_progress,
    _s3_key_exists,
    _survey_from_path,
    _table_name,
    _upload_fits,
    _upload_manifest,
)
from botocore.exceptions import ClientError
from click.testing import CliRunner

MOCK_BOTO3 = "benchmarks.commands.pipeline.boto3"

_NOT_FOUND = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_item(
    run_id: str = "run-1",
    survey: str = "grs",
    created_at: str = "2025-01-01T00:00:00+00:00",
    jobs_total: int = 10,
    jobs_completed: int = 10,
    jobs_failed: int = 0,
) -> dict[str, dict[str, str]]:
    return {
        "PK": {"S": run_id},
        "SK": {"S": "RUN"},
        "survey": {"S": survey},
        "created_at": {"S": created_at},
        "jobs_total": {"N": str(jobs_total)},
        "jobs_completed": {"N": str(jobs_completed)},
        "jobs_failed": {"N": str(jobs_failed)},
    }


# ---------------------------------------------------------------------------
# TestPipelineHelp
# ---------------------------------------------------------------------------


class TestPipelineHelp:
    """``benchmarks pipeline --help`` works and lists key options."""

    def test_help_succeeds(self) -> None:
        """Help text should exit cleanly."""
        result = CliRunner().invoke(main, ["pipeline", "--help"])
        assert result.exit_code == 0

    def test_help_lists_environment(self) -> None:
        """Help text should mention --environment."""
        result = CliRunner().invoke(main, ["pipeline", "--help"])
        assert "--environment" in result.output

    def test_help_lists_manifest(self) -> None:
        """Help text should mention --manifest."""
        result = CliRunner().invoke(main, ["pipeline", "--help"])
        assert "--manifest" in result.output


# ---------------------------------------------------------------------------
# TestPipelineValidation
# ---------------------------------------------------------------------------


class TestPipelineValidation:
    """Argument validation in direct vs manifest mode."""

    @patch(MOCK_BOTO3)
    def test_direct_mode_requires_fits_file(self, _mock_boto3: MagicMock) -> None:
        """Direct mode without FITS_FILE should fail."""
        result = CliRunner().invoke(main, ["pipeline"])
        assert result.exit_code != 0
        assert "FITS_FILE is required" in result.output

    @patch(MOCK_BOTO3)
    def test_manifest_mode_requires_cube_key(self, _mock_boto3: MagicMock) -> None:
        """Manifest mode without --cube-key should fail."""
        result = CliRunner().invoke(main, ["pipeline", "--manifest", "--survey", "grs"])
        assert result.exit_code != 0
        assert "--cube-key is required" in result.output

    @patch(MOCK_BOTO3)
    def test_manifest_mode_requires_survey(self, _mock_boto3: MagicMock) -> None:
        """Manifest mode without --survey should fail."""
        result = CliRunner().invoke(main, ["pipeline", "--manifest", "--cube-key", "cubes/test.fits"])
        assert result.exit_code != 0
        assert "--survey is required" in result.output

    def test_manifest_mode_rejects_fits_file(self, tmp_path: Path) -> None:
        """Manifest mode with a positional FITS_FILE should fail."""
        fits = tmp_path / "test.fits"
        fits.write_bytes(b"fake")
        result = CliRunner().invoke(
            main,
            [
                "pipeline",
                "--manifest",
                "--cube-key",
                "cubes/x.fits",
                "--survey",
                "grs",
                str(fits),
            ],
        )
        assert result.exit_code != 0
        assert "not allowed" in result.output


# ---------------------------------------------------------------------------
# TestPipelineDirectMode
# ---------------------------------------------------------------------------


class TestPipelineDirectMode:
    """End-to-end with mocked S3/DDB in direct mode."""

    @patch(MOCK_BOTO3)
    def test_direct_mode_uploads_and_polls(self, mock_boto3: MagicMock, tmp_path: Path) -> None:
        """Direct mode should upload FITS, create manifest, and poll DDB."""
        fits = tmp_path / "GRS.fits"
        fits.write_bytes(b"fake")
        s3 = MagicMock()
        s3.head_object.side_effect = _NOT_FOUND
        ddb = MagicMock()
        mock_boto3.client.side_effect = lambda svc: s3 if svc == "s3" else ddb

        item = _make_run_item()
        ddb.query.return_value = {"Items": [item]}
        ddb.get_item.return_value = {"Item": item}

        result = CliRunner().invoke(
            main,
            ["pipeline", str(fits), "--poll-interval", "0"],
        )
        assert result.exit_code == 0

        # S3: upload_file (FITS) + put_object (manifest)
        s3.upload_file.assert_called_once()
        s3.put_object.assert_called_once()

        # Manifest body has no params key (no --param given -> omitted)
        call_kwargs = s3.put_object.call_args
        body = json.loads(call_kwargs[1]["Body"] if call_kwargs[1] else call_kwargs.kwargs["Body"])
        assert "params" not in body
        assert body["survey"] == "grs"

    @patch(MOCK_BOTO3)
    def test_direct_mode_skips_upload_when_exists(self, mock_boto3: MagicMock, tmp_path: Path) -> None:
        """Direct mode skips FITS upload when cube already exists in S3."""
        fits = tmp_path / "GRS.fits"
        fits.write_bytes(b"fake")
        s3 = MagicMock()
        s3.head_object.return_value = {}  # file exists
        ddb = MagicMock()
        mock_boto3.client.side_effect = lambda svc: s3 if svc == "s3" else ddb

        item = _make_run_item()
        ddb.query.return_value = {"Items": [item]}
        ddb.get_item.return_value = {"Item": item}

        result = CliRunner().invoke(
            main,
            ["pipeline", str(fits), "--poll-interval", "0"],
        )
        assert result.exit_code == 0

        # FITS upload skipped, only manifest uploaded
        s3.upload_file.assert_not_called()
        s3.put_object.assert_called_once()
        assert "already exists" in result.output

    @patch(MOCK_BOTO3)
    def test_direct_mode_with_param(self, mock_boto3: MagicMock, tmp_path: Path) -> None:
        """Direct mode with --param beta=3.8 includes params in manifest."""
        fits = tmp_path / "GRS.fits"
        fits.write_bytes(b"fake")
        s3 = MagicMock()
        s3.head_object.side_effect = _NOT_FOUND
        ddb = MagicMock()
        mock_boto3.client.side_effect = lambda svc: s3 if svc == "s3" else ddb

        item = _make_run_item()
        ddb.query.return_value = {"Items": [item]}
        ddb.get_item.return_value = {"Item": item}

        result = CliRunner().invoke(
            main,
            ["pipeline", str(fits), "--param", "beta=3.8", "--poll-interval", "0"],
        )
        assert result.exit_code == 0

        call_kwargs = s3.put_object.call_args
        body = json.loads(call_kwargs[1]["Body"] if call_kwargs[1] else call_kwargs.kwargs["Body"])
        assert body["params"] == {"beta": 3.8}


# ---------------------------------------------------------------------------
# TestPipelineManifestMode
# ---------------------------------------------------------------------------


class TestPipelineManifestMode:
    """End-to-end in manifest-only mode."""

    @patch(MOCK_BOTO3)
    def test_manifest_mode_no_upload_file(self, mock_boto3: MagicMock) -> None:
        """Manifest mode should skip upload_file and only put_object once."""
        s3 = MagicMock()
        ddb = MagicMock()
        mock_boto3.client.side_effect = lambda svc: s3 if svc == "s3" else ddb

        item = _make_run_item()
        ddb.query.return_value = {"Items": [item]}
        ddb.get_item.return_value = {"Item": item}

        result = CliRunner().invoke(
            main,
            [
                "pipeline",
                "--manifest",
                "--cube-key",
                "cubes/GRS.fits",
                "--survey",
                "grs",
                "--param",
                "beta=3.8",
                "--poll-interval",
                "0",
            ],
        )
        assert result.exit_code == 0

        # No upload_file in manifest mode
        s3.upload_file.assert_not_called()
        # One put_object for the manifest
        s3.put_object.assert_called_once()


# ---------------------------------------------------------------------------
# TestDiscoverRunId
# ---------------------------------------------------------------------------


class TestDiscoverRunId:
    """Discovery loop behaviour."""

    def test_timeout_exits(self) -> None:
        """Query timeout should cause SystemExit."""
        ddb = MagicMock()
        ddb.query.return_value = {"Items": []}

        with pytest.raises(SystemExit):
            _discover_run_id(ddb, "t", "grs", "2025-01-01T00:00:00", timeout=0)

    def test_finds_run(self) -> None:
        """GSI1 query returning a matching item should return its run_id."""
        ddb = MagicMock()
        item = _make_run_item(run_id="abc-123")
        ddb.query.return_value = {"Items": [item]}

        run_id = _discover_run_id(ddb, "t", "grs", "2025-01-01T00:00:00")
        assert run_id == "abc-123"

    def test_query_uses_gsi1(self) -> None:
        """Discovery should query GSI1 with correct parameters."""
        ddb = MagicMock()
        item = _make_run_item(run_id="abc-123")
        ddb.query.return_value = {"Items": [item]}

        _discover_run_id(ddb, "t", "grs", "2025-01-01T00:00:00")
        call_kwargs = ddb.query.call_args[1]
        assert call_kwargs["IndexName"] == "GSI1"
        assert call_kwargs["ScanIndexForward"] is False
        assert call_kwargs["Limit"] == 1

    @patch("benchmarks.commands.pipeline.time.sleep")
    def test_retries_before_finding(self, mock_sleep: MagicMock) -> None:
        """Discovery retries when query returns empty, then succeeds."""
        ddb = MagicMock()
        item = _make_run_item(run_id="retry-ok")
        ddb.query.side_effect = [{"Items": []}, {"Items": [item]}]

        run_id = _discover_run_id(ddb, "t", "grs", "2025-01-01T00:00:00")
        assert run_id == "retry-ok"
        mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# TestPollProgress
# ---------------------------------------------------------------------------


class TestFinishPipeline:
    """Summary message formatting."""

    def test_prints_failures(self) -> None:
        """Summary message includes failure count when jobs_failed > 0."""
        ddb = MagicMock()
        item = _make_run_item(jobs_total=10, jobs_completed=8, jobs_failed=2)
        ddb.get_item.return_value = {"Item": item}

        _finish_pipeline(ddb, "t", "run-1", poll_interval=0, stall_timeout=0)


class TestPollProgress:
    """Progress polling with failures."""

    def test_shows_failures(self) -> None:
        """Poll result should reflect failed job count."""
        ddb = MagicMock()
        item = _make_run_item(jobs_total=10, jobs_completed=8, jobs_failed=2)
        ddb.get_item.return_value = {"Item": item}

        final = _poll_progress(ddb, "t", "run-1", poll_interval=0)
        assert int(final["jobs_failed"]["N"]) == 2

    def test_completes_when_all_done(self) -> None:
        """Poll should finish when all jobs are completed."""
        ddb = MagicMock()
        item = _make_run_item(jobs_total=5, jobs_completed=5, jobs_failed=0)
        ddb.get_item.return_value = {"Item": item}

        final = _poll_progress(ddb, "t", "run-1", poll_interval=0)
        assert int(final["jobs_completed"]["N"]) == 5

    @patch("benchmarks.commands.pipeline.time.sleep")
    def test_polls_until_done(self, mock_sleep: MagicMock) -> None:
        """Poll retries when jobs are not yet complete."""
        ddb = MagicMock()
        partial = _make_run_item(jobs_total=4, jobs_completed=2, jobs_failed=0)
        done = _make_run_item(jobs_total=4, jobs_completed=4, jobs_failed=0)
        ddb.get_item.side_effect = [{"Item": partial}, {"Item": done}]

        final = _poll_progress(ddb, "t", "run-1", poll_interval=1)
        assert int(final["jobs_completed"]["N"]) == 4
        mock_sleep.assert_called_once_with(1)

    @patch("benchmarks.commands.pipeline.time.sleep")
    @patch("benchmarks.commands.pipeline.time.monotonic")
    def test_stall_timeout_exits(self, mock_monotonic: MagicMock, _mock_sleep: MagicMock) -> None:
        """Poll exits when no progress is made within stall_timeout."""
        ddb = MagicMock()
        stalled = _make_run_item(jobs_total=10, jobs_completed=3, jobs_failed=0)
        ddb.get_item.return_value = {"Item": stalled}

        # First call: t=0 (initial), second: t=0 (first poll, done changes so reset),
        # third: t=100 (stall check after sleep), fourth: t=100 (second poll, done unchanged),
        # fifth: t=200 (stall check -> exceeds 100s timeout)
        mock_monotonic.side_effect = [0, 0, 100, 100, 200]

        with pytest.raises(SystemExit):
            _poll_progress(ddb, "t", "run-1", poll_interval=1, stall_timeout=100)


# ---------------------------------------------------------------------------
# TestUploadManifest
# ---------------------------------------------------------------------------


class TestUploadManifest:
    """Manifest body format."""

    def test_manifest_includes_params_when_set(self) -> None:
        """Manifest body contains ``params`` when non-empty."""
        s3 = MagicMock()
        _upload_manifest(s3, "bucket", "cubes/test.fits", "grs", {"beta": 3.8})
        body = json.loads(s3.put_object.call_args[1]["Body"])
        assert body["params"] == {"beta": 3.8}

    def test_manifest_omits_params_when_empty(self) -> None:
        """Manifest body omits ``params`` when empty dict."""
        s3 = MagicMock()
        _upload_manifest(s3, "bucket", "cubes/test.fits", "grs", {})
        body = json.loads(s3.put_object.call_args[1]["Body"])
        assert "params" not in body


# ---------------------------------------------------------------------------
# TestParseParams
# ---------------------------------------------------------------------------


class TestParseParams:
    """Validation of ``--param key=value`` parsing."""

    def test_invalid_key_raises(self) -> None:
        """An unknown key raises ``UsageError``."""
        with pytest.raises(click.UsageError, match="Unknown param"):
            _parse_params(("invalid_key=1.0",))

    def test_valid_key_parses(self) -> None:
        """A valid key is parsed and JSON-decoded."""
        result = _parse_params(("beta=3.8", "snr_min=2.0"))
        assert result == {"beta": 3.8, "snr_min": 2.0}

    def test_missing_equals_raises(self) -> None:
        """A param without ``=`` raises ``UsageError``."""
        with pytest.raises(click.UsageError, match="Invalid --param format"):
            _parse_params(("beta",))

    def test_cli_rejects_invalid_param(self) -> None:
        """CLI exits with error for unknown param key."""
        result = CliRunner().invoke(
            main,
            ["pipeline", "--manifest", "--cube-key", "x", "--survey", "grs", "--param", "bad_key=1"],
        )
        assert result.exit_code != 0
        assert "Unknown param" in result.output


# ---------------------------------------------------------------------------
# TestHelperFunctions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Unit tests for small helper functions."""

    def test_bucket_name(self) -> None:
        """Bucket name should follow phspectra-{env}-data pattern."""
        assert _bucket_name("staging") == "phspectra-staging-data"

    def test_table_name(self) -> None:
        """Table name is not environment-prefixed."""
        assert _table_name("production") == "phspectra-runs"

    def test_survey_from_path(self) -> None:
        """Survey name should be the lowercased file stem."""
        assert _survey_from_path("/data/GRS_cube.fits") == "grs_cube"

    def test_get_run_item_missing(self) -> None:
        """Missing item should cause SystemExit."""
        ddb = MagicMock()
        ddb.get_item.return_value = {}

        with pytest.raises(SystemExit):
            _get_run_item(ddb, "t", "missing-id")

    def test_s3_key_exists_true(self) -> None:
        """``_s3_key_exists`` returns True when head_object succeeds."""
        s3 = MagicMock()
        s3.head_object.return_value = {}
        assert _s3_key_exists(s3, "bucket", "key") is True

    def test_s3_key_exists_false(self) -> None:
        """``_s3_key_exists`` returns False when head_object raises ClientError."""
        s3 = MagicMock()
        s3.head_object.side_effect = _NOT_FOUND
        assert _s3_key_exists(s3, "bucket", "key") is False

    def test_upload_fits_skips_existing(self) -> None:
        """``_upload_fits`` skips upload when the key already exists."""
        s3 = MagicMock()
        s3.head_object.return_value = {}
        _upload_fits(s3, "bucket", "/tmp/test.fits", "cubes/test.fits")
        s3.upload_file.assert_not_called()

    def test_upload_fits_uploads_missing(self) -> None:
        """``_upload_fits`` uploads when the key does not exist."""
        s3 = MagicMock()
        s3.head_object.side_effect = _NOT_FOUND
        _upload_fits(s3, "bucket", "/tmp/test.fits", "cubes/test.fits")
        s3.upload_file.assert_called_once_with("/tmp/test.fits", "bucket", "cubes/test.fits")
