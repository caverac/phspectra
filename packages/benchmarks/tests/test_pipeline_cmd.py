"""Tests for the ``benchmarks pipeline`` command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from benchmarks.cli import main
from benchmarks.commands.pipeline import (
    _bucket_name,
    _discover_run_id,
    _get_run_item,
    _poll_progress,
    _survey_from_path,
    _table_name,
)
from click.testing import CliRunner

MOCK_BOTO3 = "benchmarks.commands.pipeline.boto3"


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
        "run_id": {"S": run_id},
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
        ddb = MagicMock()
        mock_boto3.client.side_effect = lambda svc: s3 if svc == "s3" else ddb

        item = _make_run_item()
        ddb.scan.return_value = {"Items": [item]}
        ddb.get_item.return_value = {"Item": item}

        result = CliRunner().invoke(
            main,
            ["pipeline", str(fits), "--poll-interval", "0"],
        )
        assert result.exit_code == 0

        # S3: upload_file (FITS) + put_object (manifest)
        s3.upload_file.assert_called_once()
        s3.put_object.assert_called_once()

        # Manifest body contains default beta
        call_kwargs = s3.put_object.call_args
        body = json.loads(call_kwargs[1]["Body"] if call_kwargs[1] else call_kwargs.kwargs["Body"])
        assert body["beta_values"] == [3.8]
        assert body["survey"] == "grs"


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
        ddb.scan.return_value = {"Items": [item]}
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
                "--beta",
                "3.8",
                "--beta",
                "5.0",
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
        """Scan timeout should cause SystemExit."""
        ddb = MagicMock()
        ddb.scan.return_value = {"Items": []}

        with pytest.raises(SystemExit):
            _discover_run_id(ddb, "t", "grs", "2025-01-01T00:00:00", timeout=0)

    def test_finds_run(self) -> None:
        """Scan returning a matching item should return its run_id."""
        ddb = MagicMock()
        item = _make_run_item(run_id="abc-123")
        ddb.scan.return_value = {"Items": [item]}

        run_id = _discover_run_id(ddb, "t", "grs", "2025-01-01T00:00:00")
        assert run_id == "abc-123"

    @patch("benchmarks.commands.pipeline.time.sleep")
    def test_retries_before_finding(self, mock_sleep: MagicMock) -> None:
        """Discovery retries when scan returns empty, then succeeds."""
        ddb = MagicMock()
        item = _make_run_item(run_id="retry-ok")
        ddb.scan.side_effect = [{"Items": []}, {"Items": [item]}]

        run_id = _discover_run_id(ddb, "t", "grs", "2025-01-01T00:00:00")
        assert run_id == "retry-ok"
        mock_sleep.assert_called_once()


# ---------------------------------------------------------------------------
# TestPollProgress
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestHelperFunctions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Unit tests for small helper functions."""

    def test_bucket_name(self) -> None:
        """Bucket name should follow phspectra-{env}-data pattern."""
        assert _bucket_name("staging") == "phspectra-staging-data"

    def test_table_name(self) -> None:
        """Table name should follow phspectra-{env}-runs pattern."""
        assert _table_name("production") == "phspectra-production-runs"

    def test_survey_from_path(self) -> None:
        """Survey name should be the lowercased file stem."""
        assert _survey_from_path("/data/GRS_cube.fits") == "grs_cube"

    def test_get_run_item_missing(self) -> None:
        """Missing item should cause SystemExit."""
        ddb = MagicMock()
        ddb.get_item.return_value = {}

        with pytest.raises(SystemExit):
            _get_run_item(ddb, "t", "missing-id")
