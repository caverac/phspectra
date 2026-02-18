"""Tests for the ``benchmarks pipeline-grs`` command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from benchmarks.cli import main
from benchmarks.commands.pipeline_grs import (
    _poll_all_progress,
    _survey_from_grs_filename,
)
from botocore.exceptions import ClientError
from click.testing import CliRunner

MOCK_BOTO3 = "benchmarks.commands.pipeline_grs.boto3"

_NOT_FOUND = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run_item(
    run_id: str = "run-1",
    survey: str = "grs-15",
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
# TestSurveyFromGrsFilename
# ---------------------------------------------------------------------------


class TestSurveyFromGrsFilename:
    """Unit tests for _survey_from_grs_filename."""

    def test_standard_cube_suffix(self) -> None:
        """``grs-15-cube.fits`` -> ``grs-15``."""
        assert _survey_from_grs_filename("grs-15-cube.fits") == "grs-15"

    def test_no_cube_suffix(self) -> None:
        """``grs-15.fits`` -> ``grs-15``."""
        assert _survey_from_grs_filename("grs-15.fits") == "grs-15"

    def test_uppercase(self) -> None:
        """``GRS-15-Cube.fits`` -> ``grs-15``."""
        assert _survey_from_grs_filename("GRS-15-Cube.fits") == "grs-15"

    def test_full_path(self) -> None:
        """Full path is handled correctly."""
        assert _survey_from_grs_filename("/data/tiles/grs-20-cube.fits") == "grs-20"


# ---------------------------------------------------------------------------
# TestPipelineGrsCli
# ---------------------------------------------------------------------------


class TestPipelineGrsCli:
    """CLI help and basic validation."""

    def test_help_succeeds(self) -> None:
        """Help text should exit cleanly."""
        result = CliRunner().invoke(main, ["pipeline-grs", "--help"])
        assert result.exit_code == 0
        assert "--input-dir" in result.output

    def test_empty_dir_exits(self, tmp_path: Path) -> None:
        """An empty input directory should print a message and exit 0."""
        result = CliRunner().invoke(main, ["pipeline-grs", "--input-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No FITS files" in result.output


# ---------------------------------------------------------------------------
# TestPollAllProgress
# ---------------------------------------------------------------------------


class TestPollAllProgress:
    """Tests for _poll_all_progress."""

    def test_all_complete(self) -> None:
        """All runs already complete on first poll."""
        ddb = MagicMock()
        item_a = _make_run_item(run_id="run-a", survey="grs-15")
        item_b = _make_run_item(run_id="run-b", survey="grs-16")
        ddb.get_item.side_effect = [
            {"Item": item_a},
            {"Item": item_b},
        ]

        finals = _poll_all_progress(ddb, "t", {"grs-15": "run-a", "grs-16": "run-b"}, poll_interval=0)
        assert len(finals) == 2
        assert "grs-15" in finals
        assert "grs-16" in finals

    def test_mixed_failures(self) -> None:
        """Runs with mixed pass/fail still complete."""
        ddb = MagicMock()
        item_ok = _make_run_item(jobs_total=5, jobs_completed=5, jobs_failed=0)
        item_fail = _make_run_item(jobs_total=5, jobs_completed=3, jobs_failed=2)
        ddb.get_item.side_effect = [
            {"Item": item_ok},
            {"Item": item_fail},
        ]

        finals = _poll_all_progress(ddb, "t", {"grs-a": "run-a", "grs-b": "run-b"}, poll_interval=0)
        assert len(finals) == 2

    @patch("benchmarks.commands.pipeline_grs.time.sleep")
    def test_second_iteration_uses_cache(self, _mock_sleep: MagicMock) -> None:
        """A survey that finished in iteration 1 is read from cache in iteration 2."""
        ddb = MagicMock()
        item_a_done = _make_run_item(run_id="run-a", survey="grs-15", jobs_total=5, jobs_completed=5)
        item_b_partial = _make_run_item(run_id="run-b", survey="grs-16", jobs_total=5, jobs_completed=2)
        item_b_done = _make_run_item(run_id="run-b", survey="grs-16", jobs_total=5, jobs_completed=5)

        # Iteration 1: A done, B partial.  Iteration 2: A cached (no call), B done.
        ddb.get_item.side_effect = [
            {"Item": item_a_done},
            {"Item": item_b_partial},
            {"Item": item_b_done},
        ]

        finals = _poll_all_progress(ddb, "t", {"grs-15": "run-a", "grs-16": "run-b"}, poll_interval=0)
        assert len(finals) == 2
        # Only 3 get_item calls (A cached on iteration 2)
        assert ddb.get_item.call_count == 3

    @patch("benchmarks.commands.pipeline_grs.time.sleep")
    @patch("benchmarks.commands.pipeline_grs.time.monotonic")
    def test_stall_timeout_exits(self, mock_monotonic: MagicMock, _mock_sleep: MagicMock) -> None:
        """Stall timeout triggers sys.exit."""
        ddb = MagicMock()
        stalled = _make_run_item(jobs_total=10, jobs_completed=3, jobs_failed=0)
        ddb.get_item.return_value = {"Item": stalled}

        # t=0 (init), t=0 (first poll, done changes so reset),
        # t=100 (stall check), t=100 (second poll, unchanged),
        # t=200 (stall check -> exceeds 100s)
        mock_monotonic.side_effect = [0, 0, 100, 100, 200]

        with pytest.raises(SystemExit):
            _poll_all_progress(
                ddb,
                "t",
                {"grs-15": "run-1"},
                poll_interval=1,
                stall_timeout=100,
            )


# ---------------------------------------------------------------------------
# TestPipelineGrsEndToEnd
# ---------------------------------------------------------------------------


class TestPipelineGrsEndToEnd:
    """End-to-end with mocked S3/DDB."""

    @patch(MOCK_BOTO3)
    @patch("benchmarks.commands.pipeline._discover_run_id")
    @patch("benchmarks.commands.pipeline_grs.time.sleep")
    def test_two_tiles(
        self,
        _mock_sleep: MagicMock,
        mock_discover: MagicMock,
        mock_boto3: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Two FITS files should produce 2 uploads, 2 manifests, and a summary."""
        (tmp_path / "grs-15-cube.fits").write_bytes(b"fake1")
        (tmp_path / "grs-16-cube.fits").write_bytes(b"fake2")

        s3 = MagicMock()
        s3.head_object.side_effect = _NOT_FOUND
        ddb = MagicMock()
        mock_boto3.client.side_effect = lambda svc: s3 if svc == "s3" else ddb

        mock_discover.side_effect = ["run-15", "run-16"]

        item_15 = _make_run_item(run_id="run-15", survey="grs-15")
        item_16 = _make_run_item(run_id="run-16", survey="grs-16")
        ddb.get_item.side_effect = [
            {"Item": item_15},
            {"Item": item_16},
        ]

        result = CliRunner().invoke(
            main,
            [
                "pipeline-grs",
                "--input-dir",
                str(tmp_path),
                "--poll-interval",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output

        # 2 FITS uploads
        assert s3.upload_file.call_count == 2
        # 2 manifests
        assert s3.put_object.call_count == 2
        # summary mentions 2 tiles
        assert "2 tile(s)" in result.output

    @patch(MOCK_BOTO3)
    @patch("benchmarks.commands.pipeline._discover_run_id")
    @patch("benchmarks.commands.pipeline_grs.time.sleep")
    def test_with_param(
        self,
        _mock_sleep: MagicMock,
        mock_discover: MagicMock,
        mock_boto3: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--param beta=3.8 should appear in all manifests."""
        (tmp_path / "grs-15-cube.fits").write_bytes(b"fake")

        s3 = MagicMock()
        s3.head_object.side_effect = _NOT_FOUND
        ddb = MagicMock()
        mock_boto3.client.side_effect = lambda svc: s3 if svc == "s3" else ddb

        mock_discover.return_value = "run-15"

        item = _make_run_item(run_id="run-15", survey="grs-15")
        ddb.get_item.return_value = {"Item": item}

        result = CliRunner().invoke(
            main,
            [
                "pipeline-grs",
                "--input-dir",
                str(tmp_path),
                "--param",
                "beta=3.8",
                "--poll-interval",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output

        body = json.loads(s3.put_object.call_args[1]["Body"])
        assert body["params"] == {"beta": 3.8}
