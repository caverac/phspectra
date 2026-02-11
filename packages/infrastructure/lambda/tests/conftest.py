"""Shared fixtures for Lambda handler tests."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Environment variables required at module‐level by both handlers
# ---------------------------------------------------------------------------
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("BUCKET_NAME", "test-bucket")
os.environ.setdefault("QUEUE_URL", "https://sqs.us-east-1.amazonaws.com/123456789012/test-queue")

LAMBDA_DIR = Path(__file__).resolve().parent.parent


def _load_handler(module_name: str, handler_dir: str) -> Any:
    """Import a ``handler.py`` under a distinct ``sys.modules`` name.

    Parameters
    ----------
    module_name : str
        Name to register in ``sys.modules`` (e.g. ``"splitter_handler"``).
    handler_dir : str
        Subdirectory name under ``lambda/`` (e.g. ``"splitter"``).

    Returns
    -------
    module
        The imported handler module.
    """
    handler_path = LAMBDA_DIR / handler_dir / "handler.py"
    spec = importlib.util.spec_from_file_location(module_name, handler_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def splitter() -> Any:
    """Load the splitter handler module."""
    return _load_handler("splitter_handler", "splitter")


@pytest.fixture(scope="session")
def worker() -> Any:
    """Load the worker handler module."""
    return _load_handler("worker_handler", "worker")


@pytest.fixture()
def lambda_context() -> MagicMock:
    """Return a mock ``LambdaContext``."""
    ctx = MagicMock()
    ctx.function_name = "test-function"
    ctx.memory_limit_in_mb = 128
    ctx.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test"
    ctx.aws_request_id = "test-request-id"
    return ctx


@pytest.fixture()
def eventbridge_event() -> Any:
    """Factory that builds a valid EventBridge S3 Object Created event."""

    def _build(key: str) -> dict[str, object]:
        return {
            "version": "0",
            "id": "abc-123",
            "detail-type": "Object Created",
            "source": "aws.s3",
            "account": "123456789012",
            "time": "2024-01-01T00:00:00Z",
            "region": "us-east-1",
            "resources": [],
            "detail": {
                "version": "0",
                "bucket": {"name": "test-bucket"},
                "object": {"key": key, "size": 1024},
                "request-id": "req-123",
                "requester": "123456789012",
                "source-ip-address": "127.0.0.1",
                "reason": "PutObject",
            },
        }

    return _build


@pytest.fixture()
def sqs_event() -> Any:
    """Factory that builds a valid SQS event with one record."""

    def _build(body: dict[str, object]) -> dict[str, object]:
        return {
            "Records": [
                {
                    "messageId": "msg-001",
                    "receiptHandle": "handle-001",
                    "body": json.dumps(body),
                    "attributes": {
                        "ApproximateReceiveCount": "1",
                        "SentTimestamp": "1704067200000",
                        "SenderId": "123456789012",
                        "ApproximateFirstReceiveTimestamp": "1704067200000",
                    },
                    "messageAttributes": {},
                    "md5OfBody": "abc",
                    "eventSource": "aws:sqs",
                    "eventSourceARN": "arn:aws:sqs:us-east-1:123456789012:test-queue",
                    "awsRegion": "us-east-1",
                }
            ]
        }

    return _build
