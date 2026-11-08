from __future__ import annotations

import asyncio
import base64
import json
import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("ZIMAGE_GALLERY_TABLE", "test-gallery-table")

from gallery import GalleryManager


def _run(coro):
    return asyncio.run(coro)


def _make_ddb_mock():
    table = AsyncMock()
    resource = AsyncMock()
    resource.Table = AsyncMock(return_value=table)

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resource)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, table


def _make_s3_mock():
    client = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, client


@pytest.fixture
def gallery():
    with patch("gallery.settings") as mock_settings:
        mock_settings.gallery_bucket = "test-bucket"
        mock_settings.gallery_table = "test-gallery-table"
        mock_settings.region = "us-east-1"
        mock_settings.gallery_page_size = 20
        mock_settings.gallery_presigned_url_expiry = 3600
        mgr = GalleryManager()
    return mgr


class TestListImages:
    def test_returns_images_from_dynamodb(self, gallery):
        ddb_ctx, table = _make_ddb_mock()
        s3_ctx, s3_client = _make_s3_mock()

        table.query.return_value = {
            "Items": [
                {
                    "user_id": "default",
                    "created_at": "2024-01-01T00:00:00",
                    "id": "req-001",
                    "prompt": "sunset over ocean",
                    "resolution": 512,
                    "seed": 42,
                    "steps": 8,
                    "backend": "gpu",
                    "deleted": False,
                    "image_url": "images/req-001/final.png",
                }
            ],
        }
        s3_client.generate_presigned_url.return_value = "https://s3.example.com/signed"

        with (
            patch.object(gallery, "_get_dynamodb_resource", return_value=ddb_ctx),
            patch.object(gallery, "_get_s3_client", return_value=s3_ctx),
        ):
            images, next_cursor = _run(gallery.list_images())

        assert len(images) == 1
        assert images[0]["id"] == "req-001"
        assert images[0]["prompt"] == "sunset over ocean"
        assert images[0]["url"] == "https://s3.example.com/signed"
        assert next_cursor is None

    def test_skips_deleted_images(self, gallery):
        ddb_ctx, table = _make_ddb_mock()
        s3_ctx, _ = _make_s3_mock()

        table.query.return_value = {"Items": []}

        with (
            patch.object(gallery, "_get_dynamodb_resource", return_value=ddb_ctx),
            patch.object(gallery, "_get_s3_client", return_value=s3_ctx),
        ):
            images, _ = _run(gallery.list_images(include_deleted=False))

        assert len(images) == 0

    def test_includes_deleted_when_requested(self, gallery):
        ddb_ctx, table = _make_ddb_mock()
        s3_ctx, s3_client = _make_s3_mock()

        table.query.return_value = {
            "Items": [
                {
                    "user_id": "default",
                    "created_at": "2024-01-01T00:00:00",
                    "id": "del-001",
                    "prompt": "deleted image",
                    "resolution": 512,
                    "steps": 8,
                    "deleted": True,
                    "image_url": "images/del-001/final.png",
                }
            ],
        }
        s3_client.generate_presigned_url.return_value = "https://s3.example.com/signed"

        with (
            patch.object(gallery, "_get_dynamodb_resource", return_value=ddb_ctx),
            patch.object(gallery, "_get_s3_client", return_value=s3_ctx),
        ):
            images, _ = _run(gallery.list_images(include_deleted=True))

        assert len(images) == 1
        assert images[0]["deleted"] is True

    def test_returns_cursor_when_last_evaluated_key_present(self, gallery):
        ddb_ctx, table = _make_ddb_mock()
        s3_ctx, s3_client = _make_s3_mock()

        last_key = {"user_id": "default", "created_at": "2024-01-01T00:00:00"}
        table.query.return_value = {
            "Items": [
                {
                    "user_id": "default",
                    "created_at": "2024-01-01T00:00:00",
                    "id": "req-050",
                    "prompt": "test",
                    "resolution": 768,
                    "steps": 8,
                    "deleted": False,
                    "image_url": "images/req-050/final.png",
                }
            ],
            "LastEvaluatedKey": last_key,
        }
        s3_client.generate_presigned_url.return_value = "https://s3.example.com/signed"

        with (
            patch.object(gallery, "_get_dynamodb_resource", return_value=ddb_ctx),
            patch.object(gallery, "_get_s3_client", return_value=s3_ctx),
        ):
            images, next_cursor = _run(gallery.list_images())

        assert len(images) == 1
        assert next_cursor is not None
        decoded = json.loads(base64.b64decode(next_cursor).decode())
        assert decoded == last_key

    def test_returns_none_cursor_when_no_more_pages(self, gallery):
        ddb_ctx, table = _make_ddb_mock()
        s3_ctx, s3_client = _make_s3_mock()

        table.query.return_value = {
            "Items": [
                {
                    "user_id": "default",
                    "created_at": "2026-01-01T00:00:00",
                    "id": "req-050",
                    "prompt": "test",
                    "resolution": 768,
                    "steps": 8,
                    "deleted": False,
                    "image_url": "images/req-050/final.png",
                }
            ],
        }
        s3_client.generate_presigned_url.return_value = "https://s3.example.com/signed"

        with (
            patch.object(gallery, "_get_dynamodb_resource", return_value=ddb_ctx),
            patch.object(gallery, "_get_s3_client", return_value=s3_ctx),
        ):
            images, next_cursor = _run(gallery.list_images())

        assert len(images) == 1
        assert next_cursor is None

    def test_passes_cursor_as_exclusive_start_key(self, gallery):
        ddb_ctx, table = _make_ddb_mock()
        s3_ctx, s3_client = _make_s3_mock()

        last_key = {"user_id": "default", "created_at": "2024-01-01T00:00:00"}
        cursor = base64.b64encode(json.dumps(last_key).encode()).decode()

        table.query.return_value = {"Items": []}

        with (
            patch.object(gallery, "_get_dynamodb_resource", return_value=ddb_ctx),
            patch.object(gallery, "_get_s3_client", return_value=s3_ctx),
        ):
            _run(gallery.list_images(cursor=cursor))

        call_kwargs = table.query.call_args[1]
        assert call_kwargs["ExclusiveStartKey"] == last_key


class TestGetImage:
    def test_returns_image_with_presigned_url(self, gallery):
        ddb_ctx, table = _make_ddb_mock()
        s3_ctx, s3_client = _make_s3_mock()

        table.query.return_value = {
            "Items": [
                {
                    "user_id": "default",
                    "created_at": "2024-06-01T12:00:00",
                    "id": "img-001",
                    "prompt": "a horse",
                    "resolution": 768,
                    "seed": 99,
                    "steps": 8,
                    "deleted": False,
                    "image_url": "images/img-001/final.png",
                }
            ],
        }
        s3_client.generate_presigned_url.return_value = "https://s3.example.com/img.png"

        with (
            patch.object(gallery, "_get_dynamodb_resource", return_value=ddb_ctx),
            patch.object(gallery, "_get_s3_client", return_value=s3_ctx),
        ):
            result = _run(gallery.get_image("img-001"))

        assert result is not None
        assert result["id"] == "img-001"
        assert result["prompt"] == "a horse"
        assert result["url"] == "https://s3.example.com/img.png"
        assert "frames" in result

    def test_returns_none_for_missing_image(self, gallery):
        ddb_ctx, table = _make_ddb_mock()
        s3_ctx, _ = _make_s3_mock()

        table.query.return_value = {"Items": []}

        with (
            patch.object(gallery, "_get_dynamodb_resource", return_value=ddb_ctx),
            patch.object(gallery, "_get_s3_client", return_value=s3_ctx),
        ):
            result = _run(gallery.get_image("nonexistent"))

        assert result is None


class TestDeleteImage:
    def test_soft_delete_updates_dynamodb_item(self, gallery):
        ddb_ctx, table = _make_ddb_mock()

        table.query.return_value = {
            "Items": [
                {
                    "user_id": "default",
                    "created_at": "2024-01-01T00:00:00",
                    "id": "del-001",
                    "prompt": "to delete",
                    "deleted": False,
                }
            ],
        }
        table.update_item.return_value = {}

        with patch.object(gallery, "_get_dynamodb_resource", return_value=ddb_ctx):
            result = _run(gallery.delete_image("del-001", hard_delete=False))

        assert result is True
        table.update_item.assert_called_once()
        call_kwargs = table.update_item.call_args[1]
        assert call_kwargs["Key"] == {"user_id": "default", "created_at": "2024-01-01T00:00:00"}
        assert ":d" in call_kwargs["ExpressionAttributeValues"]
        assert call_kwargs["ExpressionAttributeValues"][":d"] is True

    def test_hard_delete_removes_dynamodb_and_s3(self, gallery):
        ddb_ctx, table = _make_ddb_mock()
        s3_ctx, s3_client = _make_s3_mock()

        table.query.return_value = {
            "Items": [
                {
                    "user_id": "default",
                    "created_at": "2024-01-01T00:00:00",
                    "id": "del-002",
                    "prompt": "to hard delete",
                    "deleted": False,
                }
            ],
        }
        table.delete_item.return_value = {}
        s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "images/del-002/final.png"},
            ]
        }
        s3_client.delete_objects.return_value = {}

        with (
            patch.object(gallery, "_get_dynamodb_resource", return_value=ddb_ctx),
            patch.object(gallery, "_get_s3_client", return_value=s3_ctx),
        ):
            result = _run(gallery.delete_image("del-002", hard_delete=True))

        assert result is True
        table.delete_item.assert_called_once()
        s3_client.delete_objects.assert_called_once()

    def test_returns_false_for_missing_image(self, gallery):
        ddb_ctx, table = _make_ddb_mock()

        table.query.return_value = {"Items": []}

        with patch.object(gallery, "_get_dynamodb_resource", return_value=ddb_ctx):
            result = _run(gallery.delete_image("nonexistent", hard_delete=False))

        assert result is False


class TestSaveImageMetadata:
    def test_puts_item_to_dynamodb(self, gallery):
        ddb_ctx, table = _make_ddb_mock()
        table.put_item.return_value = {}

        with patch.object(gallery, "_get_dynamodb_resource", return_value=ddb_ctx):
            _run(
                gallery.save_image_metadata(
                    request_id="new-001",
                    prompt="a sunset",
                    resolution=512,
                    steps=8,
                    seed=42,
                    backend="gpu",
                )
            )

        table.put_item.assert_called_once()
        item = table.put_item.call_args[1]["Item"]
        assert item["id"] == "new-001"
        assert item["prompt"] == "a sunset"
        assert item["user_id"] == "default"
        assert item["resolution"] == 512
        assert item["seed"] == 42
        assert item["backend"] == "gpu"
        assert item["deleted"] is False
