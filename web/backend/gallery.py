import base64
import json
from datetime import UTC, datetime
from typing import Any

import aioboto3
from boto3.dynamodb.conditions import Key
from loguru import logger

from config import settings

_DEFAULT_USER_ID = "default"


class GalleryManager:
    def __init__(self):
        self.bucket = settings.gallery_bucket
        self.table_name = settings.gallery_table
        self.region = settings.region
        self.page_size = settings.gallery_page_size
        self._session: aioboto3.Session | None = None

    def _get_session(self) -> aioboto3.Session:
        if self._session is None:
            self._session = aioboto3.Session(region_name=self.region)
        return self._session

    async def _get_s3_client(self):
        from botocore.config import Config as BotoConfig

        return self._get_session().client("s3", config=BotoConfig(signature_version="s3v4"))

    async def _get_dynamodb_resource(self):
        return self._get_session().resource("dynamodb")

    async def list_images(
        self,
        cursor: str | None = None,
        include_deleted: bool = False,
        page_size: int | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        try:
            ddb = await self._get_dynamodb_resource()
            s3 = await self._get_s3_client()

            async with ddb as ddb_resource, s3 as s3_client:
                table = await ddb_resource.Table(self.table_name)

                query_kwargs: dict[str, Any] = {
                    "KeyConditionExpression": Key("user_id").eq(_DEFAULT_USER_ID),
                    "ScanIndexForward": False,
                    "Limit": page_size or self.page_size,
                }

                if not include_deleted:
                    from boto3.dynamodb.conditions import Attr

                    query_kwargs["FilterExpression"] = Attr("deleted").ne(True)

                if cursor:
                    try:
                        exclusive_start_key = json.loads(base64.b64decode(cursor).decode())
                        query_kwargs["ExclusiveStartKey"] = exclusive_start_key
                    except (json.JSONDecodeError, ValueError):
                        logger.warning(f"Invalid cursor: {cursor}")

                response = await table.query(**query_kwargs)

                items = response.get("Items", [])
                images = []

                for item in items:
                    image_key = item.get("image_url", f"images/{item['id']}/final.png")
                    url = await s3_client.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": self.bucket, "Key": image_key},
                        ExpiresIn=settings.gallery_presigned_url_expiry,
                    )

                    res = int(item.get("resolution", 512))
                    images.append(
                        {
                            "id": item["id"],
                            "prompt": item.get("prompt", ""),
                            "width": int(item.get("width", res)),
                            "height": int(item.get("height", res)),
                            "seed": item.get("seed"),
                            "steps": int(item.get("steps", 8)),
                            "backend": item.get("backend"),
                            "denoise_viz": item.get("denoise_viz"),
                            "elapsed_s": item.get("elapsed_s"),
                            "created_at": item.get("created_at"),
                            "deleted": item.get("deleted", False),
                            "url": url,
                        }
                    )

                next_cursor = None
                last_key = response.get("LastEvaluatedKey")
                if last_key:
                    next_cursor = base64.b64encode(json.dumps(last_key).encode()).decode()

                return images, next_cursor

        except Exception as e:
            logger.error(f"Failed to list images: {e}")
            raise

    async def get_image(self, image_id: str) -> dict[str, Any] | None:
        try:
            ddb = await self._get_dynamodb_resource()
            s3 = await self._get_s3_client()

            async with ddb as ddb_resource, s3 as s3_client:
                table = await ddb_resource.Table(self.table_name)

                response = await table.query(
                    IndexName="id-index",
                    KeyConditionExpression=Key("id").eq(image_id),
                )

                items = response.get("Items", [])
                if not items:
                    return None

                item = items[0]

                image_key = item.get("image_url", f"images/{image_id}/final.png")
                url = await s3_client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket, "Key": image_key},
                    ExpiresIn=settings.gallery_presigned_url_expiry,
                )

                frames = []
                num_steps = int(item.get("steps", 8))
                for step in range(1, num_steps + 1):
                    frame_key = f"images/{image_id}/frames/step_{step:03d}.png"
                    try:
                        frame_url = await s3_client.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": self.bucket, "Key": frame_key},
                            ExpiresIn=settings.gallery_presigned_url_expiry,
                        )
                        frames.append({"step": step, "url": frame_url})
                    except Exception:
                        break

                res = int(item.get("resolution", 512))
                return {
                    "id": image_id,
                    "prompt": item.get("prompt", ""),
                    "width": int(item.get("width", res)),
                    "height": int(item.get("height", res)),
                    "seed": item.get("seed"),
                    "steps": num_steps,
                    "created_at": item.get("created_at"),
                    "deleted": item.get("deleted", False),
                    "url": url,
                    "frames": frames,
                }

        except Exception as e:
            logger.error(f"Failed to get image {image_id}: {e}")
            raise

    async def delete_image(self, image_id: str, hard_delete: bool = False) -> bool:
        try:
            ddb = await self._get_dynamodb_resource()

            async with ddb as ddb_resource:
                table = await ddb_resource.Table(self.table_name)

                response = await table.query(
                    IndexName="id-index",
                    KeyConditionExpression=Key("id").eq(image_id),
                )

                items = response.get("Items", [])
                if not items:
                    return False

                item = items[0]
                key = {"user_id": item["user_id"], "created_at": item["created_at"]}

                if hard_delete:
                    await table.delete_item(Key=key)

                    s3 = await self._get_s3_client()
                    async with s3 as s3_client:
                        prefix = f"images/{image_id}/"
                        list_response = await s3_client.list_objects_v2(
                            Bucket=self.bucket, Prefix=prefix
                        )
                        objects = list_response.get("Contents", [])
                        if objects:
                            delete_keys = [{"Key": obj["Key"]} for obj in objects]
                            await s3_client.delete_objects(
                                Bucket=self.bucket, Delete={"Objects": delete_keys}
                            )

                    logger.info(f"Hard deleted image {image_id}")
                    return True
                else:
                    await table.update_item(
                        Key=key,
                        UpdateExpression="SET deleted = :d, deleted_at = :da",
                        ExpressionAttributeValues={
                            ":d": True,
                            ":da": datetime.now(UTC).isoformat(),
                        },
                    )
                    logger.info(f"Soft deleted image {image_id}")
                    return True

        except Exception as e:
            logger.error(f"Failed to delete image {image_id}: {e}")
            raise

    async def save_image_metadata(
        self,
        request_id: str,
        prompt: str,
        resolution: int,
        steps: int,
        seed: int | None = None,
        backend: str | None = None,
        denoise_viz: bool | None = None,
        elapsed_s: float | None = None,
    ):
        ddb = await self._get_dynamodb_resource()

        now = datetime.now(UTC).isoformat()
        item: dict[str, Any] = {
            "user_id": _DEFAULT_USER_ID,
            "created_at": now,
            "id": request_id,
            "prompt": prompt,
            "resolution": resolution,
            "steps": steps,
            "deleted": False,
            "image_url": f"images/{request_id}/final.png",
        }
        if seed is not None:
            item["seed"] = seed
        if backend is not None:
            item["backend"] = backend
        if denoise_viz is not None:
            item["denoise_viz"] = denoise_viz
        if elapsed_s is not None:
            item["elapsed_s"] = str(elapsed_s)

        try:
            async with ddb as ddb_resource:
                table = await ddb_resource.Table(self.table_name)
                await table.put_item(Item=item)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise

    async def save_image(
        self,
        request_id: str,
        image_bytes: bytes,
        prompt: str,
        resolution: int,
        steps: int,
        seed: int | None = None,
    ):
        s3 = await self._get_s3_client()

        try:
            async with s3 as s3_client:
                await s3_client.put_object(
                    Bucket=self.bucket,
                    Key=f"images/{request_id}/final.png",
                    Body=image_bytes,
                    ContentType="image/png",
                )
            logger.info(f"saved image {request_id} ({len(image_bytes)} bytes)")
        except Exception as e:
            logger.error(f"Failed to save image {request_id}: {e}")
            raise

        await self.save_image_metadata(
            request_id=request_id,
            prompt=prompt,
            resolution=resolution,
            steps=steps,
            seed=seed,
        )


gallery_manager = GalleryManager()
