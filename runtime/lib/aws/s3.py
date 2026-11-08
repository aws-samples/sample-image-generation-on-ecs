from __future__ import annotations

import boto3
from loguru import logger


def empty_versioned_bucket(bucket_name: str) -> int:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_object_versions")
    deleted = 0

    for page in paginator.paginate(Bucket=bucket_name, PaginationConfig={"PageSize": 1000}):
        objs: list[dict] = []
        for v in page.get("Versions") or []:
            objs.append({"Key": v["Key"], "VersionId": v["VersionId"]})
        for m in page.get("DeleteMarkers") or []:
            objs.append({"Key": m["Key"], "VersionId": m["VersionId"]})

        for i in range(0, len(objs), 1000):
            chunk = objs[i : i + 1000]
            if chunk:
                s3.delete_objects(Bucket=bucket_name, Delete={"Objects": chunk, "Quiet": True})
                deleted += len(chunk)

    logger.info("Deleted {} objects/versions from {}", deleted, bucket_name)
    return deleted
