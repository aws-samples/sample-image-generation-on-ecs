from __future__ import annotations

import boto3
from loguru import logger


def resolve_digest(repo_name: str, tag: str = "latest", region: str | None = None) -> str:
    client = boto3.client("ecr", region_name=region)
    try:
        resp = client.batch_get_image(
            repositoryName=repo_name,
            imageIds=[{"imageTag": tag}],
            acceptedMediaTypes=["application/vnd.docker.distribution.manifest.v2+json"],
        )
        images = resp.get("images", [])
        if not images:
            raise ValueError(f"No image found for {repo_name}:{tag}")
        digest = images[0]["imageId"]["imageDigest"]
        logger.debug("Resolved {}:{} → {}", repo_name, tag, digest)
        return digest
    except client.exceptions.ImageNotFoundException as e:
        raise ValueError(f"Image not found: {repo_name}:{tag}") from e
