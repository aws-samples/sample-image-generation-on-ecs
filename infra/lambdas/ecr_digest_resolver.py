from __future__ import annotations

import logging
import os

import boto3

LOGGER = logging.getLogger()
LOGGER.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def _resolve_digest(ecr, repo: str, tag: str, registry_id: str | None):
    kwargs = {"repositoryName": repo, "imageIds": [{"imageTag": tag}]}
    if registry_id:
        kwargs["registryId"] = registry_id
    resp = ecr.batch_get_image(**kwargs)
    images = resp.get("images", [])
    failures = resp.get("failures", [])
    if not images:
        raise RuntimeError(f"ECR image not found: repo={repo} tag={tag} failures={failures}")
    digest = images[0]["imageId"]["imageDigest"]
    acct = registry_id or boto3.client("sts").get_caller_identity()["Account"]
    region = os.environ["AWS_REGION"]
    uri = f"{acct}.dkr.ecr.{region}.amazonaws.com/{repo}@{digest}"
    return digest, uri


def handler(event, context):
    LOGGER.info("event=%s", event)
    ecr = boto3.client("ecr")
    props = event.get("ResourceProperties", {})
    repo = props["RepositoryName"]
    tag = props.get("ImageTag", "latest")
    registry_id = props.get("RegistryId") or None
    request_type = event["RequestType"]

    if request_type == "Delete":
        return {
            "PhysicalResourceId": event.get("PhysicalResourceId", "ecr-digest-deleted"),
            "Data": {},
        }

    digest, uri = _resolve_digest(ecr, repo, tag, registry_id)

    LOGGER.info("resolved %s:%s -> %s", repo, tag, uri)
    return {
        "PhysicalResourceId": uri,
        "Data": {"ImageUri": uri, "Digest": digest},
    }
