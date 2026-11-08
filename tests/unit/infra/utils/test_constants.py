from infra.utils.constants import (
    CUDA_BASE_IMAGE_URI,
    HF_REPO_ID,
    NEURON_BASE_IMAGE_URI,
    VPC_CIDR,
    ZIMAGE_UPSTREAM_COMMIT,
    ZIMAGE_UPSTREAM_REPO,
)


class TestConstants:
    def test_vpc_cidr_valid_format(self) -> None:
        parts = VPC_CIDR.split("/")
        assert len(parts) == 2
        assert int(parts[1]) in range(8, 29)

    def test_hf_repo_has_org(self) -> None:
        assert "/" in HF_REPO_ID

    def test_upstream_repo_is_github(self) -> None:
        assert "github.com" in ZIMAGE_UPSTREAM_REPO
        assert ZIMAGE_UPSTREAM_REPO.endswith(".git")

    def test_upstream_commit_is_sha(self) -> None:
        assert len(ZIMAGE_UPSTREAM_COMMIT) == 40
        assert all(c in "0123456789abcdef" for c in ZIMAGE_UPSTREAM_COMMIT)

    def test_base_images_have_region_placeholder(self) -> None:
        assert "{region}" in NEURON_BASE_IMAGE_URI
        assert "{region}" in CUDA_BASE_IMAGE_URI
