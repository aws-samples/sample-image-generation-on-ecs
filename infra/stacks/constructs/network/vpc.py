from __future__ import annotations

from aws_cdk import aws_ec2 as ec2
from constructs import Construct

from infra.utils.constants import VPC_CIDR


class Vpc(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        nat_count: int,
        account: str,
        az_count: int = 3,
    ) -> None:
        super().__init__(scope, construct_id)

        self.vpc = ec2.Vpc(
            self,
            "Vpc",
            ip_addresses=ec2.IpAddresses.cidr(VPC_CIDR),
            max_azs=az_count,
            nat_gateways=nat_count,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=23,
                ),
            ],
            restrict_default_security_group=True,
        )

        self.private_subnets = self.vpc.private_subnets
        self.public_subnets = self.vpc.public_subnets

        ec2.FlowLog(
            self,
            "FlowLogs",
            resource_type=ec2.FlowLogResourceType.from_vpc(self.vpc),
            traffic_type=ec2.FlowLogTrafficType.ALL,
        )

        self.vpc.add_gateway_endpoint(
            "S3Gw",
            service=ec2.GatewayVpcEndpointAwsService.S3,
            subnets=[ec2.SubnetSelection(subnets=self.private_subnets)],
        )

        self.endpoint_sg = ec2.SecurityGroup(
            self,
            "EndpointSg",
            vpc=self.vpc,
            allow_all_outbound=True,
            description="ECS inference + compile processing jobs",
        )
