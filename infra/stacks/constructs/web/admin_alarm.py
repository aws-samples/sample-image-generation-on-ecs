from __future__ import annotations

from aws_cdk import Duration
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cloudwatch_actions
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_sns as sns
from constructs import Construct


class AdminErrorRateAlarm(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        prefix: str,
        api_lambda: lambda_.IFunction,
        findings_topic: sns.ITopic,
        threshold: int = 20,
    ) -> None:
        super().__init__(scope, construct_id)

        self.alarm = cloudwatch.Alarm(
            self,
            "Alarm",
            alarm_name=f"{prefix}-web-admin-403-rate",
            alarm_description=("Spike in /admin/api/* errors — possible unauthorized access"),
            metric=cloudwatch.Metric(
                namespace="AWS/Lambda",
                metric_name="Errors",
                dimensions_map={"FunctionName": api_lambda.function_name},
                statistic="Sum",
                period=Duration.minutes(5),
            ),
            threshold=threshold,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )
        self.alarm.add_alarm_action(cloudwatch_actions.SnsAction(findings_topic))
