from __future__ import annotations

from aws_cdk import Duration
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_wafv2 as waf
from cdk_nag import NagSuppressions
from constructs import Construct


class WebCdnV2(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        prefix: str,
        static_bucket: s3.IBucket,
        api_url: str,
        access_logs_bucket: s3.IBucket | None = None,
    ) -> None:
        super().__init__(scope, construct_id)

        api_domain = api_url.replace("https://", "").rstrip("/")

        s3_origin = origins.S3BucketOrigin.with_origin_access_control(
            static_bucket,
        )

        api_origin = origins.HttpOrigin(
            api_domain,
            protocol_policy=cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
        )

        api_origin_request_policy = cloudfront.OriginRequestPolicy.ALL_VIEWER_EXCEPT_HOST_HEADER

        web_acl = waf.CfnWebACL(
            self,
            "WebACL",
            name=f"{prefix}-cdn-waf",
            scope="CLOUDFRONT",
            default_action=waf.CfnWebACL.DefaultActionProperty(allow={}),
            visibility_config=waf.CfnWebACL.VisibilityConfigProperty(
                cloud_watch_metrics_enabled=True,
                metric_name=f"{prefix}-cdn-waf",
                sampled_requests_enabled=True,
            ),
            rules=[
                waf.CfnWebACL.RuleProperty(
                    name="RateLimit",
                    priority=1,
                    action=waf.CfnWebACL.RuleActionProperty(block={}),
                    statement=waf.CfnWebACL.StatementProperty(
                        rate_based_statement=waf.CfnWebACL.RateBasedStatementProperty(
                            limit=1000,
                            aggregate_key_type="IP",
                        ),
                    ),
                    visibility_config=waf.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="RateLimit",
                        sampled_requests_enabled=True,
                    ),
                ),
                waf.CfnWebACL.RuleProperty(
                    name="AWSManagedCommonRules",
                    priority=2,
                    override_action=waf.CfnWebACL.OverrideActionProperty(none={}),
                    statement=waf.CfnWebACL.StatementProperty(
                        managed_rule_group_statement=waf.CfnWebACL.ManagedRuleGroupStatementProperty(
                            vendor_name="AWS",
                            name="AWSManagedRulesCommonRuleSet",
                        ),
                    ),
                    visibility_config=waf.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="CommonRules",
                        sampled_requests_enabled=True,
                    ),
                ),
                waf.CfnWebACL.RuleProperty(
                    name="AWSManagedSQLi",
                    priority=3,
                    override_action=waf.CfnWebACL.OverrideActionProperty(none={}),
                    statement=waf.CfnWebACL.StatementProperty(
                        managed_rule_group_statement=waf.CfnWebACL.ManagedRuleGroupStatementProperty(
                            vendor_name="AWS",
                            name="AWSManagedRulesSQLiRuleSet",
                        ),
                    ),
                    visibility_config=waf.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="SQLi",
                        sampled_requests_enabled=True,
                    ),
                ),
                waf.CfnWebACL.RuleProperty(
                    name="AWSManagedKnownBadInputs",
                    priority=4,
                    override_action=waf.CfnWebACL.OverrideActionProperty(none={}),
                    statement=waf.CfnWebACL.StatementProperty(
                        managed_rule_group_statement=waf.CfnWebACL.ManagedRuleGroupStatementProperty(
                            vendor_name="AWS",
                            name="AWSManagedRulesKnownBadInputsRuleSet",
                        ),
                    ),
                    visibility_config=waf.CfnWebACL.VisibilityConfigProperty(
                        cloud_watch_metrics_enabled=True,
                        metric_name="KnownBadInputs",
                        sampled_requests_enabled=True,
                    ),
                ),
            ],
        )

        security_headers = cloudfront.ResponseHeadersPolicy(
            self,
            "SecurityHeaders",
            response_headers_policy_name=f"{prefix}-web-security-headers",
            security_headers_behavior=cloudfront.ResponseSecurityHeadersBehavior(
                content_type_options=cloudfront.ResponseHeadersContentTypeOptions(override=True),
                frame_options=cloudfront.ResponseHeadersFrameOptions(
                    frame_option=cloudfront.HeadersFrameOption.DENY, override=True
                ),
                referrer_policy=cloudfront.ResponseHeadersReferrerPolicy(
                    referrer_policy=cloudfront.HeadersReferrerPolicy.STRICT_ORIGIN_WHEN_CROSS_ORIGIN,
                    override=True,
                ),
                strict_transport_security=cloudfront.ResponseHeadersStrictTransportSecurity(
                    access_control_max_age=Duration.days(365),
                    include_subdomains=True,
                    override=True,
                ),
                xss_protection=cloudfront.ResponseHeadersXSSProtection(
                    protection=True, mode_block=True, override=True
                ),
            ),
        )

        dist_kwargs: dict = {
            "comment": f"{prefix} web app (S3 + API GW)",
            "default_behavior": cloudfront.BehaviorOptions(
                origin=s3_origin,
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
                origin_request_policy=cloudfront.OriginRequestPolicy.CORS_S3_ORIGIN,
                response_headers_policy=security_headers,
            ),
            "additional_behaviors": {
                "/api/*": cloudfront.BehaviorOptions(
                    origin=api_origin,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    origin_request_policy=api_origin_request_policy,
                    response_headers_policy=security_headers,
                ),
                "/admin/api/*": cloudfront.BehaviorOptions(
                    origin=api_origin,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    origin_request_policy=api_origin_request_policy,
                    response_headers_policy=security_headers,
                ),
                "/auth/*": cloudfront.BehaviorOptions(
                    origin=api_origin,
                    viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                    allowed_methods=cloudfront.AllowedMethods.ALLOW_ALL,
                    cache_policy=cloudfront.CachePolicy.CACHING_DISABLED,
                    origin_request_policy=api_origin_request_policy,
                    response_headers_policy=security_headers,
                ),
            },
            "default_root_object": "index.html",
            "error_responses": [
                cloudfront.ErrorResponse(
                    http_status=403,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=Duration.seconds(0),
                ),
                cloudfront.ErrorResponse(
                    http_status=404,
                    response_http_status=200,
                    response_page_path="/index.html",
                    ttl=Duration.seconds(0),
                ),
            ],
            "web_acl_id": web_acl.attr_arn,
            "http_version": cloudfront.HttpVersion.HTTP2_AND_3,
            "price_class": cloudfront.PriceClass.PRICE_CLASS_100,
        }

        if access_logs_bucket:
            dist_kwargs["log_bucket"] = access_logs_bucket
            dist_kwargs["log_file_prefix"] = "cloudfront/"

        self.distribution = cloudfront.Distribution(self, "Distribution", **dist_kwargs)

        self.url = f"https://{self.distribution.distribution_domain_name}"

        suppressions = [
            {
                "id": "AwsSolutions-CFR1",
                "reason": "Geo restriction not needed for internal dev tool.",
            },
            {
                "id": "AwsSolutions-CFR4",
                "reason": "Using default CloudFront TLS certificate "
                "(*.cloudfront.net). Custom domain not required.",
            },
        ]
        if not access_logs_bucket:
            suppressions.append(
                {
                    "id": "AwsSolutions-CFR3",
                    "reason": "Access logging disabled — no log bucket provided.",
                }
            )
        NagSuppressions.add_resource_suppressions(
            self.distribution,
            suppressions,
            apply_to_children=True,
        )
