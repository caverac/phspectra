import * as cdk from 'aws-cdk-lib'
import type { Construct } from 'constructs'
import { DeploymentEnvironment } from 'utils/types'

export interface GitHubOIDCStackProps extends cdk.StackProps {
  /**
   * GitHub repository in format "owner/repo"
   * @example "caverac/phspectra"
   */
  readonly githubRepo: string

  /**
   * GitHub environments allowed to assume the role.
   * @default ["development", "production"]
   */
  readonly allowedEnvironments?: DeploymentEnvironment[]

  /**
   * ARN of an existing GitHub OIDC provider in the account.
   * When set, the stack imports the provider instead of creating one
   * (AWS allows only one provider per URL per account).
   */
  readonly existingProviderArn?: string
}

/**
 * Creates OIDC provider and IAM role for GitHub Actions deployments.
 *
 * Enables passwordless authentication from GitHub Actions to AWS
 * using OpenID Connect (OIDC) - no long-lived credentials needed.
 */
export class GitHubOIDCStack extends cdk.Stack {
  public readonly role: cdk.aws_iam.Role
  public readonly roleArn: string

  constructor(scope: Construct, id: string, props: GitHubOIDCStackProps) {
    super(scope, id, props)

    const {
      githubRepo,
      allowedEnvironments = ['development', 'production'],
      existingProviderArn
    } = props

    const providerArn = existingProviderArn
      ? existingProviderArn
      : new cdk.aws_iam.OpenIdConnectProvider(this, 'GitHubOIDCProvider', {
          url: 'https://token.actions.githubusercontent.com',
          clientIds: ['sts.amazonaws.com'],
          thumbprints: [
            '6938fd4d98bab03faadb97b34396831e3780aea1',
            '1c58a3a8518e8759bf075b76b750d4f2df264fcd'
          ]
        }).openIdConnectProviderArn

    const subjectConditions = allowedEnvironments.map(
      (env) => `repo:${githubRepo}:environment:${env}`
    )

    this.role = new cdk.aws_iam.Role(this, 'GitHubActionsRole', {
      roleName: 'PHSpectra-GitHubActions-Role',
      description: 'Role assumed by GitHub Actions for PHSpectra deployments',
      maxSessionDuration: cdk.Duration.hours(1),
      assumedBy: new cdk.aws_iam.WebIdentityPrincipal(providerArn, {
        StringEquals: {
          'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com'
        },
        StringLike: {
          'token.actions.githubusercontent.com:sub': subjectConditions
        }
      })
    })

    this.role.addToPolicy(
      new cdk.aws_iam.PolicyStatement({
        sid: 'AssumeCDKBootstrapRoles',
        effect: cdk.aws_iam.Effect.ALLOW,
        actions: ['sts:AssumeRole'],
        resources: ['arn:aws:iam::*:role/cdk-*']
      })
    )

    this.roleArn = this.role.roleArn

    new cdk.aws_ssm.StringParameter(this, 'GitHubActionsRoleArnParam', {
      parameterName: '/phspectra/github-actions/role-arn',
      description: 'IAM role ARN for GitHub Actions OIDC authentication',
      stringValue: this.roleArn
    })
  }
}
