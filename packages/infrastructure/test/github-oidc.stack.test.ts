import { Match } from 'aws-cdk-lib/assertions'

import { createGitHubOIDCTemplate } from './helpers'

describe('GitHubOIDCStack', () => {
  const template = createGitHubOIDCTemplate()

  test('creates an OIDC provider for GitHub Actions', () => {
    template.hasResourceProperties('Custom::AWSCDKOpenIdConnectProvider', {
      Url: 'https://token.actions.githubusercontent.com',
      ClientIDList: ['sts.amazonaws.com']
    })
  })

  test('IAM role has correct name', () => {
    template.hasResourceProperties('AWS::IAM::Role', {
      RoleName: 'PHSpectra-GitHubActions-Role',
      MaxSessionDuration: 3600
    })
  })

  test('trust policy requires correct audience', () => {
    template.hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Condition: Match.objectLike({
              StringEquals: Match.objectLike({
                'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com'
              })
            })
          })
        ])
      }
    })
  })

  test('trust policy allows development and production by default', () => {
    template.hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Condition: Match.objectLike({
              StringLike: Match.objectLike({
                'token.actions.githubusercontent.com:sub': [
                  'repo:caverac/phspectra:environment:development',
                  'repo:caverac/phspectra:environment:production'
                ]
              })
            })
          })
        ])
      }
    })
  })

  test('has sts:AssumeRole policy for CDK bootstrap roles', () => {
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 'sts:AssumeRole',
            Effect: 'Allow',
            Resource: 'arn:aws:iam::*:role/cdk-*'
          })
        ])
      }
    })
  })

  test('creates SSM parameter with role ARN', () => {
    template.hasResourceProperties('AWS::SSM::Parameter', {
      Name: '/phspectra/github-actions/role-arn',
      Type: 'String'
    })
  })

  test('custom environments override defaults', () => {
    const customTemplate = createGitHubOIDCTemplate('caverac/phspectra', ['production'])
    customTemplate.hasResourceProperties('AWS::IAM::Role', {
      AssumeRolePolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Condition: Match.objectLike({
              StringLike: Match.objectLike({
                'token.actions.githubusercontent.com:sub': [
                  'repo:caverac/phspectra:environment:production'
                ]
              })
            })
          })
        ])
      }
    })
  })
})

describe('GitHubOIDCStack (existing provider)', () => {
  const existingArn = 'arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com'
  const template = createGitHubOIDCTemplate('caverac/phspectra', undefined, existingArn)

  test('does not create a new OIDC provider', () => {
    template.resourceCountIs('Custom::AWSCDKOpenIdConnectProvider', 0)
  })

  test('IAM role still references the provider in trust policy', () => {
    template.hasResourceProperties('AWS::IAM::Role', {
      RoleName: 'PHSpectra-GitHubActions-Role',
      AssumeRolePolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Condition: Match.objectLike({
              StringEquals: Match.objectLike({
                'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com'
              })
            }),
            Principal: Match.objectLike({
              Federated: existingArn
            })
          })
        ])
      }
    })
  })
})
