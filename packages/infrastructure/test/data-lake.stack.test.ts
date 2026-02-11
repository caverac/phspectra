import { Match } from 'aws-cdk-lib/assertions'

import { createDataLakeTemplate } from './helpers'

describe('DataLakeStack', () => {
  const template = createDataLakeTemplate('development')

  test('creates one S3 bucket', () => {
    template.resourceCountIs('AWS::S3::Bucket', 1)
  })

  test('enables EventBridge notifications', () => {
    // CDK uses a custom resource to configure EventBridge on S3 buckets
    template.resourceCountIs('Custom::S3BucketNotifications', 1)
  })

  test('blocks public access', () => {
    template.hasResourceProperties('AWS::S3::Bucket', {
      PublicAccessBlockConfiguration: {
        BlockPublicAcls: true,
        BlockPublicPolicy: true,
        IgnorePublicAcls: true,
        RestrictPublicBuckets: true
      }
    })
  })

  test('enforces SSL via bucket policy', () => {
    template.hasResourceProperties('AWS::S3::BucketPolicy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Effect: 'Deny',
            Condition: {
              Bool: { 'aws:SecureTransport': 'false' }
            }
          })
        ])
      }
    })
  })

  test('has chunks/ lifecycle rule with 7-day expiration', () => {
    template.hasResourceProperties('AWS::S3::Bucket', {
      LifecycleConfiguration: {
        Rules: Match.arrayWith([
          Match.objectLike({
            Prefix: 'chunks/',
            ExpirationInDays: 7,
            Status: 'Enabled'
          })
        ])
      }
    })
  })

  test('has athena-results/ lifecycle rule with 7-day expiration', () => {
    template.hasResourceProperties('AWS::S3::Bucket', {
      LifecycleConfiguration: {
        Rules: Match.arrayWith([
          Match.objectLike({
            Prefix: 'athena-results/',
            ExpirationInDays: 7,
            Status: 'Enabled'
          })
        ])
      }
    })
  })

  test('production uses RETAIN removal policy', () => {
    const prodTemplate = createDataLakeTemplate('production')
    prodTemplate.hasResource('AWS::S3::Bucket', {
      DeletionPolicy: 'Retain',
      UpdateReplacePolicy: 'Retain'
    })
  })

  test('development uses DESTROY removal policy', () => {
    template.hasResource('AWS::S3::Bucket', {
      DeletionPolicy: 'Delete'
    })
  })
})
