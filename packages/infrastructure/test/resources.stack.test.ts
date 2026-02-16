import { Match } from 'aws-cdk-lib/assertions'

import { createResourcesTemplate } from './helpers'

describe('ResourcesStack', () => {
  const template = createResourcesTemplate()

  test('creates one S3 bucket', () => {
    template.resourceCountIs('AWS::S3::Bucket', 1)
  })

  test('bucket name includes environment', () => {
    template.hasResourceProperties('AWS::S3::Bucket', {
      BucketName: 'phspectra-development-resources'
    })
  })

  test('allows public access (no block)', () => {
    template.hasResourceProperties('AWS::S3::Bucket', {
      PublicAccessBlockConfiguration: {
        BlockPublicAcls: false,
        BlockPublicPolicy: false,
        IgnorePublicAcls: false,
        RestrictPublicBuckets: false
      }
    })
  })

  test('uses S3 managed encryption', () => {
    template.hasResourceProperties('AWS::S3::Bucket', {
      BucketEncryption: {
        ServerSideEncryptionConfiguration: [
          {
            ServerSideEncryptionByDefault: {
              SSEAlgorithm: 'AES256'
            }
          }
        ]
      }
    })
  })

  test('grants public s3:GetObject via bucket policy', () => {
    template.hasResourceProperties('AWS::S3::BucketPolicy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: 's3:GetObject',
            Effect: 'Allow',
            Principal: { AWS: '*' }
          })
        ])
      }
    })
  })

  test('development uses DESTROY removal policy', () => {
    template.hasResource('AWS::S3::Bucket', {
      DeletionPolicy: 'Delete'
    })
  })

  test('production uses RETAIN removal policy', () => {
    const prodTemplate = createResourcesTemplate('production')
    prodTemplate.hasResource('AWS::S3::Bucket', {
      DeletionPolicy: 'Retain',
      UpdateReplacePolicy: 'Retain'
    })
  })
})
