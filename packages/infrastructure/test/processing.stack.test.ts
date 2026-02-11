import { Match } from 'aws-cdk-lib/assertions'

import { createProcessingTemplate } from './helpers'

describe('ProcessingStack', () => {
  const template = createProcessingTemplate()

  test('creates 2 SQS queues (main + DLQ)', () => {
    template.resourceCountIs('AWS::SQS::Queue', 2)
  })

  test('main queue has 6-minute visibility timeout and 14-day retention', () => {
    template.hasResourceProperties('AWS::SQS::Queue', {
      VisibilityTimeout: 360,
      MessageRetentionPeriod: 1209600
    })
  })

  test('DLQ has maxReceiveCount of 3', () => {
    template.hasResourceProperties('AWS::SQS::Queue', {
      RedrivePolicy: Match.objectLike({
        maxReceiveCount: 3
      })
    })
  })

  test('Worker Lambda: ARM64, 512MB, 5-minute timeout', () => {
    template.hasResourceProperties('AWS::Lambda::Function', {
      FunctionName: 'phspectra__worker',
      Architectures: ['arm64'],
      MemorySize: 512,
      Timeout: 300
    })
  })

  test('Log group with correct name and 7-day retention', () => {
    template.hasResourceProperties('AWS::Logs::LogGroup', {
      LogGroupName: '/aws/lambda/phspectra__worker',
      RetentionInDays: 7
    })
  })

  test('SQS event source mapping: batchSize 1, maxConcurrency 500', () => {
    template.hasResourceProperties('AWS::Lambda::EventSourceMapping', {
      BatchSize: 1,
      ScalingConfig: {
        MaximumConcurrency: 500
      }
    })
  })

  test('Worker has bucket read/write IAM policy', () => {
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['s3:GetObject*', 's3:GetBucket*', 's3:List*']),
            Effect: 'Allow'
          })
        ])
      }
    })

    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['s3:PutObject', 's3:Abort*']),
            Effect: 'Allow'
          })
        ])
      }
    })
  })
})
