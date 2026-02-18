import { Match } from 'aws-cdk-lib/assertions'

import { createProcessingTemplate } from './helpers'

describe('ProcessingStack', () => {
  const template = createProcessingTemplate()

  test('creates 4 SQS queues (main + DLQ + slow + slow DLQ)', () => {
    template.resourceCountIs('AWS::SQS::Queue', 4)
  })

  test('main queue has 16-minute visibility timeout and 14-day retention', () => {
    template.hasResourceProperties('AWS::SQS::Queue', {
      VisibilityTimeout: 960,
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

  test('Worker Lambda: ARM64, 1024MB, 15-minute timeout', () => {
    template.hasResourceProperties('AWS::Lambda::Function', {
      FunctionName: 'phspectra__worker',
      Architectures: ['arm64'],
      MemorySize: 1024,
      Timeout: 900
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

  test('DynamoDB runs table with PAY_PER_REQUEST billing and composite key', () => {
    template.hasResourceProperties('AWS::DynamoDB::Table', {
      TableName: 'phspectra-runs',
      BillingMode: 'PAY_PER_REQUEST',
      KeySchema: [
        { AttributeName: 'PK', KeyType: 'HASH' },
        { AttributeName: 'SK', KeyType: 'RANGE' }
      ],
      AttributeDefinitions: Match.arrayWith([
        { AttributeName: 'PK', AttributeType: 'S' },
        { AttributeName: 'SK', AttributeType: 'S' }
      ])
    })
  })

  test('DynamoDB table has GSI1 for survey queries', () => {
    template.hasResourceProperties('AWS::DynamoDB::Table', {
      GlobalSecondaryIndexes: Match.arrayWith([
        Match.objectLike({
          IndexName: 'GSI1',
          KeySchema: [
            { AttributeName: 'GSI1_PK', KeyType: 'HASH' },
            { AttributeName: 'GSI1_SK', KeyType: 'RANGE' }
          ],
          Projection: { ProjectionType: 'ALL' }
        })
      ])
    })
  })

  test('DynamoDB table has DESTROY removal policy', () => {
    template.hasResource('AWS::DynamoDB::Table', {
      DeletionPolicy: 'Delete',
      UpdateReplacePolicy: 'Delete'
    })
  })

  test('Worker has TABLE_NAME environment variable', () => {
    template.hasResourceProperties('AWS::Lambda::Function', {
      FunctionName: 'phspectra__worker',
      Environment: {
        Variables: Match.objectLike({
          TABLE_NAME: Match.anyValue()
        })
      }
    })
  })

  test('Worker has dynamodb:UpdateItem and dynamodb:PutItem IAM policy', () => {
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['dynamodb:UpdateItem', 'dynamodb:PutItem']),
            Effect: 'Allow'
          })
        ])
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

  // -- Slow queue & slow worker tests ----------------------------------------

  test('slow queue has maxReceiveCount of 1', () => {
    template.hasResourceProperties('AWS::SQS::Queue', {
      QueueName: 'phspectra-slow-spectra',
      RedrivePolicy: Match.objectLike({
        maxReceiveCount: 1
      })
    })
  })

  test('Slow worker Lambda: ARM64, 1024MB, 15-minute timeout', () => {
    template.hasResourceProperties('AWS::Lambda::Function', {
      FunctionName: 'phspectra__slow_worker',
      Architectures: ['arm64'],
      MemorySize: 1024,
      Timeout: 900
    })
  })

  test('Slow worker event source: batchSize 1, maxConcurrency 10', () => {
    template.hasResourceProperties('AWS::Lambda::EventSourceMapping', {
      BatchSize: 1,
      ScalingConfig: {
        MaximumConcurrency: 10
      }
    })
  })

  test('Slow worker log group with correct name and 7-day retention', () => {
    template.hasResourceProperties('AWS::Logs::LogGroup', {
      LogGroupName: '/aws/lambda/phspectra__slow_worker',
      RetentionInDays: 7
    })
  })

  test('Worker has SLOW_QUEUE_URL environment variable', () => {
    template.hasResourceProperties('AWS::Lambda::Function', {
      FunctionName: 'phspectra__worker',
      Environment: {
        Variables: Match.objectLike({
          SLOW_QUEUE_URL: Match.anyValue()
        })
      }
    })
  })

  test('Worker has sqs:SendMessage IAM policy for slow queue', () => {
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['sqs:SendMessage']),
            Effect: 'Allow'
          })
        ])
      }
    })
  })
})
