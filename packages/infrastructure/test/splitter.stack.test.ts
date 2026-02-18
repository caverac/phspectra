import { Match } from 'aws-cdk-lib/assertions'

import { createSplitterTemplate } from './helpers'

describe('SplitterStack', () => {
  const template = createSplitterTemplate()

  test('Splitter Lambda: ARM64, 2048MB, 15-minute timeout', () => {
    template.hasResourceProperties('AWS::Lambda::Function', {
      FunctionName: 'phspectra__splitter',
      Architectures: ['arm64'],
      MemorySize: 2048,
      Timeout: 900
    })
  })

  test('Log group with correct name and 7-day retention', () => {
    template.hasResourceProperties('AWS::Logs::LogGroup', {
      LogGroupName: '/aws/lambda/phspectra__splitter',
      RetentionInDays: 7
    })
  })

  test('creates 1 EventBridge rule (manifest only)', () => {
    template.resourceCountIs('AWS::Events::Rule', 1)
  })

  test('Manifest rule matches manifests/ prefix and .json suffix', () => {
    template.hasResourceProperties('AWS::Events::Rule', {
      EventPattern: Match.objectLike({
        source: ['aws.s3'],
        'detail-type': ['Object Created'],
        detail: Match.objectLike({
          object: {
            key: [{ prefix: 'manifests/' }, { suffix: '.json' }]
          }
        })
      })
    })
  })

  test('Splitter has TABLE_NAME environment variable', () => {
    template.hasResourceProperties('AWS::Lambda::Function', {
      FunctionName: 'phspectra__splitter',
      Environment: {
        Variables: Match.objectLike({
          TABLE_NAME: Match.anyValue()
        })
      }
    })
  })

  test('Splitter has dynamodb:PutItem and dynamodb:BatchWriteItem IAM policy', () => {
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith(['dynamodb:PutItem', 'dynamodb:BatchWriteItem']),
            Effect: 'Allow'
          })
        ])
      }
    })
  })

  test('has Project=phspectra tag', () => {
    template.hasResourceProperties('AWS::Lambda::Function', {
      Tags: Match.arrayWith([Match.objectLike({ Key: 'Project', Value: 'phspectra' })])
    })
  })

  test('Splitter has bucket read/write IAM policy', () => {
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

  test('Splitter has queue sendMessage IAM policy', () => {
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith([
              'sqs:SendMessage',
              'sqs:GetQueueAttributes',
              'sqs:GetQueueUrl'
            ]),
            Effect: 'Allow'
          })
        ])
      }
    })
  })
})
