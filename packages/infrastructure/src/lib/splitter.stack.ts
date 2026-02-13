import path = require('path')

import * as cdk from 'aws-cdk-lib'
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb'
import * as events from 'aws-cdk-lib/aws-events'
import * as targets from 'aws-cdk-lib/aws-events-targets'
import * as lambda from 'aws-cdk-lib/aws-lambda'
import * as logs from 'aws-cdk-lib/aws-logs'
import * as s3 from 'aws-cdk-lib/aws-s3'
import * as sqs from 'aws-cdk-lib/aws-sqs'
import { Construct } from 'constructs'
import { DeploymentEnvironment } from 'utils/types'

export interface SplitterStackProps extends cdk.StackProps {
  deploymentEnvironment: DeploymentEnvironment
  bucket: s3.IBucket
  queue: sqs.IQueue
  table: dynamodb.ITable
}

export class SplitterStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: SplitterStackProps) {
    super(scope, id, props)

    cdk.Tags.of(this).add('Project', 'phspectra')
    cdk.Tags.of(this).add('Environment', props.deploymentEnvironment)

    const splitterLogGroup = new logs.LogGroup(this, 'SplitterLogGroup', {
      logGroupName: '/aws/lambda/phspectra__splitter',
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY
    })

    const splitterFn = new lambda.DockerImageFunction(this, 'SplitterFn', {
      functionName: 'phspectra__splitter',
      code: lambda.DockerImageCode.fromImageAsset(
        path.join(__dirname, '..', '..', 'lambda', 'splitter')
      ),
      architecture: lambda.Architecture.ARM_64,
      memorySize: 2048,
      timeout: cdk.Duration.minutes(15),
      logGroup: splitterLogGroup,
      environment: {
        BUCKET_NAME: props.bucket.bucketName,
        QUEUE_URL: props.queue.queueUrl,
        TABLE_NAME: props.table.tableName
      }
    })

    props.bucket.grantReadWrite(splitterFn)
    props.queue.grantSendMessages(splitterFn)
    props.table.grant(splitterFn, 'dynamodb:PutItem')

    // Rule 1: FITS file uploaded to cubes/ -> auto-process with default beta
    new events.Rule(this, 'FitsUploadRule', {
      eventPattern: {
        source: ['aws.s3'],
        detailType: ['Object Created'],
        detail: {
          bucket: { name: [props.bucket.bucketName] },
          object: { key: [{ suffix: '.fits' }] }
        }
      },
      targets: [new targets.LambdaFunction(splitterFn)]
    })

    // Rule 2: JSON manifest uploaded to manifests/ -> beta sweep
    new events.Rule(this, 'ManifestUploadRule', {
      eventPattern: {
        source: ['aws.s3'],
        detailType: ['Object Created'],
        detail: {
          bucket: { name: [props.bucket.bucketName] },
          object: { key: [{ prefix: 'manifests/' }, { suffix: '.json' }] }
        }
      },
      targets: [new targets.LambdaFunction(splitterFn)]
    })
  }
}
