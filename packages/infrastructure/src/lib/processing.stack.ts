import fs = require('fs')
import path = require('path')

import * as cdk from 'aws-cdk-lib'
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb'
import * as lambda from 'aws-cdk-lib/aws-lambda'
import * as lambdaEventSources from 'aws-cdk-lib/aws-lambda-event-sources'
import * as logs from 'aws-cdk-lib/aws-logs'
import * as s3 from 'aws-cdk-lib/aws-s3'
import * as sqs from 'aws-cdk-lib/aws-sqs'
import { Construct } from 'constructs'
import { DeploymentEnvironment } from 'utils/types'

export interface ProcessingStackProps extends cdk.StackProps {
  deploymentEnvironment: DeploymentEnvironment
  bucket: s3.IBucket
}

export class ProcessingStack extends cdk.Stack {
  public readonly queue: sqs.Queue
  public readonly table: dynamodb.Table

  constructor(scope: Construct, id: string, props: ProcessingStackProps) {
    super(scope, id, props)

    this.table = new dynamodb.Table(this, 'RunsTable', {
      tableName: `phspectra-runs`,
      partitionKey: { name: 'PK', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY
    })

    const deadLetterQueue = new sqs.Queue(this, 'DeadLetterQueue', {
      queueName: 'phspectra-dead-letter',
      retentionPeriod: cdk.Duration.days(14)
    })

    this.queue = new sqs.Queue(this, 'ChunkQueue', {
      queueName: 'phspectra-chunks',
      visibilityTimeout: cdk.Duration.minutes(16),
      retentionPeriod: cdk.Duration.days(14),
      deadLetterQueue: {
        queue: deadLetterQueue,
        maxReceiveCount: 3
      }
    })

    const workerLogGroup = new logs.LogGroup(this, 'WorkerLogGroup', {
      logGroupName: '/aws/lambda/phspectra__worker',
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY
    })

    // Copy phspectra source into the Docker context so it can be built
    // from source inside the container (C extension needs linux/arm64).
    const workerDir = path.join(__dirname, '..', '..', 'lambda', 'worker')
    const phspectraDir = path.resolve(workerDir, '..', '..', '..', 'phspectra')
    const vendorDir = path.join(workerDir, 'phspectra-src')

    // Copy only what pip needs to build from source
    fs.mkdirSync(vendorDir, { recursive: true })
    for (const name of ['pyproject.toml', 'setup.py', 'src']) {
      fs.cpSync(path.join(phspectraDir, name), path.join(vendorDir, name), {
        recursive: true,
        force: true
      })
    }

    const workerFn = new lambda.DockerImageFunction(this, 'WorkerFn', {
      functionName: 'phspectra__worker',
      code: lambda.DockerImageCode.fromImageAsset(workerDir),
      architecture: lambda.Architecture.ARM_64,
      memorySize: 1024,
      timeout: cdk.Duration.minutes(15),
      logGroup: workerLogGroup,
      environment: {
        BUCKET_NAME: props.bucket.bucketName,
        TABLE_NAME: this.table.tableName
      }
    })

    workerFn.addEventSource(
      new lambdaEventSources.SqsEventSource(this.queue, {
        batchSize: 1,
        maxConcurrency: 500
      })
    )

    props.bucket.grantReadWrite(workerFn)
    this.table.grant(workerFn, 'dynamodb:UpdateItem')
  }
}
