import * as cdk from 'aws-cdk-lib'
import * as s3 from 'aws-cdk-lib/aws-s3'
import { Construct } from 'constructs'
import { DeploymentEnvironment } from 'utils/types'

export interface DataLakeStackProps extends cdk.StackProps {
  deploymentEnvironment: DeploymentEnvironment
}

export class DataLakeStack extends cdk.Stack {
  public readonly bucket: s3.Bucket

  constructor(scope: Construct, id: string, props: DataLakeStackProps) {
    super(scope, id, props)

    const isProd = props.deploymentEnvironment === 'production'

    this.bucket = new s3.Bucket(this, 'DataBucket', {
      bucketName: `phspectra-${props.deploymentEnvironment}-data`,
      eventBridgeEnabled: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      versioned: false,
      removalPolicy: isProd ? cdk.RemovalPolicy.RETAIN : cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: !isProd,
      lifecycleRules: [
        {
          id: 'expire-chunks',
          prefix: 'chunks/',
          expiration: cdk.Duration.days(7)
        },
        {
          id: 'expire-athena-results',
          prefix: 'athena-results/',
          expiration: cdk.Duration.days(7)
        }
      ]
    })
  }
}
