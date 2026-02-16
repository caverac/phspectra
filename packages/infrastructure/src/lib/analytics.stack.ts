import * as cdk from 'aws-cdk-lib'
import * as athena from 'aws-cdk-lib/aws-athena'
import * as glue from 'aws-cdk-lib/aws-glue'
import * as s3 from 'aws-cdk-lib/aws-s3'
import { Construct } from 'constructs'
import { BetaComparisonQuery, RmsDistributionQuery } from 'utils/queries'
import { DeploymentEnvironment } from 'utils/types'

export interface AnalyticsStackProps extends cdk.StackProps {
  deploymentEnvironment: DeploymentEnvironment
  bucket: s3.IBucket
}

export class AnalyticsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: AnalyticsStackProps) {
    super(scope, id, props)

    const databaseName = 'phspectra'

    const database = new glue.CfnDatabase(this, 'Database', {
      catalogId: this.account,
      databaseInput: {
        name: databaseName
      }
    })

    const table = new glue.CfnTable(this, 'DecompositionsTable', {
      catalogId: this.account,
      databaseName,
      tableInput: {
        name: 'decompositions',
        tableType: 'EXTERNAL_TABLE',
        parameters: {
          classification: 'parquet',
          'projection.enabled': 'true',
          'projection.survey.type': 'enum',
          'projection.survey.values': 'grs,vgps,cgps,sgps,thor',
          'storage.location.template': `s3://${props.bucket.bucketName}/decompositions/survey=\${survey}/`
        },
        storageDescriptor: {
          location: `s3://${props.bucket.bucketName}/decompositions/`,
          inputFormat: 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
          outputFormat: 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
          serdeInfo: {
            serializationLibrary: 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
          },
          columns: [
            { name: 'x', type: 'int' },
            { name: 'y', type: 'int' },
            { name: 'beta', type: 'double' },
            { name: 'rms', type: 'double' },
            { name: 'min_persistence', type: 'double' },
            { name: 'n_components', type: 'int' },
            { name: 'component_amplitudes', type: 'array<double>' },
            { name: 'component_means', type: 'array<double>' },
            { name: 'component_stddevs', type: 'array<double>' }
          ]
        },
        partitionKeys: [{ name: 'survey', type: 'string' }]
      }
    })

    table.addDependency(database)

    const workgroup = new athena.CfnWorkGroup(this, 'Workgroup', {
      name: `phspectra`,
      recursiveDeleteOption: true,
      workGroupConfiguration: {
        resultConfiguration: {
          outputLocation: `s3://${props.bucket.bucketName}/athena-results/`
        },
        bytesScannedCutoffPerQuery: 10_737_418_240, // 10 GB
        enforceWorkGroupConfiguration: true,
        publishCloudWatchMetricsEnabled: true
      }
    })

    const betaQuery = new athena.CfnNamedQuery(this, 'BetaComparisonQuery', {
      database: databaseName,
      workGroup: workgroup.name,
      name: 'beta-component-count-comparison',
      description: 'Compare avg/median/std of component counts across beta values',
      queryString: BetaComparisonQuery
    })
    betaQuery.addDependency(workgroup)

    const rmsQuery = new athena.CfnNamedQuery(this, 'RmsDistributionQuery', {
      database: databaseName,
      workGroup: workgroup.name,
      name: 'rms-distribution-sanity-check',
      description: 'RMS distribution by survey to verify noise estimation',
      queryString: RmsDistributionQuery
    })
    rmsQuery.addDependency(workgroup)
  }
}
