import { Match } from 'aws-cdk-lib/assertions'

import { createAnalyticsTemplate } from './helpers'

describe('AnalyticsStack', () => {
  const template = createAnalyticsTemplate()

  test('Glue database named phspectra', () => {
    template.hasResourceProperties('AWS::Glue::Database', {
      DatabaseInput: {
        Name: 'phspectra'
      }
    })
  })

  test('Glue table named decompositions with EXTERNAL_TABLE type', () => {
    template.hasResourceProperties('AWS::Glue::Table', {
      TableInput: Match.objectLike({
        Name: 'decompositions',
        TableType: 'EXTERNAL_TABLE'
      })
    })
  })

  test('table has partition projection enabled', () => {
    template.hasResourceProperties('AWS::Glue::Table', {
      TableInput: Match.objectLike({
        Parameters: Match.objectLike({
          'projection.enabled': 'true',
          'projection.survey.type': 'enum',
          'projection.beta.type': 'decimal'
        })
      })
    })
  })

  test('table has 8 columns and 2 partition keys', () => {
    template.hasResourceProperties('AWS::Glue::Table', {
      TableInput: Match.objectLike({
        StorageDescriptor: Match.objectLike({
          Columns: Match.arrayWith([
            { Name: 'x', Type: 'int' },
            { Name: 'y', Type: 'int' },
            { Name: 'rms', Type: 'double' },
            { Name: 'min_persistence', Type: 'double' },
            { Name: 'n_components', Type: 'int' },
            { Name: 'component_amplitudes', Type: 'array<double>' },
            { Name: 'component_means', Type: 'array<double>' },
            { Name: 'component_stddevs', Type: 'array<double>' }
          ])
        }),
        PartitionKeys: [
          { Name: 'survey', Type: 'string' },
          { Name: 'beta', Type: 'string' }
        ]
      })
    })
  })

  test('table uses Parquet SerDe', () => {
    template.hasResourceProperties('AWS::Glue::Table', {
      TableInput: Match.objectLike({
        StorageDescriptor: Match.objectLike({
          SerdeInfo: {
            SerializationLibrary: 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
          }
        })
      })
    })
  })

  test('Athena workgroup named phspectra', () => {
    template.hasResourceProperties('AWS::Athena::WorkGroup', {
      Name: 'phspectra'
    })
  })

  test('Workgroup: 10GB cutoff, metrics enabled, enforce config', () => {
    template.hasResourceProperties('AWS::Athena::WorkGroup', {
      WorkGroupConfiguration: Match.objectLike({
        BytesScannedCutoffPerQuery: 10737418240,
        EnforceWorkGroupConfiguration: true,
        PublishCloudWatchMetricsEnabled: true
      })
    })
  })

  test('creates 2 named queries', () => {
    template.resourceCountIs('AWS::Athena::NamedQuery', 2)
  })

  test('named queries reference correct database and workgroup', () => {
    template.hasResourceProperties('AWS::Athena::NamedQuery', {
      Database: 'phspectra',
      WorkGroup: 'phspectra',
      Name: 'beta-component-count-comparison'
    })

    template.hasResourceProperties('AWS::Athena::NamedQuery', {
      Database: 'phspectra',
      WorkGroup: 'phspectra',
      Name: 'rms-distribution-sanity-check'
    })
  })
})
