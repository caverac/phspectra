import * as cdk from 'aws-cdk-lib'
import { Template } from 'aws-cdk-lib/assertions'
import { AnalyticsStack } from 'lib/analytics.stack'
import { DataLakeStack } from 'lib/data-lake.stack'
import { GitHubOIDCStack } from 'lib/github-oidc.stack'
import { ProcessingStack } from 'lib/processing.stack'
import { ResourcesStack } from 'lib/resources.stack'
import { SplitterStack } from 'lib/splitter.stack'
import { DeploymentEnvironment } from 'utils/types'

export function createDataLakeTemplate(
  environment: DeploymentEnvironment = 'development'
): Template {
  const app = new cdk.App()
  const stack = new DataLakeStack(app, 'TestDataLake', {
    deploymentEnvironment: environment,
    env: { account: '123456789012', region: 'us-east-1' }
  })
  return Template.fromStack(stack)
}

export function createProcessingTemplate(
  environment: DeploymentEnvironment = 'development'
): Template {
  const app = new cdk.App()
  const dataLake = new DataLakeStack(app, 'TestDataLake', {
    deploymentEnvironment: environment,
    env: { account: '123456789012', region: 'us-east-1' }
  })
  const stack = new ProcessingStack(app, 'TestProcessing', {
    deploymentEnvironment: environment,
    bucket: dataLake.bucket,
    env: { account: '123456789012', region: 'us-east-1' }
  })
  return Template.fromStack(stack)
}

export function createSplitterTemplate(
  environment: DeploymentEnvironment = 'development'
): Template {
  const app = new cdk.App()
  const dataLake = new DataLakeStack(app, 'TestDataLake', {
    deploymentEnvironment: environment,
    env: { account: '123456789012', region: 'us-east-1' }
  })
  const processing = new ProcessingStack(app, 'TestProcessing', {
    deploymentEnvironment: environment,
    bucket: dataLake.bucket,
    env: { account: '123456789012', region: 'us-east-1' }
  })
  const stack = new SplitterStack(app, 'TestSplitter', {
    deploymentEnvironment: environment,
    bucket: dataLake.bucket,
    queue: processing.queue,
    table: processing.table,
    env: { account: '123456789012', region: 'us-east-1' }
  })
  return Template.fromStack(stack)
}

export function createAnalyticsTemplate(
  environment: DeploymentEnvironment = 'development'
): Template {
  const app = new cdk.App()
  const dataLake = new DataLakeStack(app, 'TestDataLake', {
    deploymentEnvironment: environment,
    env: { account: '123456789012', region: 'us-east-1' }
  })
  const stack = new AnalyticsStack(app, 'TestAnalytics', {
    deploymentEnvironment: environment,
    bucket: dataLake.bucket,
    env: { account: '123456789012', region: 'us-east-1' }
  })
  return Template.fromStack(stack)
}

export function createResourcesTemplate(
  environment: DeploymentEnvironment = 'development'
): Template {
  const app = new cdk.App()
  const stack = new ResourcesStack(app, 'TestResources', {
    deploymentEnvironment: environment,
    env: { account: '123456789012', region: 'us-east-1' }
  })
  return Template.fromStack(stack)
}

export function createGitHubOIDCTemplate(
  githubRepo = 'caverac/phspectra',
  allowedEnvironments?: DeploymentEnvironment[],
  existingProviderArn?: string
): Template {
  const app = new cdk.App()
  const stack = new GitHubOIDCStack(app, 'TestGitHubOIDC', {
    githubRepo,
    allowedEnvironments,
    existingProviderArn,
    env: { account: '123456789012', region: 'us-east-1' }
  })
  return Template.fromStack(stack)
}
