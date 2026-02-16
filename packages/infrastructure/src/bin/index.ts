import * as cdk from 'aws-cdk-lib'
import { AnalyticsStack } from 'lib/analytics.stack'
import { DataLakeStack } from 'lib/data-lake.stack'
import { GitHubOIDCStack } from 'lib/github-oidc.stack'
import { ProcessingStack } from 'lib/processing.stack'
import { ResourcesStack } from 'lib/resources.stack'
import { SplitterStack } from 'lib/splitter.stack'
import { z } from 'zod'

const envSchema = z.object({
  ENVIRONMENT: z.enum(['development', 'production']),
  AWS_ACCOUNT: z.string(),
  AWS_DEFAULT_REGION: z.string().default('us-east-1')
})

const env = envSchema.parse(process.env)

const cdkEnv = {
  account: env.AWS_ACCOUNT,
  region: env.AWS_DEFAULT_REGION
}

const app = new cdk.App()

const dataLake = new DataLakeStack(app, 'PHSDataLake', {
  deploymentEnvironment: env.ENVIRONMENT,
  env: cdkEnv
})

const processing = new ProcessingStack(app, 'PHSProcessing', {
  deploymentEnvironment: env.ENVIRONMENT,
  bucket: dataLake.bucket,
  env: cdkEnv
})

new SplitterStack(app, 'PHSSplitter', {
  deploymentEnvironment: env.ENVIRONMENT,
  bucket: dataLake.bucket,
  queue: processing.queue,
  table: processing.table,
  env: cdkEnv
})

new AnalyticsStack(app, 'PHSAnalytics', {
  deploymentEnvironment: env.ENVIRONMENT,
  bucket: dataLake.bucket,
  env: cdkEnv
})

new ResourcesStack(app, 'PHSResources', {
  deploymentEnvironment: env.ENVIRONMENT,
  env: cdkEnv
})

new GitHubOIDCStack(app, 'PHSGitHubOIDC', {
  githubRepo: 'caverac/phspectra',
  existingProviderArn: `arn:aws:iam::${env.AWS_ACCOUNT}:oidc-provider/token.actions.githubusercontent.com`,
  env: cdkEnv
})
