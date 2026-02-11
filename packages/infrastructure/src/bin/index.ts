import * as cdk from 'aws-cdk-lib'
import { z } from 'zod'
import { PhspectraStack } from 'lib/phspectra.stack'

const envSchema = z.object({
  ENVIRONMENT: z.enum(['development', 'staging', 'production']),
  AWS_ACCOUNT: z.string().optional(),
  AWS_REGION: z.string().default('us-east-1')
})

const env = envSchema.parse(process.env)

const app = new cdk.App()

new PhspectraStack(app, `phspectra-${env.ENVIRONMENT}`, {
  deploymentEnvironment: env.ENVIRONMENT,
  env: {
    account: env.AWS_ACCOUNT ?? process.env.CDK_DEFAULT_ACCOUNT,
    region: env.AWS_REGION ?? process.env.CDK_DEFAULT_REGION
  }
})
