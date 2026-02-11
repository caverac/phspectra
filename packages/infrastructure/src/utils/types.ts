import { z } from 'zod'

export const DeploymentEnvironmentSchema = z.enum(['development', 'staging', 'production'])

export type DeploymentEnvironment = z.infer<typeof DeploymentEnvironmentSchema>

export const EnvironmentVariablesSchema = z.object({
  ENVIRONMENT: DeploymentEnvironmentSchema
})

export type EnvironmentVariables = z.infer<typeof EnvironmentVariablesSchema>
