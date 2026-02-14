import {
  DeploymentEnvironmentSchema,
  EnvironmentVariablesSchema
} from 'utils/types'

describe('DeploymentEnvironmentSchema', () => {
  test.each(['development', 'staging', 'production'])('accepts "%s"', (env) => {
    expect(DeploymentEnvironmentSchema.parse(env)).toBe(env)
  })

  test('rejects invalid value', () => {
    expect(() => DeploymentEnvironmentSchema.parse('invalid')).toThrow()
  })
})

describe('EnvironmentVariablesSchema', () => {
  test('parses valid input', () => {
    const result = EnvironmentVariablesSchema.parse({ ENVIRONMENT: 'development' })
    expect(result.ENVIRONMENT).toBe('development')
  })

  test('rejects missing ENVIRONMENT', () => {
    expect(() => EnvironmentVariablesSchema.parse({})).toThrow()
  })
})
