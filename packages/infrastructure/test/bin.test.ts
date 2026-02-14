describe('bin/index.ts', () => {
  const originalEnv = process.env

  afterEach(() => {
    process.env = originalEnv
  })

  test('creates the CDK app with valid env vars', () => {
    process.env = {
      ...originalEnv,
      ENVIRONMENT: 'development',
      AWS_ACCOUNT: '123456789012',
      AWS_DEFAULT_REGION: 'us-east-1'
    }
    expect(() => {
      jest.isolateModules(() => {
        require('../src/bin/index')
      })
    }).not.toThrow()
  })

  test('uses default region when AWS_DEFAULT_REGION is omitted', () => {
    process.env = {
      ...originalEnv,
      ENVIRONMENT: 'production',
      AWS_ACCOUNT: '123456789012'
    }
    delete process.env.AWS_DEFAULT_REGION
    expect(() => {
      jest.isolateModules(() => {
        require('../src/bin/index')
      })
    }).not.toThrow()
  })

  test('throws with invalid ENVIRONMENT', () => {
    process.env = {
      ...originalEnv,
      ENVIRONMENT: 'invalid',
      AWS_ACCOUNT: '123456789012'
    }
    expect(() => {
      jest.isolateModules(() => {
        require('../src/bin/index')
      })
    }).toThrow()
  })

  test('throws when AWS_ACCOUNT is missing', () => {
    process.env = {
      ...originalEnv,
      ENVIRONMENT: 'development'
    }
    delete process.env.AWS_ACCOUNT
    expect(() => {
      jest.isolateModules(() => {
        require('../src/bin/index')
      })
    }).toThrow()
  })
})
