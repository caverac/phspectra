import * as cdk from 'aws-cdk-lib'
import { Construct } from 'constructs'

export interface PhspectraStackProps extends cdk.StackProps {
  deploymentEnvironment: 'development' | 'staging' | 'production'
}

export class PhspectraStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: PhspectraStackProps) {
    super(scope, id, props)

    cdk.Tags.of(this).add('Project', 'morse-smale-spectra')
    cdk.Tags.of(this).add('Environment', props.deploymentEnvironment)
  }
}
