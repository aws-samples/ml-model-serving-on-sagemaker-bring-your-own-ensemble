# Amazon SageMaker Bring Your Own Ensemble

This git repo was developed based on the git repository [Amazon SageMaker Local Mode Examples](https://github.com/aws-samples/amazon-sagemaker-local-mode), which shows different examples of running SageMaker Training and Serving in local instance/machine.

In this git repo, we added examples of running the SageMaker Training and Hosting jobs using Jupyter Notebook (On SageMaker Studio, Notebook instance or your local environment) using **local** mode with some commonly used frameworks, e.g. Scikit Learn, XGBoost, bring your own algorithm and etc.

For more details, please refer to the original git repository or read the Machine Learning Blog post at: TBD

### Prerequisites:
To run this github code, you need to have an AWS account and an AWS Identity and Access Management (IAM) user. For instructions on how to set up an AWS account, see [How do I create and activate a new AWS account?](http://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/). For instructions on how to secure your account with an IAM administrator user, see [Creating your first IAM admin user and user group in the IAM User Guide](https://docs.aws.amazon.com/IAM/latest/UserGuide/getting-started_create-admin-group.html). If you are using [SageMaker Studio](https://aws.amazon.com/sagemaker/studio/), you can use the AWS managed policy [`AmazonSageMakerFullAccess`](https://docs.aws.amazon.com/sagemaker/latest/dg/security-iam-awsmanpol.html) to run this code example. 

You will also need an Amazon S3 bucket to store your training data and model artifacts. To learn how to create a bucket, see [Create your first S3 bucket](To learn how to create a bucket, see Create your first S3 bucket in the Amazon S3 User Guide.) in the Amazon S3 User Guide. If you don't provide your own bucket, the code will create a default SageMaker S3 bucket in the region where you run the notebook.

## Running Costs

If you are running the above example on your own computer/machine in local mode, which means the resources are not running in the cloud, then there should be very minimum. Please refer to the below listed items for detail regarding cost:

- **SageMaker** – Prices vary based on EC2 instance usage for the SageMaker notebook instances. Model Hosting and Model Training only occurs charge when you are running them in the remote instances (NOT local mode); each charged per hour of use. For more information, see [Amazon SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/).
  
- **S3** – Low cost, prices will vary depending on the size of the models/artifacts stored. The first 50 TB each month will cost only $0.023 per GB stored. For more information, see [Amazon S3 Pricing](https://aws.amazon.com/s3/pricing/).

It is important to delete resources after you finish experimenting with the example. We have the `Clean up` section in the sample notebook. Please uncomment and execute the code in the section to delete any SageMaker endpoint you have created. You also need to delete data that has been stored on the S3 bucket, you can do this from the S3 console. 

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
