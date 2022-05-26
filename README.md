# Amazon SageMaker Bring Your Own Scripts and Container

This git repo was developed based on the git repository [Amazon SageMaker Local Mode Examples](https://github.com/aws-samples/amazon-sagemaker-local-mode), which shows different examples of running SageMaker Training and Serving in local instance/machine.

In this git repo, we added examples of running the SageMaker Training and Hosting jobs on SageMaker Notebook Instances using **local** mode with some commonly used frameworks, e.g. Scikit Learn, XGBoost, bring your own algorithm and etc.

For more details, please refer to the original git repository or read the Machine Learning Blog post at: https://aws.amazon.com/blogs/machine-learning/use-the-amazon-sagemaker-local-mode-to-train-on-your-notebook-instance/



## Running Costs

If you are running the above example on your own computer/machine in local mode, which means the resources are not running in the cloud, then there should be very minimum. Please refer to the below listed items for detail regarding cost:

- **SageMaker** – Prices vary based on EC2 instance usage for the SageMaker notebook instances. Model Hosting and Model Training only occurs charge when you are running them in the remote instances (NOT local mode); each charged per hour of use. For more information, see [Amazon SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/).
  
- **S3** – Low cost, prices will vary depending on the size of the models/artifacts stored. The first 50 TB each month will cost only $0.023 per GB stored. For more information, see [Amazon S3 Pricing](https://aws.amazon.com/s3/pricing/).

- **ECR** – Low cost, you pay only for the amount of data you store in your public or private repositories. For more information, see [Amazon Elastic Container Registry pricing](https://aws.amazon.com/ecr/pricing/).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.