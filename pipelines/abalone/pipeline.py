"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.inputs import CreateModelInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    TrainingStep,
    CreateModelStep
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.huggingface import HuggingFace

from stepfunctions.steps import ProcessingStep

import time


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_name=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """

    
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )

    PREPROCESSING_SCRIPT_LOCATION = os.path.join(BASE_DIR, "preprocess.py")

    processing_code = sagemaker_session.upload_data(
        PREPROCESSING_SCRIPT_LOCATION,
        bucket=sagemaker_session.default_bucket(),
        key_prefix="code/processing",
    )
  
    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )

    raw_input_data_s3_uri = f"s3://{sagemaker_session.default_bucket()}/data/dataset.csv"
    
    input_data = ParameterString(
        name="InputData",
        default_value=raw_input_data_s3_uri,
    )

    processing_inputs=[
      ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
      ProcessingInput(
        source=processing_code,
        destination="/opt/ml/code/processing",
        input_name="code",
    ),
    ]
    processing_outputs=[
            ProcessingOutput(destination=f"s3://{sagemaker_session.default_bucket()}/data/train.csv", source="/opt/ml/processing/train"),
            ProcessingOutput(destination=f"s3://{sagemaker_session.default_bucket()}/data/validation.csv", source="/opt/ml/processing/validation"),
            ProcessingOutput(destination=f"s3://{sagemaker_session.default_bucket()}/data/test.csv", source="/opt/ml/processing/test"),
    ]
  
    step_process = ProcessingStep(
        name="PreprocessData",
        code="/opt/ml/code/processing/preprocessing.py",
        processor = sklearn_processor,
        inputs=processing_inputs,
        outputs=processing_outputs,
        #container_entrypoint=["python3", "/opt/ml/code/processing/preprocessing.py"]
    )
  
    # training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/model"
        
    train_est = HuggingFace(entry_point= "./train.py",
                            instance_type=training_instance_type,
                            instance_count=1,
                            role=role,
                            transformers_version='4.26',
                            pytorch_version='1.13',
                            py_version='py39',
                            output_path=model_path,
                            sagemaker_session=pipeline_session,
                            base_job_name=f"{base_job_prefix}/training", 
                           )
  
    training_step = TrainingStep(
      name="Train",
      estimator=train_est,
      inputs={
          "train": TrainingInput(
              s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
              content_type="text/csv",
          ),
          "validation": TrainingInput(
              s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
              content_type="text/csv",
          ),
          "test": TrainingInput(
              s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
              content_type="text/csv",
          ),
    }
)


    # processing step for evaluation
    evaluation_processor = SKLearnProcessor(
        framework_version="0.23-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        env={"AWS_DEFAULT_REGION": region},
        max_runtime_in_seconds=7200,
    )
  
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="metrics",
        path="evaluation.json",
    )
    
    evaluation_step = ProcessingStep(
        name="EvaluateModel",
        processor=evaluation_processor,
        code="./evaluate.py",
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/data/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="metrics", s3_upload_mode="EndOfJob", source="/opt/ml/processing/output/metrics/"
            ),
        ],
        property_files=[evaluation_report],
    )

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=evaluation_step.properties.ProcessingOutputConfig.Outputs["metrics"].S3Output.S3Uri,
            content_type="application/json"
        )
    )

    inference_image_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-cpu-py39-ubuntu20.04"

    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    
    register_step = RegisterModel(
        name="RegisterModel",
        #    entry_point='inference.py', # Adds a Repack Step:  https://github.com/aws/sagemaker-python-sdk/blob/01c6ee3a9ec1831e935e86df58cf70bc92ed1bbe/src/sagemaker/workflow/_utils.py#L44
        #    source_dir='src',
        estimator=train_est,
        image_uri=inference_image_uri,  # we have to specify, by default it's using training image
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/jsonlines"],
        response_types=["application/jsonlines"],
        inference_instances=[processing_instance_type],
        transform_instances=["ml.m4.xlarge"],
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    
    timestamp = int(time.time())
  
    model_name = "bert-model-{}".format(timestamp)
    
    model = Model(
        name=model_name,
        image_uri=inference_image_uri,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session ,
        role=role,
    )

    create_inputs = CreateModelInput(
        instance_type=processing_instance_type,
    )

    create_step = CreateModelStep(
        name="CreateModel",
        model=model,
        inputs=create_inputs,
    )

    
  
    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value"
        ),
        right=6.0,
    )
    step_cond = ConditionStep(
        name="CheckMSEAbaloneEvaluation",
        conditions=[cond_lte],
        if_steps=[register_step,create_step],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
        ],
        steps=[step_process, training_step, evaluation_step, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
