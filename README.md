## Jenkins CI/CD integration with SageMaker preprocessing, training and registration  

The template provides a starting point for bringing your SageMaker Pipeline development to production.

```
|-- CONTRIBUTING.md
|-- jenkins
|   |-- Jenkinsfile
|   `-- seed_job.groovy
|-- pipelines
|   |-- project
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   |-- preprocess.py
|   |   `-- train.py
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
|-- README.md
|-- sagemaker-pipelines-project.ipynb
|-- setup.cfg
|-- setup.py
|-- tests
|   `-- test_pipelines.py
`-- tox.ini
```

## Start here
This project focuses on implementing extractive text summarization using the Thext architecture, a state-of-the-art model for scientific paper summarization. I have fine-tuned this model on the CNN/Dailymail dataset from HuggingFace to adapt it for extractive summarization on a new dataset.

A description of some of the artifacts is provided below:
<br/><br/>
Your jenkins pipeline building and execution instructions. File `seed_job.groovy` contains the instructions for creation of jenkins pipeline using `Jenkinsfile`. `Jenkinsfile` contains pipeline definition for stages in jenkins pipeline, and you can modify it to add/delete/update stages in the pipeline. 

```
|-- jenkins
|   |-- Jenkinsfile
|   `-- seed_job.groovy
```

<br/><br/>
Your pipeline artifacts, which includes a pipeline module defining the required `get_pipeline` method that returns an instance of a SageMaker pipeline, a preprocessing script that is used in feature engineering, and a model evaluation script to measure the Mean Squared Error of the model that's trained by the pipeline. This is the core business logic, and if you want to create your own folder, you can do so, and implement the `get_pipeline` interface as illustrated here.

```
|-- pipelines
|   |-- project
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   |-- preprocess.py
|   |   `-- train.py

```
<br/><br/>
Utility modules for getting pipeline definition jsons and running pipelines :

```
|-- pipelines
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
```
### Organization of the pipeline
This project presents a comprehensive pipeline for extractive text summarization using the Thext architecture. The pipeline includes preprocessing, training, evaluation, and a condition step for model registration and deployment.
![Pipeline Diagram](/img/pipeline-full.png)

#### 1. Preprocessing Step

In this step, we prepare the dataset for training, validation, and testing:

- **Article Sentences**: The sentences from the articles are extracted and used as input data.
- **Associated Context**: The context of the articles is collected to provide additional information for the model.
- **Rouge Scores**: Relative Rouge scores between the extracted sentences and the expected summary are computed to serve as labels.

The dataset is then split into three sets: training, validation, and test.

#### 2. Training Step

In the training step, we use the Hugging Face Estimator to fine-tune the Thext model with the new dataset. The goal is to adapt the model's embeddings to the new domain of application. The Thext model is initialized with pretrained BERT weights and includes a fully connected layer for summarization.

#### 3. Evaluation Step

After training, the model is evaluated using a test set to assess its performance. We use standard extractive summarization metrics such as ROUGE and BLEU to measure how well the model extracts sentences that match the expected summary.

#### 4. Condition Step

In the condition step, we assess the results of the newly fine-tuned model. If the model achieves better performance compared to a predefined baseline or previous models, it is registered in the model registry. The model registry keeps track of model versions and serves as a repository for deploying models.

### Usage

Detailed instructions for using the pipeline, including commands and configurations, can be found in the project's documentation.

### Model Deployment

Once a model is registered in the model registry and meets the desired performance criteria, it can be deployed as an endpoint in AWS SageMaker. Deploying the model allows it to be accessible for automated or on-demand summarization tasks.

